import os
import numpy as np
import math
import cv2
import torch
from ..basetopdown import BaseTopDownModelCache
from .hrnet import HRNet

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim: {}'.format(batch_heatmaps.shape)

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
pairs = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]
def coco17tobody25(points2d):
    kpts = np.zeros((points2d.shape[0], 25, 3))
    kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
    kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
    # 需要交换一下
    # kpts = kpts[:, :, [1,0,2]]
    return kpts

class MyHRNet(BaseTopDownModelCache):
    def __init__(self, ckpt):
        super().__init__(name='hand2d', bbox_scale=1.25, res_input=[288, 384])
        model = HRNet(48, 17, 0.1)
        if not os.path.exists(ckpt) and ckpt.endswith('pose_hrnet_w48_384x288.pth'):
            url = "11ezQ6a_MxIRtj26WqhH3V3-xPI3XqYAw"
            text = '''Download `models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth` from (OneDrive)[https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ],
            And place it into {}'''.format(os.path.dirname(ckpt))
            print(text)
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            cmd = 'gdown "{}" -O {}'.format(url, ckpt)
            print('\n', cmd, '\n')
            os.system(cmd)
        assert os.path.exists(ckpt), f'{ckpt} not exists'
        checkpoint = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    @staticmethod
    def get_max_preds(batch_heatmaps):
        coords, maxvals = get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # post-processing
        if True:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = batch_heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                        diff = np.array(
                            [
                                hm[py][px+1] - hm[py][px-1],
                                hm[py+1][px]-hm[py-1][px]
                            ]
                        )
                        coords[n][p] += np.sign(diff) * .25
        coords = coords.astype(np.float32) * 4
        pred = np.dstack((coords, maxvals))
        return pred

    def __call__(self, bbox, images, imgnames):
        squeeze = False
        if not isinstance(images, list):
            images = [images]
            imgnames = [imgnames]
            bbox = [bbox]
            squeeze = True
        nViews = len(images)
        kpts_all = []
        for nv in range(nViews):
            _bbox = bbox[nv]
            if _bbox.shape[0] == 0:
                kpts_all.append(np.zeros((17, 3)))
                continue
            img = images[nv]
            # TODO: add flip test
            out = super().__call__(_bbox, img, imgnames[nv])
            output = out['params']['output']
            kpts = self.get_max_preds(output)
            kpts_ori = self.batch_affine_transform(kpts, out['params']['inv_trans'])
            kpts = np.concatenate([kpts_ori, kpts[..., -1:]], axis=-1)
            kpts = coco17tobody25(kpts)
            if len(kpts.shape) == 3:
                kpts = kpts[0]
            kpts_all.append(kpts)
        kpts_all = np.stack(kpts_all)
        if squeeze:
            kpts_all = kpts_all[0]
        return {
            'keypoints': kpts_all
        }