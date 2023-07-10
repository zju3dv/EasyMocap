import os
from os.path import join
import numpy as np
import cv2
import torch
import torch.nn as nn
import pickle
import math

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, inv_trans

# TODO: add UDP
def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix

def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):

    trans, inv_trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(cvimg, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img_patch, trans, inv_trans

def get_single_image_crop_demo(image, bbox, scale=1.2, crop_size=224,
                               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], fliplr=False):

    crop_image, trans, inv_trans = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=crop_size[0],
        patch_height=crop_size[1],
        do_flip=False,
        scale=scale,
        rot=0,
    )
    if fliplr:
        crop_image = cv2.flip(crop_image, 1)
    # cv2.imwrite('debug_crop.jpg', crop_image[:,:,::-1])
    # cv2.imwrite('debug_crop_full.jpg', image[:,:,::-1])
    crop_image = crop_image.transpose(2,0,1)
    mean1=np.array(mean, dtype=np.float32).reshape(3,1,1)
    std1= np.array(std, dtype=np.float32).reshape(3,1,1)
    crop_image = (crop_image.astype(np.float32))/255.
    # _max = np.max(abs(crop_image))
    # crop_image = np.divide(crop_image, _max)
    crop_image = (crop_image - mean1)/std1

    return crop_image, inv_trans

def xyxy2ccwh(bbox):
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    cx = (bbox[:, 2] + bbox[:, 0])/2
    cy = (bbox[:, 3] + bbox[:, 1])/2
    return np.stack([cx, cy, w, h], axis=1)

class BaseTopDownModel(nn.Module):
    def __init__(self, bbox_scale, res_input,
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.bbox_scale = bbox_scale
        if not isinstance(res_input, list):
            res_input = [res_input, res_input]
        self.crop_size = res_input
        self.mean = mean
        self.std = std

    def load_checkpoint(self, model, state_dict, prefix, strict):
        state_dict_new = {}
        for key, val in state_dict.items():
            if key.startswith(prefix):
                key_new = key.replace(prefix, '')
                state_dict_new[key_new] = val
        model.load_state_dict(state_dict_new, strict=strict)

    def infer(self, image, bbox, to_numpy=False, flips=None):
        if isinstance(image, str):
            image = cv2.imread(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        squeeze = False
        if len(bbox.shape) == 1:
            bbox = bbox[None]
            squeeze = True
        # TODO: 兼容多张图片的
        bbox = xyxy2ccwh(bbox)
        # convert the bbox to the aspect of input bbox
        aspect_ratio = self.crop_size[1] / self.crop_size[0]
        w, h = bbox[:, 2], bbox[:, 3]
        # 如果height大于w*ratio，那么增大w
        flag = h > aspect_ratio * w
        bbox[flag, 2] = h[flag] / aspect_ratio
        # 否则增大h
        bbox[~flag, 3] = w[~flag] * aspect_ratio
        inputs = []
        inv_trans_ = []
        for i in range(bbox.shape[0]):
            if flips is None:
                fliplr=False
            else:
                fliplr=flips[i]
            norm_img, inv_trans = get_single_image_crop_demo(
                img,
                bbox[i],
                scale=self.bbox_scale,
                crop_size=self.crop_size,
                mean=self.mean,
                std=self.std,
                fliplr=fliplr
            )
            inputs.append(norm_img)
            inv_trans_.append(inv_trans)
        if False:
            vis = np.hstack(inputs)
            mean, std = np.array(self.mean), np.array(self.std)
            mean = mean.reshape(3, 1, 1)
            std = std.reshape(3, 1, 1)
            vis = (vis * std) + mean
            vis = vis.transpose(1, 2, 0)
            vis = (vis[:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite('debug_crop.jpg', vis)
        inputs = np.stack(inputs)
        inv_trans_ = np.stack(inv_trans_)
        inputs = torch.FloatTensor(inputs).to(self.device)
        with torch.no_grad():
            output = self.model(inputs)
        if squeeze:
            for key, val in output.items():
                output[key] = val[0]
        if to_numpy:
            for key, val in output.items():
                if torch.is_tensor(val):
                    output[key] = val.detach().cpu().numpy()
        output['inv_trans'] = inv_trans_
        return output

    @staticmethod
    def batch_affine_transform(points, trans):
        # points: (Bn, J, 2), trans: (Bn, 2, 3)
        points = np.dstack((points[..., :2], np.ones((*points.shape[:-1], 1))))
        out = np.matmul(points, trans.swapaxes(-1, -2))
        return out

class BaseTopDownModelCache(BaseTopDownModel):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name
    
    def cachename(self, imgname):
        basename = os.sep.join(imgname.split(os.sep)[-2:])
        cachename = join(self.output, self.name, basename.replace('.jpg', '.pkl'))
        return cachename

    def dump(self, cachename, output):
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        with open(cachename, 'wb') as f:
            pickle.dump(output, f)
        return output
    
    def load(self, cachename):
        with open(cachename, 'rb') as f:
            output = pickle.load(f)
        return output

    def __call__(self, bbox, images, imgname, flips=None):
        cachename = self.cachename(imgname)
        if os.path.exists(cachename):
            output = self.load(cachename)
        else:
            output = self.infer(images, bbox, to_numpy=True, flips=flips)
            output = self.dump(cachename, output)

        ret = {
            'params': output
        }
        return ret

# post processing
def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

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

def get_preds_from_heatmaps(batch_heatmaps):
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

def gdown_models(ckpt, url):
    print('Try to download model from {} to {}'.format(url, ckpt))
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    cmd = 'gdown "{}" -O {}'.format(url, ckpt)
    print('\n', cmd, '\n')
    os.system(cmd)