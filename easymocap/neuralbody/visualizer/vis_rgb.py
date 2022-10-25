'''
  @ Date: 2021-09-03 17:44:26
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-03 17:44:26
  @ FilePath: /EasyMocap/easymocap/neuralbody/visualizer/rgb.py
'''
import numpy as np
import cv2
from termcolor import colored
import os
from os.path import join
from ..dataset.utils_reader import palette

colors_rgb = [
    (1, 1, 1),
    (94/255, 124/255, 226/255), # 青色
    (255/255, 200/255, 87/255), # yellow
    (74/255.,  189/255.,  172/255.), # green
    (8/255, 76/255, 97/255), # blue
    (219/255, 58/255, 52/255), # red
    (77/255, 40/255, 49/255), # brown
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (110/255, 211/255, 207/255), # light green
    (1, 1, 1),
    (94/255, 124/255, 226/255), # 青色
    (255/255, 200/255, 87/255), # yellow
    (74/255.,  189/255.,  172/255.), # green
    (8/255, 76/255, 97/255), # blue
    (219/255, 58/255, 52/255), # red
    (77/255, 40/255, 49/255), # brown
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (110/255, 211/255, 207/255), # light green
    (1, 1, 1),
]
def get_rgb_01(pid):
    return colors_rgb[pid][::-1]

class BaseVisualizer:
    def __init__(self, out, **kwargs) -> None:
        self.data_dir = out
    
    def write_image(self, imgname, image):
        os.makedirs(os.path.dirname(imgname), exist_ok=True)
        image = (np.clip(image, 0., 1.)*255).astype(np.uint8)
        image = image[..., ::-1]
        cv2.imwrite(imgname, image)
    
    def write_01map(self, imgname, image):
        assert len(image.shape) == 2, image.shape
        pred = (np.clip(image, 0, 1.) * 255).astype(np.uint8)
        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        cv2.imwrite(imgname, pred)

    def __call__(self, output, batch):
        basename = '{:06d}'.format(output['meta']['index'].item())
        imgname = join(self.data_dir, 'rgb_map_{}.jpg'.format(basename))
        self.write_image(imgname, output['rgb_map'])
        imgname = join(self.data_dir, 'acc_map_{}.jpg'.format(basename))
        self.write_01map(imgname, output['acc_map'])
    
    def get_blank(self, batch):
        H, W = batch['meta']['H'], batch['meta']['W']
        res = np.zeros((H, W, 3))
        return res
    
    def to_numpy(self, tensor):
        return tensor[0].detach().cpu().numpy()

class EvalVisualizer(BaseVisualizer):        
    def __call__(self, output, batch):
        basename = '{:06d}'.format(batch['meta']['index'].item())
        coord = batch['coord'][0].detach().cpu().numpy()
        rgb = self.to_numpy(batch['rgb'])
        rgb_map = self.to_numpy(output['rgb_map'])
        gt_all = self.get_blank(batch)
        out_all = self.get_blank(batch)
        gt_all[coord[:, 0], coord[:, 1]] = rgb
        out_all[coord[:, 0], coord[:, 1]] = rgb_map
        imgname = join(self.data_dir, 'rgb_{}.jpg'.format(basename))
        self.write_image(imgname, np.vstack([gt_all, out_all]))

class Visualizer:
    def __init__(self, out, H=-1, W=-1, depth_range=[0.1, 10.], format='index',
        back_col=[0., 0., 0.],
        subs=[],
        keys=['rgb_map'], concat='none') -> None:
        self.data_dir = out
        # os.makedirs(out, exist_ok=True)
        print(colored('the results are saved at {}'.format(self.data_dir),
                      'yellow'))
        self.H = H
        self.W = W
        self.depth_range = depth_range
        self.keys = keys
        self.concat = concat
        self.back_col = np.array(back_col).reshape(1, 1, 3)
        self.format = format
        self.subs = subs

    @staticmethod
    def to_numpy(output):
        for key in output.keys():
            if 'cache' in key:continue
            output[key] = output[key][0].detach().cpu().numpy()
        return output

    def __call__(self, output, batch):
        os.makedirs(self.data_dir, exist_ok=True)
        keys = output.pop('keys', [])
        output = self.to_numpy(output)
        H, W = batch['meta']['H'], batch['meta']['W']
        coord = batch['coord'][0].detach().cpu().numpy()
        if self.format == 'index':
            basename = '{:06d}'.format(batch['meta']['index'].item())
        elif self.format == 'nvnf':
            basename = '{:02d}_{:06d}'.format(
                batch['meta']['nview'].item(), batch['meta']['nframe'].item())
        elif self.format == 'nvnfnp':
            basename = '{}_{}_{}'.format(
                batch['meta']['nview'].item(),
                batch['meta']['nframe'].item(),
                batch['meta']['pid'].item()
            )
        elif self.format == 'index_np':
            basename = '{:06d}_{}'.format(
                batch['meta']['index'].item(),
                batch['meta']['pid'].item(),
            )
        elif self.format == 'camnf':
            basename = '{}_{:06d}'.format(
                self.subs[batch['meta']['nview'].item()], batch['meta']['nframe'].item())
        elif self.format == 'camnfnp':
            basename = '{}_{:06d}_{}'.format(
                self.subs[batch['meta']['nview'].item()], batch['meta']['nframe'].item(), batch['meta']['pid'].item())
        else:
            raise NotImplementedError
        msk_all = None
        for key_ in batch['meta']['object_keys']:
            key = key_[0]
            if msk_all is None:
                msk_all = batch[key+'_mask'].detach().cpu().numpy()
            else:
                msk_all = msk_all & batch[key+'_mask'].detach().cpu().numpy()
        # coord = coord[msk_all[0]]
        outputs = {}
        for key in ['rgb', 'rgb_map']:
            if key not in self.keys:continue
            res = np.zeros((H, W, 3)) + self.back_col
            if key == 'rgb':
                out_ = batch[key][0].detach().cpu().numpy()
            else:
                out_ = output[key]
            res[coord[:, 0], coord[:, 1]] = out_
            if False:
                res[coord[:, 0], coord[:, 1]] = out_ * output['acc_map'][:, None] + res[coord[:, 0], coord[:, 1]] *(1-output['acc_map'][:, None])
            pred = (np.clip(res, 0, 1.) * 255).astype(np.uint8)
            pred = pred[..., [2, 1, 0]]
            outputs[key] = pred
        if 'human_0' in keys:
            key_index_offset = keys.index('human_0') + 1
        else:
            key_index_offset = 2
        for key in ['instance_map']:
            if key not in self.keys:continue
            if key not in output.keys():continue

            val = np.hstack([output[key], 1.-output[key].sum(axis=-1, keepdims=True)])
            
            argmax = val.argmax(axis=1)
            THRESHOLD = 0.0
            valid = val[np.arange(0, argmax.shape[0]), argmax] > THRESHOLD
            res = np.zeros((H, W), dtype=np.float32)
            res[coord[valid, 0], coord[valid, 1]] = argmax[valid] + key_index_offset
            res[res==val.shape[1]] = 0
            outputs[key] = res * 50
            # visualize
            colors = []
            for i in range(0, val.shape[1] - 1):
                colors.append(get_rgb_01(i - key_index_offset + 2))
            colors.append(get_rgb_01(0))
            # (N, 3)
            colors = np.array(colors)
            if False:
                res = np.zeros((H, W), dtype=np.int)
                res[coord[:, 0], coord[:, 1]] = output[key].argmax(axis=-1)
                pred = (res * 40).astype(np.uint8)
                pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            for name in ['feat_map', 'feat_argmax_map']:
                if name == 'feat_map':
                    color = val @ colors
                else:
                    color = colors[val.argmax(axis=-1)]
                res = np.ones((H, W, 3), dtype=np.float32)
                res[coord[:, 0], coord[:, 1]] = color
                empty = np.linalg.norm(output[key], axis=-1) < 0.01
                res[coord[empty, 0], coord[empty, 1]] = colors[0]
                pred = (res * 255).astype(np.uint8)
                outputs[name] = pred
        for key in ['semantic_map']:
            if key not in self.keys:continue
            if key not in output.keys():continue
            background = 1. - output[key].sum(axis=-1, keepdims=True)
            semantic = np.hstack([background, output[key]])
            res = np.ones((H, W, 3), dtype=np.float32)
            # color = palette[[0, 17, 5, 9]][:semantic.shape[-1]]/255.
            color = np.stack([get_rgb_01(0), get_rgb_01(1), get_rgb_01(2), get_rgb_01(3)])[:semantic.shape[-1]]
            color[0] = 1.
            color = semantic @ color
            res[coord[:, 0], coord[:, 1]] = color
            pred = (res * 255).astype(np.uint8)
            outputs[key] = pred
        for key in ['acc_map', 'human_0_acc_map', 'disp_map']:
            if key not in self.keys:continue
            if key not in output.keys():continue
            res = np.zeros((H, W))
            maxs = output[key].shape[0]
            res[coord[:maxs, 0], coord[:maxs, 1]] = output[key]
            pred = (np.clip(res, 0, 1.) * 255).astype(np.uint8)
            pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            outputs[key] = pred
        depth_min, depth_max = self.depth_range
        for key in ['depth_map']:
            if key not in self.keys:continue
            res = np.zeros((H, W))
            maxs = output[key].shape[0]
            val = (output[key] - depth_min)/(depth_max-depth_min)
            res[coord[:maxs, 0], coord[:maxs, 1]] = val
            pred = (np.clip(res, 0, 1.) * 255).astype(np.uint8)
            pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            outputs[key] = pred
        for key in ['raw_depth']:
            if key not in self.keys:continue
            res = np.zeros((H, W), dtype=np.uint16)
            val = (output['depth_map'] * 1000).astype(np.uint16)
            maxs = output['depth_map'].shape[0]
            res[coord[:maxs, 0], coord[:maxs, 1]] = val
            outputs[key] = res
        if self.concat == 'none':
            for key, pred in outputs.items():
                outname = join(self.data_dir, key+'_'+basename)
                if key in ['raw_depth', 'instance_map']:
                    outname += '.png'
                else:
                    outname += '.jpg'
                cv2.imwrite(outname, pred)
            return 0
        elif self.concat == 'hstack':
            outputs = np.hstack(list(outputs.values()))
            outname = join(self.data_dir, 'concat' + '_' + basename + '.jpg')
            cv2.imwrite(outname, outputs)
