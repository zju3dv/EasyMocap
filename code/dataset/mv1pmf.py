'''
  @ Date: 2021-01-12 17:12:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-21 14:51:45
  @ FilePath: /EasyMocap/code/dataset/mv1pmf.py
'''
import os
import ipdb
import torch
from os.path import join
import numpy as np
import cv2
from .base import MVBase

class MV1PMF(MVBase):
    def __init__(self, root, cams=[], pid=0, out=None, config={}, 
        image_root='images', annot_root='annots', mode='body15',
        undis=True, no_img=False) -> None:
        super().__init__(root, cams, out, config, image_root, annot_root, 
            mode, undis, no_img)
        self.pid = pid
    
    def write_keypoints3d(self, keypoints3d, nf):
        results = [{'id': 0, 'keypoints3d': keypoints3d.tolist()}]
        self.writer.write_keypoints3d(results, nf)
        
    def write_smpl(self, params, nf, images=[], to_img=False):
        result = {'id': 0}
        result.update(params)
        self.writer.write_smpl([result], nf)

    def vis_smpl(self, vertices, faces, images, nf, sub_vis=[], 
        mode='smpl', extra_data=[], add_back=True):
        render_data = {}
        if len(vertices.shape) == 3:
            vertices = vertices[0]
        pid = self.pid
        render_data[pid] = {'vertices': vertices, 'faces': faces, 
            'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        cameras = {'K': [], 'R':[], 'T':[]}
        if len(sub_vis) == 0:
            sub_vis = self.cams
        for key in cameras.keys():
            cameras[key] = [self.cameras[cam][key] for cam in sub_vis]
        images = [images[self.cams.index(cam)] for cam in sub_vis]
        self.writer.vis_smpl(render_data, nf, images, cameras, mode, add_back=add_back)
    
    def vis_detections(self, images, annots, nf, to_img=True, sub_vis=[]):
        lDetections = []
        for nv in range(len(images)):
            det = {
                'id': self.pid,
                'bbox': annots['bbox'][nv],
                'keypoints': annots['keypoints'][nv]
            }
            lDetections.append([det])
        if len(sub_vis) != 0:
            valid_idx = [self.cams.index(i) for i in sub_vis]
            images = [images[i] for i in valid_idx]
            lDetections = [lDetections[i] for i in valid_idx]
        return self.writer.vis_detections(images, lDetections, nf, 
            key='keypoints', to_img=to_img, vis_id=False)

    def vis_repro(self, images, annots, kpts_repro, nf, to_img=True, sub_vis=[]):
        lDetections = []
        for nv in range(len(images)):
            det = {
                'id': -1,
                'repro': kpts_repro[nv]
            }
            lDetections.append([det])
        if len(sub_vis) != 0:
            valid_idx = [self.cams.index(i) for i in sub_vis]
            images = [images[i] for i in valid_idx]
            lDetections = [lDetections[i] for i in valid_idx]
        return self.writer.vis_detections(images, lDetections, nf, key='repro',
            to_img=to_img, vis_id=False)

    def __getitem__(self, index: int):
        images, annots_all = super().__getitem__(index)
        annots = {'bbox': [], 'keypoints': []}
        for nv, cam in enumerate(self.cams):
            data = [d for d in annots_all[nv] if d['id'] == self.pid]
            if len(data) == 1:
                data = data[0]
                bbox = data['bbox']
                keypoints = data['keypoints']
            else:
                print('not found pid {} in {}, {}'.format(self.pid, index, nv))
                if self.add_hand_face:
                    keypoints = np.zeros((137, 3))
                else:
                    keypoints = np.zeros((25, 3))
                bbox = np.array([0, 0, 100., 100., 0.])
            annots['bbox'].append(bbox)
            annots['keypoints'].append(keypoints)
        for key in ['bbox', 'keypoints']:
            annots[key] = np.stack(annots[key])
        return images, annots


if __name__ == "__main__":
    root = '/home/qian/zjurv2/mnt/data/ftp/Human/vis/lightstage/CoreView_302_sync/'
    dataset = MV1PMF(root)
    images, annots = dataset[0]