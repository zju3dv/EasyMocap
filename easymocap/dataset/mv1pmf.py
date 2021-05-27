'''
  @ Date: 2021-01-12 17:12:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-27 20:25:24
  @ FilePath: /EasyMocap/easymocap/dataset/mv1pmf.py
'''
from ..mytools.file_utils import get_bbox_from_pose
from os.path import join
import numpy as np
from os.path import join
from .base import MVBase

class MV1PMF(MVBase):
    def __init__(self, root, cams=[], pid=0, out=None, config={}, 
        image_root='images', annot_root='annots', kpts_type='body15',
        undis=True, no_img=False, verbose=False) -> None:
        super().__init__(root=root, cams=cams, out=out, config=config, 
            image_root=image_root, annot_root=annot_root, 
            kpts_type=kpts_type, undis=undis, no_img=no_img)
        self.pid = pid
        self.verbose = verbose

    def write_keypoints3d(self, keypoints3d, nf):
        results = [{'id': self.pid, 'keypoints3d': keypoints3d}]
        super().write_keypoints3d(results, nf)

    def write_smpl(self, params, nf, mode='smpl'):
        result = {'id': 0}
        result.update(params)
        super().write_smpl([result], nf, mode)

    def vis_smpl(self, vertices, faces, images, nf, sub_vis=[], 
        mode='smpl', extra_data=[], add_back=True):
        outname = join(self.out, 'smpl', '{:06d}.jpg'.format(nf))
        render_data = {}
        assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
        pid = self.pid
        render_data[pid] = {'vertices': vertices, 'faces': faces, 
            'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        cameras = {'K': [], 'R':[], 'T':[]}
        if len(sub_vis) == 0:
            sub_vis = self.cams
        for key in cameras.keys():
            cameras[key] = np.stack([self.cameras[cam][key] for cam in sub_vis])
        images = [images[self.cams.index(cam)] for cam in sub_vis]
        self.writer.vis_smpl(render_data, images, cameras, outname, add_back=add_back)
    
    def vis_detections(self, images, annots, nf, to_img=True, sub_vis=[]):
        lDetections = []
        for nv in range(len(images)):
            det = {
                'id': self.pid,
                'bbox': annots['bbox'][nv],
                'keypoints2d': annots['keypoints'][nv]
            }
            lDetections.append([det])
        return super().vis_detections(images, lDetections, nf, sub_vis=sub_vis)

    def vis_repro(self, images, kpts_repro, nf, to_img=True, sub_vis=[], mode='repro'):
        lDetections = []
        for nv in range(len(images)):
            det = {
                'id': -1,
                'keypoints2d': kpts_repro[nv],
                'bbox': get_bbox_from_pose(kpts_repro[nv], images[nv])
            }
            lDetections.append([det])
        return super().vis_detections(images, lDetections, nf, mode=mode, sub_vis=sub_vis)

    def __getitem__(self, index: int):
        images, annots_all = super().__getitem__(index)
        annots = self.select_person(annots_all, index, self.pid)
        return images, annots
    

if __name__ == "__main__":
    root = '/home/qian/zjurv2/mnt/data/ftp/Human/vis/lightstage/CoreView_302_sync/'
    dataset = MV1PMF(root)
    images, annots = dataset[0]