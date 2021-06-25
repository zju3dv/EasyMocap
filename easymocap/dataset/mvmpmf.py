'''
  @ Date: 2021-01-13 17:15:46
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-05 19:29:33
  @ FilePath: /EasyMocap/easymocap/dataset/mvmpmf.py
'''
from .base import MVBase
from ..mytools.file_utils import get_bbox_from_pose

class MVMPMF(MVBase):
    """ Dataset for multi-view, multiperson, multiframe.
    This class is compatible with single-view, multiperson, multiframe if use specify only one `cams`
    """
    def __init__(self, root, cams=[], out=None, config={}, 
        image_root='images', annot_root='annots', kpts_type='body25',
        undis=False, no_img=False, filter2d=None) -> None:
        super().__init__(root, cams, out, config, image_root, annot_root, 
            kpts_type=kpts_type, undis=undis, no_img=no_img, filter2d=filter2d)
    
    def write_keypoints3d(self, peopleDict, nf):
        results = []
        for pid, people in peopleDict.items():
            result = {'id': pid, 'keypoints3d': people.keypoints3d}
            results.append(result)
        super().write_keypoints3d(results, nf)
    
    def vis_repro(self, images, peopleDict, nf, sub_vis=[]):
        lDetections = []
        for nv in range(len(images)):
            res = []
            for pid, people in peopleDict.items():
                det = {
                    'id': people.id,
                    'keypoints2d': people.kptsRepro[nv],
                    'bbox': get_bbox_from_pose(people.kptsRepro[nv], images[nv])
                }
                res.append(det)
            lDetections.append(res)
        super().vis_detections(images, lDetections, nf, mode='repro', sub_vis=sub_vis)

    def __getitem__(self, index: int):
        images, annots_all = super().__getitem__(index)
        # 筛除不需要的2d
        return images, annots_all

    def __len__(self) -> int:
        return self.nFrames