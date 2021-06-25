'''
  @ Date: 2021-06-04 21:34:19
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-05 16:26:06
  @ FilePath: /EasyMocapRelease/easymocap/affinity/ray.py
'''
import numpy as np
from .plucker import computeRay, dist_ll_pointwise_conf

class Affinity:
    def __init__(self, cameras, cams, MAX_DIST) -> None:
        self.cameras = cameras
        self.cams = cams
        self.MAX_DIST = MAX_DIST
    
    def __call__(self, annots, dimGroups):
        # calculate the ray
        nViews = len(annots)
        distance = np.zeros((dimGroups[-1], dimGroups[-1])) + self.MAX_DIST*2

        lPluckers = []
        for nv, annot in enumerate(annots):
            cam = self.cameras[self.cams[nv]]
            pluckers = []
            for det in annot:
                lines = computeRay(det['keypoints'][None, :15, :], 
                    cam['invK'], cam['R'], cam['T'])[0]
                pluckers.append(lines)
            if len(pluckers) > 0:
                pluckers = np.stack(pluckers)
            lPluckers.append(pluckers)
        for nv0 in range(nViews-1):
            for nv1 in range(nv0+1, nViews):
                if dimGroups[nv0]==dimGroups[nv0+1] or dimGroups[nv1]==dimGroups[nv1+1]:
                    continue
                p0 = lPluckers[nv0][:, None]
                p1 = lPluckers[nv1][None, :]
                dist = dist_ll_pointwise_conf(p0, p1)
                distance[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]] = dist
                distance[dimGroups[nv1]:dimGroups[nv1+1], dimGroups[nv0]:dimGroups[nv0+1]] = dist.T
        distance[distance > self.MAX_DIST] = self.MAX_DIST
        affinity = 1 - distance / self.MAX_DIST
        return affinity