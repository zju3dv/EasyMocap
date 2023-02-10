'''
  @ Date: 2021-06-04 20:40:12
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-04 21:43:45
  @ FilePath: /EasyMocapRelease/easymocap/affinity/affinity.py
'''
import numpy as np
from ..config import load_object
from .matchSVT import matchSVT

def getDimGroups(lDetections):
    dimGroups = [0]
    for data in lDetections:
        dimGroups.append(dimGroups[-1] + len(data))
    views = np.zeros(dimGroups[-1], dtype=int)
    for nv in range(len(dimGroups) - 1):
        views[dimGroups[nv]:dimGroups[nv+1]] = nv
    return dimGroups, views

def composeAff(out, vis=False):
    names = list(out.keys())
    N = len(names)
    aff = out[names[0]].copy()
    for i in range(1, N):
        aff = aff * out[names[i]]
    aff = np.power(aff, 1/N)
    return aff

def SimpleConstrain(dimGroups):
    constrain = np.ones((dimGroups[-1], dimGroups[-1]))
    for i in range(len(dimGroups)-1):
        start, end = dimGroups[i], dimGroups[i+1]
        constrain[start:end, start:end] = 0
    N = constrain.shape[0]
    constrain[range(N), range(N)] = 1
    return constrain

class ComposedAffinity:
    def __init__(self, cameras, basenames, cfg):
        affinity = {}
        for key, args in cfg.aff_funcs.items():
            args['cameras'] = cameras
            args['cams'] = basenames
            affinity[key] = load_object(key, args)
        self.cameras = cameras
        self.affinity = affinity
        self.cfg = cfg

    def __call__(self, annots, images=None):
        dimGroups, maptoview = getDimGroups(annots)
        out = {}
        for key, model in self.affinity.items():
            out[key] = model(annots, dimGroups)
        aff = composeAff(out, self.cfg.vis_aff)
        constrain = SimpleConstrain(dimGroups)
        observe = np.ones_like(aff)
        aff = constrain * aff
        if self.cfg.svt_py:
            aff = matchSVT(aff, dimGroups, constrain, observe, self.cfg.svt_args)
        aff[aff<self.cfg.aff_min] = 0
        return aff, dimGroups