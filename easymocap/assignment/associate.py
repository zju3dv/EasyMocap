'''
  @ Date: 2021-06-04 21:58:37
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 11:50:10
  @ FilePath: /EasyMocapRelease/easymocap/assignment/associate.py
'''
import numpy as np
from ..mytools.reconstruction import batch_triangulate, projectN3
from ..config import load_object

def views_from_dimGroups(dimGroups):
    views = np.zeros(dimGroups[-1], dtype=int)
    for nv in range(len(dimGroups) - 1):
        views[dimGroups[nv]:dimGroups[nv+1]] = nv
    return views

def set_keypoints2d(indices, annots, Pall, dimGroups):
    Vused = np.where(indices!=-1)[0]
    if len(Vused) < 1:
        return [], [], [], []
    keypoints2d = np.stack([annots[nv][indices[nv]-dimGroups[nv]]['keypoints'].copy() for nv in Vused])
    bboxes = np.stack([annots[nv][indices[nv]-dimGroups[nv]]['bbox'].copy() for nv in Vused])
    Pused = Pall[Vused]
    return keypoints2d, bboxes, Pused, Vused

def load_criterions(cfg):
    criterions = []
    for key, val in cfg.items():
        crit = load_object(key, val)
        criterions.append(crit)
    return criterions

def simple_associate(annots, affinity, dimGroups, Pall, group, cfg):
    nViews = len(annots)
    criterions = load_criterions(cfg.criterions)
    n2D = dimGroups[-1]
    views = views_from_dimGroups(dimGroups)

    views_cnt = np.zeros((affinity.shape[0], nViews))
    for nv in range(nViews):
        views_cnt[:, nv] = affinity[:, dimGroups[nv]:dimGroups[nv+1]].sum(axis=1)
    views_cnt = (views_cnt>0.5).sum(axis=1)
    sortidx = np.argsort(-views_cnt)
    p2dAssigned = np.zeros(n2D, dtype=int) - 1
    indices_zero = np.zeros((nViews), dtype=int) - 1
    for idx in sortidx:
        if p2dAssigned[idx] != -1:
            continue
        proposals = [indices_zero.copy()]
        for nv in range(nViews):
            match = np.where( 
                (affinity[idx, dimGroups[nv]:dimGroups[nv+1]] > 0.) 
              & (p2dAssigned[dimGroups[nv]:dimGroups[nv+1]] == -1) )[0]
            if len(match) > 0:
                match = match + dimGroups[nv]
                for proposal in proposals:
                    proposal[nv] = match[0]
            if len(match) > 1:
                proposals_new = []
                for proposal in proposals:
                    for col in match[1:]:
                        p = proposal.copy()
                        p[nv] = col
                        proposals_new.append(p)
                proposals += proposals_new
        results = []
        while len(proposals) > 0:
            proposal = proposals.pop()
            # less than two views
            if (proposal != -1).sum() < cfg.min_views:
                continue
            # print('[associate] pop proposal: {}'.format(proposal))
            keypoints2d, bboxes, Pused, Vused = set_keypoints2d(proposal, annots, Pall, dimGroups)
            keypoints3d = batch_triangulate(keypoints2d, Pused)
            kptsRepro = projectN3(keypoints3d, Pused)
            err = ((kptsRepro[:, :, 2]*keypoints2d[:, :, 2]) > 0.) * np.linalg.norm(kptsRepro[:, :, :2] - keypoints2d[:, :, :2], axis=2)
            size = (bboxes[:, [2, 3]] - bboxes[:, [0, 1]]).max(axis=1, keepdims=True)
            err = err / size
            err_view = err.sum(axis=1)/((err>0. + 1e-9).sum(axis=1))
            flag = (err_view < cfg.max_repro_error).all()
            err = err.sum()/(err>0 + 1e-9).sum()
            # err_view = err.sum(axis=1)/((err>0.).sum(axis=1))
            # err = err.sum()/(err>0.).sum()
            # flag = err_view.max() < err_view.mean() * 2
            flag = True
            for crit in criterions:
                if not crit(keypoints3d):
                    flag = False
                    break
            if flag:
                # print('[associate]: view {}'.format(Vused))
                results.append({
                    'indices': proposal,
                    'keypoints2d': keypoints2d,
                    'keypoints3d': keypoints3d,
                    'Vused': Vused,
                    'error': err
                })
            else:
                # make new proposals
                outlier_view = Vused[err_view.argmax()]
                proposal[outlier_view] = -1
                proposals.append(proposal)
        if len(results) == 0:
            continue
        if len(results) > 1:
            # print('[associate] More than one avalible results')
            results.sort(key=lambda x:x['error'])
        result = results[0]
        proposal = result['indices']
        Vused = result['Vused']
        # proposal中有-1的，所以需要使用Vused进行赋值
        p2dAssigned[proposal[Vused]] = 1
        group.add(result)
    group.dimGroups = dimGroups
    return group