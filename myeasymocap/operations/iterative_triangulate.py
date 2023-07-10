import numpy as np
import itertools
from easymocap.mytools.triangulator import batch_triangulate, project_points
from easymocap.mytools.debug_utils import log, mywarn, myerror

def project_and_distance(kpts3d, RT, kpts2d):
    kpts_proj = project_points(kpts3d, RT)
    # 1. distance between input and projection
    conf = (kpts3d[None, :, -1] > 0) * (kpts2d[:, :, -1] > 0)
    dist = np.linalg.norm(kpts_proj[..., :2] - kpts2d[..., :2], axis=-1) * conf
    return kpts_proj[..., -1], dist, conf

def remove_outview(kpts2d, out_view, debug):
    if len(out_view) == 0:
        return False
    elif len(out_view) == 1:
        # directly remove the outlier view
        outv = out_view[0]
        if debug:
            log('[triangulate] remove outview: {}'.format(outv))
    else:
        # only remove the first outlier view
        outv = out_view[0]
        if debug:
            mywarn('[triangulate] remove first outview: {} from {}'.format(outv, out_view))
    kpts2d[outv] = 0.
    return True

def remove_outjoint(kpts2d, Pall, out_joint, dist_max, dist_track, min_view=3, previous=None, debug=False):
    MIN_CONF_3D = 0.1
    if len(out_joint) == 0:
        return False
    if debug:
        mywarn('[triangulate] remove outjoint: {}'.format(out_joint))
    nviews = np.arange(kpts2d.shape[0])
    for nj in out_joint:
        valid = np.where(kpts2d[:, nj, -1] > 0)[0]
        if len(valid) < min_view:
            # if less than 3 visible view, set these unvisible
            kpts2d[:, nj, -1] = 0
            continue
        kpts_nj = kpts2d[valid, nj]
        Pall_nj = Pall[valid]
        view_index = nviews[valid]
        view_local = np.arange(valid.shape[0])
        comb_views = np.array(list(itertools.combinations(view_local.tolist(), min_view))).T
        comb_kpts = kpts_nj[comb_views]
        comb_Pall = Pall_nj[comb_views]
        comb_k3d = batch_triangulate(comb_kpts, comb_Pall)
        depth, dist, conf = project_and_distance(comb_k3d, comb_Pall, comb_kpts)
        # 依次选择置信度最高的
        sort_by_conf = (-comb_kpts[..., -1].sum(axis=0)).argsort()
        flag = (dist[:, sort_by_conf]<dist_max).all(axis=0)
        if previous is not None:
            dist3d = np.linalg.norm(previous[[nj], :3] - comb_k3d[:, :3], axis=-1) * 1000
            flag = flag & ((dist3d[sort_by_conf] < dist_track) | (previous[nj, 3] < MIN_CONF_3D))
        valid = sort_by_conf[flag]
        if valid.shape[0] == 0:
            if debug:
                mywarn('[triangulate] cannot find valid combinations of joint {}'.format(nj))
            kpts2d[:, nj, -1] = 0
        else:
            # check all 2D keypoints
            k3d = comb_k3d[valid[0]].reshape(1, 4)
            depth, dist, conf = project_and_distance(k3d, Pall_nj, kpts_nj[:, None])
            valid_view = view_index[np.where(dist < dist_max)[0]]
            # 这里需要尝试三角化一下，如果按照新的三角化之后误差更大的话，不应该用新的，而是使用老的
            if debug:
                log('[triangulate] {} find valid combinations of joint: {}'.format(nj, valid_view))
                log('[triangulate] {} distance 2d pixel (max {}): {}'.format(nj, dist_max, dist[np.where(dist < dist_max)[0], 0]))
                if previous is not None and previous[nj, 3] > MIN_CONF_3D:
                    _dist3d = np.linalg.norm(previous[[nj], :3] - k3d[:, :3], axis=-1) * 1000
                    log('[triangulate] {} distance 3d mm (max {}): {}'.format(nj, dist_track, _dist3d))
                    if _dist3d > dist_track:
                        import ipdb; ipdb.set_trace()
            set0 = np.zeros(kpts2d.shape[0])
            set0[valid_view] = 1.
            kpts2d[:, nj, -1] *= set0
    return True

def iterative_triangulate(kpts2d, RT,
    min_conf=0.1, min_view=3, min_joints=3, dist_max=0.05, dist_track=50,
    thres_outlier_view=0.4, thres_outlier_joint=0.4, debug=True,
    previous=None,
    **kwargs):
    kpts2d = kpts2d.copy()
    conf = kpts2d[..., -1]
    kpts2d[conf<min_conf] = 0.
    if debug:
        log('[triangulate] kpts2d: {}'.format(kpts2d.shape))
    while True:
        # 0. triangulate and project
        kpts3d = batch_triangulate(kpts2d, RT, min_view=min_view)
        depth, dist, conf = project_and_distance(kpts3d, RT, kpts2d)
        # 2. find the outlier
        vv, jj = np.where(dist > dist_max)
        if vv.shape[0] < 1:
            if debug:
                log('[triangulate] Not found outlier, break')
            break
        ratio_outlier_view = (dist>dist_max).sum(axis=1)/(1e-5 + (conf > 0.).sum(axis=1))
        ratio_outlier_joint = (dist>dist_max).sum(axis=0)/(1e-5 + (conf > 0.).sum(axis=0))
        # 3. find the totally wrong detections
        out_view = np.where(ratio_outlier_view > thres_outlier_view)[0]
        error_joint = dist.sum(axis=0)/(1e-5 + (conf > 0.).sum(axis=0))
        # for joint, we calculate the mean distance of this joint
        out_joint = np.where((ratio_outlier_joint > thres_outlier_joint) & (error_joint > dist_max))[0]
        if len(out_view) > 1:
            # TODO: 如果全都小于0的话，相当于随机丢了，应该增加视角的置信度
            # 应该生成多个proposal；然后递归的去寻找
            # 不应该直接抛弃的
            # 如果有previous的情况，应该用previous来作为判断标准
            # cfg = dict(min_conf=min_conf, min_view=min_view, min_joints=min_joints, dist_max=dist_max, dist_track=dist_track,
            #            thres_outlier_view=thres_outlier_view, thres_outlier_joint=0.4, debug=True, previous=None)
            if debug: mywarn('[triangulate] More than one outlier view: {}, stop triangulation.'.format(ratio_outlier_view))
            return kpts3d, np.zeros_like(kpts2d)
            if debug: mywarn('[triangulate] Remove outlier view give outlier ratio: {}'.format(ratio_outlier_view))
            dist_view = dist.sum(axis=1)/(1e-5 + (conf > 0.).sum(axis=1))
            out_view = out_view.tolist()
            out_view.sort(key=lambda x:-dist_view[x])
        if remove_outview(kpts2d, out_view, debug): continue
        if len(out_joint) > 0:
            if debug: 
                print(dist[:, out_joint])
                mywarn('[triangulate] Remove outlier joint {} given outlier ratio: {}'.format(out_joint, ratio_outlier_joint[out_joint]))
            remove_outjoint(kpts2d, RT, out_joint, dist_max, dist_track, previous=previous, debug=debug)
            continue
        if debug:
            log('[triangulate] Directly remove {}, {}'.format(vv, jj))
        kpts2d[vv, jj, -1] = 0.
    if debug:
        log('[triangulate] finally {} valid points, {} not valid'.format((kpts3d[..., -1]>0).sum(), np.where(kpts3d[..., -1]<=0)[0]))
    if (kpts3d[..., -1]>0).sum() < min_joints:
        kpts3d[..., -1] = 0.
        kpts2d[..., -1] = 0.
        return kpts3d, kpts2d
    return kpts3d, kpts2d