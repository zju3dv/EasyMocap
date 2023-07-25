import numpy as np
import cv2
from easymocap.datasets.base import crop_image

from easymocap.estimator.wrapper_base import bbox_from_keypoints
from easymocap.mytools.vis_base import merge, plot_keypoints_auto
from .debug_utils import log, mywarn, myerror

def batch_triangulate(keypoints_, Pall, min_view=2):
    """ triangulate the keypoints of whole body

    Args:
        keypoints_ (nViews, nJoints, 3): 2D detections
        Pall (nViews, 3, 4) | (nViews, nJoints, 3, 4): projection matrix of each view
        min_view (int, optional): min view for visible points. Defaults to 2.

    Returns:
        keypoints3d: (nJoints, 4)
    """
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v >= min_view)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    if len(Pall.shape) == 3:
        P0 = Pall[None, :, 0, :]
        P1 = Pall[None, :, 1, :]
        P2 = Pall[None, :, 2, :]
    else:
        P0 = Pall[:, :, 0, :].swapaxes(0, 1)
        P1 = Pall[:, :, 1, :].swapaxes(0, 1)
        P2 = Pall[:, :, 2, :].swapaxes(0, 1)
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d #* (conf[..., 0].sum(axis=-1)>min_view)
    return result

def project_points(keypoints, RT, einsum=None):
    homo = np.concatenate([keypoints[..., :3], np.ones_like(keypoints[..., :1])], axis=-1)
    if einsum is None:
        if len(homo.shape) == 2 and len(RT.shape) == 3:
            kpts2d = np.einsum('vab,kb->vka', RT, homo)
        elif len(homo.shape) == 2 and len(RT.shape) == 4:
            kpts2d = np.einsum('vkab,kb->vka', RT, homo)
        else:
            import ipdb; ipdb.set_trace()
    else:
        kpts2d = np.einsum(einsum, RT, homo)
    kpts2d[..., :2] /= kpts2d[..., 2:]
    return kpts2d

def make_Cnk(n, k):
    import itertools
    res = {}
    for n_ in range(3, n+1):
        n_0 = [i for i in range(n_)]
        for k_ in range(2, k+1):
            res[(n_, k_)] = list(map(list, itertools.combinations(n_0, k_)))
    return res

MAX_VIEWS = 30
Cnk = make_Cnk(MAX_VIEWS, 3)

def robust_triangulate_point(kpts2d, Pall, dist_max, min_v = 3):
    nV = kpts2d.shape[0]
    if len(kpts2d) < min_v:# 重建失败
        return [], None
    # min_v = max(2, nV//2)
    # 1. choose the combination of min_v
    index_ = Cnk[(len(kpts2d), min(min_v, len(kpts2d)))]
    # 2. proposals: store the reconstruction points of each proposal
    proposals = np.zeros((len(index_), 4))
    weight_self = np.zeros((nV, len(index_)))
    for i, index in enumerate(index_):
        weight_self[index, i] = 100.
        point = batch_triangulate(kpts2d[index, :], Pall[index], min_view=min_v)
        proposals[i] = point
    # 3. project the proposals to each view
    #    and calculate the reprojection error
    # (nViews, nProposals, 4)
    kpts_repro = project_points(proposals, Pall)
    conf = (proposals[None, :, -1] > 0) * (kpts2d[..., -1] > 0)
    # err: (nViews, nProposals)
    err = np.linalg.norm(kpts_repro[..., :2] - kpts2d[..., :2], axis=-1) * conf
    valid = 1. - err/dist_max
    valid[valid<0] = 0
    # consider the weight of different view
    # TODO:naive weight:
    conf = kpts2d[..., -1]
    weight = conf
    # (valid > 0)*weight_self 一项用于强制要求使用到的两个视角都需要被用到
    # 增加一项使用的视角数的加成
    weight_sum = (weight * valid).sum(axis=0) + ((valid > 0)*weight_self).sum(axis=0) - min_v * 100
    if weight_sum.max() < 0:# 重建失败
        return [], None
    best = weight_sum.argmax()
    if (err[index_[best], best] > dist_max).any():
        return [], None
    # 对于选出来的proposal，寻找其大于0的其他视角
    point = proposals[best]
    best_add = np.where(valid[:, best])[0].tolist()
    index = list(index_[best])
    best_add.sort(key=lambda x:-weight[x])
    for add in best_add:
        if add in index:
            continue
        index.append(add)
        point = batch_triangulate(kpts2d[index, :], Pall[index], min_view=min_v)
        kpts_repro = project_points(point, Pall[index])
        err = np.linalg.norm(kpts_repro[..., :2] - kpts2d[index, ..., :2], axis=-1)
        if (err > dist_max).any():
            index.remove(add)
            break
    return index, point

def remove_outview(kpts2d, out_view, debug):
    if len(out_view) == 0:
        return False
    outv = out_view[0]
    if debug:
        mywarn('[triangulate] remove outview: {} from {}'.format(outv, out_view))
    kpts2d[outv] = 0.
    return True

def remove_outjoint(kpts2d, Pall, out_joint, dist_max, min_view=3, debug=False):
    if len(out_joint) == 0:
        return False
    if debug:
        mywarn('[triangulate] remove outjoint: {}'.format(out_joint))
    for nj in out_joint:
        valid = np.where(kpts2d[:, nj, -1] > 0)[0]
        if len(valid) < min_view:
            # if less than 3 visible view, set these unvisible
            kpts2d[:, nj, -1] = 0
            continue
        if len(valid) > MAX_VIEWS:
            # only select max points
            conf = -kpts2d[:, nj, -1]
            valid = conf.argsort()[:MAX_VIEWS]
        index_j, point = robust_triangulate_point(kpts2d[valid, nj:nj+1], Pall[valid], dist_max=dist_max, min_v=3)
        index_j = valid[index_j]
        # print('select {} for joint {}'.format(index_j, nj))
        set0 = np.zeros(kpts2d.shape[0])
        set0[index_j] = 1.
        kpts2d[:, nj, -1] *= set0
    return True

def project_and_distance(kpts3d, RT, kpts2d):
    kpts_proj = project_points(kpts3d, RT)
    # 1. distance between input and projection
    conf = (kpts3d[None, :, -1] > 0) * (kpts2d[:, :, -1] > 0)
    dist = np.linalg.norm(kpts_proj[..., :2] - kpts2d[..., :2], axis=-1) * conf
    return dist, conf

def iterative_triangulate(kpts2d, RT, previous=None,
    min_conf=0.1, min_view=3, min_joints=3, dist_max=0.05, dist_vel=0.05,
    thres_outlier_view=0.4, thres_outlier_joint=0.4, debug=False):
    kpts2d = kpts2d.copy()
    conf = kpts2d[..., -1]
    kpts2d[conf<min_conf] = 0.
    if debug:
        log('[triangulate] kpts2d: {}'.format(kpts2d.shape))
    # TODO: consider large motion
    if previous is not None:
        dist, conf = project_and_distance(previous, RT, kpts2d)
        nottrack = (dist > dist_vel) & conf
        if nottrack.sum() > 0:
            kpts2d[nottrack] = 0.
            if debug:
                log('[triangulate] Remove with track {}'.format(np.where(nottrack)))
    while True:
        # 0. triangulate and project
        kpts3d = batch_triangulate(kpts2d, RT, min_view=min_view)
        dist, conf = project_and_distance(kpts3d, RT, kpts2d)
        # 2. find the outlier
        vv, jj = np.where(dist > dist_max)
        if vv.shape[0] < 1:
            if debug:
                log('[triangulate] Not found outlier, break')
            break
        ratio_outlier_view = (dist>dist_max).sum(axis=1)/(1e-5 + conf.sum(axis=1))
        ratio_outlier_joint = (dist>dist_max).sum(axis=0)/(1e-5 + conf.sum(axis=0))
        # 3. find the totally wrong detections
        out_view = np.where(ratio_outlier_view > thres_outlier_view)[0]
        out_joint = np.where(ratio_outlier_joint > thres_outlier_joint)[0]
        if len(out_view) > 1:
            dist_view = dist.sum(axis=1)/(1e-5 + conf.sum(axis=1))
            out_view = out_view.tolist()
            out_view.sort(key=lambda x:-dist_view[x])
            if debug: mywarn('[triangulate] Remove outlier view: {}'.format(ratio_outlier_view))
        if remove_outview(kpts2d, out_view, debug): continue
        if remove_outjoint(kpts2d, RT, out_joint, dist_max, debug=debug): continue
        if debug:
            log('[triangulate] Directly remove {}, {}'.format(vv, jj))
        kpts2d[vv, jj, -1] = 0.
    if debug:
        log('[triangulate] finally {} valid points'.format((kpts3d[..., -1]>0).sum()))
    if (kpts3d[..., -1]>0).sum() < min_joints:
        kpts3d[..., -1] = 0.
        kpts2d[..., -1] = 0.
        return kpts3d, kpts2d
    return kpts3d, kpts2d

class BaseTriangulator:
    def __init__(self, config, debug, keys) -> None:
        self.config = config
        self.debug = debug
        self.keys = keys

    def project_and_check(self, kpts3d, kpts2d, RT):
        kpts_proj = project_points(kpts3d, RT)
        conf = (kpts3d[None, :, -1] > 0) * (kpts2d[:, :, -1] > 0)
        dist = np.linalg.norm(kpts_proj[..., :2] - kpts2d[..., :2], axis=-1) * conf
        return conf, dist
    
    def triangulate_with_results(self, pid, data, results):
        new = {'id': pid}
        for key in self.keys:
            key3d = key.replace('2d', '3d')
            if len(results) == 0:
                kpts3d, kpts2d = iterative_triangulate(data[key + '_unproj'], data['RT'],
                    debug=self.debug, **self.config[key])
            else:
                if len(results) == 1:
                    previous = results[-1][key3d] # TODO: mean previous frame
                elif len(results) >= 2:
                    # TODO: mean previous velocity
                    previous0 = results[-2][key3d] # TODO: mean previous frame
                    previous1 = results[-1][key3d] # TODO: mean previous frame
                    vel = (previous1[:, :3] - previous0[:, :3])*((previous0[:, -1:]>0)&(previous0[:, -1:]>0))
                    previous = previous1.copy()
                    previous[:, :3] += vel
                kpts3d, kpts2d = iterative_triangulate(data[key + '_unproj'], data['RT'],
                    debug=self.debug, previous=previous, **self.config[key])
                vel = np.linalg.norm(kpts3d[:, :3] - previous[:, :3], axis=-1)
            new[key] = np.concatenate([data[key+'_distort'][..., :-1], kpts2d[..., -1:]], axis=-1)
            new[key3d] = kpts3d
        return new

class SimpleTriangulator(BaseTriangulator):
    def __init__(self, keys, debug, config,
        pid=0, disable_previous=False) -> None:
        super().__init__(config, debug, keys)
        self.results = []
        self.infos = []
        self.dim_name = ['_joints', '_views']
        self.pid = pid
        self.disable_previous = disable_previous

    def __call__(self, data, results=None):
        info = {}
        if results is None:
            results = self.results
        if self.disable_previous:
            results = []
        new = {'id': self.pid}
        for key in self.keys:
            if key not in data.keys(): continue
            key3d = key.replace('2d', '3d')
            if self.debug:
                log('[triangulate] {}'.format(key))
            if len(results) == 0:
                kpts3d, kpts2d = iterative_triangulate(data[key + '_unproj'], data['RT'],
                    debug=self.debug, **self.config[key])
            else:
                if len(results) == 1:
                    previous = results[-1][key3d] # TODO: mean previous frame
                elif len(results) >= 2:
                    # TODO: mean previous velocity
                    previous0 = results[-2][key3d] # TODO: mean previous frame
                    previous1 = results[-1][key3d] # TODO: mean previous frame
                    vel = (previous1[:, :3] - previous0[:, :3])*((previous0[:, -1:]>0)&(previous0[:, -1:]>0))
                    previous = previous1.copy()
                    previous[:, :3] += vel
                kpts3d, kpts2d = iterative_triangulate(data[key + '_unproj'], data['RT'],
                    debug=self.debug, previous=previous, **self.config[key])
                vel = np.linalg.norm(kpts3d[:, :3] - previous[:, :3], axis=-1)
            new[key] = np.concatenate([data[key+'_distort'][..., :-1], kpts2d[..., -1:]], axis=-1)
            new[key.replace('2d', '3d')] = kpts3d
            if self.debug:
                conf, dist = self.project_and_check(kpts3d, kpts2d, data['RT'])
                for dim in [0, 1]:
                    info_dim = {
                        'valid': conf.sum(axis=dim),
                        'dist': 10000*dist.sum(axis=dim)/(1e-5 + conf.sum(axis=dim)),
                    }
                    info[key+self.dim_name[dim]] = info_dim
                info[key+'_joints']['valid3d'] = kpts3d[:, -1] >0
        results.append(new)
        self.infos.append(info)
        return [new]

    def report(self):
        if not self.debug:
            return 0
        from .debug_utils import print_table
        for key in self.infos[0].keys():
            metrics = list(self.infos[0][key].keys())
            values = [np.mean(np.stack([info[key][metric] for info in self.infos]), axis=0) for metric in metrics]
            metrics = [key] + metrics
            values = [[i for i in range(values[0].shape[0])]] + values
            print_table(metrics, values)

class SimpleTriangulatorMulti(SimpleTriangulator):
    def __init__(self, pids, **cfg) -> None:
        super().__init__(**cfg)
        self.results = {}
    
    def __call__(self, data, results=None):
        res_now = []
        for ipid, pid in enumerate(data['pid']):
            if pid not in self.results.keys():
                self.results[pid] = []
            data_ = {'RT': data['RT']}
            for key in self.keys:
                data_[key+'_distort'] = data[key+'_distort'][:, ipid]
                data_[key+'_unproj'] = data[key+'_unproj'][:, ipid]
                data_[key] = data[key][:, ipid]
            res = self.triangulate_with_results(pid, data_, self.results[pid])
            self.results[pid].append(res)
            res_now.append(res)
        return res_now

def skew_op(x):
    skew_op = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    res = np.zeros((3, 3), dtype=x.dtype)
    # 0, -z, y
    res[0, 1] = -x[2, 0]
    res[0, 2] =  x[1, 0]
    # z, 0, -x
    res[1, 0] =  x[2, 0]
    res[1, 2] = -x[0, 0]
    # -y, x, 0
    res[2, 0] = -x[1, 0]
    res[2, 1] =  x[0, 0]
    return res

def fundamental_op(K0, K1, R_0, T_0, R_1, T_1):
    invK0 = np.linalg.inv(K0)
    return invK0.T @ (R_0 @ R_1.T) @ K1.T @ skew_op(K1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1 = list(map(lambda x:int(x+0.5), pt1[:2].tolist()))
        pt2 = list(map(lambda x:int(x+0.5), pt2[:2].tolist()))
        if pt1[0] < 0 or pt1[1] < 0:
            continue
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def SimpleConstrain(dimGroups):
    constrain = np.ones((dimGroups[-1], dimGroups[-1]))
    for i in range(len(dimGroups)-1):
        start, end = dimGroups[i], dimGroups[i+1]
        constrain[start:end, start:end] = 0
    N = constrain.shape[0]
    constrain[range(N), range(N)] = 1
    return constrain

def check_cluster(affinity, row, views, dimGroups, indices, p2dAssigned, visited):
    affinity_row = affinity[row].copy()
    # given affinity and row, select the combine of all possible set
    cluster = np.where((affinity[row]>0)&(p2dAssigned==-1)&(visited==0))[0].tolist()
    cluster.sort(key=lambda x:-affinity[row, x])
    views_ = views[cluster]
    view_count = np.bincount(views[cluster])
    indices_all = [indices]
    for col in cluster:
        v = views[col]
        nOld = len(indices_all)
        if indices[v] != -1: # already assigned, copy and make new 
            for i in range(nOld):
                ind = indices_all[i].copy()
                ind[v] = col
                indices_all.append(ind)
        else: # not assigned, assign
            for i in range(nOld):
                indices_all[i][v] = col
    return indices_all

def views_from_dimGroups(dimGroups):
    views = np.zeros(dimGroups[-1], dtype=np.int)
    for nv in range(len(dimGroups) - 1):
        views[dimGroups[nv]:dimGroups[nv+1]] = nv
    return views

class SimpleMatchAndTriangulator(SimpleTriangulator):
    def __init__(self, num_joints, min_views, min_joints, cfg_svt, cfg_track, **cfg) -> None:
        super().__init__(**cfg)
        self.nJoints = num_joints
        self.cfg_svt = cfg_svt
        self.cfg_track = cfg_track
        self.min_views = min_views
        self.min_joints = min_joints
        self.time = -1
        self.max_id = 0
        self.tracks = {}
        self.loglevel_dict = {
            'info': 0,
            'warn': 1,
            'error': 2,
        }
        self.loglevel = self.loglevel_dict['info'] # ['info', 'warn', 'error']
        self.debug = False
        self.data = None
        self.people = None
    
    def log(self, text):
        if self.loglevel > 0:
            return 0
        log(text)
    
    def warn(self, text):
        if self.loglevel > 1:
            return 0
        mywarn(text)
    

    @staticmethod
    def distance_by_epipolar(pts0, pts1, K0, K1, R0, T0, R1, T1):
        F = fundamental_op(K0, K1, R0, T0, R1, T1)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines0 = cv2.computeCorrespondEpilines(pts0[..., :2].reshape (-1,1,2), 2, F)
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts1[..., :2].reshape(-1,1,2), 1, F)
        if False:
            H, W = 1080, 1920
            img0 = np.zeros((H, W, 3), dtype=np.uint8) +255
            img4, img3 = drawlines(img0.copy(), img0.copy(), lines0.reshape(-1, 3), pts1.reshape(-1, 3), pts0.reshape(-1,3))
            img5,img6 = drawlines(img0.copy(), img0.copy(), lines1.reshape(-1, 3), pts0.reshape(-1,3), pts1.reshape(-1,3))
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.imshow(img5)
            plt.subplot(122)
            plt.imshow(img4)
            plt.show()        
        lines0 = lines0.reshape(pts0.shape)
        lines1 = lines1.reshape(pts1.shape)
        # dist: (D_v0, D_v1, nJoints)
        # TODO: / sqrt(A^2 + B^2)
        dist01 = np.abs(np.sum(lines0[:, None, :, :2] * pts1[None, :, :, :2], axis=-1) + lines0[:, None, :, 2])
        conf = pts0[:, None, :, 2] * pts1[None, :, :, 2]
        dist10 = np.abs(np.sum(lines1[:, None, :, :2] * pts0[None, :, :, :2], axis=-1) + lines1[:, None, :, 2])

        dist = np.sum(dist01 * conf + dist10.transpose(1, 0, 2) * conf, axis=-1)/(conf.sum(axis=-1) + 1e-5)/2
        return dist

    def _simple_associate2d_triangulate(self, data, affinity, dimGroups, prev_id):
        # sum1 = affinity.sum(axis=1)
        # 注意：这里的排序应该是对每个视角，挑选最大的一个
        sum1 = np.zeros((affinity.shape[0]))
        for i in range(len(dimGroups)-1):
            start, end = dimGroups[i], dimGroups[i+1]
            if end == start:continue
            sum1 += affinity[:, start:end].max(axis=-1)
        n2d = affinity.shape[0]
        nViews = len(dimGroups) - 1
        idx_zero = np.zeros(nViews, dtype=np.int) - 1
        views = views_from_dimGroups(dimGroups)
        # the assigned results of each person
        p2dAssigned = np.zeros(n2d, dtype=np.int) - 1
        visited = np.zeros(n2d, dtype=np.int)
        sortidx = np.argsort(-sum1)
        pid = 0
        k3dresults = []
        for idx in sortidx:
            if p2dAssigned[idx] != -1:
                continue
            if prev_id[idx] != -1:
                results = [self.people[prev_id[idx]]]
            else:
                results = []
            proposals = check_cluster(affinity, row=idx, views=views, 
                dimGroups=dimGroups, indices=idx_zero.copy(), p2dAssigned=p2dAssigned, visited=visited)
            for indices in proposals:
                if (indices > -1).sum() < self.min_views - (len(results)):
                    continue
                # set keypoints2d
                info = {'RT': data['RT']}
                for name in ['keypoints2d', 'keypoints2d_unproj', 'keypoints2d_distort']:
                    info[name] = np.zeros((nViews, self.nJoints, 3), dtype=np.float32)
                for nv in range(nViews):
                    if indices[nv] == -1: continue
                    for name in ['keypoints2d', 'keypoints2d_unproj', 'keypoints2d_distort']:
                        info[name][nv] = data[name][nv][indices[nv]-dimGroups[nv]]

                res = super().__call__(info, results=results)[0]

                k2d = res['keypoints2d']
                valid_view = (k2d[..., 2] > 0).sum(axis=-1) > self.min_joints
                # if valid_view.sum() < self.min_views - len(results): # 这里如果是有前一帧的话，len(results)会是2；不知道之前为啥有这个条件使用
                if valid_view.sum() < self.min_views:
                    self.log('[associate] Skip proposal {}->{} with not enough valid view {}'.format(idx, indices, (k2d[..., 2] > 0).sum(axis=-1)))
                    continue
                valid_joint = res['keypoints3d'][:, -1] > 0.1
                if valid_joint.sum() < self.min_joints:
                    self.log('[associate] Skip proposal {}->{} as not enough joints'.format(idx, indices))
                    continue
                indices[~valid_view] = -1
                if (indices < 0).all():
                    import ipdb; ipdb.set_trace()
                self.log('[associate] Add indices {}, valid {}'.format(indices, (k2d[..., 2] > 0).sum(axis=-1)))
                res['id'] = pid
                res['indices'] = indices
                res['valid_view'] = valid_view
                res['valid_joints'] = res['keypoints3d'][:, -1] > 0.1
                k3dresults.append(res)
                for nv in range(nViews):
                    if valid_view[nv] and indices[nv] != -1:
                        p2dAssigned[indices[nv]] = pid
                        visited[indices[nv]] = 1
                pid += 1
                break
            visited[idx] = 1
        self.log('[associate] {} points not visited, {} not assigned'.format(visited.shape[0] - visited.sum(), (p2dAssigned==-1).sum()))
        k3dresults.sort(key=lambda x: -x['keypoints2d'][..., -1].sum())
        return k3dresults

    @staticmethod
    def calculate_affinity_MxM(dims, dimGroups, data, key, DIST_MAX):
        M = dimGroups[-1]
        distance = np.zeros((M, M), dtype=np.float32)
        nViews = len(dims)
        for v0 in range(nViews-1):
            for v1 in range(1, nViews):
                # calculate distance between (v0, v1)
                if v0 >= v1:
                    continue
                if dims[v0] == 0 or dims[v1] == 0:
                    continue
                if True:
                    pts0 = data[key][v0] # (nPerson0, nKeypoints, 3)
                    pts1 = data[key][v1] # (nPerson1, nKeypoints, 3)
                    K0, K1 = data['K'][v0], data['K'][v1] # K0, K1: (3, 3)
                    R0, T0 = data['Rc'][v0], data['Tc'][v0]
                    R1, T1 = data['Rc'][v1], data['Tc'][v1]
                    dist = SimpleMatchAndTriangulator.distance_by_epipolar(pts0, pts1, K0, K1, R0, T0, R1, T1)
                    dist /= (K0[0, 0] + K1[0, 0])/2
                else:
                    dist = self.distance_by_ray(pts0, pts1, R0, T0, R1, T1)
                distance[dimGroups[v0]:dimGroups[v0+1], dimGroups[v1]:dimGroups[v1+1]] = dist
                distance[dimGroups[v1]:dimGroups[v1+1], dimGroups[v0]:dimGroups[v0+1]] = dist.T

        for nv in range(nViews):
            distance[dimGroups[nv]:dimGroups[nv+1], dimGroups[nv]:dimGroups[nv+1]] = DIST_MAX
        distance -= np.eye(M) * DIST_MAX
        aff = (DIST_MAX - distance)/DIST_MAX
        aff = np.clip(aff, 0, 1)
        return aff

    def _calculate_affinity_MxM(self, dims, dimGroups, data, key):
        DIST_MAX = self.cfg_track.track_dist_max
        return self.calculate_affinity_MxM(dims, dimGroups, data, key, DIST_MAX=DIST_MAX)

    def _calculate_affinity_MxN(self, dims, dimGroups, data, key, results):
        M = dimGroups[-1]
        N = len(results)
        distance = np.zeros((M, N), dtype=np.float32)
        nViews = len(dims)
        k3d = np.stack([r['keypoints3d'] for r in results])
        kpts_proj = project_points(k3d, data['KRT'], einsum='vab,pkb->vpka')
        depth = kpts_proj[..., -1]
        kpts_proj[depth<0] = -10000
        for v in range(nViews):
            if dims[v] == 0:
                continue
            focal = data['K'][v][0, 0]
            pts2d = data[key][v][:, None]
            pts_repro = kpts_proj[v][None]
            conf = np.sqrt(pts2d[..., -1]*k3d[None, ..., -1])
            diff = np.linalg.norm(pts2d[..., :2] - pts_repro[..., :2], axis=-1)
            diff = np.sum(diff*conf, axis=-1)/(1e-5 + np.sum(conf, axis=-1))
            dist = diff / focal
            distance[dimGroups[v]:dimGroups[v+1], :] = dist
        DIST_MAX = self.cfg_track.track_repro_max
        aff = (DIST_MAX - distance)/DIST_MAX
        aff = np.clip(aff, 0, 1)
        return aff

    def _svt_optimize_affinity(self, affinity, dimGroups):
        # match SVT
        import pymatchlr
        observe = np.ones_like(affinity)
        aff_svt = pymatchlr.matchSVT(affinity, dimGroups, SimpleConstrain(dimGroups), observe, self.cfg_svt)
        aff_svt[aff_svt<self.cfg_svt.aff_min] = 0.
        if False:
            import matplotlib.pyplot as plt
            M = affinity.shape[0]
            plt.subplot(121)
            plt.imshow(affinity)
            plt.hlines([i-0.5 for i in dimGroups[1:]], -0.5, M-0.5, 'w')
            plt.vlines([i-0.5 for i in dimGroups[1:]], -0.5, M-0.5, 'w')
            plt.subplot(122)
            sum_row = aff_svt.sum(axis=1, keepdims=True)/(len(dimGroups) - 1)
            plt.imshow(np.hstack([aff_svt, sum_row]))
            plt.hlines([i-0.5 for i in dimGroups[1:]], -0.5, M-0.5, 'w')
            plt.vlines([i-0.5 for i in dimGroups[1:]], -0.5, M-0.5, 'w')
            plt.ioff()
            plt.show()
        return aff_svt

    def _track_add(self, res):
        pid = res['id']
        if pid == -1:
            pid = self.max_id
            res['id'] = pid
            self.max_id += 1
            self.log('[{:06d}] Create track {} <- {}'.format(self.time, pid, res['indices']))
            if False:
                crops = []
                data = self.data
                kpts = np.vstack(data['keypoints2d'])
                for nv in range(len(data['imgname'])):
                    img = cv2.imread(data['imgname'][nv])
                    if res['indices'][nv] == -1: continue
                    _kpts = kpts[res['indices'][nv]]
                    bbox = bbox_from_keypoints(_kpts)
                    plot_keypoints_auto(img, _kpts, pid)
                    crop = crop_image(img, bbox, crop_square=True)
                    crops.append(crop)
                debug = merge(crops)
                cv2.imwrite('debug/{:06d}.jpg'.format(pid), debug)
        else:
            self.max_id = max(self.max_id, pid+1)
            self.log('[{:06d}] Initialize track {}, valid joints={}'.format(self.time, pid, (res['keypoints3d'][:, -1]>0.01).sum()))
        self.tracks[pid] = {
            'start_time': self.time,
            'end_time': self.time+1,
            'missing_frame': [],
            'infos': [res]
        }
    
    def _track_update(self, res, pid):
        res['id'] = pid
        info = self.tracks[pid]
        self.log('[{:06d}] Update track {} [{}->{}], valid joints={}'.format(self.time, pid, info['start_time'], info['end_time'], (res['keypoints3d'][:, -1]>0.1).sum()))
        self.tracks[pid]['end_time'] = self.time + 1
        self.tracks[pid]['infos'].append(res)
    
    def _track_merge(self, res, pid):
        res['id'] = -1
        # TODO: merge

    def _track_and_update(self, data, results):
        cfg = self.cfg_track
        self.time += 1
        if self.time == 0:
            # initialize the tracks
            for res in results:
                self._track_add(res)
            return results
        # filter the missing frames
        for pid in list(self.tracks.keys()):
            if self.time - self.tracks[pid]['end_time'] > cfg.max_missing_frame:
                self.warn('[{:06d}] Remove track {}'.format(self.time, pid))
                self.tracks.pop(pid)
        # track the results with greedy matching
        for idx_match, res in enumerate(results):
            res['id'] = -1
            # compute the distance
            k3d = res['keypoints3d'][None]
            pids_free = [pid for pid in self.tracks.keys() if self.tracks[pid]['end_time'] != self.time+1]
            pids_used = [pid for pid in self.tracks.keys() if self.tracks[pid]['end_time'] == self.time+1]
            def check_dist(k3d_check):
                dist = np.linalg.norm(k3d[..., :3] - k3d_check[..., :3], axis=-1)
                conf = np.sqrt(k3d[..., 3] * k3d_check[..., 3])
                dist_mean = ((conf>0.1).sum(axis=-1) < self.min_joints)*cfg.track_dist_max + np.sum(dist * conf, axis=-1)/(1e-5 + np.sum(conf, axis=-1))
                argmin = dist_mean.argmin()
                dist_min = dist_mean[argmin]
                return dist_mean, argmin, dist_min
            # check free
            NOT_VISITED = -2
            NOT_FOUND = -1
            flag_tracked, flag_current = NOT_VISITED, NOT_VISITED
            if len(pids_free) > 0:
                k3d_check = np.stack([self.tracks[pid]['infos'][-1]['keypoints3d'] for pid in pids_free])
                dist_track, best, best_dist_track = check_dist(k3d_check)
                if best_dist_track < cfg.track_dist_max:
                    flag_tracked = best
                else:
                    flag_tracked = NOT_FOUND
            # check used
            if len(pids_used) > 0:
                k3d_check = np.stack([self.tracks[pid]['infos'][-1]['keypoints3d'] for pid in pids_used])
                dist_cur, best, best_dist_curr = check_dist(k3d_check)
                if best_dist_curr < cfg.track_dist_max:
                    flag_current = best
                else:
                    flag_current = NOT_FOUND
            if flag_tracked >= 0 and (flag_current == NOT_VISITED or flag_current == NOT_FOUND):
                self._track_update(res, pids_free[flag_tracked])
            elif (flag_tracked == NOT_FOUND or flag_tracked==NOT_VISITED) and flag_current >= 0:
                # 没有跟踪到，但是有当前帧的3D的，合并
                self.log('[{:06d}] Merge track {} to {}'.format(self.time, idx_match, pids_used[flag_current]))
                self._track_merge(res, pids_used[flag_current])
            elif flag_tracked == NOT_FOUND and flag_current == NOT_FOUND:
                # create a new track
                self._track_add(res)
            else:
                # 丢弃
                self.log('[{:06d}] Remove track {}. No close points'.format(self.time, idx_match))

        for pid in list(self.tracks.keys()):
            if self.tracks[pid]['end_time'] != self.time + 1:
                self.warn('[{:06d}] Tracking {} missing'.format(self.time, pid))
        results = [r for r in results if r['id']!=-1]
        return results

    def __call__(self, data):
        # match the data
        self.data = data
        key = 'keypoints2d'
        dims = [d.shape[0] for d in data[key]]
        dimGroups = np.cumsum([0] + dims)
        # 1. compute affinity
        affinity = self._calculate_affinity_MxM(dims, dimGroups, data, key)
        N2D = affinity.shape[0]
        if self.people is not None and len(self.people) > 0:
            # add 3d affinity
            _affinity = affinity
            affinity_3d = self._calculate_affinity_MxN(dims, dimGroups, data, key, self.people)
            affinity = np.concatenate([affinity, affinity_3d], axis=1)
            eye3d = np.eye(affinity_3d.shape[1])
            affinity = np.concatenate([affinity, np.hstack((affinity_3d.T, eye3d))], axis=0)
            dimGroups = dimGroups.tolist()
            dimGroups.append(dimGroups[-1]+affinity_3d.shape[1])
            affinity = self._svt_optimize_affinity(affinity, dimGroups)
            # affinity = self._svt_optimize_affinity(_affinity, dimGroups[:-1])
            # recover
            affinity_3d = np.hstack([np.ones((N2D, 1))*0.5, affinity[:N2D, N2D:]])
            prev_id = affinity_3d.argmax(axis=-1) - 1
            affinity = affinity[:N2D, :N2D]
            dimGroups = np.array(dimGroups[:-1])
        else:
            affinity = self._svt_optimize_affinity(affinity, dimGroups)
            prev_id = np.zeros(N2D) - 1
        # 2. associate and triangulate
        results = self._simple_associate2d_triangulate(data, affinity, dimGroups, prev_id)
        # 3. track, filter and return
        results = self._track_and_update(data, results)
        results.sort(key=lambda x:x['id'])
        self.people = results
        return results

def simple_match(data):
    key = 'keypoints2d'
    dims = [d.shape[0] for d in data[key]]
    dimGroups = np.cumsum([0] + dims)
    affinity = SimpleMatchAndTriangulator.calculate_affinity_MxM(dims, dimGroups, data, key, DIST_MAX=0.1)
    import pymatchlr
    observe = np.ones_like(affinity)
    cfg_svt = {
        'debug': 1,
        'maxIter': 10,
        'w_sparse': 0.1,
        'w_rank': 50,
        'tol': 0.0001,
        'aff_min': 0.3,
    }
    affinity = pymatchlr.matchSVT(affinity, dimGroups, SimpleConstrain(dimGroups), observe, cfg_svt)
    return affinity, dimGroups