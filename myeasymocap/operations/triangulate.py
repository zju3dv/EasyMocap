import numpy as np
from itertools import combinations
from easymocap.mytools.camera_utils import Undistort
from easymocap.mytools.triangulator import iterative_triangulate

def batch_triangulate(keypoints_, Pall, min_view=2):
    """ triangulate the keypoints of whole body

    Args:
        keypoints_ (nViews, nJoints, 3): 2D detections
        Pall (nViews, 3, 4): projection matrix of each view
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
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
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

def project_wo_dist(keypoints, RT, einsum='vab,kb->vka'):
    homo = np.concatenate([keypoints[..., :3], np.ones_like(keypoints[..., :1])], axis=-1)
    kpts2d = np.einsum(einsum, RT, homo)
    depth = kpts2d[..., 2]
    kpts2d[..., :2] /= kpts2d[..., 2:]
    return kpts2d, depth

class SimpleTriangulate:
    def __init__(self, mode):
        self.mode = mode
    
    @staticmethod
    def undistort(points, cameras):
        nViews = len(points)
        pelvis_undis = []
        for nv in range(nViews):
            camera = {key:cameras[key][nv] for key in ['R', 'T', 'K', 'dist']}
            if points[nv].shape[0] > 0:
                pelvis = Undistort.points(points[nv], camera['K'], camera['dist'])
            else:
                pelvis = points[nv].copy()
            pelvis_undis.append(pelvis)
        return pelvis_undis

    def __call__(self, keypoints, cameras):
        '''
            keypoints: [nViews, nJoints, 3]
        
        output: 
            keypoints3d: (nJoints, 4)
        '''
        keypoints = self.undistort(keypoints, cameras)
        keypoints = np.stack(keypoints)
        if self.mode == 'naive':
            keypoints3d = batch_triangulate(keypoints, cameras['P'])
        else:
            keypoints3d, k2d = iterative_triangulate(keypoints, cameras['P'], dist_max=25)
        return {'keypoints3d': keypoints3d}

class RobustTriangulate(SimpleTriangulate):
    def __init__(self, mode, cfg):
        super().__init__(mode)
        self.cache_view = {}
        self.cfg = cfg

    def try_to_triangulate_and_project(self, index, keypoints, cameras):
        # 选择最好的3个视角
        P = cameras['P'][index]
        kpts = keypoints[index][:, None]
        k3d = batch_triangulate(kpts, P)
        k2d, depth = project_wo_dist(k3d, P)
        dist_repro = np.linalg.norm(k2d[..., :2] - kpts[..., :2], axis=-1).mean(axis=-1)
        return k3d, dist_repro

    def robust_triangulate(self, keypoints, cameras):
        # 选择最好的3个视角
        # TODO: 移除不合理的视角
        nViews = keypoints.shape[0]
        if nViews not in self.cache_view:
            views = list(range(nViews))
            combs = list(combinations(views, self.cfg.triangulate.init_views))
            combs = np.array(combs)
            self.cache_view[nViews] = combs
        combs = self.cache_view[nViews]
        keypoints_comb = keypoints[combs]
        conf_sum = keypoints_comb[..., 2].mean(axis=1) * (keypoints_comb[..., 2]>0.05).all(axis=1)
        comb_sort_id = (-conf_sum).argsort()
        flag_find_init = False
        for comb_id in comb_sort_id:
            if conf_sum[comb_id] < 0.1:
                break
            comb = combs[comb_id]
            k3d, dist_repro = self.try_to_triangulate_and_project(comb, keypoints, cameras)
            if (dist_repro < self.cfg.triangulate.repro_init).all():
                flag_find_init = True
                init = comb.tolist()
                break
        if not flag_find_init:
            print('Cannot find good initialize pair')
            import ipdb; ipdb.set_trace()
        view_idxs = (-keypoints[:, -1]).argsort()
        for view_idx in view_idxs:
            if view_idx in init:
                continue
            if keypoints[view_idx, 2] < 0.1:
                continue
            k3d, dist_repro = self.try_to_triangulate_and_project(init+[view_idx], keypoints, cameras)
            if (dist_repro < self.cfg.triangulate.repro_2d).all():
                # print('Add view {}'.format(view_idx))
                init.append(view_idx)
        return k3d, init

    def __call__(self, keypoints, cameras):
        """
            keypoints: (nViews, nJoints, 3)
            cameras: (nViews, 3, 4)
        """
        nViews, nJoints, _ = keypoints.shape
        keypoints_undis = np.stack(self.undistort(keypoints, cameras))
        # for each points, find good initial pairs
        points_all = np.zeros((nJoints, 4))
        keypoints_copy = keypoints.copy()
        for nj in range(nJoints):
            point, select_views = self.robust_triangulate(keypoints_undis[:, nj], cameras)
            points_all[nj:nj+1] = point
            keypoints_copy[select_views, nj, 2] += 10
            keypoints_copy[:, nj, 2] = np.clip(keypoints_copy[:, nj, 2]-10, 0, 1)
        return {'keypoints3d': points_all, 'keypoints_select': keypoints_copy}