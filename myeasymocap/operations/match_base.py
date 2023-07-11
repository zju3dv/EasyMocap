import numpy as np
import cv2
from easymocap.mytools.camera_utils import Undistort
from easymocap.mytools.debug_utils import log, mywarn, myerror
from .iterative_triangulate import iterative_triangulate
from easymocap.mytools.triangulator import project_points, batch_triangulate
from easymocap.mytools.timer import Timer

class DistanceBase:
    # 这个类用于计算affinity
    # 主要基于关键点计算；未来可以考虑支持其他东西
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def calculate_affinity_MxM(self, keypoints, cameras):
        raise NotImplementedError
    
    @staticmethod
    def SimpleConstrain(dimGroups):
        constrain = np.ones((dimGroups[-1], dimGroups[-1]))
        for i in range(len(dimGroups)-1):
            start, end = dimGroups[i], dimGroups[i+1]
            constrain[start:end, start:end] = 0
        N = constrain.shape[0]
        constrain[range(N), range(N)] = 1
        return constrain

def skew_op(x):
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

class EpipolarDistance(DistanceBase):
    @staticmethod
    def distance2d2d(pts0, pts1, K0, K1, R0, T0, R1, T1):
        F = fundamental_op(K0, K1, R0, T0, R1, T1)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines0 = cv2.computeCorrespondEpilines(pts0[..., :2].reshape (-1,1,2), 2, F)
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts1[..., :2].reshape(-1,1,2), 1, F)   
        lines0 = lines0.reshape(pts0.shape)
        lines1 = lines1.reshape(pts1.shape)
        # dist: (D_v0, D_v1, nJoints)
        # TODO: / sqrt(A^2 + B^2)
        dist01 = np.abs(np.sum(lines0[:, None, :, :2] * pts1[None, :, :, :2], axis=-1) + lines0[:, None, :, 2])
        conf = (pts0[:, None, :, 2] * pts1[None, :, :, 2]) > 0
        dist10 = np.abs(np.sum(lines1[:, None, :, :2] * pts0[None, :, :, :2], axis=-1) + lines1[:, None, :, 2])

        dist = np.sum(dist01 * conf + dist10.transpose(1, 0, 2) * conf, axis=-1)/(conf.sum(axis=-1) + 1e-5)/2
        return dist

    def vis_affinity(self, aff, dimGroups, scale=10):
        aff = cv2.resize(aff, (aff.shape[1]*scale, aff.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        aff_float = aff.copy()
        aff = (aff * 255).astype(np.uint8)
        aff = cv2.applyColorMap(aff, cv2.COLORMAP_JET)
        transp = (aff_float * 255).astype(np.float32)
        for dim in dimGroups[1:-1]:
            cv2.line(aff, (0, dim*scale), (aff.shape[0], dim*scale), (255, 255, 255), thickness=1)
            cv2.line(aff, (dim*scale, 0), (dim*scale, aff.shape[0]), (255, 255, 255), thickness=1)
            cv2.line(transp, (0, dim*scale), (aff.shape[0], dim*scale), (255,), thickness=1)
            cv2.line(transp, (dim*scale, 0), (dim*scale, aff.shape[0]), (255,), thickness=1)
        # last line
        cv2.rectangle(aff, (0, 0), (aff.shape[0]-1, aff.shape[0]-1), (0, 0, 255), thickness=1)
        cv2.rectangle(transp, (0, 0), (aff.shape[0]-1, aff.shape[0]-1), (255,), thickness=1)
        aff = np.dstack([aff, transp])
        return aff

    def calculate_affinity_MxM(self, keypoints, cameras):
        # 计算一下总长度
        dims = [d.shape[0] for d in keypoints]
        dimGroups = np.cumsum([0] + dims)
        M = dimGroups[-1]
        distance = np.eye((M), dtype=np.float32)
        nViews = len(keypoints)
        for v0 in range(nViews-1):
            # set the diag block
            for v1 in range(1, nViews):
                # calculate distance between (v0, v1)
                if v0 >= v1:
                    continue
                pts0 = keypoints[v0]
                pts1 = keypoints[v1]
                if pts0.shape[0] == 0 or pts1.shape[0] == 0:
                    continue
                K0, K1 = cameras['K'][v0], cameras['K'][v1] # K0, K1: (3, 3)
                R0, T0 = cameras['R'][v0], cameras['T'][v0]
                R1, T1 = cameras['R'][v1], cameras['T'][v1]
                dist = self.distance2d2d(pts0, pts1, K0, K1, R0, T0, R1, T1)
                conf0 = pts0[..., -1]
                conf1 = pts1[..., -1]
                common_count = ((conf0[:, None] > 0) & (conf1[None] > 0)).sum(axis=-1)
                common_affinity = np.sqrt(conf0[:, None] * conf1[None])
                dist /= (K0[0, 0] + K1[0, 0])/2
                dist[common_count < self.cfg.min_common_joints] = self.cfg.threshold * 10
                aff_geo = (self.cfg.threshold - dist)/self.cfg.threshold
                aff_conf = common_affinity.mean(axis=-1)
                aff_compose = aff_geo * aff_conf
                distance[dimGroups[v0]:dimGroups[v0+1], dimGroups[v1]:dimGroups[v1+1]] = aff_compose
                distance[dimGroups[v1]:dimGroups[v1+1], dimGroups[v0]:dimGroups[v0+1]] = aff_compose.T
            
        affinity = np.clip(distance, 0, 1)
        return affinity, dimGroups
    
    def _calculate_affinity_MxN(self, keypoints3d, keypoints, cameras):
        DEPTH_NEAR = 0.5
        dims = [d.shape[0] for d in keypoints]
        dimGroups = np.cumsum([0] + dims)
        M = dimGroups[-1]
        N = keypoints3d.shape[0]
        distance = np.zeros((M, N), dtype=np.float32)
        nViews = len(keypoints)
        kpts_proj = project_points(keypoints3d, cameras['P'], einsum='vab,pkb->vpka')
        depth = kpts_proj[..., -1]
        kpts_proj[depth<DEPTH_NEAR] = -10000
        # TODO: constrain the depth far
        affinity_all = []
        for v in range(nViews):
            if dims[v] == 0:
                continue
            focal = (cameras['K'][v][0, 0] + cameras['K'][v][1, 1])/2
            # pts2d: (N, J, 3)
            pts2d = keypoints[v]
            # pts_repro: (N3D, J, 3)
            pts_repro = kpts_proj[v]
            # conf: (N2D, N3D, J)
            conf = np.sqrt(pts2d[:, None, ..., -1]*keypoints3d[None, ..., -1])
            diff = np.linalg.norm(pts2d[:, None, ..., :2] - pts_repro[None, ..., :2], axis=-1)
            # (N2D, N3D)
            diff = np.sum(diff*conf, axis=-1)/(1e-5 + np.sum(conf, axis=-1))
            dist = diff / focal
            aff_geo = (self.cfg.threshold_track - dist)/self.cfg.threshold_track
            affinity_all.append(aff_geo)
        aff = np.vstack(affinity_all)
        aff = np.clip(aff, 0, 1)
        return aff

    def low_rank_optimization(self, affinity, dimGroups, vis=False):
        if True:
            import pymatchlr
            observe = np.ones_like(affinity)
            aff_svt = pymatchlr.matchSVT(affinity, dimGroups, self.SimpleConstrain(dimGroups), observe, self.cfg.cfg_svt)
        else:
            aff_svt = affinity
        aff_svt[aff_svt<self.cfg.cfg_svt.aff_min] = 0.
        if vis:
            cv2.imwrite('debug.png', np.hstack([self.vis_affinity(affinity, dimGroups), self.vis_affinity(aff_svt, dimGroups)]))
            import ipdb; ipdb.set_trace()
        return aff_svt

class MatchBase:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        if cfg.distance.mode == 'epipolar':
            self.distance = EpipolarDistance(cfg.distance)
        else:
            raise NotImplementedError

    def set_previous(self, previous):
        prev_ids = [p['id'] for p in previous]
        prev_keypoints = [p['keypoints3d'] for p in previous]
        self.prev_ids = prev_ids
        self.prev_keypoints = prev_keypoints
        if len(prev_ids) > 0:
            self.prev_keypoints = np.stack(prev_keypoints)

    @staticmethod
    def undistort(points, cameras):
        nViews = len(points)
        pelvis_undis = []
        for nv in range(nViews):
            K = cameras['K'][nv]
            dist = cameras['dist'][nv]
            points_nv = points[nv]
            points_nv_flat = points_nv.reshape(-1, 3)
            if points_nv_flat.shape[0] > 0:
                points_nv_flat = Undistort.points(points_nv_flat, K, dist)
            pelvis_undis.append(points_nv_flat.reshape(*points_nv.shape))
        return pelvis_undis

    def _prepare_associate(self, affinity, keypoints):
        dimGroups = [0]
        views = []
        nViews = len(keypoints)
        affinity_sum = np.zeros((affinity.shape[0],))
        for nv in range(nViews):
            dimGroups.append(dimGroups[-1] + keypoints[nv].shape[0])
            views.extend([nv] * keypoints[nv].shape[0])
            start, end = dimGroups[nv], dimGroups[nv+1]
            if end > start:
                affinity_sum += affinity[:, start:end].max(axis=-1)
        return affinity_sum, dimGroups, views

    def try_to_triangulate(self, keypoints, cameras, indices, previous=None):
        Pall, keypoints2d = [], []
        for nv in range(indices.shape[0]):
            if indices[nv] == -1:
                Pall.append(cameras['P'][nv])
                keypoints2d.append(np.zeros((25, 3), dtype=np.float32))
                # keypoints2d.append(keypoints[nv][indices[nv]])
            else:
                Pall.append(cameras['P'][nv])
                keypoints2d.append(keypoints[nv][indices[nv]])
        Pall = np.stack(Pall)
        keypoints2d = np.stack(keypoints2d)
        if previous is not None:
            kpts_proj = project_points(previous, cameras['P'], einsum='vab,kb->vka')
            # 注意，这里需要考虑深度，因为深度是已知的
            # 越近的地方这个阈值应该越大，越远的地方阈值越小
            # radius / depth * focal
            depth = kpts_proj[..., -1]
            # 超出这个track阈值的直接丢掉了；这样可以保证三角化出来的一定是小于阈值的
            # 如果对这个阈值有意见，应该增大这个阈值条件
            radius = self.cfg.triangulate.dist_track * cameras['K'][:, 0, 0][:, None] / depth
            dist = np.linalg.norm(kpts_proj[..., :2] - keypoints2d[..., :2], axis=-1)
            conf = np.sqrt(kpts_proj[..., -1] * keypoints2d[..., -1]) > 0
            not_track = (dist > radius) & conf
            if not_track.sum() > 0:
                log('[Tri] {} 2d joints not tracked'.format(not_track.sum()))
                keypoints2d[not_track] = 0.
        keypoints3d, k2d = iterative_triangulate(keypoints2d, Pall, previous=previous, **self.cfg.triangulate)
        not_valid_view = np.where((k2d[..., -1] < 0.1).all(axis=1))[0]
        indices[not_valid_view] = -1
        result = {
            'keypoints3d': keypoints3d,
            'indices': indices,
            'keypoints2d': k2d
        }
        return result, indices

    @staticmethod
    def _indices_from_affinity(dimGroups, affinit_row, assigned, visited, nViews):
        proposals = []
        indices = np.zeros((nViews), dtype=np.int) - 1
        for nv in range(nViews):
            start, end = dimGroups[nv], dimGroups[nv+1]
            block = affinit_row[start:end]
            to_select = np.where((block>0.1) & \
                                 (~assigned[start:end]) & \
                                 (~visited[start:end]))[0]
            if to_select.shape[0] == 1:
                # 只有唯一的一个候选
                indices[nv] = to_select[0]
            elif to_select.shape[0] > 1:
                to_select_sort = sorted(to_select, key=lambda x:-block[x])
                indices[nv] = to_select_sort[0]
                for select_id in to_select_sort[1:]:
                    proposals.append((nv, select_id, block[select_id]))
            elif to_select.shape[0] == 0:
                # empty
                pass
        return indices, proposals

    def _check_indices(self, indices, keypoints3d=None):
        flag_ind = (indices > -1).sum() >= self.cfg.triangulate.min_view_body
        if keypoints3d is not None:
            conf = keypoints3d[:, 3]
            flag_3d = (conf > self.cfg.triangulate.min_conf_3d).sum() > self.cfg.min_joints
            flag_ind = flag_ind & flag_3d
        return flag_ind

    def _simple_associate2d_triangulate(self, affinity, keypoints, cameras, assigned=None):
        # sum1 = affinity.sum(axis=1)
        # 注意：这里的排序应该是对每个视角，挑选最大的一个
        affinity_sum, dimGroups, views = self._prepare_associate(affinity, keypoints)
        nViews = len(keypoints)
        n2d = affinity.shape[0]
        # the assigned results of each person
        if assigned is None:
            assigned = np.zeros(n2d, dtype=np.bool)
        visited = np.zeros(n2d, dtype=np.bool)
        sortidx = np.argsort(-affinity_sum)
        k3dresults = []
        for idx in sortidx:
            if assigned[idx]:continue
            log('[Tri] Visited view{}: {}'.format(views[idx], idx-dimGroups[views[idx]]))
            affinit_row = affinity[idx]
            indices, proposals = self._indices_from_affinity(dimGroups, affinit_row, assigned, visited, nViews)
            # 注意：要再生成所有的proposal之后再设置visited
            visited[idx] = True
            if not self._check_indices(indices):continue
            # 只考虑有候选的；不考虑移除某个视角的
            log('[Tri] First try to triangulate of {}'.format(indices))
            indices_origin = indices.copy()
            result, indices = self.try_to_triangulate(keypoints, cameras, indices)
            if not self._check_indices(indices, result['keypoints3d']):
                # if the proposals is valid
                if len(proposals) > 0:
                    proposals.sort(key=lambda x:-x[2])
                    for (nviews, select_id, conf) in proposals:
                        indices = indices_origin.copy()
                        indices[nviews] = select_id
                        log('[Tri] Max fail, then try to triangulate of {}'.format(indices))
                        result, indices = self.try_to_triangulate(keypoints, cameras, indices)
                        if self._check_indices(indices, result['keypoints3d']):
                            break
                    else:
                        # overall proposals, not find any valid
                        continue
                else:
                    continue
            for nv in range(nViews):
                if indices[nv] == -1:
                    continue
                assigned[indices[nv]+dimGroups[nv]] = True
            result['id'] = -1
            k3dresults.append(result)
        return k3dresults
    
    def _check_speed(self, previous, current, verbo=False):
        conf = np.sqrt(previous[:, -1] * current[:, -1])
        conf[conf < self.cfg.triangulate.min_conf_3d] = 0.
        dist = np.linalg.norm(previous[:, :3] - current[:, :3], axis=-1)
        conf_mean = (conf * dist).sum()/(1e-5 + conf.sum()) * 1000
        if verbo:
            log('Track distance of each joints:')
            print(dist)
            print(conf_mean)
        return conf_mean < self.cfg.triangulate.dist_track
    
    def _simple_associate2d3d_triangulate(self, keypoints3d, affinity, keypoints, dimGroups, cameras):
        nViews = len(keypoints)
        n2d = affinity.shape[0]
        # the assigned results of each person
        assigned = np.zeros(n2d, dtype=np.bool)
        visited = np.zeros(n2d, dtype=np.bool)
        affinity_sum = affinity.sum(axis=0)
        sortidx = np.argsort(-affinity_sum)
        k3dresults = []
        for idx3d in sortidx:
            log('[Tri] Visited 3D {}'.format(self.prev_ids[idx3d]))
            affinit_row = affinity[:, idx3d]
            indices, proposals = self._indices_from_affinity(dimGroups, affinit_row, assigned, visited, nViews)
            if not self._check_indices(indices):continue
            # 只考虑有候选的；不考虑移除某个视角的
            log('[Tri] First try to triangulate of {}'.format(indices))
            indices_origin = indices.copy()
            result, indices = self.try_to_triangulate(keypoints, cameras, indices, previous=keypoints3d[idx3d])
            
            if not (self._check_indices(indices, result['keypoints3d']) and self._check_speed(keypoints3d[idx3d], result['keypoints3d'])):
                # if the proposals is valid
                previous = keypoints3d[idx3d]
                # select the best keypoints of each view
                previous_proj = project_points(previous, cameras['P'])
                dist_all = np.zeros((previous_proj.shape[0],)) + 999.
                indices_all = np.zeros((previous_proj.shape[0],), dtype=int)
                keypoints_all = np.zeros_like(previous_proj)
                for nv in range(previous_proj.shape[0]):
                    dist = np.linalg.norm(previous_proj[nv, :, :2][None] - keypoints[nv][:, :, :2], axis=-1)
                    conf = (previous[..., -1] > 0.1)[None] & (keypoints[nv][:, :, -1] > 0.1)
                    dist_mean = (dist * conf).sum(axis=-1) / (1e-5 + conf.sum(axis=-1))
                    dist_all[nv] = dist_mean.min()
                    indices_all[nv] = dist_mean.argmin()
                    keypoints_all[nv] = keypoints[nv][indices_all[nv]]
                want_view = dist_all.argsort()[:self.cfg.triangulate.min_view_body]
                # TODO: add more proposal instead of the top K
                proposal = (want_view, indices_all[want_view], -dist_all[want_view])
                proposals = [proposal]
                if len(proposals) > 0:
                    proposals.sort(key=lambda x:-x[2])
                    for (nv, select_id, conf) in proposals:
                        indices = np.zeros_like(indices_origin) - 1
                        indices[nv] = select_id
                        log('[Tri] Max fail, then try to triangulate of {}'.format(indices))
                        result, indices = self.try_to_triangulate(keypoints, cameras, indices, previous=keypoints3d[idx3d])
                        if (self._check_indices(indices, result['keypoints3d']) and self._check_speed(keypoints3d[idx3d], result['keypoints3d'])):
                            # 检测合格了，需要计算一下所有的view里面，那些是合格的，再一起计算
                            k2d_repro = project_points(result['keypoints3d'], cameras['P'])
                            dist = np.linalg.norm(k2d_repro[..., :2] - keypoints_all[..., :2], axis=-1)
                            conf = (result['keypoints3d'][:, -1][None] > 0.1) & (keypoints_all[..., 2] > 0.1)
                            dist[~conf] = 0.
                            valid_2d = dist < self.cfg.triangulate.dist_max
                            valid_ratio_view = valid_2d.mean(axis=-1)
                            valid_view = np.where(valid_ratio_view > 0.4)[0]
                            indices_new = np.zeros_like(indices_origin) - 1
                            indices_new[valid_view] = indices_all[valid_view]
                            keypoints_all[~valid_2d] = 0.
                            k3d_new = batch_triangulate(keypoints_all, cameras['P'], min_view=3)
                            result = {
                                'keypoints3d': k3d_new,
                                'indices': indices_new,
                                'keypoints2d': keypoints_all
                            }
                            log('[Tri] Max success, Refine the indices to {}'.format(indices))
                            # result, indices = self.try_to_triangulate(keypoints, cameras, indices_new, previous=result['keypoints3d'])
                            break
                        else:
                            log('[Tri] triangulation failed')
                            self._check_speed(keypoints3d[idx3d], result['keypoints3d'], verbo=True)
                    else:
                        # overall proposals, not find any valid
                        mywarn('[Tri] {} Track fail after {} proposal'.format(self.prev_ids[idx3d], len(proposals)))
                        continue
                else:
                    mywarn('[Tri] Track fail {}'.format(indices))
                    self._check_speed(keypoints3d[idx3d], result['keypoints3d'], verbo=True)
                    continue
            log('[Tri] finally used indices: {}'.format(indices))
            for nv in range(nViews):
                if indices[nv] == -1:
                    continue
                assigned[indices[nv]+dimGroups[nv]] = True
            result['id'] = self.prev_ids[idx3d]
            k3dresults.append(result)
        return k3dresults, assigned

    def associate(self, cameras, keypoints):
        keypoints = self.undistort(keypoints, cameras)
        for kpts in keypoints:
            conf = kpts[..., -1]
            conf[conf < self.cfg.min_conf] = 0.
        if len(self.prev_ids) > 0:
            # naive track
            with Timer('affinity 2d'):
                affinity2d2d, dimGroups = self.distance.calculate_affinity_MxM(keypoints, cameras)
            with Timer('affinity 3d'):
                affinity2d3d = self.distance._calculate_affinity_MxN(self.prev_keypoints, keypoints, cameras)
            affinity_comp = np.vstack([
                np.hstack([affinity2d2d, affinity2d3d]),
                np.hstack([affinity2d3d.T, np.eye(len(self.prev_ids))])
            ])
            with Timer('svt'):
                affinity2d2d_2d3d = self.distance.low_rank_optimization(
                    affinity_comp, 
                    dimGroups.tolist() + [dimGroups[-1] + len(self.prev_ids)],
                    vis=False)
            # 先associate2d 3d
            affinity2d3d = affinity2d2d_2d3d[:affinity2d2d.shape[0], affinity2d2d.shape[1]:]
            with Timer('associate 3d'):
                k3dresults, assigned = self._simple_associate2d3d_triangulate(self.prev_keypoints, affinity2d3d, keypoints, dimGroups, cameras)
            # 再associate2d 2d
            with Timer('associate 2d'):
                affinity2d2d = affinity2d2d_2d3d[:affinity2d2d.shape[0], :affinity2d2d.shape[1]]
                match_results = self._simple_associate2d_triangulate(affinity2d2d, keypoints, cameras, assigned=assigned)
            match_results = k3dresults + match_results
        else:
            affinity2d2d, dimGroups = self.distance.calculate_affinity_MxM(keypoints, cameras)
            affinity2d2d = self.distance.low_rank_optimization(affinity2d2d, dimGroups)
            # 直接associate2d
            match_results = self._simple_associate2d_triangulate(affinity2d2d, keypoints, cameras)
        return match_results

class TrackBase:
    # 这个类用于维护一般的track操作
    # 主要提供的接口：
    # 1. add
    # 2. remove
    # 3. smooth
    # 4. naive fit
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.kintree = np.array(cfg.kintree)
        self.max_id = 0
        self.current_frame = -1
        self.record = {}

    def update_frame(self, frame):
        # remove the results that are not in the frame
        self.current_frame = frame
        remove_id = []
        for pid, record in self.record.items():
            if frame - record['frames'][-1] > self.cfg.max_missing:
                mywarn('[Track] remove track {} with frames {}'.format(pid, record['frames']))
                remove_id.append(pid)
        for pid in remove_id:
            self.record.pop(pid)
        return True

    def query_current(self, ret_final=False):
        # return the results that are in the frame
        prevs = []
        for pid, record in self.record.items():
            k3d = record['records'][-1]
            valid = k3d[:, -1] > 0.1
            if ret_final:
                # 判断一下valid range
                k3d_valid = k3d[valid]
                flag = (k3d_valid[:, 0] > self.cfg.final_ranges[0][0]) & \
                      (k3d_valid[:, 0] < self.cfg.final_ranges[1][0]) & \
                      (k3d_valid[:, 1] > self.cfg.final_ranges[0][1]) & \
                      (k3d_valid[:, 1] < self.cfg.final_ranges[1][1]) & \
                      (k3d_valid[:, 2] > self.cfg.final_ranges[0][2]) & \
                      (k3d_valid[:, 2] < self.cfg.final_ranges[1][2])
                if flag.sum() < 5:
                    continue
            prevs.append({
                'id': pid,
                'keypoints3d': record['records'][-1],
                'ages': len(record['frames'])
            })
        if ret_final:
            prevs.sort(key=lambda x:-x['ages'])
            prevs = prevs[:self.cfg.final_max_person]
        prevs.sort(key=lambda x:x['id'])
        return prevs
    
    def add_track(self, res):
        # add a new track
        pid = self.max_id
        mywarn('[Track] add new person {}'.format(pid))
        res['id'] = pid
        self.record[pid] = {
            'frames': [self.current_frame],
            'records': [res['keypoints3d']]
        }
        self.max_id += 1
    
    def update_track(self, res):
        pid = res['id']
        N_UPDATE_LENGTH = 10
        if len(self.record[pid]['frames']) >= N_UPDATE_LENGTH and len(self.record[pid]['frames']) % N_UPDATE_LENGTH == 0:
            # 更新骨长
            # (nFrames, nJoints, 4)
            history = np.stack(self.record[pid]['records'])
            left = history[:, self.kintree[:, 0]]
            right = history[:, self.kintree[:, 1]]
            conf = np.minimum(left[..., -1], right[..., -1])
            conf[conf < 0.1] = 0.
            limb_length = np.linalg.norm(left[..., :3] - right[..., :3], axis=-1)
            limb_mean = (conf * limb_length).sum(axis=0)/(1e-5 + conf.sum(axis=0))
            conf_mean = conf.sum(axis=0)
            log('[Track] Update limb length of {} to \n {}'.format(pid, limb_mean))
            self.record[pid]['limb_length'] = (limb_mean, conf_mean)
        k3d = res['keypoints3d']
        if 'limb_length' in self.record[pid].keys():
            left = k3d[self.kintree[:, 0]]
            right = k3d[self.kintree[:, 1]]
            limb_now = np.linalg.norm(left[:, :3] - right[:, :3], axis=-1)
            limb_mean, conf_mean = self.record[pid]['limb_length']
            not_valid = ((limb_now > limb_mean * 1.5) | (limb_now < limb_mean * 0.5)) & (conf_mean > 0.1)
            if not_valid.sum() > 0:
                leaf = self.kintree[not_valid, 1]
                res['keypoints3d'][leaf] = 0.
                mywarn('[Track] {} remove {} joints'.format(pid, leaf))
                mywarn('[Track] mean: {}'.format(limb_mean[not_valid]))
                mywarn('[Track] current: {}'.format(limb_now[not_valid]))

        self.record[pid]['frames'].append(self.current_frame)
        self.record[pid]['records'].append(res['keypoints3d'])

    def track(self, match_results):
        wo_id_results = [r for r in match_results if r['id'] == -1]
        w_id_results = [r for r in match_results if r['id'] != -1]
        wo_id_results.sort(key=lambda x:-(x['indices']!=-1).sum())
        for res in wo_id_results:
            self.add_track(res)
        for res in w_id_results:
            self.update_track(res)
        return w_id_results + wo_id_results

class MatchAndTrack():
    def __init__(self, cfg_match, cfg_track) -> None:
        self.matcher = MatchBase(cfg_match)
        self.tracker = TrackBase(cfg_track)
    
    def __call__(self, cameras, keypoints, meta):
        frame = meta['frame']
        # 1. query the previous frame
        self.tracker.update_frame(frame)
        previous = self.tracker.query_current()
        # 2. associate the current frame
        self.matcher.set_previous(previous)
        match_results = self.matcher.associate(cameras, keypoints)
        # 3. update the tracker
        self.tracker.track(match_results)
        results = self.tracker.query_current(ret_final=True)
        pids = [p['id'] for p in results]
        if len(pids) > 0:
            keypoints3d = np.stack([p['keypoints3d'] for p in results])
        else:
            keypoints3d = []
        log('[Match&Triangulate] Current ID: {}'.format(pids))
        return {'results': results, 'keypoints3d': keypoints3d, 'pids': pids}