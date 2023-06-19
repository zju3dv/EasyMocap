import numpy as np
import cv2
from easymocap.mytools.camera_utils import Undistort
from easymocap.mytools.debug_utils import mywarn
from .triangulate import batch_triangulate, project_wo_dist
from collections import defaultdict
LOG_FILE = 'log.txt'
LOG_LEVEL = 0 #2
FULL_LOG = (lambda x: print(x, file=open(LOG_FILE, 'a'))) if LOG_LEVEL > 1 else (lambda x: None)
LOG = (lambda x: print(x, file=open(LOG_FILE, 'a'))) if LOG_LEVEL > 0 else (lambda x: None)

def LOG_ARRAY(array2d, format='{:>8.2f} '):
    res = ''
    for i in range(array2d.shape[0]):
        for j in range(array2d.shape[1]):
            res += format.format(array2d[i, j])
        res += '\n'
    return res

class MatchBase:
    def __init__(self, mode, cfg) -> None:
        self.mode = mode
        self.cfg = cfg
        print('[{}]'.format(self.__class__.__name__))
        print(self.cfg)
        self.max_id = 0

    def make_grids(self, grids, grids_step):
        grid_x = np.arange(grids[0][0], grids[1][0], grids_step)
        grid_y = np.arange(grids[0][1], grids[1][1], grids_step)
        grid_z = np.arange(grids[0][2], grids[1][2], grids_step)
        grid_xyz = np.meshgrid(grid_x, grid_y, grid_z)
        grid_xyz = np.stack(grid_xyz, axis=-1)
        grids = grid_xyz.reshape(-1, 3)
        print('[{}] Generate {} => {} grids'.format(self.__class__.__name__, grid_xyz.shape, grids.shape[0]))
        return grids
    
    @staticmethod
    def stack_array(arrays):
        dimGroups = [0]
        results = []
        views_all = []
        for nv, array in enumerate(arrays):
            dimGroups.append(dimGroups[-1] + array.shape[0])
            views_all.extend([nv for _ in range(array.shape[0])])
            results.append(array)
        results = np.concatenate(results, axis=0)
        return results, np.array(views_all), np.array(dimGroups)

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
    
    @staticmethod
    def distance_by_triangulate(p_src, p_dst, camera_src, camera_dst, ranges):
        dist = np.zeros((p_src.shape[0], p_dst.shape[0]), dtype=np.float32)
        # generate (m, n) points and distance
        idx_src = np.arange(p_src.shape[0])
        idx_dst = np.arange(p_dst.shape[0])
        idx_src, idx_dst = np.meshgrid(idx_src, idx_dst)
        idx_src = idx_src.reshape(-1)
        idx_dst = idx_dst.reshape(-1)
        p_src = p_src[idx_src]
        p_dst = p_dst[idx_dst]
        keypoints = np.stack([p_src, p_dst], axis=0)
        keypoints_flat = keypoints.reshape(keypoints.shape[0], -1, keypoints.shape[-1])
        P = np.stack([camera_src['P'], camera_dst['P']], axis=0)
        k3d = batch_triangulate(keypoints_flat, P, min_view=2)
        repro, depth = project_wo_dist(k3d, P)
        dist_repro = np.linalg.norm(repro[..., :2] - keypoints_flat[..., :2], axis=-1).mean(axis=0)
        valid = (k3d[:, 0] > ranges[0][0]) & (k3d[:, 0] < ranges[1][0]) & \
                (k3d[:, 1] > ranges[0][1]) & (k3d[:, 1] < ranges[1][1]) & \
                (k3d[:, 2] > ranges[0][2]) & (k3d[:, 2] < ranges[1][2])
        dist_repro[~valid] = 1e5
        dist[idx_src, idx_dst] = dist_repro
        return dist

    def calculate_distance(self, pelvis_undis, cameras, dimGroups):
        DIST_MAX = 10000.
        distance = np.zeros((dimGroups[-1], dimGroups[-1]), dtype=np.float32) + DIST_MAX
        nViews = len(dimGroups) - 1
        ray0 = np.array([0, 0, 1], dtype=np.float32).reshape(1, 3, 1)
        ray_cam = cameras['R'].transpose(0, 2, 1) @ ray0
        ray_cam = ray_cam[..., 0]
        cos_theta = np.sum(ray_cam[:, None] * ray_cam[None], axis=-1)
        theta = np.rad2deg(np.arccos(np.clip(cos_theta, -1., 1.)))
        valid_theta = np.logical_and(theta > self.cfg.valid_angle[0], theta < self.cfg.valid_angle[1])
        for src in range(nViews - 1):
            for dst in range(src + 1, nViews):
                # TODO: 计算两个射线的夹角
                # 这里对于不相邻或者对角的视角，我们直接跳过距离的计算
                # 这样后面在进行初始化的时候就无法挑到两个比较接近的视角了
                # if not valid_theta[src, dst]:
                #     continue
                p_src = pelvis_undis[src][:, None] #(m, 2)
                p_dst = pelvis_undis[dst][:, None] #(n, 2)
                if p_src.shape[0] == 0 or p_dst.shape[0] == 0:
                    continue
                camera_src = {key:cameras[key][src] for key in ['R', 'T', 'K', 'dist', 'P']}
                camera_dst = {key:cameras[key][dst] for key in ['R', 'T', 'K', 'dist', 'P']}
                dist = self.distance_by_triangulate(p_src, p_dst, camera_src, camera_dst, self.cfg.valid_ranges)
                distance[dimGroups[src]:dimGroups[src+1], dimGroups[dst]:dimGroups[dst+1]] = dist
                distance[dimGroups[dst]:dimGroups[dst+1], dimGroups[src]:dimGroups[src+1]] = dist.T
        cameras['valid_theta'] = valid_theta
        return distance
    
    def calculate_repro(self, results, pelvis_undis, cameras, views_all):
        nViews = len(cameras['P'])
        n3D = len(results)
        distance = np.zeros((pelvis_undis.shape[0], n3D), dtype=np.float32)
        if n3D == 0:
            return distance
        keypoints3d = np.stack([d['pelvis'] for d in results], axis=0)
        Pall = np.stack([cameras['P'][nv] for nv in range(nViews)])
        # k2d: (nViews, nPerson, nPoints, 3)
        k2d, depth = project_wo_dist(keypoints3d, Pall, einsum='vab,pkb->vpka')
        repro_select = k2d[views_all]
        # dist: (nPoints, n3D)
        dist = np.linalg.norm(repro_select[..., :2] - pelvis_undis[:, None, None, :2], axis=-1).mean(axis=2)
        # for nv in range(nViews):
        return dist
    
    def triangulate_and_repro(self, cameras, views, proposals):
        Pall = np.stack([cameras['P'][v] for v in views])
        kpts = np.stack(proposals)
        kpts = kpts[:, None]
        k3d = batch_triangulate(kpts, Pall)
        k2d, depth = project_wo_dist(k3d, Pall)
        dist_repro = np.linalg.norm(k2d[..., :2] - kpts[..., :2], axis=-1).mean(axis=-1)
        return k3d, dist_repro, depth


    @staticmethod
    def check_is_best_3d_of_2d(distance, idx3d, idx2d, visited3d):
        isbest3d = True
        distance_2d = distance[idx2d]
        for i3d in distance_2d.argsort():
            if i3d != idx3d and i3d not in visited3d:
                isbest3d = False
                break
            elif i3d == idx3d:
                break
        return isbest3d

    @staticmethod
    def sort_with_affinity(distance, dimGroups, INLIER_REPRO):
        nViews = len(dimGroups) - 1
        # 排序计算affinity
        count_rows = np.zeros((dimGroups[-1]), dtype=int)
        distance_rows = np.zeros((dimGroups[-1]))
        for nv in range(nViews):
            if dimGroups[nv] == dimGroups[nv+1]:continue
            valid_view = np.clip((distance[:, dimGroups[nv]:dimGroups[nv+1]] < INLIER_REPRO).sum(axis=-1), 0, 1)
            count_rows += valid_view # 最多也只累计一个
            distance_rows += valid_view * (distance[:, dimGroups[nv]:dimGroups[nv+1]].min(axis=-1))
        index = list(range(dimGroups[-1]))
        # index.sort(key=lambda x: (-count_rows[x], distance_rows[x]))
        # sort with 2D confidence
        # index.sort(key=lambda x: -pelvis_all[x, 2])
        # sort with valid matches
        # 选择2D的依据改为：根据有效的2D重投影距离的数量
        valid_count = (distance < INLIER_REPRO * 2).sum(axis=0)
        index = (-valid_count).argsort()
        return index

    def assign_by_3D(self, used_index, distance, pelvis_all, views_all, dimGroups, cameras):
        INLIER_TRACK = self.cfg.track_pixel
        INLIER_REPRO = self.cfg.max_pixel
        # 使用前一帧的可见性来进行排序
        index_3d = list(range(len(self.results)))
        index_3d.sort(key=lambda x:-len(self.results[x]['views']))
        results = []
        visited3d = set()
        for idx3d in index_3d:
            visited3d.add(idx3d)
            self.results[idx3d]['tracked'] = False
            pid = self.results[idx3d]['id']
            dist = distance[:, idx3d]
            FULL_LOG('[Assign 3D] Check 3D {}'.format(pid))
            FULL_LOG('[Assign 3D] Distance {}'.format(LOG_ARRAY(dist[None])))
            current = []
            views = []
            proposal = dist.argsort()
            # 初始化一下：
            for idx2d in proposal:
                # 不满足视角关系
                # if not valid_theta[views_all[row], views_all[idx2d]]:
                #     continue
                # 不满足距离关系
                if dist[idx2d] > INLIER_TRACK:
                    break
                if used_index[idx2d] > -1:
                    continue
                if views_all[idx2d] in views:
                    continue
                if not self.check_is_best_3d_of_2d(distance, idx3d, idx2d, visited3d):
                    continue
                if len(current) == 1: # 已经有一个了，如果还要再添加，那么需要判断一下三角化出来的距离关系
                    k3d, dist_repro, depth = self.triangulate_and_repro(cameras, views + [views_all[idx2d]], current + [pelvis_all[idx2d]])
                    _dist = np.linalg.norm(k3d[:, :3] - self.results[idx3d]['pelvis'][:, :3], axis=-1).mean()
                    if _dist > self.cfg.max_movement:
                        continue
                # 找到了合理的pair，作为一个良好的初始化
                current.append(pelvis_all[idx2d])
                views.append(views_all[idx2d])
                used_index[idx2d] = pid
                FULL_LOG(f'[Assign 3D] First track 3D {pid} with {idx2d}, view ({views_all[idx2d]})')
                if len(current) == 2:
                    break
            if len(current) < 2:
                # 没有找到良好的初始化
                continue
            for idx2d in proposal:
                # 这个视角已经有了 ｜ 这个2D已经被使用过了
                if views_all[idx2d] in views:
                    continue
                if used_index[idx2d] > -1:
                    continue
                if not self.check_is_best_3d_of_2d(distance, idx3d, idx2d, visited3d):
                    continue
                # 尝试添加
                FULL_LOG('[Assign 3D] 3D {} add {}, distance={:.2f}'.format(pid, idx2d, dist[idx2d]))
                new = current + [pelvis_all[idx2d]]
                views_new = views + [views_all[idx2d]]
                k3d, dist_repro, depth = self.triangulate_and_repro(cameras, views_new, new)
                _dist = np.linalg.norm(k3d[:, :3] - self.results[idx3d]['pelvis'][:, :3], axis=-1).mean()
                flag_movement = _dist < self.cfg.max_movement
                flag_depth = (depth > 0.5).all()
                flag_repro = dist_repro.mean() < INLIER_REPRO
                flag = flag_repro & flag_depth
                FULL_LOG('[Assign 3D] repro: \n{}, \ndepth: \n{}'.format(LOG_ARRAY(dist_repro[None]), LOG_ARRAY(depth.T)))
                if flag:
                    # 添加
                    current = new
                    views = views_new
                    used_index[idx2d] = pid
                    FULL_LOG('[Assign 3D] {} => {}'.format(idx2d, np.where(used_index == pid)[0]))
                else:
                    FULL_LOG('[Assign 3D] Failed')
            # check the results
            if len(views) < self.cfg.min_views: #不足以添加
                continue
            k3d, dist_repro, depth = self.triangulate_and_repro(cameras, views, current)
            select = np.where(used_index == pid)[0]
            results.append({
                'id': pid,
                'pelvis': k3d, 
                'keypoints3d': k3d, # 这里保存两个，这样即使后面覆盖掉了keypoints3d还能取出pelvis来
                'views': views_all[select],
                'select': select,
                'indices': select - dimGroups[views_all[select]],
                'frames': self.results[idx3d]['frames'] + [self.frames]
            })
            self.results[idx3d]['tracked'] = True
        for res in results:
            text = f'''  - Track {res['id']} with {len(res['views'])} views
      views: {' '.join(list(map(lambda x:'{:2d}'.format(x), res['views'])))}
      id   : {' '.join(list(map(lambda x:'{:2d}'.format(x), res['select'])))}'''
            LOG(text)
            print(text)
        for res in self.results:
            if not res['tracked']:
                mywarn('- 3D {} not tracked'.format(res['id']))
                # 对于没有被跟踪到的：检查是否有两个距离很小的视角
                # 如果有，并且被其他人占用了，那么把这个2D也给他；在极端情况下，有的视角下会有人恰好被另一个人挡住
                print(res)
                if len(res['frames']) < 3:
                    mywarn('- 3D {} not tracked, but only {} frames'.format(res['id'], len(res['frames'])))
                else:
                    pass
                    # import ipdb; ipdb.set_trace()
        return results
    
    def find_initial_3_pair(self, distance, pelvis_all, views_all, dimGroups):
        # 生成所有可能的候选的3个pair
        index_0 = np.arange(pelvis_all.shape[0])
        index_0 = np.stack(np.meshgrid(index_0, index_0, index_0), axis=-1).reshape(-1, 3)
        flag_order = (index_0[:, 0] < index_0[:, 1]) & (index_0[:, 1] < index_0[:, 2])
        # flag_views = (views_all[index_0[:, 0]] != views_all[index_0[:, 1]]) & \
        #                 (views_all[index_0[:, 1]] != views_all[index_0[:, 2]]) & \
        #                 (views_all[index_0[:, 0]] != views_all[index_0[:, 2]])
        valid_index = index_0[flag_order]
        distance_circle = distance[valid_index[:, 0], valid_index[:, 1]] + \
                            distance[valid_index[:, 1], valid_index[:, 2]] + \
                            distance[valid_index[:, 2], valid_index[:, 0]]
        distance_circle = distance_circle / 3
        valid_dist = distance_circle < self.cfg.max_pixel
        valid_ = valid_index[valid_dist]
        dist_sum = distance_circle[valid_dist]
        arg_idx = dist_sum.argsort()
        FULL_LOG('[Assign 2D] find {} 3 pair: '.format(len(arg_idx)))
        return valid_[arg_idx], dist_sum[arg_idx]

    def try_to_add_index(self, dist_row, cameras, pelvis_all, views_all, dimGroups,
                         used_index, views, current, pid):
        INLIER_REPRO = self.cfg.max_pixel
        proposal = dist_row.argsort()
        indices = []
        for idx2d in proposal:
            if dist_row[idx2d] > INLIER_REPRO:
                break
            # 这个视角已经有了 ｜ 这个2D已经被使用过了
            if views_all[idx2d] in views:
                continue
            if used_index[idx2d] > -1:
                continue
            FULL_LOG('[Assign 2D] Try to add {}, distance={:.2f}'.format(idx2d, dist_row[idx2d]))
            # 尝试三角化并进行重投影
            new = current + [pelvis_all[idx2d]]
            views_new = views + [views_all[idx2d]]
            k3d, dist_repro, depth = self.triangulate_and_repro(cameras, views_new, new)
            flag_depth = (depth > 0.5).all()
            flag_repro = dist_repro.mean() < INLIER_REPRO
            flag = flag_repro & flag_depth
            FULL_LOG('[Assign 2D] repro: \n{}, \ndepth: \n{}'.format(LOG_ARRAY(dist_repro[None]), LOG_ARRAY(depth.T)))
            if flag:
                # 添加
                current.append(pelvis_all[idx2d])
                views.append(views_all[idx2d])
                indices.append(idx2d)
                FULL_LOG('[Assign 2D] Add {}'.format(idx2d ))
            else:
                FULL_LOG('[Assign 2D] Failed')
        return indices

    def assign_by_2D_3pair(self, results, distance, dimGroups, used_index, valid_3pairs, views_all, pelvis_all, cameras):
        INLIER_REPRO = self.cfg.max_pixel
        for ipair, valid_3pair in enumerate(valid_3pairs):
            # 先检查是否被使用过了
            if (used_index[valid_3pair] > -1).any():
                continue
            # 先检查是否是合理的
            FULL_LOG('[Assign 2D] Check 3 pair {}'.format(valid_3pair))
            k3d, dist_repro, depth = self.triangulate_and_repro(cameras, views_all[valid_3pair], pelvis_all[valid_3pair])
            flag_depth = (depth > 0.5).all()
            flag_repro = dist_repro.mean() < INLIER_REPRO
            # TODO: flag range
            flag = flag_repro & flag_depth
            if not flag: continue
            # 添加其余的点
            pid = self.max_id
            self.max_id += 1
            dist_pair = distance[valid_3pair].mean(axis=0)
            views = views_all[valid_3pair].tolist()
            current = [pelvis_all[i] for i in valid_3pair]
            indices = self.try_to_add_index(dist_pair, cameras, pelvis_all, views_all, dimGroups,
                         used_index, views, current, pid)
            select = np.array(valid_3pair.tolist() + indices)
            k3d, dist_repro, depth = self.triangulate_and_repro(cameras, views, current)

            used_index[select] = pid
            results.append({
                'id': pid,
                'pelvis': k3d, 
                'keypoints3d': k3d, # 这里保存两个，这样即使后面覆盖掉了keypoints3d还能取出pelvis来
                'views': views_all[select],
                'select': select,
                'indices': select - dimGroups[views_all[select]],
                'frames': [self.frames],
            })
        return results

    def assign_by_2D(self, used_index, distance, pelvis_all, views_all, dimGroups, cameras):
        def log_index_2d(index2d):
            return '({}|{}-{})'.format(index2d, views_all[index2d], index2d-dimGroups[views_all[index2d]])
        def log_indexes_2d(index2d_):
            return ', '.join(['({}|{}-{})'.format(index2d, views_all[index2d], index2d-dimGroups[views_all[index2d]]) for index2d in index2d_])
        INLIER_REPRO = self.cfg.max_pixel
        new_id_start = 10000
        new_max_id = new_id_start
        valid_3pairs, dist_3pair = self.find_initial_3_pair(distance, pelvis_all, views_all, dimGroups=dimGroups)
        results = []
        if valid_3pairs.sum() > 0:
            results = self.assign_by_2D_3pair(results, distance, dimGroups, used_index, valid_3pairs, views_all, pelvis_all, cameras)
        valid_theta = cameras['valid_theta']
        nViews = len(dimGroups)-1
        # 排序计算affinity
        count_rows = np.zeros((dimGroups[-1]), dtype=int)
        distance_rows = np.zeros((dimGroups[-1]))
        for nv in range(nViews):
            if dimGroups[nv] == dimGroups[nv+1]:continue
            valid_view = np.clip((distance[:, dimGroups[nv]:dimGroups[nv+1]] < INLIER_REPRO).sum(axis=-1), 0, 1)
            count_rows += valid_view # 最多也只累计一个
            distance_rows += valid_view * (distance[:, dimGroups[nv]:dimGroups[nv+1]].min(axis=-1))
        index = list(range(dimGroups[-1]))
        # index.sort(key=lambda x: (-count_rows[x], distance_rows[x]))
        # sort with 2D confidence
        # index.sort(key=lambda x: -pelvis_all[x, 2])
        # sort with valid matches
        # 选择2D的依据改为：根据有效的2D重投影距离的数量
        valid_count = (distance < INLIER_REPRO * 2).sum(axis=0)
        index = (-valid_count).argsort()

        visited2d = set()
        for row in index:
            visited2d.add(row)
            if used_index[row] > -1:continue
            FULL_LOG('[Assign 2D] Check 2D {}'.format(log_index_2d(row)))
            pid = new_max_id
            new_max_id += 1
            dist_row = distance[row]
            proposal = dist_row.argsort()
            current = [pelvis_all[row]]
            views = [views_all[row]]
            used_index[row] = pid
            # 初始化一下：
            for idx2d in proposal:
                # 不满足视角关系
                if not valid_theta[views_all[row], views_all[idx2d]]:
                    continue
                # 不满足距离关系
                if dist_row[idx2d] > INLIER_REPRO:
                    break
                if used_index[idx2d] > -1:
                    continue
                if views_all[idx2d] in views:
                    continue
                # self.triangulate_and_repro(cameras, [views_all[18], views_all[34]], [pelvis_all[18], pelvis_all[34]])
                # 2D的时候不能选择是最好的，因为2D可能还有其他视角的在
                # 顶多判断一下，是对于这个视角来说最好的
                # if not self.check_is_best_3d_of_2d(distance, row, idx2d, visited2d):
                #     continue
                # 找到了合理的pair，作为一个良好的初始化
                current.append(pelvis_all[idx2d])
                views.append(views_all[idx2d])
                used_index[idx2d] = pid
                FULL_LOG(f'[Assign 2D] Init with {log_index_2d(idx2d)}')
                break
            if len(current) < 2:
                # 没有找到良好的初始化
                continue
            for idx2d in proposal:
                if dist_row[idx2d] > INLIER_REPRO:
                    break
                # 这个视角已经有了 ｜ 这个2D已经被使用过了
                if views_all[idx2d] in views:
                    continue
                if used_index[idx2d] > -1:
                    continue
                # if not self.check_is_best_3d_of_2d(distance, row, idx2d, visited2d):
                #     continue
                # 尝试三角化并进行重投影
                new = current + [pelvis_all[idx2d]]
                views_new = views + [views_all[idx2d]]
                k3d, dist_repro, depth = self.triangulate_and_repro(cameras, views_new, new)
                flag_depth = (depth > 0.5).all()
                flag_repro = dist_repro.mean() < INLIER_REPRO
                flag = flag_repro & flag_depth
                FULL_LOG('[Assign 2D] repro: \n{}, \ndepth: \n{}'.format(LOG_ARRAY(dist_repro[None]), LOG_ARRAY(depth.T)))
                if flag:
                    # 添加
                    current = new
                    views = views_new
                    used_index[idx2d] = pid
                    _current_id = np.where(used_index == pid)[0]
                    FULL_LOG('[Assign 2D] {} => {}'.format(idx2d, log_indexes_2d(_current_id)))
                else:
                    FULL_LOG('[Assign 2D] Failed')
            if len(views) < self.cfg.min_views_init: #不足以添加
                continue
            k3d, dist_repro, depth = self.triangulate_and_repro(cameras, views, current)
            select = np.where(used_index == pid)[0]
            final_id = self.max_id
            self.max_id += 1
            used_index[select] = final_id
            results.append({
                'id': final_id,
                'pelvis': k3d, 
                'keypoints3d': k3d, # 这里保存两个，这样即使后面覆盖掉了keypoints3d还能取出pelvis来
                'views': views_all[select],
                'select': select,
                'indices': select - dimGroups[views_all[select]],
                'frames': [self.frames],
            })
        for res in results:
            text = f'''  - Init {res['id']} with {len(res['views'])} views
      views: {res['views']}
      id   : {res['select']}'''
            LOG(text)
            print(text)
        return results

class MatchRoot(MatchBase):
    def __init__(self, mode, cfg):
        super().__init__(mode, cfg)
        self.results = []
        self.frames = -1

    def __call__(self, pelvis, cameras, self_results=None):
        """
            cameras: {K, R, T, dist, P}
        """
        self.frames += 1
        LOG('>>> Current frames: {}'.format(self.frames))
        if self_results is None:
            self_results = self.results
        nViews = len(pelvis)
        pelvis_all, views_all, dimGroups = self.stack_array(pelvis)
        # Undistort
        pelvis_undis = self.undistort(pelvis, cameras)
        pelvis_undis_all, _, _ = self.stack_array(pelvis_undis)
        # distance3D => 2D
        distance3d_2d = self.calculate_repro(self_results, pelvis_all, cameras, views_all)
        # FULL_LOG('distance3d_2d: {}'.format(LOG_ARRAY(distance3d_2d)))
        # distance: triangulate and project
        distance2d_2d = self.calculate_distance(pelvis_undis, cameras, dimGroups)
        # FULL_LOG('distance2d_2d: {}'.format(LOG_ARRAY(distance2d_2d)))
        # set assign index
        used_index = np.zeros((dimGroups[-1]), dtype=int) - 1
        results = []
        # assign by 3D => 2D
        results3d = self_results
        if len(results3d) > 0:
            results3d = self.assign_by_3D(used_index, distance3d_2d, pelvis_undis_all, views_all, dimGroups, cameras)
        # assign by 2D + 2D
        results2d = self.assign_by_2D(used_index, distance2d_2d, pelvis_undis_all, views_all, dimGroups, cameras)
        results = results3d + results2d
        # distance = np.linalg.norm(keypoints3d[:, None, ..., :3] - keypoints3d[None, ..., :3], axis=-1).mean(axis=-1)
        # print(LOG_ARRAY(distance, format='{:6.2f}'))
        results.sort(key=lambda x: -len(x['views']))
        results = results[:self.cfg.max_person]
        
        if self.mode == 'track':
            self.results = results
        results.sort(key=lambda x:x['id'])
        # TODO: 增加结果的NMS检查和合并
        if len(results) == 0:
            keypoints3d = np.zeros((0, 25, 3))
        else:
            keypoints3d = np.stack([d['keypoints3d'] for d in results])
        return {'keypoints3d': keypoints3d, 'results': results}

class MatchTwoRoot(MatchRoot):
    def __init__(self, mode, cfg):
        keys = ['pelvis', 'neck']
        self._max_id_add = -1
        self._max_id = {key: 0 for key in keys}
        self.current = 'pelvis'
        self._results = {key: [] for key in keys}
        super().__init__(mode, cfg)
        self.results_limb = []
        self.mapping = {key: {} for key in keys}
    
    @property
    def max_id_add(self):
        self._max_id_add += 1
        return self._max_id_add
    
    @property
    def max_id(self):
        return self._max_id[self.current]
    
    @max_id.setter
    def max_id(self, index):
        self._max_id[self.current] = index
    
    @property
    def results(self):
        return self._results[self.current]
    
    @results.setter
    def results(self, val):
        self._results[self.current] = val

    @staticmethod
    def check_tracked(key, record_pelvis, current_3d, mapping):
        for ires, res in enumerate(record_pelvis):
            pid = res['id']
            res['limb_id'] = -1
            if pid in mapping[key]:
                p3d = mapping[key][pid]
                res['limb_id'] = p3d
                current_3d[p3d][key] = ires

    def __call__(self, cameras, openpose):
        pelvis_id = 8
        neck_id = 1
        pelvis = [openpose[v][pelvis_id] for v in range(len(openpose))]
        neck = [openpose[v][neck_id] for v in range(len(openpose))]
        self.current = 'pelvis'
        record_pelvis = super().__call__(pelvis, cameras)['results']
        self.current = 'neck'
        record_neck = super().__call__(neck, cameras)['results']
        current_3d = {p['id']: {'pelvis': -1, 'neck': -1} for p in self.results_limb}
        # 先检查是否已经track过了
        self.check_tracked('pelvis', record_pelvis, current_3d, self.mapping)
        self.check_tracked('neck', record_neck, current_3d, self.mapping)
        # 先整体记录一下ID；然后如果某一帧有丢掉的；就更新
        for p in self.results_limb:
            # 检查一下当前帧
            current_a, current_b = current_3d[p['id']]['pelvis'], current_3d[p['id']]['neck']
            if current_a != -1 and current_b != -1:
                assert current_a < len(record_pelvis) and current_b < len(record_neck), 'Index Error {}/{}, {}/{}'.format(current_a, current_b, len(record_pelvis), len(record_neck))
                p['pelvis'] = record_pelvis[current_a]['pelvis']
                p['neck'] = record_neck[current_b]['pelvis']
            elif current_a == -1 and current_b != -1:
                # a没有检测到，但b检测到了
                # 保持相对值
                mywarn('Missing Pelvis')
                p['neck'] = record_neck[current_b]['pelvis']
                pre_direc = p['pelvis'][:, :3] - p['neck'][:, :3]
                p['pelvis'][:, :3] = p['neck'][:, :3] + pre_direc
                # 得把补全的这个点设置回去
                self._results['pelvis'].append({
                    'id': p['pelvis_id'],
                    'pelvis': p['pelvis'],
                    'views': [],
                    'frames': [],
                    'indices': [],
                    'limb_id': p['id'],
                })
            elif current_a != -1 and current_b == -1:
                mywarn('Missing Neck')
                pre_direc = p['neck'][:, :3] - p['pelvis'][:, :3]
                p['pelvis'] = record_pelvis[current_a]['pelvis']
                p['neck'][:, :3] = p['pelvis'][:, :3] + pre_direc
                # 得把补全的这个点设置回去
                self._results['neck'].append({
                    'id': p['neck_id'],
                    'pelvis': p['neck'],
                    'views': [],
                    'frames': [],
                    'indices': [],
                    'limb_id': p['id'],
                })
            else:
                import ipdb; ipdb.set_trace()
                raise NotImplementedError
        # 遍历所有没有跟踪上的组合
        n_pelvis = len(record_pelvis)
        n_neck = len(record_neck)
        dist = np.zeros((n_pelvis, n_neck))
        # TODO: 用2D PAF来关联
        for i in range(n_pelvis):
            if record_pelvis[i]['limb_id'] > -1:
                continue
            for j in range(n_neck):
                if record_neck[j]['limb_id'] > -1:
                    continue
                pa = record_pelvis[i]['pelvis']
                pb = record_neck[j]['pelvis']
                length = np.linalg.norm(pa[:, :3] - pb[:, :3])
                dist[i, j] = length
        LIMB_MEAN = 0.489
        dist_to_mean = np.exp(-(dist - LIMB_MEAN)**2/(2*(LIMB_MEAN/3)**2))
        for i in range(n_pelvis):
            if record_pelvis[i]['limb_id'] > -1:
                continue
            for j in range(n_neck):
                if record_neck[j]['limb_id'] > -1:
                    continue
                pa = record_pelvis[i]['pelvis']
                pb = record_neck[j]['pelvis']
                if dist_to_mean[i, j] > 0.8:
                    # 可以接受
                    limb = {
                        'id': self.max_id_add,
                        'pelvis_id': record_pelvis[i]['id'],
                        'neck_id': record_neck[j]['id'],
                        'pelvis': pa,
                        'neck': pb,
                        'frames': [self.frames],
                    }
                    self.mapping['pelvis'][limb['pelvis_id']] = limb['id']
                    self.mapping['neck'][limb['neck_id']] = limb['id']
                    self.results_limb.append(limb)
        # 丢掉没有跟踪上的
        results = []
        for limb in self.results_limb:
            k3d = np.vstack([limb['pelvis'], limb['neck']])
            results.append({
                'id': limb['id'],
                'keypoints3d': k3d,
            })
        return {'results': results}
        

class MatchTorso(MatchBase):
    def __init__(self, mode, cfg):
        super().__init__(mode, cfg)
        self.results = []
        self.frames = -1

    @staticmethod
    def stack_pafs(pafs):
        dimGroups = [0]
        results = defaultdict(list)
        views_all = []
        for nv, paf in enumerate(pafs):
            src = paf['src']
            dimGroups.append(dimGroups[-1] + src.shape[0])
            views_all.extend([nv for _ in range(src.shape[0])])
            results['src'].append(src)
            results['dst'].append(paf['dst'])
            results['value'].append(paf['value'])
        results = {key: np.concatenate(val, axis=0) for key, val in results.items()}
        return results, np.array(views_all), np.array(dimGroups)
    
    def check_used_index(self, info_limb, index, info_joints):
        idx_src = info_limb['src'][index]
        idx_dst = info_limb['dst'][index]
        if info_joints['src']['used_index'][idx_src] > -1:
            return True
        if info_joints['dst']['used_index'][idx_dst] > -1:
            return True
        return False
    
    def set_used_index(self, info_limb, index, info_joints, pid):
        idx_src = info_limb['src'][index]
        idx_dst = info_limb['dst'][index]
        info_joints['src']['used_index'][idx_src] = pid
        info_joints['dst']['used_index'][idx_dst] = pid
        return True

    def triangulate_limb(self, info_limb, info_joints, index, views, cameras):
        flag = True
        k3d_all = []
        dist_all = []
        for key in ['src', 'dst']:
            proposals = []
            for idx in index:
                idx_ = info_limb[key][idx]
                proposals.append(info_joints[key]['detect_undis'][idx_])
            k3d, dist_repro, depth = self.triangulate_and_repro(cameras, views, proposals)
            dist_all.append(dist_repro)
            k3d_all.append(k3d)
        k3d_all = np.vstack(k3d_all)
        limb_length = np.linalg.norm(k3d_all[1, ..., :3] - k3d_all[0, ..., :3])
        if limb_length < 0.3 or limb_length > 0.7:
            flag = False
        dist_all = np.stack(dist_all)
        dist_all = np.max(dist_all, axis=0)
        return flag, k3d_all, dist_all

    # def assign_limb_by_2D(self, used_index, distance, pelvis_all, views_all, dimGroups, cameras):
    def assign_limb_by_2D(self, info_limb, info_joints, distance, views_all, dimGroups, cameras):
        def log_index_2d(index2d):
            src = info_limb['src'][index2d]
            dst = info_limb['dst'][index2d]
            src = src - info_joints['src']['dimGroups'][views_all[index2d]]
            dst = dst - info_joints['dst']['dimGroups'][views_all[index2d]]
            return '({}|{}-({},{}))'.format(index2d, views_all[index2d], src, dst)
        # def log_indexes_2d(index2d_):
        #     return ', '.join(['({}|{}-{})'.format(index2d, views_all[index2d], index2d-dimGroups[views_all[index2d]]) for index2d in index2d_])

        INLIER_REPRO = self.cfg.max_pixel
        valid_theta = cameras['valid_theta']
        index = self.sort_with_affinity(distance, dimGroups, INLIER_REPRO)
        visited2d = set()
        results = []
        new_id_start = 10000
        new_max_id = new_id_start
        for row in index:
            visited2d.add(row)
            if self.check_used_index(info_limb, row, info_joints):
                continue
            pid = new_max_id
            new_max_id += 1
            FULL_LOG('[Assign 2D] Check 2D {}'.format(log_index_2d(row)))
            dist_row = distance[row]
            proposal = dist_row.argsort()
            # 尝试初始化
            views = [views_all[row]]
            current = [row]
            for idx2d in proposal:
                # 不满足视角关系
                if not valid_theta[views_all[row], views_all[idx2d]]:
                    continue
                # 不满足距离关系
                if dist_row[idx2d] > INLIER_REPRO:
                    break
                if self.check_used_index(info_limb, idx2d, info_joints):
                    continue
                if views_all[idx2d] in views:
                    continue
                # 检查骨长
                flag, k3d, repro_error = self.triangulate_limb(info_limb, info_joints, [row, idx2d], [views_all[row], views_all[idx2d]], cameras)
                length = np.linalg.norm(k3d[1, ..., :3] - k3d[0, ..., :3])
                if flag:
                    views.append(views_all[idx2d])
                    current.append(idx2d)
                    FULL_LOG(f'[Assign 2D] Init with {log_index_2d(idx2d)}, length={length:.4f}')
                    break
                else:
                    FULL_LOG(f'[Assign 2D] Init failed with {log_index_2d(idx2d)}, length = {length:.4f}')
            if len(current) < 2:
                # 没有找到良好的初始化
                FULL_LOG(f'[Assign 2D] Cannot find a good initialization pair {log_index_2d(row)}')
                continue
            for idx2d in proposal:
                if dist_row[idx2d] > INLIER_REPRO:break
                # 这个视角已经有了 ｜ 这个2D已经被使用过了
                if views_all[idx2d] in views:
                    continue
                if self.check_used_index(info_limb, idx2d, info_joints):
                    continue
                FULL_LOG('[Assign 2D] Try to add 2D {} => {}'.format(idx2d, log_index_2d(idx2d)))
                # 尝试三角化并进行重投影
                new = current + [idx2d]
                views_new = views + [views_all[idx2d]]
                flag_limb, k3d, dist_repro = self.triangulate_limb(info_limb, info_joints, new, views_new, cameras)
                # flag_depth = (depth > 0.5).all()
                flag_depth = True
                flag_repro = dist_repro.mean() < INLIER_REPRO
                flag = flag_repro & flag_depth & flag_limb
                FULL_LOG('[Assign 2D] repro: \n{}'.format(LOG_ARRAY(dist_repro[None])))
                if flag:
                    # 添加
                    current = new
                    views = views_new
                    self.set_used_index(info_limb, idx2d, info_joints, pid)
                    FULL_LOG('[Assign 2D] {} => {}'.format(idx2d, current))
                else:
                    FULL_LOG('[Assign 2D] Failed')
                    new = None
                    views_new = None
            if len(views) < self.cfg.min_views: #不足以添加
                continue
            flag_limb, k3d, dist_repro = self.triangulate_limb(info_limb, info_joints, current, views, cameras)
            final_id = self.max_id
            self.max_id += 1
            results.append({
                'id': final_id,
                'torso': k3d, 
                'keypoints3d': k3d, # 这里保存两个，这样即使后面覆盖掉了keypoints3d还能取出pelvis来
                'views': views,
                'select': current,
                # 'indices': select - dimGroups[views_all[select]],
                'frames': [self.frames],
            })
        for res in results:
            text = f'''  - Init {res['id']} with {len(res['views'])} views
      views: {res['views']}
      id   : {res['select']}'''
            LOG(text)
            print(text)
        return results

    def calculte_distance_src_dst(self, src, dst, cameras):
        info = {}
        for name, detect in zip(['src', 'dst'], [src, dst]):
            detect_all, views_all, dimGroups = self.stack_array(detect)
            # Undistort
            detect_undis = self.undistort(detect, cameras)
            detect_undis_all, _, _ = self.stack_array(detect_undis)
            # # distance3D => 2D
            # distance3d_2d = self.calculate_repro(self.results, pelvis_all, cameras, views_all)
            # FULL_LOG('distance3d_2d: {}'.format(LOG_ARRAY(distance3d_2d)))
            # distance: triangulate and project
            distance2d_2d = self.calculate_distance(detect_undis, cameras, dimGroups)
            info[name] = {
                'detect_all': detect_all,
                'views_all': views_all,
                'dimGroups': dimGroups,
                'distance2d_2d': distance2d_2d,
                'detect_undis': detect_undis_all,
                'used_index': np.zeros((dimGroups[-1]), dtype=int) - 1
            }
        return info
    
    def get_valid_limbs(self, pafs, info_joint):
        nViews = len(pafs)
        valid_paf = []
        for nv in range(nViews):
            paf = pafs[nv]
            src, dst = np.where(paf > 0.3)
            value = paf[src, dst]
            valid_paf.append({
                'src': src + info_joint['src']['dimGroups'][nv],
                'dst': dst + info_joint['dst']['dimGroups'][nv],
                'value': value,
                'view': nv,
            })
        results, views_all, dimGroups = self.stack_pafs(valid_paf)
        return results, views_all, dimGroups

    def calculate_distance_limb(self, results, cameras, dimGroups, distance_src, distance_dst):
        src_idx, dst_idx = results['src'], results['dst']
        src_idx0, src_idx1 = np.meshgrid(src_idx, src_idx)
        dist_src_src = distance_src[src_idx0, src_idx1]
        dst_idx0, dst_idx1 = np.meshgrid(dst_idx, dst_idx)
        dist_dst_dst = distance_dst[dst_idx0, dst_idx1]
        # TODO: 考虑每个视角的 limb的置信度，joint的置信度
        dist_spatial = np.maximum(dist_src_src, dist_dst_dst)
        return dist_spatial

    def __call__(self, cameras, openpose, openpose_paf):
        """
            cameras: {K, R, T, dist, P}
        """
        self.frames += 1
        pelvis_id = 8
        neck_id = 1
        nViews = len(openpose)
        LOG('>>> Current frames: {}'.format(self.frames))
        pelvis = [openpose[v][pelvis_id] for v in range(len(openpose))]
        neck = [openpose[v][neck_id] for v in range(len(openpose))]
        info_joint = self.calculte_distance_src_dst(pelvis, neck, cameras)
        pafs = [openpose_paf[v][(pelvis_id, neck_id)] for v in range(len(openpose_paf))]
        info_limb, views_all, dimGroups = self.get_valid_limbs(pafs, info_joint)
        distance2d_2d = self.calculate_distance_limb(info_limb, cameras, dimGroups, 
                            info_joint['src']['distance2d_2d'], info_joint['dst']['distance2d_2d'])
        results = self.assign_limb_by_2D(info_limb, info_joint, distance2d_2d, views_all, dimGroups, cameras)
        results.sort(key=lambda x: -len(x['views']))
        results = results[:self.cfg.max_person]
        # if self.mode == 'track':
        #     self.results = results
        results.sort(key=lambda x:x['id'])

        # TODO: 增加结果的NMS检查和合并
        if len(results) == 0:
            keypoints3d = np.zeros((0, 2, 3))
        else:
            keypoints3d = np.stack([d['keypoints3d'] for d in results])
        return {'keypoints3d': keypoints3d, 'results': results}

class TriangulateAll:
    def __init__(self, mode) -> None:
        self.mode = mode
    
    def __call__(self, bbox, keypoints, cameras, results):
        for res in results:
            bbox_, k2d, Pall = [], [], []
            for i in range(len(res['views'])):
                v = res['views'][i]
                bbox_.append(bbox[v][res['indices'][i]])
                k2d.append(keypoints[v][res['indices'][i]])
                Pall.append(cameras['P'][v])
            k2d = np.stack(k2d)
            Pall = np.stack(Pall)
            bbox_ = np.stack(bbox_)
            if self.mode == 'naive':
                k3d = batch_triangulate(k2d, Pall)
            elif self.mode == 'robust':
                from easymocap.mytools.triangulator import iterative_triangulate
                k3d, k2d = iterative_triangulate(k2d, Pall,
                    dist_max=25)
            res['keypoints3d'] = k3d
            res['keypoints2d'] = k2d
            res['bbox'] = bbox_
        return {'keypoints3d': np.stack([d['keypoints3d'] for d in results]), 'results': results}

class MatchHandLR:
    def __init__(self, mode, cfg):
        self.model_l = MatchRoot(mode,cfg)
        self.model_r = MatchRoot(mode,cfg)
    def __call__(self, pelvis_l, pelvis_r, cameras):
        ret = {}
        outl = self.model_l(pelvis_l, cameras)
        outr = self.model_r(pelvis_r, cameras)
        for k in outl.keys():
            ret[k+'_l'] = outl[k]
        for k in outr.keys():
            ret[k+'_r'] = outr[k]
        return ret
    
class MatchBodyHand:
    def __init__(self, mode) -> None:
        pass
    
    def projectPoints(self, X, K, R, t, Kd):    
        x = R @ X + t
        x[0:2,:] = x[0:2,:]/x[2,:]#到归一化平面
        r = x[0,:]*x[0,:] + x[1,:]*x[1,:]

        x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
        x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])
        x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
        x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
        return x
    def match3d_step(self, results, keypoints3d,wristid):
        match_results=(np.zeros((len(keypoints3d)),dtype=np.int)-1).tolist()
        vis = (np.zeros((len(keypoints3d)))-1).tolist()
        dis = []
        for i in range(len(keypoints3d)):
            for j in range(len(results)):
                dis.append([i,j,((keypoints3d[i][wristid][:3]-results[j]['pelvis'][0,:3].reshape(-1))**2).sum()])
        if(len(dis)>0):
            dis = np.array(dis)
            dis = dis[np.argsort(dis[:,-1])]
            for i in range(len(dis)):
                bid =int(dis[i][0])
                hid =int(dis[i][1])
                if vis[bid]>=0 or hid in vis:
                    continue
                if dis[i][2]>0.5:
                    continue
                tmp_results = results[hid].copy()
                # tmp_results['dis_bh'] = dis[i][2]
                match_results[bid]=tmp_results
                vis[bid]=hid
        return match_results
    def match2d_step(self, bbox_hand, keypoints3d, wristid, results_match_l, cameras):
        lack_body_id=[]
        mv_use_hand=[]
        for i in range(cameras['R'].shape[0]):
            mv_use_hand.append([])

        for i in range(len(results_match_l)):
            if isinstance(results_match_l[i],int) and results_match_l[i]==-1:
                lack_body_id.append(i)
            else:
                mv = results_match_l[i]['views']#[cid]
                indices = results_match_l[i]['indices']#[cid]
                for j in range(len(mv)):
                    mv_use_hand[mv[j]].append(indices[j])

        wrist3dkpts = keypoints3d[lack_body_id,wristid,:3] #(nperson,3)每个人呢的wrist关键点
        dis = []
        for nv in range(len(bbox_hand)):
            for hid in range(len(bbox_hand[nv])):
                if hid in mv_use_hand[nv]:
                    continue
                if bbox_hand[nv][hid][-1]==0:
                    continue
                bx_ = bbox_hand[nv][hid]
                k2d = np.array([(bx_[0]+bx_[2])/2,(bx_[1]+bx_[3])/2,bx_[-1]])
                K =  cameras['K'][nv]                 
                Kd = cameras['dist'][nv].reshape(5)
                R = cameras['R'][nv]                 
                t = cameras['T'][nv]                 
                wristkpts2d = self.projectPoints(wrist3dkpts.T[0:3,:], K, R, t, Kd).T
                for bid in range(len(lack_body_id)):
                    D = ((wristkpts2d[bid][:2]-k2d[:2])**2).sum()
                    dis.append([D,lack_body_id[bid],nv,hid]) # 误差，3d身体id ,视角编号  ,2d图像上手box id

        if(len(dis)>0):
            vis = (np.zeros((len(keypoints3d)))-1).tolist()
            dis = np.array(dis)
            dis = dis[np.argsort(dis[:,0])]
            # TODO 判断dis大小，将dis过大的删除掉
            for i in range(len(dis)):
                bid = int(dis[i][1])
                nv =  int(dis[i][2])
                hid = int(dis[i][3])
                if vis[bid]>=0 or hid in vis or results_match_l[bid]!=-1:
                    continue
                if dis[i][0]>50: #人和手的在2D中距离
                    continue
                results_match_l[bid]={
                    'views': np.array([nv]), 
                    'indices': np.array([hid]), # ?indices是在对应的视角下第几个Box 
                    # 'dis_bh': dis[i][0]
                }

                vis[bid]=hid
        return results_match_l


    def __call__(self, results_l, results_r, keypoints3d, cameras, bbox_handl, bbox_handr):
        '''
        results: list nhand
        keypoints3d: (nperson,25,3)
        '''
        results_match_l = self.match3d_step(results_l, keypoints3d, 7)
        results_match_r = self.match3d_step(results_r, keypoints3d, 4)

        if(-1 in results_match_l):
            # TODO: dis为空，则表示没有身体，或者所有视角都未检测到手，尝试启动单视角检测
            # TODO: dis不为空，也有可能有的身体缺少与手的匹配，可以尝试单视角检测，或者之后尝试补全。
            # 单视角匹配，从匹配列表中找出-1的部分，将其投影到多视角中，在多视角找出未被选择的box，然后匹配，记录在a
            results_match_l = self.match2d_step(bbox_handl, keypoints3d, 7, results_match_l, cameras)
        if(-1 in results_match_r):
            results_match_r = self.match2d_step(bbox_handr, keypoints3d, 4, results_match_r, cameras)
        return  {'match3d_l':results_match_l ,'match3d_r':results_match_r}
