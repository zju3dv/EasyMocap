'''
  @ Date: 2021-06-27 16:21:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-28 10:59:59
  @ FilePath: /EasyMocapRelease/easymocap/assignment/track.py
'''
from tqdm import tqdm
import numpy as np
import os
from os.path import join
from glob import glob
from ..affinity.affinity import getDimGroups
from ..affinity.matchSVT import matchSVT
from ..mytools.reader import read_keypoints2d, read_keypoints3d
from ..mytools.file_utils import read_annot, read_json, save_annot, save_json, write_keypoints3d

def check_path(x):
    assert os.path.exists(x), '{} not exists!'.format(x)

class BaseTrack:
    def __init__(self, path, out, WINDOW_SIZE, MIN_FRAMES, SMOOTH_SIZE) -> None:
        self.path = path
        self.out = out
        self.WINDOW_SIZE = WINDOW_SIZE
        self.SMOOTH_SIZE = SMOOTH_SIZE
        self.MIN_FRAMES = MIN_FRAMES
        self.svt_args = {
            'maxIter': 1000,
            'w_sparse': 0.3,
            'w_rank': 10,
            'tol': 1e-4,
            'log': False
        }
    
    def auto_track(self):
        results = self.read()
        edges = self.compute_dist(results)
        results = self.associate(results, edges)
        results, occupancy = self.reset_id(results)
        results, occupancy = self.smooth(results, occupancy)
        self.write(results, occupancy)

    def read(self):
        return []

    def write(self, results, occupancy):
        return 0

    def compute_dist(self, results):
        nFrames = len(results)
        WINDOW_SIZE = self.WINDOW_SIZE
        edges = {}
        for start in tqdm(range(0, nFrames - 1), desc='affinity'):
            window_size = min(WINDOW_SIZE, nFrames - start)
            results_window = results[start:start+window_size]
            dimGroups, frames = getDimGroups(results_window)
            dist = self._compute_dist(dimGroups, results_window)
            res = matchSVT(dist, dimGroups, control=self.svt_args)
            xx, yy = np.where(res)
            for x, y in zip(xx, yy):
                if x >= y:continue
                nf0, nf1 = frames[x], frames[y]
                ni0, ni1 = x - dimGroups[nf0], y - dimGroups[nf1]
                edge = ((nf0+start, ni0), (nf1+start, ni1))
                if edge not in edges:
                    edges[edge] = []
                edges[edge].append(res[x, y])
        return edges

    def associate(self, results, edges):
        WINDOW_SIZE = self.WINDOW_SIZE
        connects = list(edges.keys())
        connects.sort(key=lambda x:-sum(edges[x]))
        maxid = 0
        frames_of_id = {}
        log = print
        log = lambda x:x
        for (nf0, ni0), (nf1, ni1) in connects:
            if abs(nf1 - nf0) > WINDOW_SIZE//2:
                continue
            # create
            id0 = results[nf0][ni0]['id']
            id1 = results[nf1][ni1]['id']
            if id0 == -1 and id1 == -1:
                results[nf0][ni0]['id'] = maxid
                log('Create person {}'.format(maxid))
                frames_of_id[maxid] = {nf0:ni0, nf1:ni1}
                maxid += 1
            # directly assign
            if id0 != -1 and id1 == -1:
                if nf1 in frames_of_id[id0].keys():
                    log(f'Merge conflict1 nf0: {nf0} ni0: {ni0} id0: {id0} nf1: {nf1} ni1: {ni1} id1: {id1}')
                    continue
                results[nf1][ni1]['id'] = id0
                # log('Merge person {}'.format(maxid))
                frames_of_id[id0][nf1] = ni1
                continue
            if id0 == -1 and id1 != -1:
                if nf0 in frames_of_id[id1].keys():
                    log(f'Merge conflict2 nf0: {nf0} ni0: {ni0} id0: {id0} nf1: {nf1} ni1: {ni1} id1: {id1}')
                    continue
                results[nf0][ni0]['id'] = id1
                frames_of_id[id1][nf0] = ni0
                continue
            if id0 == id1:
                continue
            # merge
            if id0 != id1:
                common = frames_of_id[id0].keys() & frames_of_id[id1].keys()
                for key in common: # conflict
                    if frames_of_id[id0][key] == frames_of_id[id1][key]:
                        pass
                    else:
                        break
                else: # merge
                    log('Merge {} to {}'.format(id1, id0))
                    for key in frames_of_id[id1].keys():
                        results[key][frames_of_id[id1][key]]['id'] = id0
                        frames_of_id[id0][key] = frames_of_id[id1][key]
                    frames_of_id.pop(id1)
                    continue
                log('Conflict; not merged')
        return results
    
    def reset_id(self, results):
        mapid = {}
        maxid = 0
        occupancy = []
        nFrames = len(results)
        for nf, res in enumerate(results):
            for info in res:
                if info['id'] == -1:
                    continue
                if info['id'] not in mapid.keys():
                    mapid[info['id']] = maxid
                    maxid += 1
                    occupancy.append([0 for _ in range(nFrames)])
                pid = mapid[info['id']]
                info['id'] = pid
                occupancy[pid][nf] = 1
        occupancy = np.array(occupancy)
        results, occupancy = self.remove_outlier(results, occupancy)
        results, occupancy = self.interpolate(results, occupancy)
        return results, occupancy

    def remove_outlier(self, results, occupancy):
        nFrames = len(results)
        pids = []
        for pid in range(occupancy.shape[0]):
            if occupancy[pid].sum() > self.MIN_FRAMES:
                pids.append(pid)
            else:
                print('[track] remove {} with {} frames'.format(pid, occupancy[pid].sum()))
        occupancy = occupancy[pids]
        for nf in range(nFrames):
            result = results[nf]
            result_filter = []
            for info in result:
                if info['id'] == -1 or info['id'] not in pids:
                    continue
                info['id'] = pids.index(info['id'])
                result_filter.append(info)
            results[nf] = result_filter
        return results, occupancy

    def interpolate(self, results, occupancy):
        # find missing frames
        WINDOW_SIZE = self.WINDOW_SIZE
        for pid in range(occupancy.shape[0]):
            for nf in range(1, occupancy.shape[1]-1):
                if occupancy[pid, nf-1] < 1 or occupancy[pid, nf] > 0:
                    continue
                left = nf - 1
                right = np.where(occupancy[pid, nf+1:])[0]
                if len(right) > 0:
                    right = right.min() + nf + 1
                else:
                    continue
                print('[interp] {} in [{}, {}]'.format(pid, left, right))
                # find valid (left, right)
                # interpolate 3d pose
                info_left = [res for res in results[left] if res['id'] == pid][0]
                info_right = [res for res in results[right] if res['id'] == pid][0]
                for nf_i in range(left+1, right):
                    weight = 1 - (nf_i - left)/(right - left)
                    res = self._interpolate(info_left, info_right, weight)
                    res['id'] = pid
                    results[nf_i].append(res)
                    occupancy[pid, nf_i] = 1
        return results, occupancy
    
    def smooth(self, results, occupancy):
        return results, occupancy

    def _interpolate(self, info_left, info_right, weight):
        return info_left.copy()

class Track3D(BaseTrack):
    def __init__(self, with2d=False, mode='body25', **cfg) -> None:
        super().__init__(**cfg)
        self.with2d = with2d
        self.mode = mode

    def read(self):
        k3dpath = join(self.path, 'keypoints3d')
        check_path(k3dpath)
        filenames = sorted(glob(join(k3dpath, '*.json')))
        if self.with2d:
            k2dpath = join(self.path, 'keypoints2d')
            check_path(k2dpath)
            subs = sorted(os.listdir(k2dpath))
        else:
            k2dpath = ''
            subs = []
        results = []
        for nf, filename in enumerate(filenames):
            basename = os.path.basename(filename)
            infos = read_keypoints3d(filename)
            for n, info in enumerate(infos):
                info['id'] = -1
                info['index'] = n

            results.append(infos)
            if self.with2d:
                # load 2d keypoints
                for nv, sub in enumerate(subs):
                    k2dname = join(k2dpath, sub, basename)
                    annots = read_keypoints2d(k2dname, self.mode)
                    for annot in annots:
                        pid = annot['id']
                        bbox = annot['bbox']
                        keypoints = annot['keypoints']
                        import ipdb; ipdb.set_trace()
        return results
    
    def write(self, results, occupancy):
        os.makedirs(self.out, exist_ok=True)
        for nf, res in enumerate(tqdm(results)):
            outname = join(self.out, 'keypoints3d', '{:06d}.json'.format(nf))
            result = results[nf]
            write_keypoints3d(outname, result)

    def _compute_dist(self, dimGroups, results_window):
        max_dist = 0.15
        max_dist_step = 0.01
        window_size = len(results_window)
        dist = np.eye(dimGroups[-1])
        for i in range(window_size-1):
            if len(results_window[i]) == 0:
                continue
            k3d_i = np.stack([info['keypoints3d'] for info in results_window[i]])
            for j in range(i+1, window_size):
                if len(results_window[j]) == 0:
                    continue
                k3d_j = np.stack([info['keypoints3d'] for info in results_window[j]])
                conf = np.sqrt(k3d_i[:, None, :, 3] * k3d_j[None, :, :, 3])
                d_ij = np.linalg.norm(k3d_i[:, None, :, :3] - k3d_j[None, :, :, :3], axis=3)
                a_ij = 1 - d_ij / (max_dist + (j-i)*max_dist_step )
                a_ij[a_ij < 0] = 0
                weight  =(conf*a_ij).sum(axis=2)/(1e-4 + conf.sum(axis=2))
                dist[dimGroups[i]:dimGroups[i+1], dimGroups[j]:dimGroups[j+1]] = weight
                dist[dimGroups[j]:dimGroups[j+1], dimGroups[i]:dimGroups[i+1]] = weight.T
        return dist
    
    def _interpolate(self, info_left, info_right, weight):
        kpts_new = info_left['keypoints3d'] * weight + info_right['keypoints3d'] * (1-weight)
        res = {'keypoints3d': kpts_new}
        return res

class Track2D(BaseTrack):
    def __init__(self, **cfg) -> None:
        super().__init__(**cfg)

    def read(self):
        filenames = sorted(glob(join(self.path, '*.json')))
        results = []
        for filename in tqdm(filenames, desc='loading'):
            result = read_json(filename)['annots']
            for n, info in enumerate(result):
                info['id'] = -1
            results.append(result)
        return results
    
    def write(self, results, occupancy):
        os.makedirs(self.out, exist_ok=True)
        filenames = sorted(glob(join(self.path, '*.json')))
        for nf, res in enumerate(tqdm(results, desc='writing')):
            outname = join(self.out, '{:06d}.json'.format(nf))
            result = results[nf]
            annots = read_json(filenames[nf])
            annots['annots'] = result
            for res in result:
                res['personID'] = res.pop('id')
            save_annot(outname, annots)
        annot = os.path.basename(os.path.dirname(self.out))
        occpath = self.out.replace(annot, 'track') + '.json'
        save_json(occpath, occupancy.tolist())

    def _compute_dist(self, dimGroups, results_window):
        window_size = len(results_window)
        dist = np.eye(dimGroups[-1])
        for i in range(window_size-1):
            if len(results_window[i]) == 0:
                continue
            bbox_pre = np.stack([info['bbox'] for info in results_window[i]])
            bbox_pre = bbox_pre[:, None]
            for j in range(i+1, window_size):
                if len(results_window[j]) == 0:
                    continue
                bbox_now = np.stack([info['bbox'] for info in results_window[j]])
                bbox_now = bbox_now[None, :]
                areas_pre = (bbox_pre[..., 2] - bbox_pre[..., 0]) * (bbox_pre[..., 3] - bbox_pre[..., 1])
                areas_now = (bbox_now[..., 2] - bbox_now[..., 0]) * (bbox_now[..., 3] - bbox_now[..., 1])
                # 左边界的大值
                xx1 = np.maximum(bbox_pre[..., 0], bbox_now[..., 0])
                yy1 = np.maximum(bbox_pre[..., 1], bbox_now[..., 1])
                # 右边界的小值
                xx2 = np.minimum(bbox_pre[..., 2], bbox_now[..., 2])
                yy2 = np.minimum(bbox_pre[..., 3], bbox_now[..., 3])
                
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                over = inter / (areas_pre + areas_now - inter)
                weight  = over
                dist[dimGroups[i]:dimGroups[i+1], dimGroups[j]:dimGroups[j+1]] = weight
                dist[dimGroups[j]:dimGroups[j+1], dimGroups[i]:dimGroups[i+1]] = weight.T
        return dist
    
    def reset_id(self, results):
        results[0].sort(key=lambda x:-(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
        return super().reset_id(results)
    
    def _interpolate(self, info_left, info_right, weight):
        bbox = [info_left['bbox'][i]*weight+info_left['bbox'][i]*(1-weight) for i in range(5)]
        kpts_l = info_left['keypoints']
        kpts_r = info_right['keypoints']
        kpts = []
        for nj in range(len(kpts_l)):
            if kpts_l[nj][2] < 0.1 or kpts_r[nj][2] < 0.1:
                kpts.append([0., 0., 0.])
            else:
                kpts.append([kpts_l[nj][i]*weight + kpts_r[nj][i]*(1-weight) for i in range(3)])
        res = {'bbox': bbox, 'keypoints': kpts}
        return res
    
    def smooth(self, results, occupancy):
        for pid in range(occupancy.shape[0]):
            # the occupancy must be continuous
            pass
        return results, occupancy