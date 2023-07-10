import numpy as np
from tqdm import tqdm

class CheckFramePerson:
    def __init__(self, key) -> None:
        self.key = key
        self.pids = []
        self.frames = 0
    
    def __call__(self, keypoints3d, pids):
        k3d_, pid_ = [], []
        for i, pid in enumerate(pids):
            if pid not in self.pids:
                if self.frames == 0:
                    print('[{}]/{:06d} Add person {}'.format(self.__class__.__name__, self.frames, pid))
                    self.pids.append(pid)
                else:
                    continue
            k3d_.append(keypoints3d[i])
            pid_.append(pid)
        self.frames += 1
        k3d_ = np.stack(k3d_)
        return {
            'keypoints3d': k3d_,
            'pids': pid_
        }

class CollectMultiPersonMultiFrame:
    def __init__(self, key, min_frame=10) -> None:
        self.key = key
        self.min_frame = min_frame
    
    def __call__(self, keypoints3d, pids):
        records = {}
        for frame in tqdm(range(len(pids)), desc='Reading'):
            pid_frame = pids[frame]
            for i, pid in enumerate(pid_frame):
                if pid not in records:
                    records[pid] = {
                        'frames': [],
                        'keypoints3d': []
                    }
                records[pid]['frames'].append(frame)
                records[pid]['keypoints3d'].append(keypoints3d[frame][i])
        remove_id = []
        for pid, record in records.items():
            print('[{}] Collect person {} with {} frames'.format(self.__class__.__name__, pid, len(record['frames'])))
            record['keypoints3d'] = np.stack(record['keypoints3d']).astype(np.float32)
            if len(record['frames']) < self.min_frame:
                remove_id.append(pid)
        for pid in remove_id:
            records.pop(pid)
        return {'results': records}