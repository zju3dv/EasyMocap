'''
  @ Date: 2021-05-28 16:36:45
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-30 12:21:15
  @ FilePath: /EasyMocap/easymocap/assignment/criterion.py
'''
import numpy as np

class BaseCrit:
    def __init__(self, min_conf, min_joints=3) -> None:
        self.min_conf = min_conf
        self.min_joints = min_joints
        self.name = self.__class__.__name__

    def __call__(self, keypoints3d, **kwargs):
        # keypoints3d: (N, 4)
        conf = keypoints3d[..., -1]
        conf[conf<self.min_conf] = 0
        idx = keypoints3d[..., -1] > self.min_conf
        return len(idx) > self.min_joints

class CritWithTorso(BaseCrit):
    def __init__(self, torso_idx, min_conf, **kwargs) -> None:
        super().__init__(min_conf)
        self.idx = torso_idx
        self.min_conf = min_conf
    
    def __call__(self, keypoints3d, **kwargs) -> bool:
        self.log = '{}'.format(keypoints3d[self.idx, -1])
        return (keypoints3d[self.idx, -1] > self.min_conf).all()

class CritLenTorso(BaseCrit):
    def __init__(self, src, dst, min_torso_length, max_torso_length, min_conf) -> None:
        super().__init__(min_conf)
        self.src = src
        self.dst = dst
        self.min_torso_length = min_torso_length
        self.max_torso_length = max_torso_length
    
    def __call__(self, keypoints3d, **kwargs):
        """length of torso"""
        # eps = 0.1
        # MIN_TORSO_LENGTH = 0.3
        # MAX_TORSO_LENGTH = 0.8
        if (keypoints3d[[self.src, self.dst], -1] < self.min_conf).all():
            # low confidence, skip
            return True
        length = np.linalg.norm(keypoints3d[self.dst] - keypoints3d[self.src])
        self.log = '{}: {:.3f}'.format(self.name, length)
        if length < self.min_torso_length or length > self.max_torso_length:
            return False
        return True

class CritRange(BaseCrit):
    def __init__(self, minr, maxr, rate_inlier, min_conf) -> None:
        super().__init__(min_conf)
        self.min = minr
        self.max = maxr
        self.rate = rate_inlier
    
    def __call__(self, keypoints3d, **kwargs):
        idx = keypoints3d[..., -1] > self.min_conf
        k3d = keypoints3d[idx, :3]
        crit = (k3d[:, 0] > self.min[0]) & (k3d[:, 0] < self.max[0]) &\
        (k3d[:, 1] > self.min[1]) & (k3d[:, 1] < self.max[1]) &\
        (k3d[:, 2] > self.min[2]) & (k3d[:, 2] < self.max[2])
        self.log = '{}: {}'.format(self.name, k3d)
        return crit.sum()/crit.shape[0] > self.rate

class CritMinMax(BaseCrit):  
    def __init__(self, max_human_length, min_conf) -> None:
        super().__init__(min_conf)
        self.max_human_length = max_human_length

    def __call__(self, keypoints3d, **kwargs):
        idx = keypoints3d[..., -1] > self.min_conf
        k3d = keypoints3d[idx, :3]
        mink = np.min(k3d, axis=0)
        maxk = np.max(k3d, axis=0)
        length = max(np.abs(maxk - mink))
        self.log = '{}: {:.3f}'.format(self.name, length)
        return length < self.max_human_length