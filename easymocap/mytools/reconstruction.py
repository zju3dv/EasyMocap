'''
 * @ Date: 2020-09-14 11:01:52
 * @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-13 20:31:34
  @ FilePath: /EasyMocapRelease/media/qing/Project/mirror/EasyMocap/easymocap/mytools/reconstruction.py
'''

import numpy as np

def solveZ(A):
    u, s, v = np.linalg.svd(A)
    X = v[-1, :]
    X = X / X[3]
    return X[:3]

def projectN3(kpts3d, Pall):
    # kpts3d: (N, 3)
    nViews = len(Pall)
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    if kpts3d.shape[-1] == 4:
        kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)
    return kp2ds

def simple_reprojection_error(kpts1, kpts1_proj):
    # (N, 3)
    error = np.mean((kpts1[:, :2] - kpts1_proj[:, :2])**2)
    return error
    
def simple_triangulate(kpts, Pall):
    # kpts: (nViews, 3)
    # Pall: (nViews, 3, 4)
    #   return: kpts3d(3,), conf: float
    nViews = len(kpts)
    A = np.zeros((nViews*2, 4), dtype=np.float)
    result = np.zeros(4)
    result[3] = kpts[:, 2].sum()/(kpts[:, 2]>0).sum()
    for i in range(nViews):
        P = Pall[i]
        A[i*2, :] = kpts[i, 2]*(kpts[i, 0]*P[2:3,:] - P[0:1,:])
        A[i*2 + 1, :] = kpts[i, 2]*(kpts[i, 1]*P[2:3,:] - P[1:2,:])
    result[:3] = solveZ(A)

    return result

def batch_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v > 1)[0]
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
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result

eps = 0.01
def simple_recon_person(keypoints_use, Puse):
    out = batch_triangulate(keypoints_use, Puse)
    # compute reprojection error
    kpts_repro = projectN3(out, Puse)
    square_diff = (keypoints_use[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = np.repeat(out[None, :, -1:], len(Puse), 0)
    kpts_repro = np.concatenate((kpts_repro, conf), axis=2)
    return out, kpts_repro

def check_limb(keypoints3d, limb_means, thres=0.5):
    # keypoints3d: (nJ, 4)
    valid = True
    cnt = 0
    for (src, dst), val in limb_means.items():
        if not (keypoints3d[src, 3] > 0 and keypoints3d[dst, 3] > 0):
            continue
        cnt += 1 
        # 计算骨长
        l_est = np.linalg.norm(keypoints3d[src, :3] - keypoints3d[dst, :3])
        if abs(l_est - val['mean'])/val['mean']/val['std'] > thres:
            valid = False
            break
    # 至少两段骨头可以使用
    valid = valid and cnt > 2
    return valid
