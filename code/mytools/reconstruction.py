'''
 * @ Date: 2020-09-14 11:01:52
 * @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-25 16:06:41
  @ FilePath: /EasyMocap/code/mytools/reconstruction.py
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
        
def simple_recon_person(keypoints_use, Puse, config=None, ret_repro=False):
    eps = 0.01
    nJoints = keypoints_use[0].shape[0]
    if isinstance(keypoints_use, list):
        keypoints_use = np.stack(keypoints_use)
    out = np.zeros((nJoints, 4))
    for nj in range(nJoints):
        keypoints = keypoints_use[:, nj]
        if (keypoints[:, 2] > 0.01).sum() < 2:
            continue
        out[nj] = simple_triangulate(keypoints, Puse)
    if config is not None:
        # remove the false limb with the help of limb
        for (i, j), mean_std in config['skeleton'].items():
            ii, jj = min(i, j), max(i, j)
            if out[ii, -1] < eps:
                out[jj, -1] = 0
            if out[jj, -1] < eps:
                continue
            length = np.linalg.norm(out[ii, :3] - out[jj, :3])
            if abs(length - mean_std['mean'])/(3*mean_std['std']) > 1:
                # print((i, j), length, mean_std)
                out[jj, :] = 0
    # 计算重投影误差
    kpts_repro = projectN3(out, Puse)
    square_diff = (keypoints_use[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = np.repeat(out[None, :, -1:], len(Puse), 0)
    kpts_repro = np.concatenate((kpts_repro, conf), axis=2)
    if conf.sum() < 3: # 至少得有3个有效的关节
        repro_error = 1e3
    else:
        conf2d = conf *(keypoints_use[:, :, -1:] > 0.01)
        # (nViews, nJoints): reprojection error for each joint in each view
        repro_error_joint = np.sqrt(square_diff.sum(axis=2, keepdims=True))*conf2d
        # remove the not valid joints
        # remove the bad views
        repro_error = repro_error_joint.sum()/conf.sum()
    
    if ret_repro:
        return out, repro_error, kpts_repro
    return out, repro_error

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