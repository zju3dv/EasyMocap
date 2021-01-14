'''
 * @ Date: 2020-09-14 11:01:52
 * @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-13 11:30:38
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
    # kpts_proj = projectN3(result, Pall)
    # repro_error = simple_reprojection_error(kpts, kpts_proj)
    #     return kpts3d, conf/nViews, repro_error/nViews
    # else:
    #     return kpts3d, conf
        
def simple_recon_person(keypoints_use, Puse, ret_repro=False, max_error=100):
    nJoints = keypoints_use[0].shape[0]
    if isinstance(keypoints_use, list):
        keypoints_use = np.stack(keypoints_use)
    out = np.zeros((nJoints, 4))
    for nj in range(nJoints):
        keypoints = keypoints_use[:, nj]
        if (keypoints[:, 2] > 0.01).sum() < 2:
            continue
        out[nj] = simple_triangulate(keypoints, Puse)
    # 计算重投影误差
    kpts_repro = projectN3(out, Puse)
    square_diff = (keypoints_use[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = (out[None, :, -1] > 0.01) * (keypoints_use[:, :, 2] > 0.01)
    if conf.sum() < 3: # 至少得有3个有效的关节
        repro_error = 1e3
    else:
        repro_error_joint = np.sqrt(square_diff.sum(axis=2))*conf
        num_valid_view = conf.sum(axis=0)
        # 对于可见视角少的，强行设置为不可见
        repro_error_joint[:, num_valid_view==0] = max_error * 2
        num_valid_view[num_valid_view==0] = 1
        repro_error_joint_ = repro_error_joint.sum(axis=0)/num_valid_view
        # print(repro_error_joint_)
        not_valid = np.where(repro_error_joint_>max_error)[0]
        out[not_valid, -1] = 0
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