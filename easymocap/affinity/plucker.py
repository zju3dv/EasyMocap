'''
  @ Date: 2021-01-25 21:27:56
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-07-28 17:18:20
  @ FilePath: /EasyMocap/easymocap/affinity/plucker.py
'''
import numpy as np

def plucker_from_pl(point, line):
    """ construct plucker line from a point and directions
    
    Arguments:
        point {tensor} -- N, 3
        line {tensor} -- N, 3
    """
    norm = np.linalg.norm(line, axis=-1, keepdims=True)
    lunit = line/norm
    moment = np.cross(point, lunit, axis=-1)
    return lunit, moment

def plucker_from_pp(point1, point2):
    line = point2 - point1
    return plucker_from_pl(point1, line)

def dist_pl(query_points, line, moment):
    moment_q = moment - np.cross(query_points, line)
    dist = np.linalg.norm(moment_q, axis=1)
    return dist

def reciprocal_product(l1, m1, l2, m2):
    l1 = l1[:, None]
    m1 = m1[:, None]
    l2 = l2[None, :]
    m2 = m2[None, :]
    product = np.sum(l1*m2, axis=2) + np.sum(l2*m1, axis=2)
    return np.abs(product)

def dist_pl_pointwise(p0, p1):
    moment_q = p1[..., 3:6] - np.cross(p0[..., :3], p1[..., :3])
    dist = np.linalg.norm(moment_q, axis=-1)
    return dist

def dist_ll_pointwise(p0, p1):
    product = np.sum(p0[..., :3] * p1[..., 3:6], axis=-1) + np.sum(p1[..., :3] * p0[..., 3:6], axis=-1)
    return np.abs(product)

def dist_ll_pointwise_conf(p0, p1):
    dist = dist_ll_pointwise(p0, p1)
    conf = np.sqrt(p0[..., -1] * p1[..., -1])
    dist = np.sum(dist*conf, axis=-1)/(1e-5 + conf.sum(axis=-1))
    dist[conf.sum(axis=-1)<0.1] = 1e5
    return dist

def computeRay(keypoints2d, invK, R, T):
    # 将点转为世界坐标系下plucker坐标
    # points: (nJoints, 3)
    # invK: (3, 3)
    # R: (3, 3)
    # T: (3, 1)
    # cam_center: (3, 1)
    if len(keypoints2d.shape) == 3:
        keypoints2d = keypoints2d[0]
    conf = keypoints2d[..., -1:]
    cam_center = - R.T @ T
    N = keypoints2d.shape[0]
    kp_pixel = np.hstack([keypoints2d[..., :2], np.ones_like(conf)])
    kp_all_3d = (kp_pixel @ invK.T - T.T) @ R
    l, m = plucker_from_pp(cam_center.T, kp_all_3d)
    res = np.hstack((l, m, conf))
    # 兼容cpp版本，所以补一个维度
    return res[None, :, :]

def computeRaynd(keypoints2d, invK, R, T):
    # keypoints2d: (..., 3)
    conf = keypoints2d[..., 2:]
    # cam_center: (1, 3)
    cam_center = - (R.T @ T).T
    kp_pixel = np.concatenate([keypoints2d[..., :2], np.ones_like(conf)], axis=-1)
    kp_all_3d = (kp_pixel @ invK.T - T.T) @ R
    while len(cam_center.shape) < len(kp_all_3d.shape):
        cam_center = cam_center[None]
    l, m = plucker_from_pp(cam_center, kp_all_3d)
    res = np.concatenate((l, m, conf), axis=-1)
    return res