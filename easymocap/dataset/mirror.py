'''
  @ Date: 2021-01-21 19:34:48
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-06 18:57:47
  @ FilePath: /EasyMocap/code/dataset/mirror.py
'''
import numpy as np
from os.path import join
import os
import cv2

FLIP_BODY25 = [0,1,5,6,7,2,3,4,8,12,13,14,9,10,11,16,15,18,17,22,23,24,19,20,21]
FLIP_BODYHAND = [
    0,1,5,6,7,2,3,4,8,12,13,14,9,10,11,16,15,18,17,22,23,24,19,20,21, # body 25
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, # right hand
    ]
FLIP_SMPL_VERTICES = np.loadtxt(join(os.path.dirname(__file__), 'smpl_vert_sym.txt'), dtype=int)

def flipPoint2D(point):
    if point.shape[-2] == 25:
        return point[..., FLIP_BODY25, :]
    elif point.shape[-2] == 15:
        return point[..., FLIP_BODY25[:15], :]
    elif point.shape[-2] == 6890:
        return point[..., FLIP_SMPL_VERTICES, :]
        import ipdb; ipdb.set_trace()
    elif point.shape[-1] == 67:
        import ipdb; ipdb.set_trace()

# Permutation of SMPL pose parameters when flipping the shape
_PERMUTATION = {
    'smpl': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22],
    'smplh': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 24, 25, 23, 24],
    'smplx': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 24, 25, 23, 24, 26, 28, 27],
    'smplhfull': [
        0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, # body
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
    ],
    'smplxfull': [
        0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, # body
        22, 24, 23,  # jaw, left eye, right eye
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, # right hand
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, # left hand
    ]
}
PERMUTATION = {}
for key in _PERMUTATION.keys():
    res = []
    for i in _PERMUTATION[key]:
        res.extend([3*i + j for j in range(3)])
    PERMUTATION[max(res)+1] = res

def flipSMPLPoses(pose):
    """Flip pose. 
    const input: (N, 72) -> (N, 72)
    The flipping is based on SMPL parameters.
    """
    pose = pose[:, PERMUTATION[pose.shape[-1]]]
    if pose.shape[1] in [72, 156, 165]:
        pose[:, 1::3] = -pose[:, 1::3]
        pose[:, 2::3] = -pose[:, 2::3]
    elif pose.shape[1] in [78, 87]:
        pose[:, 1:66:3] = -pose[:, 1:66:3]
        pose[:, 2:66:3] = -pose[:, 2:66:3]
    else:
        import ipdb; ipdb.set_trace()
    # we also negate the second and the third dimension of the axis-angle
    return pose

def mirrorPoint3D(point, M):
    point_homo = np.hstack([point, np.ones([point.shape[0], 1])])
    point_m = (M @ point_homo.T).T[..., :3]
    return flipPoint2D(point_m)

def calc_mirror_transform(m):
    coeff_mat = np.eye(4)[None, :, :]
    coeff_mat = coeff_mat.repeat(m.shape[0], 0)
    norm = np.linalg.norm(m[:, :3], keepdims=True, axis=1)
    m[:, :3] /= norm
    coeff_mat[:, 0, 0] = 1 - 2*m[:, 0]**2
    coeff_mat[:, 0, 1] = -2*m[:, 0]*m[:, 1]
    coeff_mat[:, 0, 2] = -2*m[:, 0]*m[:, 2]
    coeff_mat[:, 0, 3] = -2*m[:, 0]*m[:, 3]
    coeff_mat[:, 1, 0] = -2*m[:, 1]*m[:, 0]
    coeff_mat[:, 1, 1] = 1-2*m[:, 1]**2
    coeff_mat[:, 1, 2] = -2*m[:, 1]*m[:, 2]
    coeff_mat[:, 1, 3] = -2*m[:, 1]*m[:, 3]
    coeff_mat[:, 2, 0] = -2*m[:, 2]*m[:, 0]
    coeff_mat[:, 2, 1] = -2*m[:, 2]*m[:, 1]
    coeff_mat[:, 2, 2] = 1-2*m[:, 2]**2
    coeff_mat[:, 2, 3] = -2*m[:, 2]*m[:, 3]
    return coeff_mat

def get_rotation_from_two_directions(direc0, direc1):
    direc0 = direc0/np.linalg.norm(direc0)
    direc1 = direc1/np.linalg.norm(direc1)
    rotdir = np.cross(direc0, direc1)
    if np.linalg.norm(rotdir) < 1e-2:
        return np.eye(3)
    rotdir = rotdir/np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc0, direc1))
    rotmat, _ = cv2.Rodrigues(rotdir)
    return rotmat

def mirror_Rh(Rh, normals):
    rvecs = np.zeros_like(Rh)
    for nf in range(Rh.shape[0]):
        normal = normals[nf]
        rotmat = cv2.Rodrigues(Rh[nf])[0]
        rotmat_m = np.zeros((3, 3))
        for i in range(3):
            rot = rotmat[:, i] - 2*(rotmat[:, i] * normal).sum()*normal
            rotmat_m[:, i] = rot
        rotmat_m[:, 0] *= -1
        rvecs[nf] = cv2.Rodrigues(rotmat_m)[0].T
    return rvecs

def flipSMPLParams(params, mirror):
    """Flip pose. 
    const input: (1, 72) -> (1, 72)
    The flipping is based on SMPL parameters.
    """
    mirror[:, :3] /= np.linalg.norm(mirror[:, :3], keepdims=True, axis=1)
    if mirror.shape[0] == 1 and mirror.shape[0] != params['Rh'].shape[0]:
        mirror = mirror.repeat(params['Rh'].shape[0], 0)
    M = calc_mirror_transform(mirror)
    T = params['Th']
    rvecm = mirror_Rh(params['Rh'], mirror[:, :3])
    Tnew = np.einsum('bmn,bn->bm', M[:, :3, :3], params['Th']) + M[:, :3, 3]
    params = {
        'poses': flipSMPLPoses(params['poses']),
        'shapes': params['shapes'],
        'Rh': rvecm,
        'Th': Tnew
    }
    return params
