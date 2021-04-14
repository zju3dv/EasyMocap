'''
  @ Date: 2021-01-17 21:38:19
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-22 23:08:18
  @ FilePath: /EasyMocap/code/visualize/skelmodel.py
'''
import numpy as np
import cv2
from os.path import join
import os

def calTransformation(v_i, v_j, r, adaptr=False, ratio=10):
    """ from to vertices to T
    
    Arguments:
        v_i {} -- [description]
        v_j {[type]} -- [description]
    """
    xaxis = np.array([1, 0, 0])
    v = (v_i + v_j)/2
    direc = (v_i - v_j)
    length = np.linalg.norm(direc)
    direc = direc/length
    rotdir = np.cross(xaxis, direc)
    rotdir = rotdir/np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc, xaxis))
    rotmat, _ = cv2.Rodrigues(rotdir)
    # set the minimal radius for the finger and face
    shrink = max(length/ratio, 0.005)
    eigval = np.array([[length/2/r, 0, 0], [0, shrink, 0], [0, 0, shrink]])
    T = np.eye(4)
    T[:3,:3] = rotmat @ eigval @ rotmat.T
    T[:3, 3] = v
    return T, r, length

class SkelModel:
    def __init__(self, nJoints, kintree) -> None:
        self.nJoints = nJoints
        self.kintree = kintree
        cur_dir = os.path.dirname(__file__)
        faces = np.loadtxt(join(cur_dir, 'sphere_faces_20.txt'), dtype=np.int)
        self.vertices = np.loadtxt(join(cur_dir, 'sphere_vertices_20.txt'))
        # compose faces
        faces_all = []
        for nj in range(nJoints):
            faces_all.append(faces + nj*self.vertices.shape[0])
        for nk in range(len(kintree)):
            faces_all.append(faces + nJoints*self.vertices.shape[0] + nk*self.vertices.shape[0])
        self.faces = np.vstack(faces_all)

    def __call__(self, keypoints3d, id=0, return_verts=True, return_tensor=False):
        vertices_all = []
        r = 0.02
        # joints 
        for nj in range(self.nJoints):
            if nj > 25:
                r_ = r * 0.4
            else:
                r_ = r
            if keypoints3d[nj, -1] < 0.01:
                vertices_all.append(self.vertices*0.001)
                continue
            vertices_all.append(self.vertices*r_ + keypoints3d[nj:nj+1, :3])
        # limb
        for nk, (i, j) in enumerate(self.kintree):
            if keypoints3d[i][-1] < 0.1 or keypoints3d[j][-1] < 0.1:
                vertices_all.append(self.vertices*0.001)
                continue
            T, _, length = calTransformation(keypoints3d[i, :3], keypoints3d[j, :3], r=1)
            if length > 2: # 超过两米的
                vertices_all.append(self.vertices*0.001)
                continue
            vertices = self.vertices @ T[:3, :3].T + T[:3, 3:].T
            vertices_all.append(vertices)
        vertices = np.vstack(vertices_all)
        return vertices[None, :, :]