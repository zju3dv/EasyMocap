'''
  @ Date: 2022-09-26 16:32:19
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-17 13:05:28
  @ FilePath: /EasyMocapPublic/apps/calibration/vis_camera_by_open3d.py
'''
import open3d as o3d
import os
import cv2
import numpy as np
from easymocap.mytools.camera_utils import read_cameras
from easymocap.visualize.o3dwrapper import Vector3dVector, create_pcd
from easymocap.mytools.vis_base import generate_colorbar

def transform_cameras(cameras):
    dims = {'x': 0, 'y': 1, 'z': 2}
    R_global = np.eye(3)
    T_global = np.zeros((3, 1))
    # order: trans0, rot, trans
    if len(args.trans0) == 3:
        trans = np.array(args.trans0).reshape(3, 1)
        T_global += trans
    if len(args.rot) > 0:
        for i in range(len(args.rot)//2):
            dim = args.rot[2*i]
            val = float(args.rot[2*i+1])
            rvec = np.zeros((3,))
            rvec[dims[dim]] = np.deg2rad(val)
            R = cv2.Rodrigues(rvec)[0]
            R_global = R @ R_global
        T_global = R_global @ T_global
    # 平移相机
    if len(args.trans) == 3:
        trans = np.array(args.trans).reshape(3, 1)
        T_global += trans
    trans = np.eye(4)
    trans[:3, :3] = R_global
    trans[:3, 3:] = T_global
    # apply the transformation of each camera
    for key, cam in cameras.items():
        RT = np.eye(4)
        RT[:3, :3] = cam['R']
        RT[:3, 3:] = cam['T']
        RT = RT @ np.linalg.inv(trans)
        cam.pop('Rvec', '')
        cam['R'] = RT[:3, :3]
        cam['T'] = RT[:3, 3:]
    return cameras, trans

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--subs', type=str, default=[], nargs='+')
    parser.add_argument('--pcd', type=str, default=[], nargs='+')
    parser.add_argument('--trans0', type=float, nargs=3, 
        default=[], help='translation')
    parser.add_argument('--rot', type=str, nargs='+',
        default=[], help='control the rotation')
    parser.add_argument('--trans', type=float, nargs=3, 
        default=[], help='translation')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    grids = []
    cameras = read_cameras(args.path)
    cameras, trans = transform_cameras(cameras)

    print(repr(trans))
    for pcd in args.pcd:
        if not os.path.exists(pcd):
            print(pcd, ' not exist')
            continue
        if pcd.endswith('.npy'):
            data = np.load(pcd)
            points = data[:, :3]
            colors = data[:, 3:]
            points = (trans[:3,:3] @ points.T + trans[:3,3:]).T
            p = create_pcd(points, colors=data[:, 3:])
            grids.append(p)
        elif pcd.endswith('.ply'):
            print('Load pcd: ', pcd)
            p = o3d.io.read_point_cloud(pcd)
            print(p)
            grids.append(p)
        elif pcd.endswith('.pkl'):
            p = o3d.io.read_triangle_mesh(pcd)
            grids.append(p)
        elif pcd.endswith('.obj'):
            p = o3d.io.read_triangle_mesh(pcd)
            vertices = np.asarray(p.vertices)
            print(vertices.shape)
            print(vertices.min(axis=0))
            print(vertices.max(axis=0))
            grids.append(p)
        
    center = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])
    grids.append(center)
    colorbar = generate_colorbar(len(cameras), rand=False)
    camera_locations = []
    for ic, (cam, camera) in enumerate(cameras.items()):
        if len(args.subs) > 0 and cam not in args.subs:continue
        center = - camera['R'].T @ camera['T']
        print('{}: {}'.format(cam, center.T[0]))
        camera_locations.append(center)
        center = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[center[0, 0], center[1, 0], center[2, 0]])
        center.rotate(camera['R'].T)
        grids.append(center)
        # TODO: add label
    camera_locations = np.stack(camera_locations).reshape(-1, 3)
    o3d.visualization.draw_geometries(grids)