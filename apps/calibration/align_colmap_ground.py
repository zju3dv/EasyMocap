# 这个脚本用于对colmap的相机标定结果，寻找地面与场景中心
# 方法：
# 1. 使用棋盘格
# 2. 估计点云里的地面
import os
from os.path import join
from easymocap.annotator.file_utils import save_json
from easymocap.mytools.debug_utils import myerror, run_cmd, mywarn, log
from easymocap.mytools.camera_utils import read_cameras, write_camera
from easymocap.mytools import read_json
from easymocap.mytools import batch_triangulate, projectN3, Undistort
import numpy as np
import cv2
from apps.calibration.calib_extri import solvePnP

def guess_ground(pcdname):
    pcd = o3d.io.read_point_cloud(pcdname)

def compute_rel(R_src, T_src, R_tgt, T_tgt):
    R_rel = R_src.T @ R_tgt
    T_rel = R_src.T @ (T_tgt - T_src)
    return R_rel, T_rel

def triangulate(cameras, areas):
    Ps, k2ds = [], []
    for cam, _, k2d, k3d in areas:
        k2d = Undistort.points(k2d, cameras[cam]['K'], cameras[cam]['dist'])
        P = cameras[cam]['K'] @ np.hstack([cameras[cam]['R'], cameras[cam]['T']])
        Ps.append(P)
        k2ds.append(k2d)
    Ps = np.stack(Ps)
    k2ds = np.stack(k2ds)
    k3d = batch_triangulate(k2ds, Ps)
    return k3d

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    return R, t

def align_by_chessboard(cameras, path):
    camnames = sorted(os.listdir(join(path, 'chessboard')))
    areas = []
    for ic, cam in enumerate(camnames):
        imagename = join(path, 'images', cam, '000000.jpg')
        chessname = join(path, 'chessboard', cam, '000000.json')        
        data = read_json(chessname)
        k3d = np.array(data['keypoints3d'], dtype=np.float32)
        k2d = np.array(data['keypoints2d'], dtype=np.float32)
        # TODO
        # pattern = (11, 8)
        if 'pattern' in data.keys():
            pattern = data['pattern']
        else:
            pattern = None
        img = cv2.imread(imagename)
        if args.scale2d is not None:
            k2d[:, :2] *= args.scale2d
            img = cv2.resize(img, None, fx=args.scale2d, fy=args.scale2d)
        if args.origin is not None:
            cameras[args.prefix+cam] = cameras.pop(args.origin+cam.replace('VID_', '0000'))
        cam = args.prefix + cam
        if cam not in cameras.keys():
            myerror('camera {} not found in {}'.format(cam, cameras.keys()))
            continue
        cameras[cam]['shape'] = img.shape[:2]
        if k2d[:, -1].sum() < 1:
            continue
        # calculate the area of the chessboard
        mask = np.zeros_like(img[:, :, 0])
        k2d_int = np.round(k2d[:, :2]).astype(int)
        if pattern is not None:
            cv2.fillPoly(mask, [k2d_int[[0, pattern[0]-1, -1, -pattern[0]]]], 1)
        else:
            cv2.fillPoly(mask, [k2d_int[[0, 1, 2, 3, 0]]], 1)
        area = mask.sum()
        print(cam, area)
        areas.append([cam, area, k2d, k3d])
    areas.sort(key=lambda x: -x[1])
    best_cam, area, k2d, k3d = areas[0]
    # 先解决尺度问题
    ref_point_id = np.linalg.norm(k3d - k3d[:1], axis=-1).argmax()
    k3d_pre = triangulate(cameras, areas)
    length_gt = np.linalg.norm(k3d[0, :3] - k3d[ref_point_id, :3])
    length = np.linalg.norm(k3d_pre[0, :3] - k3d_pre[ref_point_id, :3])
    log('gt diag={:.3f}, est diag={:.3f}, scale={:.3f}'.format(length_gt, length, length_gt/length))
    scale_colmap = length_gt / length
    for cam, camera in cameras.items():
        camera['T'] *= scale_colmap
    k3d_pre = triangulate(cameras, areas)
    length = np.linalg.norm(k3d_pre[0, :3] - k3d_pre[-1, :3])
    log('gt diag={:.3f}, est diag={:.3f}, scale={:.3f}'.format(length_gt, length, length_gt/length))
    # 计算相机相对于棋盘格的RT
    if False:
        for cam, _, k2d, k3d in areas:
            K, dist = cameras[cam]['K'], cameras[cam]['dist']
            R, T = cameras[cam]['R'], cameras[cam]['T']
            err, rvec, tvec, kpts_repro = solvePnP(k3d, k2d, K, dist, flag=cv2.SOLVEPNP_ITERATIVE)
            # 不同视角的计算的相对变换应该是一致的
            R_tgt = cv2.Rodrigues(rvec)[0]
            T_tgt = tvec.reshape(3, 1)
            R_rel, T_rel = compute_rel(R, T, R_tgt, T_tgt)
            break
    else:
        # 使用估计的棋盘格坐标与实际的棋盘格坐标
        X = k3d_pre[:, :3]
        X_gt = k3d[:, :3]
        R_rel, T_rel = best_fit_transform(X_gt, X)
        # 从棋盘格坐标系映射到colmap坐标系
        T_rel = T_rel.reshape(3, 1)
    centers = []
    for cam, camera in cameras.items():
        camera.pop('Rvec')
        R_old, T_old = camera['R'], camera['T']
        R_new = R_old @ R_rel
        T_new = T_old + R_old @ T_rel
        camera['R'] = R_new
        camera['T'] = T_new
        center = - camera['R'].T @ camera['T']
        centers.append(center)
        print('{}: ({:6.3f}, {:.3f}, {:.3f})'.format(cam, *np.round(center.T[0], 3)))
    # 使用棋盘格估计一下尺度
    k3d_pre = triangulate(cameras, areas)
    length = np.linalg.norm(k3d_pre[0, :3] - k3d_pre[ref_point_id, :3])
    log('{} {} {}'.format(length_gt, length, length_gt/length))
    log(k3d_pre)
    transform = np.eye(4)
    transform[:3, :3] = R_rel
    transform[:3, 3:] = T_rel
    return cameras, scale_colmap, np.linalg.inv(transform)

# for 3D points X,
# in origin world: \Pi(RX + T) = x
# in new world: \Pi(R'Y+T') = x
#    , where X = R_pY + T_p

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--plane_by_chessboard', type=str, default=None)
    parser.add_argument('--plane_by_point', type=str, default=None)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--scale2d', type=float, default=None)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--origin', type=str, default=None)
    parser.add_argument('--guess_plane', action='store_true')
    parser.add_argument('--noshow', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(join(args.path, 'intri.yml')):
        assert os.path.exists(join(args.path, 'cameras.bin')), os.listdir(args.path)
        cmd = f'python3 apps/calibration/read_colmap.py {args.path} .bin'
        run_cmd(cmd)
    
    cameras = read_cameras(args.path)
    if args.plane_by_point is not None:
        # 读入点云
        import ipdb; ipdb.set_trace()
    if args.plane_by_chessboard is not None:
        cameras, scale, transform = align_by_chessboard(cameras, args.plane_by_chessboard)
        if not args.noshow:
            import open3d as o3d
            sparse_name = join(args.path, 'sparse.ply')
            dense_name = join(args.path, '..', '..', 'dense', 'fused.ply')
            if os.path.exists(dense_name):
                pcd = o3d.io.read_point_cloud(dense_name)
            else:
                pcd = o3d.io.read_point_cloud(sparse_name)
            save_json(join(args.out, 'transform.json'), {'scale': scale, 'transform': transform.tolist()})
            points = np.asarray(pcd.points)
            # TODO: read correspondence of points3D and points2D
            points_new = (scale*points) @ transform[:3, :3].T + transform[:3, 3:].T
            pcd.points = o3d.utility.Vector3dVector(points_new)
            o3d.io.write_point_cloud(join(args.out, 'sparse_aligned.ply'), pcd)
            grids = []
            center = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1, origin=[0, 0, 0])
            grids.append(center)
            for cam, camera in cameras.items():
                center = - camera['R'].T @ camera['T']
                center = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.5, origin=[center[0, 0], center[1, 0], center[2, 0]])
                if cam.startswith(args.prefix):
                    center.paint_uniform_color([1, 0, 1])
                center.rotate(camera['R'].T)
                grids.append(center)
            o3d.visualization.draw_geometries([pcd] + grids)
    write_camera(cameras, args.out)
    if args.prefix is not None:
        cameras_ = {}
        for key, camera in cameras.items():
            if args.prefix not in key:
                continue
            cameras_[key.replace(args.prefix, '')] = camera
        os.makedirs(join(args.out, args.prefix), exist_ok=True)
        write_camera(cameras_, join(args.out, args.prefix))