import os
import shutil
import numpy as np
import cv2
import open3d as o3d
from os.path import join
from os.path import join
from glob import glob
from tqdm import tqdm
from easymocap.mytools.file_utils import read_json, write_keypoints3d, write_vertices, write_smpl

# Input and output path
database = '/nas/home/shuaiqing/datasets/HI4D'
outdatabase = '/nas/home/shuaiqing/datasets/HI4D_easymocap'

# After convering, use
#     python3 apps/calibration/vis_camera_by_open3d.py ${data} --pcd ${data}/mesh-test.obj
# for visualization

seqlist = [
    'pair10/dance10',
    'pair32/pose32',
    # 'pair09/hug09',
    # 'pair00/fight00',
    # 'pair00/hug00',
    # 'pair12/fight12',
    # 'pair12/hug12',
    # 'pair14/dance14',
    # 'pair28/dance28',
    # 'pair32/pose32',
    # 'pair37/pose37',
]


from easymocap.bodymodel.smpl import SMPLModel
body_model = SMPLModel(
    model_path='data/bodymodels/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
    device = 'cuda',
    regressor_path = 'data/smplx/J_regressor_body25.npy',
    NUM_SHAPES = 10,
    use_pose_blending =True
)

for seq in seqlist:
    root = join(database, seq)
    outroot = join(outdatabase, seq.replace('/', '_'))
    
    cameraname = join(root, 'cameras', 'rgb_cameras.npz')

    cameras = dict(np.load(cameraname))

    cameras_out = {}
    for i, cam in enumerate(cameras['ids']):
        K = cameras['intrinsics'][i]
        dist = cameras['dist_coeffs'][i:i+1]
        RT = cameras['extrinsics'][i]
        R = RT[:3, :3]
        T = RT[:3, 3:]
        cameras_out[str(cam)] = {
            'K': K,
            'dist': dist,
            'R': R,
            'T': T
        }
        # 绕x轴转90度
        center = - R.T @ T
        print(cam, center.T[0])

    cameras = cameras_out

    meshanme = sorted(glob(join(root, 'frames', '*.obj')))[0]

    mesh = o3d.io.read_triangle_mesh(meshanme)
    vertices = np.asarray(mesh.vertices)

    R_global = cv2.Rodrigues(np.array([np.pi/2, 0, 0]))[0]

    vertices_R = vertices @ R_global.T
    z_min = np.min(vertices_R[:, 2])
    T_global = np.array([0, 0, -z_min]).reshape(3, 1)
    vertices_RT = vertices_R + T_global.T

    mesh.vertices = o3d.utility.Vector3dVector(vertices_RT)

    for key, cam in cameras.items():
        cam['R'] = cam['R'] @ R_global.T
        cam.pop('Rvec', '')
        center = - cam['R'].T @ cam['T']
        newcenter = center + T_global
        newT = -cam['R'] @ newcenter
        cam['T'] = newT
        center = - cam['R'].T @ cam['T']
        print(center.T)

    from easymocap.mytools.camera_utils import write_camera
    write_camera(cameras, outroot)
    o3d.io.write_triangle_mesh(join(outroot, 'mesh-test.obj'), mesh)

    regressor = np.load('data/smplx/J_regressor_body25.npy')
    filenames = sorted(glob(join(root, 'smpl', '*.npz')))
    for filename in tqdm(filenames):
        data = dict(np.load(filename))
        vertices = data['verts']
        vertices = vertices @ R_global.T + T_global.T
        joints = np.matmul(regressor[None], vertices)
        # 绕x轴旋转90度
        results = [{
            'id': 0,
            'keypoints3d': joints[0]
        },
        {
            'id': 1,
            'keypoints3d': joints[1]
        }]
        outname = join(outroot, 'body25', os.path.basename(filename).replace('.npz', '.json'))
        if not os.path.exists(outname):
            write_keypoints3d(outname, results)
        results = [{
            'id': 0,
            'vertices': vertices[0]
        },
        {
            'id': 1,
            'vertices': vertices[1]
        }]
        outname = join(outroot, 'vertices-gt', os.path.basename(filename).replace('.npz', '.json'))
        if not os.path.exists(outname):
            write_vertices(outname, results)
        params0 = {
            'poses': np.hstack([data['global_orient'][0], data['body_pose'][0]]),
            'shapes': data['betas'][0],
            'Th': data['transl'][0]
        }
        params1 = {
            'poses': np.hstack([data['global_orient'][1], data['body_pose'][1]]),
            'shapes': data['betas'][1],
            'Th': data['transl'][1]
        }
        outname = join(outroot, 'smpl-gt', os.path.basename(filename).replace('.npz', '.json'))
        if not os.path.exists(outname) or True:
            params = [params0, params1]
            for i, param in enumerate(params):
                for key in param.keys():
                    param[key] = param[key][None]
                param['Rh'] = np.zeros_like(param['Th'])
                param = body_model.convert_from_standard_smpl(param)
                Rold = cv2.Rodrigues(param['Rh'])[0]
                Told = param['Th']
                Rnew = R_global @ Rold
                Tnew = (R_global @ Told.T + T_global).T
                param['Rh'] = cv2.Rodrigues(Rnew)[0].reshape(1, 3)
                param['Th'] = Tnew.reshape(1, 3)
                param['id'] = i
                params[i] = param
            write_smpl(outname, params)
    if not os.path.exists(join(outroot, 'images')):
        shutil.copytree(join(root, 'images'), join(outroot, 'images'))