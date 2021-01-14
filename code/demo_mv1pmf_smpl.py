'''
  @ Date: 2021-01-12 17:08:25
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-14 20:49:25
  @ FilePath: /EasyMocap/code/demo_mv1pmf_smpl.py
'''
# show skeleton and reprojection
import pyrender # first import the pyrender
from pyfitting.optimize_simple import optimizeShape, optimizePose
from dataset.mv1pmf import MV1PMF
from dataset.config import CONFIG
from mytools.reconstruction import simple_recon_person, projectN3
from smplmodel import select_nf, init_params, Config

from tqdm import tqdm
import numpy as np

def load_model(use_cuda=True):
    # prepare SMPL model
    import torch
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    from smplmodel import SMPLlayer
    body_model = SMPLlayer('data/smplx/smpl', gender='neutral', device=device,
        regressor_path='data/smplx/J_regressor_body25.npy')
    body_model.to(device)
    return body_model

def load_weight_shape():
    weight = {'s3d': 1., 'reg_shape': 5e-3}
    return weight

def load_weight_pose():
    weight = {
        'k3d': 1., 'reg_poses_zero': 1e-2, 
        'smooth_Rh': 1e-2, 'smooth_Th': 1e-2, 'smooth_poses': 1e-2
    }
    return weight

def mv1pmf_smpl(path, sub, out, mode, args):
    config = CONFIG[mode]
    MIN_CONF_THRES = 0.5
    no_img = False
    dataset = MV1PMF(path, cams=sub, config=CONFIG[mode], add_hand_face=False,
        undis=args.undis, no_img=no_img, out=out)
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    dataset.no_img = True
    annots_all = []
    for nf in tqdm(range(start, end), desc='triangulation'):
        images, annots = dataset[nf]
        conf = annots['keypoints'][..., -1]
        conf[conf < MIN_CONF_THRES] = 0
        keypoints3d, _, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall, ret_repro=True)
        kp3ds.append(keypoints3d)
        annots_all.append(annots)
    # smooth the skeleton
    kp3ds = np.stack(kp3ds)
    # optimize the human shape
    body_model = load_model()
    params_init = init_params(nFrames=1)
    weight = load_weight_shape()
    params_shape = optimizeShape(body_model, params_init, kp3ds, weight_loss=weight, kintree=config['kintree'])
    # optimize 3D pose
    cfg = Config()
    params = init_params(nFrames=kp3ds.shape[0])
    params['shapes'] = params_shape['shapes'].copy()
    weight = load_weight_pose()
    cfg.OPT_R = True
    cfg.OPT_T = True
    params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
    cfg.OPT_POSE = True
    params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
    # optimize 2D pose
    # render the mesh
    dataset.no_img = not args.vis_smpl
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        dataset.write_smpl(select_nf(params, nf-start), nf)
        if args.vis_smpl:
            vertices = body_model(return_verts=True, return_tensor=False, **select_nf(params, nf-start))
            dataset.vis_smpl(vertices=vertices, faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('multi_view one_person multi_frame skel')
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the sub folder lists when in video mode')
    parser.add_argument('--start', type=int, default=0,
        help='frame start')
    parser.add_argument('--end', type=int, default=10000,
        help='frame end')    
    parser.add_argument('--step', type=int, default=1,
        help='frame step')
    parser.add_argument('--body', type=str, default='body15', choices=['body15', 'body25', 'total'])
    parser.add_argument('--undis', action='store_true')
    parser.add_argument('--add_hand_face', action='store_true')
    parser.add_argument('--vis_smpl', action='store_true')
    parser.add_argument('--sub_vis', type=str, nargs='+', default=[],
        help='the sub folder lists for visualization')
    args = parser.parse_args()
    mv1pmf_smpl(args.path, args.sub, args.out, args.body, args)