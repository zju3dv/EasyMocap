'''
  @ Date: 2021-01-12 17:08:25
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-25 19:32:44
  @ FilePath: /EasyMocapRelease/code/demo_mv1pmf_smpl.py
'''
# show skeleton and reprojection
import pyrender # first import the pyrender
from pyfitting.optimize_simple import optimizeShape, optimizePose
from dataset.mv1pmf import MV1PMF
from dataset.config import CONFIG
from mytools.utils import Timer
from smplmodel import select_nf, init_params, Config, load_model, check_keypoints
from os.path import join
from tqdm import tqdm
import numpy as np

def load_weight_shape():
    weight = {'s3d': 1., 'reg_shapes': 5e-3}
    return weight

def load_weight_pose(model):
    if model == 'smpl':
        weight = {
            'k3d': 1., 'reg_poses_zero': 1e-2, 'smooth_body': 1e-2
        }
    elif model == 'smplh':
        weight = {
            'k3d': 1., 'reg_poses_zero': 1e-3,
            'smooth_body': 1e-2, 'smooth_hand': 1e-2
        }
    elif model == 'smplx':
        weight = {
            'k3d': 1., 'reg_poses_zero': 1e-3,
            'reg_expression': 1e-2,
            'smooth_body': 1e-2, 'smooth_hand': 1e-2
        }
    else:
        raise NotImplementedError
    return weight

def print_mean_skel(mode):
    with Timer('Loading {}, {}'.format(args.model, args.gender)):
        body_model = load_model(args.gender, model_type=args.model)
    params_init = init_params(nFrames=1, model_type=args.model)
    skel = body_model(return_verts=False, return_tensor=False, **params_init)[0]
    # skel: nJoints, 3
    config = CONFIG[mode]
    skeleton = {}
    for i, j_ in config['kintree']:
        if j_ == 25:
            j = 7
        elif j_ == 46:
            j = 4
        else:
            j = j_
        key = tuple(sorted([i, j]))
        limb_length = np.linalg.norm(skel[i] - skel[j])
        skeleton[key] = {'mean': limb_length, 'std': limb_length*0.2}
    print('{')
    for key, val in skeleton.items():
        res = '    ({:2d}, {:2d}): {{\'mean\': {:.3f}, \'std\': {:.3f}}}, '.format(*key, val['mean'], val['std'])
        if 'joint_names' in config.keys():
            res += '# {:9s}->{:9s}'.format(config['joint_names'][key[0]], config['joint_names'][key[1]])
        print(res)
    print('}')

def mv1pmf_smpl(path, sub, out, mode, args):
    config = CONFIG[mode]
    no_img = True
    dataset = MV1PMF(path, cams=sub, config=CONFIG[mode], mode=args.body,
        undis=args.undis, no_img=no_img, out=out)
    if args.skel is None:
        from demo_mv1pmf_skel import mv1pmf_skel
        mv1pmf_skel(path, sub, out, mode, args)
        args.skel = join(out, 'keypoints3d')
    dataset.skel_path = args.skel
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    dataset.no_img = True
    annots_all = []
    for nf in tqdm(range(start, end), desc='loading'):
        images, annots = dataset[nf]
        infos = dataset.read_skel(nf)
        kp3ds.append(infos[0]['keypoints3d'])
        annots_all.append(annots)
    kp3ds = np.stack(kp3ds)
    kp3ds = check_keypoints(kp3ds, 1)
    # optimize the human shape
    with Timer('Loading {}, {}'.format(args.model, args.gender)):
        body_model = load_model(args.gender, model_type=args.model)
    params_init = init_params(nFrames=1, model_type=args.model)
    weight = load_weight_shape()
    if args.model in ['smpl', 'smplh', 'smplx']:
        # when use SMPL model, optimize the shape only with first 14 limbs
        params_shape = optimizeShape(body_model, params_init, kp3ds, weight_loss=weight, kintree=CONFIG['body15']['kintree'])
    else:
        params_shape = optimizeShape(body_model, params_init, kp3ds, weight_loss=weight, kintree=config['kintree'])
    # optimize 3D pose
    cfg = Config()
    cfg.VERBOSE = args.verbose
    cfg.MODEL = args.model
    params = init_params(nFrames=kp3ds.shape[0], model_type=args.model)
    params['shapes'] = params_shape['shapes'].copy()
    weight = load_weight_pose(args.model)
    with Timer('Optimize global RT'):
        cfg.OPT_R = True
        cfg.OPT_T = True
        params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
    with Timer('Optimize Pose/{} frames'.format(end-start)):
        cfg.OPT_POSE = True
        params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
        if args.model in ['smplh', 'smplx']:
            cfg.OPT_HAND = True
            params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
        if args.model == 'smplx':
            cfg.OPT_EXPR = True
            params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
    # TODO:optimize 2D pose
    # write out the results
    dataset.no_img = not args.vis_smpl
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        dataset.write_smpl(select_nf(params, nf-start), nf)
        if args.vis_smpl:
            vertices = body_model(return_verts=True, return_tensor=False, **select_nf(params, nf-start))
            dataset.vis_smpl(vertices=vertices, faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis, add_back=True)

if __name__ == "__main__":
    from mytools.cmd_loader import load_parser
    parser = load_parser()
    parser.add_argument('--skel', type=str, default=None, 
        help='path to keypoints3d')
    parser.add_argument('--vis_smpl', action='store_true')
    args = parser.parse_args()
    # print_mean_skel(args.body)
    mv1pmf_smpl(args.path, args.sub, args.out, args.body, args)