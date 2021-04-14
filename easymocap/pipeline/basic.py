'''
  @ Date: 2021-04-13 20:43:16
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-14 13:38:34
  @ FilePath: /EasyMocapRelease/easymocap/pipeline/basic.py
'''
from ..pyfitting import optimizeShape, optimizePose2D, optimizePose3D
from ..smplmodel import init_params
from ..mytools import Timer
from ..dataset import CONFIG
from .weight import load_weight_pose, load_weight_shape
from .config import Config

def multi_stage_optimize(body_model, params, kp3ds, kp2ds=None, bboxes=None, Pall=None, weight={}, cfg=None):
    with Timer('Optimize global RT'):
        cfg.OPT_R = True
        cfg.OPT_T = True
        params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
        # params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
    with Timer('Optimize 3D Pose/{} frames'.format(kp3ds.shape[0])):
        cfg.OPT_POSE = True
        cfg.ROBUST_3D = False
        params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
        if False:
            cfg.ROBUST_3D = True
            params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
        if cfg.model in ['smplh', 'smplx']:
            cfg.OPT_HAND = True
            params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
        if cfg.model == 'smplx':
            cfg.OPT_EXPR = True
            params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
    if kp2ds is not None:
        with Timer('Optimize 2D Pose/{} frames'.format(kp3ds.shape[0])):
            # bboxes => (nFrames, nViews, 5), keypoints2d => (nFrames, nViews, nJoints, 3)
            params = optimizePose2D(body_model, params, bboxes, kp2ds, Pall, weight=weight, cfg=cfg)
    return params

def smpl_from_keypoints3d2d(body_model, kp3ds, kp2ds, bboxes, Pall, config, args,
    weight_shape=None, weight_pose=None):
    model_type = body_model.model_type
    params_init = init_params(nFrames=1, model_type=model_type)
    if weight_shape is None:
        weight_shape = load_weight_shape(args.opts)
    if model_type in ['smpl', 'smplh', 'smplx']:
        # when use SMPL model, optimize the shape only with first 1-14 limbs, 
        # don't use (nose, neck)
        params_shape = optimizeShape(body_model, params_init, kp3ds, 
            weight_loss=weight_shape, kintree=CONFIG['body15']['kintree'][1:])
    else:
        params_shape = optimizeShape(body_model, params_init, kp3ds, 
            weight_loss=weight_shape, kintree=config['kintree'])
    # optimize 3D pose
    cfg = Config(args)
    cfg.device = body_model.device
    params = init_params(nFrames=kp3ds.shape[0], model_type=model_type)
    params['shapes'] = params_shape['shapes'].copy()
    if weight_pose is None:
        weight_pose = load_weight_pose(model_type, args.opts)
    # We divide this step to two functions, because we can have different initialization method
    params = multi_stage_optimize(body_model, params, kp3ds, kp2ds, bboxes, Pall, weight_pose, cfg)
    return params

def smpl_from_keypoints3d(body_model, kp3ds, config, args, 
    weight_shape=None, weight_pose=None):
    model_type = body_model.model_type
    params_init = init_params(nFrames=1, model_type=model_type)
    if weight_shape is None:
        weight_shape = load_weight_shape(args.opts)
    if model_type in ['smpl', 'smplh', 'smplx']:
        # when use SMPL model, optimize the shape only with first 1-14 limbs, 
        # don't use (nose, neck)
        params_shape = optimizeShape(body_model, params_init, kp3ds, 
            weight_loss=weight_shape, kintree=CONFIG['body15']['kintree'][1:])
    else:
        params_shape = optimizeShape(body_model, params_init, kp3ds, 
            weight_loss=weight_shape, kintree=config['kintree'])
    # optimize 3D pose
    cfg = Config(args)
    cfg.device = body_model.device
    cfg.model_type = model_type
    params = init_params(nFrames=kp3ds.shape[0], model_type=model_type)
    params['shapes'] = params_shape['shapes'].copy()
    if weight_pose is None:
        weight_pose = load_weight_pose(model_type, args.opts)
    # We divide this step to two functions, because we can have different initialization method
    params = multi_stage_optimize(body_model, params, kp3ds, None, None, None, weight_pose, cfg)
    return params