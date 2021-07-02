'''
  @ Date: 2021-04-13 20:43:16
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-14 15:32:24
  @ FilePath: /EasyMocap/easymocap/pipeline/basic.py
'''
from ..pyfitting import optimizeShape, optimizePose2D, optimizePose3D
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

def multi_stage_optimize2d(body_model, params, kp2ds, bboxes, Pall, weight={}, args=None):
    cfg = Config(args)
    cfg.device = body_model.device
    cfg.device = body_model.device
    cfg.model = body_model.model_type
    with Timer('Optimize global RT'):
        cfg.OPT_R = True
        cfg.OPT_T = True
        params = optimizePose2D(body_model, params, bboxes, kp2ds, Pall, weight=weight, cfg=cfg)
    with Timer('Optimize 2D Pose/{} frames'.format(kp2ds.shape[0])):
        cfg.OPT_POSE = True
        cfg.OPT_SHAPE = True
        # bboxes => (nFrames, nViews, 5), keypoints2d => (nFrames, nViews, nJoints, 3)
        params = optimizePose2D(body_model, params, bboxes, kp2ds, Pall, weight=weight, cfg=cfg)
    return params

def smpl_from_keypoints3d2d(body_model, kp3ds, kp2ds, bboxes, Pall, config, args,
    weight_shape=None, weight_pose=None):
    model_type = body_model.model_type
    params_init = body_model.init_params(nFrames=1)
    if weight_shape is None:
        weight_shape = load_weight_shape(model_type, args.opts)
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
    params = body_model.init_params(nFrames=kp3ds.shape[0])
    params['shapes'] = params_shape['shapes'].copy()
    if weight_pose is None:
        weight_pose = load_weight_pose(model_type, args.opts)
    # We divide this step to two functions, because we can have different initialization method
    params = multi_stage_optimize(body_model, params, kp3ds, kp2ds, bboxes, Pall, weight_pose, cfg)
    return params

def smpl_from_keypoints3d(body_model, kp3ds, config, args, 
    weight_shape=None, weight_pose=None):
    model_type = body_model.model_type
    params_init = body_model.init_params(nFrames=1)
    if weight_shape is None:
        weight_shape = load_weight_shape(model_type, args.opts)
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
    params = body_model.init_params(nFrames=kp3ds.shape[0])
    params['shapes'] = params_shape['shapes'].copy()
    if weight_pose is None:
        weight_pose = load_weight_pose(model_type, args.opts)
    # We divide this step to two functions, because we can have different initialization method
    params = multi_stage_optimize(body_model, params, kp3ds, None, None, None, weight_pose, cfg)
    return params