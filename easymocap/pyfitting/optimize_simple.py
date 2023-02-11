'''
  @ Date: 2020-11-19 10:49:26
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-25 19:51:12
  @ FilePath: /EasyMocap/easymocap/pyfitting/optimize_simple.py
'''
import numpy as np
import torch
from .lbfgs import LBFGS 
from .optimize import FittingMonitor, grad_require, FittingLog
from .lossfactory import LossSmoothBodyMean, LossRegPoses
from .lossfactory import LossKeypoints3D, LossKeypointsMV2D, LossSmoothBody, LossRegPosesZero, LossInit, LossSmoothPoses

def optimizeShape(body_model, body_params, keypoints3d,
    weight_loss, kintree, cfg=None):
    """ simple function for optimizing model shape given 3d keypoints

    Args:
        body_model (SMPL model)
        params_init (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        keypoints (nFrames, nJoints, 3): 3D keypoints
        weight (Dict): string:float
        kintree ([[src, dst]]): list of list:int
        cfg (Config): Config Node controling running mode
    """
    device = body_model.device
    # 计算不同的骨长
    kintree = np.array(kintree, dtype=int)
    # limb_length: nFrames, nLimbs, 1
    limb_length = np.linalg.norm(keypoints3d[:, kintree[:, 1], :3] - keypoints3d[:, kintree[:, 0], :3], axis=2, keepdims=True)
    # conf: nFrames, nLimbs, 1
    limb_conf = np.minimum(keypoints3d[:, kintree[:, 1], 3:], keypoints3d[:, kintree[:, 0], 3:])
    limb_length = torch.Tensor(limb_length).to(device)
    limb_conf = torch.Tensor(limb_conf).to(device)
    body_params = {key:torch.Tensor(val).to(device) for key, val in body_params.items()}
    body_params_init = {key:val.clone() for key, val in body_params.items()}
    opt_params = [body_params['shapes']]
    grad_require(opt_params, True)
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe', max_iter=10)
    nFrames = keypoints3d.shape[0]
    verbose = False
    def closure(debug=False):
        optimizer.zero_grad()
        keypoints3d = body_model(return_verts=False, return_tensor=True, only_shape=True, **body_params)
        src = keypoints3d[:, kintree[:, 0], :3] #.detach()
        dst = keypoints3d[:, kintree[:, 1], :3]
        direct_est = (dst - src).detach()
        direct_norm = torch.norm(direct_est, dim=2, keepdim=True)
        direct_normalized = direct_est/(direct_norm + 1e-4)
        err = dst - src - direct_normalized * limb_length
        loss_dict = {
            's3d': torch.sum(err**2*limb_conf)/nFrames, 
            'reg_shapes': torch.sum(body_params['shapes']**2)}
        if 'init_shape' in weight_loss.keys():
            loss_dict['init_shape'] = torch.sum((body_params['shapes'] - body_params_init['shapes'])**2)
        # fittingLog.step(loss_dict, weight_loss)
        if verbose:
            print(' '.join([key + ' %.3f'%(loss_dict[key].item()*weight_loss[key]) 
                for key in loss_dict.keys() if weight_loss[key]>0]))
        loss = sum([loss_dict[key]*weight_loss[key]
                    for key in loss_dict.keys()])
        if not debug:
            loss.backward()
            return loss
        else:
            return loss_dict

    fitting = FittingMonitor(ftol=1e-4)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    loss_dict = closure(debug=True)
    for key in loss_dict.keys():
        loss_dict[key] = loss_dict[key].item()
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe')
    body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params

N_BODY = 25
N_HAND = 21

def interp(left_value, right_value, weight, key='poses'):
    if key == 'Rh':
        return left_value * weight + right_value * (1 - weight)
    elif key == 'Th':
        return left_value * weight + right_value * (1 - weight)
    elif key == 'poses':
        return left_value * weight + right_value * (1 - weight)

def get_interp_by_keypoints(keypoints):
    if len(keypoints.shape) == 3: # (nFrames, nJoints, 3)
        conf = keypoints[..., -1]
    elif len(keypoints.shape) == 4: # (nViews, nFrames, nJoints)
        conf = keypoints[..., -1].sum(axis=0)
    else:
        raise NotImplementedError
    not_valid_frames = np.where(conf.sum(axis=1) < 0.01)[0].tolist()
    # 遍历空白帧，选择起点和终点
    ranges = []
    if len(not_valid_frames) > 0:
        start = not_valid_frames[0]
        for i in range(1, len(not_valid_frames)):
            if not_valid_frames[i] == not_valid_frames[i-1] + 1:
                pass
            else:# 改变位置了
                end = not_valid_frames[i-1]
                ranges.append((start, end))
                start = not_valid_frames[i]
        ranges.append((start, not_valid_frames[-1]))
    def interp_func(params):
        for start, end in ranges:
            # 对每个需要插值的区间: 这里直接使用最近帧进行插值了
            left = start - 1
            right = end + 1
            for nf in range(start, end+1):
                weight = (nf - left)/(right - left)
                for key in ['Rh', 'Th', 'poses']:
                    params[key][nf] = interp(params[key][left], params[key][right], 1-weight, key=key)
        return params
    return interp_func

def interp_by_k3d(conf, params):
    for key in ['Rh', 'Th', 'poses']:
        params[key] = params[key].clone()
    # Totally invalid frames
    not_valid_frames = torch.nonzero(conf.sum(dim=1).squeeze() < 0.01)[:, 0].detach().cpu().numpy().tolist()
    # 遍历空白帧，选择起点和终点
    ranges = []
    if len(not_valid_frames) > 0:
        start = not_valid_frames[0]
        for i in range(1, len(not_valid_frames)):
            if not_valid_frames[i] == not_valid_frames[i-1] + 1:
                pass
            else:# 改变位置了
                end = not_valid_frames[i-1]
                ranges.append((start, end))
                start = not_valid_frames[i]
        ranges.append((start, not_valid_frames[-1]))
    for start, end in ranges:
        # 对每个需要插值的区间: 这里直接使用最近帧进行插值了
        left = start - 1
        right = end + 1
        for nf in range(start, end+1):
            weight = (nf - left)/(right - left)
            for key in ['Rh', 'Th', 'poses']:
                params[key][nf] = interp(params[key][left], params[key][right], 1-weight, key=key)
    return params

def deepcopy_tensor(body_params):
    for key in body_params.keys():
        body_params[key] = body_params[key].clone()
    return body_params

def dict_of_tensor_to_numpy(body_params):
    body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params

def get_prepare_smplx(body_params, cfg, nFrames):
    zero_pose = torch.zeros((nFrames, 3), device=cfg.device)
    if not cfg.OPT_HAND and cfg.model in ['smplh', 'smplx']:
        zero_pose_hand = torch.zeros((nFrames, body_params['poses'].shape[1] - 66), device=cfg.device)
    elif cfg.OPT_HAND and not cfg.OPT_EXPR and cfg.model == 'smplx':
        zero_pose_face = torch.zeros((nFrames, body_params['poses'].shape[1] - 78), device=cfg.device)

    def pack(new_params):
        if not cfg.OPT_HAND and cfg.model in ['smplh', 'smplx']:
            new_params['poses'] = torch.cat([zero_pose, new_params['poses'][:, 3:66], zero_pose_hand], dim=1)
        else:
            new_params['poses'] = torch.cat([zero_pose, new_params['poses'][:, 3:]], dim=1)
        return new_params
    return pack

def get_optParams(body_params, cfg, extra_params):
    for key, val in body_params.items():
        body_params[key] = torch.Tensor(val).to(cfg.device)
    if cfg is None:
        opt_params = [body_params['Rh'], body_params['Th'], body_params['poses']]
    else:
        if extra_params is not None:
            opt_params = extra_params
        else:
            opt_params = []
        if cfg.OPT_R:
            opt_params.append(body_params['Rh'])
        if cfg.OPT_T:
            opt_params.append(body_params['Th'])
        if cfg.OPT_POSE:
            opt_params.append(body_params['poses'])
        if cfg.OPT_SHAPE:
            opt_params.append(body_params['shapes'])
        if cfg.OPT_EXPR and cfg.model == 'smplx':
            opt_params.append(body_params['expression'])
    return opt_params

def _optimizeSMPL(body_model, body_params, prepare_funcs, postprocess_funcs, 
    loss_funcs, extra_params=None,
    weight_loss={}, cfg=None):
    """ A common interface for different optimization.

    Args:
        body_model (SMPL model)
        body_params (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        prepare_funcs (List): functions for prepare
        loss_funcs (Dict): functions for loss
        weight_loss (Dict): weight
        cfg (Config): Config Node controling running mode
    """
    loss_funcs = {key: val for key, val in loss_funcs.items() if key in weight_loss.keys() and weight_loss[key] > 0.}
    if cfg.verbose:
        print('Loss Functions: ')
        for key, func in loss_funcs.items():
            print('  -> {:15s}: {}'.format(key, func.__doc__))
    opt_params = get_optParams(body_params, cfg, extra_params)
    grad_require(opt_params, True)
    optimizer = LBFGS(opt_params, 
        line_search_fn='strong_wolfe')
    PRINT_STEP = 100
    records = []
    def closure(debug=False):
        # 0. Prepare body parameters => new_params
        optimizer.zero_grad()
        new_params = body_params.copy()
        for func in prepare_funcs:
            new_params = func(new_params)
        # 1. Compute keypoints => kpts_est
        kpts_est = body_model(return_verts=False, return_tensor=True, **new_params)
        # 2. Compute loss => loss_dict
        loss_dict = {key:func(kpts_est=kpts_est, **new_params) for key, func in loss_funcs.items()}
        # 3. Summary and log
        cnt = len(records)
        if cfg.verbose and cnt % PRINT_STEP == 0:
            print('{:-6d}: '.format(cnt) + ' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key]) 
                for key in loss_dict.keys() if weight_loss[key]>0]))
        loss = sum([loss_dict[key]*weight_loss[key]
                    for key in loss_dict.keys()])
        records.append(loss.item())
        if debug:
            return loss_dict
        loss.backward()
        return loss

    fitting = FittingMonitor(ftol=1e-4)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    loss_dict = closure(debug=True)
    if cfg.verbose:
        print('{:-6d}: '.format(len(records)) + ' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key]) 
            for key in loss_dict.keys() if weight_loss[key]>0]))
    loss_dict = {key:val.item() for key, val in loss_dict.items()}
    # post-process the body_parameters
    for func in postprocess_funcs:
        body_params = func(body_params)
    return body_params

def optimizePose3D(body_model, params, keypoints3d, weight, cfg):
    """ 
        simple function for optimizing model pose given 3d keypoints

    Args:
        body_model (SMPL model)
        params (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        keypoints3d (nFrames, nJoints, 4): 3D keypoints
        weight (Dict): string:float
        cfg (Config): Config Node controling running mode
    """
    nFrames = keypoints3d.shape[0]
    prepare_funcs = [
        deepcopy_tensor,
        get_prepare_smplx(params, cfg, nFrames),
        get_interp_by_keypoints(keypoints3d)
    ]
    loss_funcs = {
        'k3d': LossKeypoints3D(keypoints3d, cfg).body,
        'smooth_body': LossSmoothBodyMean(cfg).body,
        'smooth_poses': LossSmoothPoses(1, nFrames, cfg).poses,
        'reg_poses': LossRegPoses(cfg).reg_body,
        'init_poses': LossInit(params, cfg).init_poses,
    }
    if body_model.model_type != 'mano':
        loss_funcs['reg_poses_zero'] = LossRegPosesZero(keypoints3d, cfg).__call__
    if cfg.OPT_HAND:
        loss_funcs['k3d_hand'] = LossKeypoints3D(keypoints3d, cfg, norm='l1').hand
        loss_funcs['reg_hand'] = LossRegPoses(cfg).reg_hand
        # loss_funcs['smooth_hand'] = LossSmoothPoses(1, nFrames, cfg).hands
        loss_funcs['smooth_hand'] = LossSmoothBodyMean(cfg).hand

    if cfg.OPT_EXPR:
        loss_funcs['k3d_face'] = LossKeypoints3D(keypoints3d, cfg, norm='l1').face
        loss_funcs['reg_head'] = LossRegPoses(cfg).reg_head
        loss_funcs['reg_expr'] = LossRegPoses(cfg).reg_expr
        loss_funcs['smooth_head'] = LossSmoothPoses(1, nFrames, cfg).head

    postprocess_funcs = [
        get_interp_by_keypoints(keypoints3d),
        dict_of_tensor_to_numpy
    ]
    params = _optimizeSMPL(body_model, params, prepare_funcs, postprocess_funcs, loss_funcs, weight_loss=weight, cfg=cfg)
    return params

def optimizePose2D(body_model, params, bboxes, keypoints2d, Pall, weight, cfg):
    """ 
        simple function for optimizing model pose given 3d keypoints

    Args:
        body_model (SMPL model)
        params (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        keypoints2d (nFrames, nViews, nJoints, 4): 2D keypoints of each view
        bboxes: (nFrames, nViews, 5)
        weight (Dict): string:float
        cfg (Config): Config Node controling running mode
    """
    # transpose to (nViews, nFrames, 5)
    bboxes = bboxes.transpose(1, 0, 2)
    # transpose to => keypoints2d: (nViews, nFrames, nJoints, 3)
    keypoints2d = keypoints2d.transpose(1, 0, 2, 3)
    nViews, nFrames = keypoints2d.shape[:2]
    prepare_funcs = [
        deepcopy_tensor,
        get_prepare_smplx(params, cfg, nFrames),
        get_interp_by_keypoints(keypoints2d)
    ]
    loss_funcs = {
        'k2d': LossKeypointsMV2D(keypoints2d, bboxes, Pall, cfg).__call__,
        'smooth_body': LossSmoothBodyMean(cfg).body,
        'init_poses': LossInit(params, cfg).init_poses,
        'smooth_poses': LossSmoothPoses(nViews, nFrames, cfg).poses,
        'reg_poses': LossRegPoses(cfg).reg_body,
    }
    if body_model.model_type != 'mano':
        loss_funcs['reg_poses_zero'] = LossRegPosesZero(keypoints2d, cfg).__call__
    if cfg.OPT_SHAPE:
        loss_funcs['init_shapes'] = LossInit(params, cfg).init_shapes
    if cfg.OPT_HAND:
        loss_funcs['reg_hand'] = LossRegPoses(cfg).reg_hand
        # loss_funcs['smooth_hand'] = LossSmoothPoses(1, nFrames, cfg).hands
        loss_funcs['smooth_hand'] = LossSmoothBodyMean(cfg).hand

    if cfg.OPT_EXPR:
        loss_funcs['reg_head'] = LossRegPoses(cfg).reg_head
        loss_funcs['reg_expr'] = LossRegPoses(cfg).reg_expr
        loss_funcs['smooth_head'] = LossSmoothPoses(1, nFrames, cfg).head

    loss_funcs = {key:val for key, val in loss_funcs.items() if key in weight.keys()}
    
    postprocess_funcs = [
        get_interp_by_keypoints(keypoints2d),
        dict_of_tensor_to_numpy
    ]
    params = _optimizeSMPL(body_model, params, prepare_funcs, postprocess_funcs, loss_funcs, weight_loss=weight, cfg=cfg)
    return params