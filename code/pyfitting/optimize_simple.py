'''
  @ Date: 2020-11-19 10:49:26
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-24 21:29:12
  @ FilePath: /EasyMocapRelease/code/pyfitting/optimize_simple.py
'''
import numpy as np
import torch
from .lbfgs import LBFGS 
from .optimize import FittingMonitor, grad_require, FittingLog
from .lossfactory import SMPLAngleLoss, SmoothLoss, RegularizationLoss, ReprojectionLoss

def create_closure(optimizer, body_model, body_params, body_params_init, cam_params,
    keypoints2d, bboxes, weight_loss_, debug=False, verbose=False):
    K, Rc, Tc = cam_params['K'], cam_params['Rc'], cam_params['Tc']
    bbox_sizes = np.maximum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1])/100
    inv_bbox_sizes = torch.Tensor(1./bbox_sizes[:, None, None]).to(body_model.device)**2
    nFrames = keypoints2d.shape[0]
    nPoses = body_params['poses'].shape[0]
    if nFrames == nPoses:
        weight_loss = {key:val/nFrames for key, val in weight_loss_.items()} # normalize by frames
    else:
        # the multiple views case
        weight_loss = weight_loss_.copy()
        weight_loss['repro'] /= nFrames
    angle_prior = SMPLAngleLoss(keypoints2d)

    if len(K.shape) == 2:
        K, Rc, Tc = K[None, :, :], Rc[None, :, :], Tc[None, :, :]
    def closure(debug=False):
        optimizer.zero_grad()
        keypoints3d = body_model(return_verts=False, return_tensor=True, **body_params)
        loss_dict = {}
        loss_dict['repro'] = ReprojectionLoss(keypoints3d, keypoints2d, K, Rc, Tc, inv_bbox_sizes)
        # smooth
        loss_dict.update(SmoothLoss(body_params, ['poses', 'Th'], weight_loss))
        # regularize
        loss_dict.update(RegularizationLoss(body_params, body_params_init, weight_loss))
        loss_dict['reg_poses_zero'] = angle_prior.loss(body_params['poses'])
        # fittingLog.step(loss_dict, weight_loss)
        if verbose:
            print(' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key]) 
                for key in loss_dict.keys() if weight_loss[key]>0]))
        loss = sum([loss_dict[key]*weight_loss[key]
                    for key in loss_dict.keys()])
        if not debug:
            loss.backward()
            return loss
        else:
            return loss_dict
    return closure

def findNeighborIdx(nf, nF, nspan):
    idx = [i for i in range(nf-nspan, nf+nspan+1) if i >=0 and i<nF and i!= nf]
    return idx
    
def viewSelection(body_params, span=5):
    """ Apply view selection for body parameters

    Args:
        body_params (DictParams)
    """
    assert span % 2 == 1, 'span = {} must be an odd number'.format(span)
    nspan = (span - 1)//2
    # for linear data
    nF = body_params['poses'].shape[0]
    min_thres = {'Th': 0.2, 'poses': 0.2}
    for key in ['poses', 'Th']:
        weight = np.ones((nF))
        for nf in range(nF):
            # first find neighbor
            neighbor_idx = findNeighborIdx(nf, nF, nspan)
            dist = np.linalg.norm(body_params[key][neighbor_idx, :] - body_params[key][nf:nf+1, :], axis=1)
            conf = dist.min()/np.linalg.norm(body_params[key][nf])
            weight[nf] = min_thres[key] - conf
            if conf > min_thres[key]:
                weight[nf] = 0
        for nf in range(nF):
            neighbor_idx = findNeighborIdx(nf, nF, nspan) + [nf]
            weight_cur = weight[neighbor_idx]
            weight_sum = weight_cur.sum()
            if weight_sum == 0:
                pass
            else:
                val = body_params[key][neighbor_idx, :]*weight_cur[:, None]
                val = val.sum(axis=0)/weight_sum
                # simply remove outliers
                body_params[key][nf] = val
    # for rotation data
    pass
    return body_params


def optimizeVideo(body_model, params_init, cam_params, 
    keypoints, bbox,
    weight, cfg=None):
    """ simple function for optimizing video/image

    Args:
        body_model (SMPL model)
        params_init (DictParam): poses(F, 72), shapes(1/F, 10), Rh(F, 3), Th(F, 3)
        cam_params (DictParam): K, R, T
        keypoints (F, J, 3): 2D keypoints
        bbox (F, 7): bounding box
        weight (Dict): string:float
        cfg (Config): Config Node controling running mode
    """
    device = body_model.device
    cam_params = {key:torch.Tensor(val).to(device) for key, val in cam_params.items()}
    keypoints = torch.Tensor(keypoints).to(device)
    body_params = {key:torch.Tensor(val).to(device) for key, val in params_init.items()}
    body_params_init = {key:val.clone() for key, val in body_params.items()}
    if cfg is None:
        opt_params = [body_params['Rh'], body_params['Th'], body_params['poses']]
    else:
        opt_params = []
        if cfg.OPT_R:
            opt_params.append(body_params['Rh'])
        if cfg.OPT_T:
            opt_params.append(body_params['Th'])
        if cfg.OPT_POSE:
            opt_params.append(body_params['poses'])
        
    grad_require(opt_params, True)
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe')
    
    closure = create_closure(optimizer, body_model, body_params, 
        body_params_init, cam_params,
        keypoints, bbox, weight, verbose=cfg.VERBOSE)

    fitting = FittingMonitor(ftol=1e-4)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    loss_dict = closure(debug=True)
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe')
    body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params

def optimizeMultiImage(body_model, params_init, cam_params, 
    keypoints, bbox,
    weight, cfg):
    """ simple function for optimizing multiple images

    Args:
        body_model (SMPL model)
        params_init (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        cam_params (DictParam): K(nV, 3, 3), R(nV, 3, 3), T(nV, 3, 1)
        keypoints (nV, J, 3): 2D keypoints
        bbox (nV, 7): bounding box
        weight (Dict): string:float
        cfg (Config): Config Node controling running mode
    """
    device = body_model.device
    cam_params = {key:torch.Tensor(val).to(device) for key, val in cam_params.items()}
    keypoints = torch.Tensor(keypoints).to(device)
    body_params = {key:torch.Tensor(val).to(device) for key, val in params_init.items()}
    body_params_init = {key:val.clone() for key, val in body_params.items()}
    if cfg is None:
        opt_params = [body_params['Rh'], body_params['Th'], body_params['poses']]
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
    grad_require(opt_params, True)
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe')
    
    closure = create_closure(optimizer, body_model, body_params, 
        body_params_init, cam_params,
        keypoints, bbox, weight, verbose=cfg.VERBOSE)

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
    return body_params, loss_dict

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
    kintree = np.array(kintree, dtype=np.int)
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

def optimizePose(body_model, body_params, keypoints3d,
    weight_loss, kintree, cfg=None):
    """ simple function for optimizing model pose given 3d keypoints

    Args:
        body_model (SMPL model)
        params_init (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        keypoints (nFrames, nJoints, 3): 3D keypoints
        weight (Dict): string:float
        kintree ([[src, dst]]): list of list:int
        cfg (Config): Config Node controling running mode
    """
    device = body_model.device
    model_type = body_model.model_type
    # 计算不同的骨长
    kintree = np.array(kintree, dtype=np.int)
    nFrames = keypoints3d.shape[0]
    nJoints = keypoints3d.shape[1]
    keypoints3d = torch.Tensor(keypoints3d).to(device)
    angle_prior = SMPLAngleLoss(keypoints3d, body_model.model_type)

    body_params = {key:torch.Tensor(val).to(device) for key, val in body_params.items()}
    body_params_init = {key:val.clone() for key, val in body_params.items()}
    if cfg is None:
        opt_params = [body_params['Rh'], body_params['Th'], body_params['poses']]
        verbose = False
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
        if cfg.OPT_EXPR and model_type == 'smplx':
            opt_params.append(body_params['expression'])
        verbose = cfg.VERBOSE
    grad_require(opt_params, True)
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe')
    zero_pose = torch.zeros((nFrames, 3), device=device)
    if not cfg.OPT_HAND and model_type in ['smplh', 'smplx']:
        zero_pose_hand = torch.zeros((nFrames, body_params['poses'].shape[1] - 66), device=device)
        nJoints = N_BODY
        keypoints3d = keypoints3d[:, :nJoints]
    elif cfg.OPT_HAND and not cfg.OPT_EXPR and model_type == 'smplx':
        zero_pose_face = torch.zeros((nFrames, body_params['poses'].shape[1] - 78), device=device)
        nJoints = N_BODY + N_HAND * 2
        keypoints3d = keypoints3d[:, :nJoints]
    else:
        nJoints = keypoints3d.shape[1]
    def closure(debug=False):
        optimizer.zero_grad()
        new_params = body_params.copy()
        if not cfg.OPT_HAND and cfg.MODEL in ['smplh', 'smplx']:
            new_params['poses'] = torch.cat([zero_pose, body_params['poses'][:, 3:66], zero_pose_hand], dim=1)
        else:
            new_params['poses'] = torch.cat([zero_pose, body_params['poses'][:, 3:]], dim=1)
        kpts_est = body_model(return_verts=False, return_tensor=True, **new_params)[:, :nJoints, :]
        diff_square = (kpts_est[:, :nJoints, :3] - keypoints3d[..., :3])**2
        # TODO:add robust loss
        conf = keypoints3d[..., 3:]
        loss_3d = torch.sum(conf * diff_square)
        loss_dict = {
            'k3d': loss_3d,
            'reg_poses_zero': angle_prior.loss(body_params['poses'])
        }
        # regularize
        loss_dict.update(RegularizationLoss(body_params, body_params_init, weight_loss))
        # smooth
        smooth_conf = keypoints3d[1:, ..., -1:]**2
        loss_dict['smooth_body'] = torch.sum(smooth_conf[:, :N_BODY] * torch.abs(kpts_est[:-1, :N_BODY] - kpts_est[1:, :N_BODY]))
        if cfg.OPT_HAND and cfg.MODEL in ['smplh', 'smplx']:
            loss_dict['smooth_hand'] = torch.sum(smooth_conf[:, N_BODY:N_BODY+N_HAND*2] * torch.abs(kpts_est[:-1, N_BODY:N_BODY+N_HAND*2] - kpts_est[1:, N_BODY:N_BODY+N_HAND*2]))
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key]/nFrames
        # fittingLog.step(loss_dict, weight_loss)
        if verbose:
            print(' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key]) 
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