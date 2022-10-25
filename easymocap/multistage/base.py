# 这个脚本用于通用的多阶段的优化
import numpy as np
import torch

from ..annotator.file_utils import read_json
from ..mytools import Timer
from .lossbase import print_table
from ..config.baseconfig import load_object
from ..bodymodel.base import Params
from torch.utils.data import DataLoader
from tqdm import tqdm

def dict_of_numpy_to_tensor(body_model, body_params, *args, **kwargs):
    device = body_model.device
    body_params = {key:torch.Tensor(val).to(device) for key, val in body_params.items()}
    return body_params

class AddExtra:
    def __init__(self, vals) -> None:
        self.vals = vals

    def __call__(self, body_model, body_params, *args, **kwargs):
        shapes = body_params['poses'].shape[:-1]
        for key in self.vals:
            if key in body_params.keys():
                continue
            if key.startswith('R_') or key.startswith('T_'):
                val = np.zeros((*shapes, 3), dtype=np.float32)
                body_params[key] = val
        return body_params

def dict_of_tensor_to_numpy(body_params):
    body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params

def grad_require(params, flag=False):
    if isinstance(params, list):
        for par in params:
            par.requires_grad = flag 
    elif isinstance(params, dict):
        for key, par in params.items():
            par.requires_grad = flag

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([1e-5, abs(prev_val), abs(curr_val)])

def make_optimizer(opt_params, optim_type='lbfgs', max_iter=20,
    lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, **kwargs):
    if isinstance(opt_params, dict):
        # LBFGS 不支持参数字典
        opt_params = list(opt_params.values())
    if optim_type == 'lbfgs':
        from ..pyfitting.lbfgs import LBFGS
        optimizer = LBFGS(opt_params, line_search_fn='strong_wolfe', max_iter=max_iter, **kwargs)
    elif optim_type == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def make_lossfuncs(stage, infos, device, irepeat, verbose=False):
    loss_funcs, weights = {}, {}
    for key, val in stage.loss.items():
        loss_args = dict(val.args)
        if 'infos' in val.keys():
            for k in val.infos:
                loss_args[k] = infos[k]
        module = load_object(val.module, loss_args)
        module.to(device)
        if 'weights' in val.keys():
            weights[key] = val.weights[irepeat]
        else:
            weights[key] = val.weight
        if weights[key] < 0:
            weights.pop(key)
        else:
            loss_funcs[key] = module
    if verbose or True:
        print('Loss functions: ')
        for key, func in loss_funcs.items():
            print('  - {:15s}: {}, {}'.format(key, weights[key], func))
    return loss_funcs, weights

def make_before_after(before_after, body_model, body_params, infos):
    modules = []
    for key, val in before_after.items():
        args = dict(val.args)
        if 'body_model' in args.keys():
            args['body_model'] = body_model
        try:
            module = load_object(val.module, args)
        except:
            print('[Fitting] Failed to load module {}'.format(key))
            raise NotImplementedError
        module.infos = infos
        modules.append(module)
    return modules

def process(start_or_end, body_model, body_params, infos):
    for key, val in start_or_end.items():
        if isinstance(val, dict):
            module = load_object(val.module, val.args)
        else:
            if key == 'convert' and val == 'numpy_to_tensor':
                module = dict_of_numpy_to_tensor
            if key == 'add':
                module = AddExtra(val)
        body_params = module(body_model, body_params, infos)
    return body_params

def plot_meshes(img, meshes, K, R, T):
    import cv2
    mesh_camera = []
    for mesh in meshes:
        vertices = mesh['vertices'] @ R.T + T.T
        v2d = vertices @ K.T
        v2d[:, :2] = v2d[:, :2] / v2d[:, 2:3]
        lw=1
        col=(0,0,255)
        for (x, y, d) in v2d[::10]:
            cv2.circle(img, (int(x+0.5), int(y+0.5)), lw*2, col, -1)
    return img

class MultiStage:
    def __init__(self, batch_size, optimizer, monitor, initialize, stages) -> None:
        self.batch_size = batch_size
        self.optimizer_args = optimizer
        self.monitor = monitor
        self.initialize = initialize
        self.stages = stages
    
    def make_closure(self, body_model, body_params, infos, loss_funcs, weights, optimizer, before_after_module):
        def closure(debug=False, ret_kpts=False):
            # 0. Prepare body parameters => new_params
            optimizer.zero_grad()
            new_params = body_params.copy()
            for module in before_after_module:
                new_params = module.before(new_params)
            # 1. Compute keypoints => kpts_est
            poses_full = body_model.extend_poses(**new_params)
            kpts_est = body_model(return_verts=False, return_tensor=True, **new_params)
            if ret_kpts:
                return kpts_est
            verts_est = None
            # 2. Compute loss => loss_dict
            loss_dict = {}
            for key, loss_func in loss_funcs.items():
                if key.startswith('v'):
                    if verts_est is None:
                        verts_est = body_model(return_verts=True, return_tensor=True, **new_params)
                    loss_dict[key] = loss_func(verts_est=verts_est, **new_params, **infos)
                elif key.startswith('pf-'):
                    loss_dict[key] = loss_func(poses_full=poses_full, **new_params, **infos)
                else:
                    loss_dict[key] = loss_func(kpts_est=kpts_est, **new_params, **infos)
            loss = sum([loss_dict[key]*weights[key]
                        for key in loss_dict.keys()])
            if debug:
                return loss_dict
            loss.backward()
            return loss
        return closure
    
    def optimizer_step(self, optimizer, closure, weights):
        prev_loss = None
        for iter_ in range(self.monitor.maxiters):
            with torch.no_grad():
                loss_dict = closure(debug=True)
            if self.monitor.printloss or (self.monitor.verbose and iter_ == 0):
                print('{:-6d}: '.format(iter_) + ' '.join([key + ' %f'%(loss_dict[key].item()*weights[key]) for key in loss_dict.keys()]))
            loss = optimizer.step(closure)
            # check the loss
            if torch.isnan(loss).sum() > 0:
                print('[optimize] NaN loss value, stopping!')
                break
            if torch.isinf(loss).sum() > 0:
                print('[optimize] Infinite loss value, stopping!')
                break
            # check the delta
            if iter_ > 0 and prev_loss is not None:
                loss_rel_change = rel_change(prev_loss, loss.item())
                if loss_rel_change <= self.monitor.ftol:
                    if self.monitor.printloss or self.monitor.verbose:
                        print('{:-6d}: '.format(iter_) + ' '.join([key + ' %f'%(loss_dict[key].item()*weights[key]) for key in loss_dict.keys()]))
                    break
            # log
            if self.monitor.vis2d:
                pass
            if self.monitor.vis3d:
                pass
            prev_loss = loss.item()
        return True

    def fit_stage(self, body_model, body_params, infos, stage, irepeat):
        # 单独拟合一个stage, 返回body_params
        optimizer_args = stage.get('optimizer', self.optimizer_args)
        dtype, device = body_model.dtype, body_model.device
        body_params = process(stage.get('at_start', {'convert': 'numpy_to_tensor'}), body_model, body_params, infos)
        opt_params = {}
        if 'optimize' in stage.keys():
            optimize_names = stage.optimize
        else:
            optimize_names = stage.optimizes[irepeat]
        for key in optimize_names:
            if key in infos.keys(): # 优化的参数
                infos[key] = infos[key].to(device)
                opt_params[key] = infos[key]
            elif key in body_params.keys():
                opt_params[key] = body_params[key]
            else:
                raise ValueError('{} is not in infos or body_params'.format(key))
        if self.monitor.verbose:
            print('[optimize] optimizing {}'.format(optimize_names))
        for key, val in opt_params.items():
            infos['init_'+key] = val.clone().detach().cpu()
        # initialize keypoints
        with torch.no_grad():
            kpts_est = body_model.keypoints(body_params)
            infos['init_kpts_est'] = kpts_est.clone().detach().cpu()
        before_after_module = make_before_after(stage.get('before_after', {}), body_model, body_params, infos)
        for module in before_after_module:
            # Input to this module is tensor
            body_params = module.start(body_params)
        grad_require(opt_params, True)
        optimizer = make_optimizer(opt_params, **optimizer_args)
        loss_funcs, weights = make_lossfuncs(stage, infos, device, irepeat, self.monitor.verbose)
        closure = self.make_closure(body_model, body_params, infos, loss_funcs, weights, optimizer, before_after_module)
        if self.monitor.check:
            new_params = body_params.copy()
            for module in before_after_module:
                new_params = module.before(new_params)
            kpts_est = body_model.keypoints(new_params)
            for key, loss in loss_funcs.items():
                loss.check_at_start(kpts_est=kpts_est, **new_params)
        self.optimizer_step(optimizer, closure, weights)
        grad_require(opt_params, False)
        if self.monitor.check:
            new_params = body_params.copy()
            for module in before_after_module:
                new_params = module.before(new_params)
            kpts_est = body_model.keypoints(new_params)
            for key, loss in loss_funcs.items():
                loss.check_at_end(kpts_est=kpts_est, **new_params)
        for module in before_after_module:
            # Input to this module is tensor
            body_params = module.final(body_params)
        body_params = dict_of_tensor_to_numpy(body_params)
        for key, val in opt_params.items():
            if key in infos.keys():
                infos[key] = val.detach().cpu()
        return body_params

    def fit_data(self, data, body_model):
        infos = data.copy()
        init_params = body_model.init_params(nFrames=infos['nFrames'], nPerson=infos.get('nPerson', 1))
        # first initialize the model
        for name, init_func in self.initialize.items():
            if 'loss' in init_func.keys():
                # fitting to initialize
                init_params = self.fit_stage(body_model, init_params, infos, init_func, 0)
            else:
                # use initialize module
                init_module = load_object(init_func.module, init_func.args)
                init_params = init_module(body_model, init_params, infos)
        # if there are multiple initialization params
        # then fit each of them
        if not isinstance(init_params, list):
            init_params = [init_params]
        results = []
        for init_param in init_params:
            # check the repeat params
            body_params = init_param
            for stage_name, stage in self.stages.items():
                for irepeat in range(stage.get('repeat', 1)):
                    with Timer('optimize {}'.format(stage_name), not self.monitor.timer):
                        body_params = self.fit_stage(body_model, body_params, infos, stage, irepeat)
            results.append(body_params)
        # select the best results
        if len(results) > 1:
            # check the result
            loss = load_object(self.check.module, self.check.args, **{key:infos[key] for key in self.check.infos})
            metrics = [loss(body_model.keypoints(body_params, return_tensor=True).cpu()).item() for body_params in results]
            best_idx = np.argmin(metrics)
        else:
            best_idx = 0
        body_params = Params(**results[best_idx])
        return body_params, infos

    def fit(self, body_model, dataset):
        batch_size = len(dataset) if self.batch_size == -1 else self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        if len(dataloader) > 1:
            dataloader = tqdm(dataloader, desc='optimizing')
        for data in dataloader:
            data = dataset.reshape_data(data)
            body_params, infos = self.fit_data(data, body_model)
            if 'sync_offset' in body_params.keys():
                offset = body_params.pop('sync_offset')
                dataset.write_offset(offset)
            if data['nFrames'] != body_params['poses'].shape[0]:
                for key in body_params.keys():
                    if body_params[key].shape[0] == 1:continue
                    body_params[key] = body_params[key].reshape(data['nFrames'], -1, *body_params[key].shape[1:])
                    print(key, body_params[key].shape)
            if 'K' in infos.keys():
                camera = Params(K=infos['K'].numpy(), R=infos['Rc'].numpy(), T=infos['Tc'].numpy())
                if 'mirror' in infos.keys():
                    camera['mirror'] = infos['mirror'].numpy()[None]
                dataset.write(body_model, body_params, data, camera)
            else:
                # write data without camera
                dataset.write(body_model, body_params, data)