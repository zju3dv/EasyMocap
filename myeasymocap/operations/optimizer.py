import torch
import torch.nn as nn
from easymocap.config import Config, load_object
from easymocap.mytools.debug_utils import log

def dict_of_numpy_to_tensor(body_params, device):
    params_ = {}
    for key, val in body_params.items():
        if isinstance(val, dict):
            params_[key] = dict_of_numpy_to_tensor(val, device)
        else:
            params_[key] = torch.Tensor(val).to(device)
    return params_

def dict_of_tensor_to_numpy(body_params):
    params_ = {}
    for key, val in body_params.items():
        if isinstance(val, dict):
            params_[key] = dict_of_tensor_to_numpy(val)
        else:
            params_[key] = val.cpu().numpy()
    return params_

def make_optimizer(opt_params, optim_type='lbfgs', max_iter=20,
    lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, **kwargs):
    if isinstance(opt_params, dict):
        # LBFGS 不支持参数字典
        opt_params = list(opt_params.values())
    if optim_type == 'lbfgs':
        # optimizer = torch.optim.LBFGS(
        #     opt_params, max_iter=max_iter, lr=lr, line_search_fn='strong_wolfe',
        #     tolerance_grad= 0.0000001, # float32的有效位数是7位
        #     tolerance_change=0.0000001,
        # )
        from easymocap.pyfitting.lbfgs import LBFGS
        optimizer = LBFGS(opt_params, line_search_fn='strong_wolfe', max_iter=max_iter,
                          tolerance_grad= 0.0000001, # float32的有效位数是7位
                            tolerance_change=0.0000001,
                          **kwargs)
    elif optim_type == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def grad_require(params, flag=False):
    if isinstance(params, list):
        for par in params:
            par.requires_grad = flag 
    elif isinstance(params, dict):
        for key, par in params.items():
            par.requires_grad = flag

def make_closure(optimizer, model, params, infos, loss, device):
    loss_func = {}
    for key, val in loss.items():
        loss_func[key] = load_object(val['module'], val['args'])
        if isinstance(loss_func[key], nn.Module):
            loss_func[key].to(device)
    
    def closure(debug=False):
        optimizer.zero_grad()
        new_params = params.copy()
        output = model(new_params)
        loss_dict = {}
        loss_weight = {key:loss[key].weight for key in loss_func.keys()}
        for key, func in loss_func.items():
            output_ = {k: output[k] for k in loss[key].key_from_output}
            infos_ = {k: infos[k] for k in loss[key].key_from_infos}
            loss_now = func(output_, infos_)
            if isinstance(loss_now, dict):
                for k, _loss in loss_now.items():
                    loss_dict[key+'_'+k] = _loss
                    loss_weight[key+'_'+k] = loss_weight[key]
                loss_weight.pop(key)
            else:
                loss_dict[key] = loss_now
        loss_sum = sum([loss_dict[key]*loss_weight[key]
                        for key in loss_dict.keys()])
        # for key in loss_dict.keys():
        #     print(key, loss_dict[key] * loss_weight[key])
        # print(loss_sum)
        if debug:
            return loss_dict, loss_weight
        loss_sum.backward()
        return loss_sum
    return closure

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([1e-5, abs(prev_val), abs(curr_val)])

class Optimizer:
    def __init__(self, optimize_keys, optimizer_args, loss) -> None:
        self.optimize_keys = optimize_keys
        self.optimizer_args = optimizer_args
        self.loss = loss
        self.used_infos = []
        for key, val in loss.items():
            self.used_infos.extend(val.key_from_infos)
        self.used_infos = list(set(self.used_infos))
        self.iter = 0

    def log_loss(self, iter_, closure, print_loss=False):
        if iter_ % 10 == 0 or print_loss:
            with torch.no_grad():
                loss_dict, loss_weight = closure(debug=True)
            print('{:-6d}: '.format(iter_) + ' '.join([key + ' %7.4f'%(loss_dict[key].item()*loss_weight[key]) for key in loss_dict.keys()]))
        
    def optimizer_step(self, optimizer, closure):
        prev_loss = None
        self.log_loss(0, closure, True)
        for iter_ in range(1, 1000):
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
                if loss_rel_change <= 0.0000001:
                    break
            self.log_loss(iter_, closure)
            prev_loss = loss.item()
        self.log_loss(iter_, closure, True)
        return True

    def __call__(self, params, model, **infos):
        """
            待优化变量一定要在params中，但params中不一定会被优化
            infos中的变量不一定会被优化
        """
        # TODO: 应该使用model的device，但考虑到model可能是一个函数，所以暂时当场计算
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        params = dict_of_numpy_to_tensor(params, device=device)
        infos_used = {key: infos[key] for key in self.used_infos if key in infos.keys()}
        infos_used = dict_of_numpy_to_tensor(infos_used, device=device)
        
        optimize_keys = self.optimize_keys
        if isinstance(optimize_keys[0], list):
            optimize_keys = optimize_keys[self.iter]
        log('[{}] Optimize {}'.format(self.__class__.__name__, optimize_keys))
        log('[{}] Loading {}'.format(self.__class__.__name__, self.used_infos))
        opt_params = {}
        for key in optimize_keys:
            if key in infos.keys(): # 优化的参数
                opt_params[key] = infos_used[key]
            elif key in params.keys():
                opt_params[key] = params[key]
            else:
                raise ValueError('{} is not in infos or body_params'.format(key))
        for key, val in opt_params.items():
            infos_used['init_'+key] = val.clone()
        optimizer = make_optimizer(opt_params, **self.optimizer_args)
        closure = make_closure(optimizer, model, params, infos_used, self.loss, device)
        # 准备开始优化
        grad_require(opt_params, True)
        self.optimizer_step(optimizer, closure)
        grad_require(opt_params, False)
        # 直接返回
        ret = {
            'params': params
        }
        for key in optimize_keys:
            if key in infos.keys():
                ret[key] = opt_params[key]
        ret = dict_of_tensor_to_numpy(ret)
        return ret