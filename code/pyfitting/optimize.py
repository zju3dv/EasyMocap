'''
    @ Date: 2020-06-26 12:06:25
    @ LastEditors: Qing Shuai
    @ LastEditTime: 2020-06-26 12:08:37
    @ Author: Qing Shuai
    @ Mail: s_q@zju.edu.cn
'''
import numpy as np
import os
from tqdm import tqdm
import torch
import json

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])

class FittingMonitor:
    def __init__(self, ftol=1e-5, gtol=1e-6, maxiters=100, visualize=False, verbose=False, **kwargs):
        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol
        self.visualize = visualize
        self.verbose = verbose
        if self.visualize:
            from utils.mesh_viewer import MeshViewer
            self.mv = MeshViewer(width=1024, height=1024, bg_color=[1.0, 1.0, 1.0, 1.0], 
                body_color=[0.65098039, 0.74117647, 0.85882353, 1.0],
            offscreen=False)

    def run_fitting(self, optimizer, closure, params, smpl_render=None, **kwargs):
        prev_loss = None
        grad_require(params, True)
        if self.verbose:
            trange = tqdm(range(self.maxiters), desc='Fitting')
        else:
            trange = range(self.maxiters)
        for iter in trange:
            loss = optimizer.step(closure)
            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break
            
            # if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
            #         for var in params if var.grad is not None]):
            #     print('Small grad, stopping!')                
            #     break

            if iter > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break
            
            if self.visualize:
                vertices = smpl_render.GetVertices(**kwargs)
                self.mv.update_mesh(vertices[::10], smpl_render.faces)
            prev_loss = loss.item()
        grad_require(params, False)        
        return prev_loss

    def close(self):
        if self.visualize:
            self.mv.close_viewer()
class FittingLog:
    if False:
        from tensorboardX import SummaryWriter
        swriter = SummaryWriter()
    def __init__(self, log_name, useVisdom=False):
        if not os.path.exists(log_name):
            log_file = open(log_name, 'w')
            self.index = {log_name:0}

        else:
            log_file = open(log_name, 'r')
            log_pre = log_file.readlines()
            log_file.close()
            self.index = {log_name:len(log_pre)}
            log_file = open(log_name, 'a')
        self.log_file = log_file
        self.useVisdom = useVisdom
        if useVisdom:
            import visdom
            self.vis = visdom.Visdom(env=os.path.realpath(
                join(os.path.dirname(log_name), '..')).replace(os.sep, '_'))
        elif False:
            self.writer = FittingLog.swriter
        self.log_name = log_name
    
    def step(self, loss_dict, weight_loss):
        print(' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key]) 
            for key in loss_dict.keys() if weight_loss[key]>0]), file=self.log_file)
        loss = {key:loss_dict[key].item()*weight_loss[key]
            for key in loss_dict.keys() if weight_loss[key]>0}
        if self.useVisdom:
            name = list(loss.keys())
            val = list(loss.values())
            x = self.index.get(self.log_name, 0)
            if len(val) == 1:
                y = np.array(val)
            else:
                y = np.array(val).reshape(-1, len(val))
            self.vis.line(Y=y,X=np.ones(y.shape)*x,
                        win=str(self.log_name),#unicode
                        opts=dict(legend=name,
                            title=self.log_name),
                        update=None if x == 0 else 'append'
                        )
        elif False:
            self.writer.add_scalars('data/{}'.format(self.log_name), loss, self.index[self.log_name])
        self.index[self.log_name] += 1
    
    def log_loss(self, weight_loss):
        loss = json.dumps(weight_loss, indent=4)
        self.log_file.writelines(loss)
        self.log_file.write('\n')

    def close(self):
        self.log_file.close()


def grad_require(paras, flag=False):
    if isinstance(paras, list):
        for par in paras:
            par.requires_grad = flag 
    elif isinstance(paras, dict):
        for key, par in paras.items():
            par.requires_grad = flag