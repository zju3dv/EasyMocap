'''
  @ Date: 2022-03-11 12:13:01
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-11 21:52:00
  @ FilePath: /EasyMocapPublic/easymocap/multistage/synchronization.py
'''
import numpy as np
import torch

class AddTime:
    def __init__(self, gt) -> None:
        self.gt = gt

    def __call__(self, body_model, body_params, infos):
        nViews = infos['keypoints2d'].shape[1]
        offset = np.zeros((nViews,), dtype=np.float32)
        body_params['sync_offset'] = offset
        return body_params

class Interpolate:
    def __init__(self, actfn) -> None:
        # self.act_fn = lambda x: 2*torch.nn.functional.softsign(x)
        self.act_fn = lambda x: 2*torch.tanh(x)
        self.use0asref = False

    def get_offset(self, time_offset):
        if self.use0asref:
            off = self.act_fn(torch.cat([torch.zeros(1, device=time_offset.device), time_offset[1:]]))
        else:
            off = self.act_fn(time_offset)
        return off

    def start(self, body_params):
        return body_params

    def before(self, body_params):
        off = self.get_offset(body_params['sync_offset'])
        nViews = off.shape[0]
        if len(body_params['poses'].shape) == 2:
            off = off[None, :, None]
        else:
            off = off[None, :, None, None]
        for key in body_params.keys():
            if key in ['sync_offset', 'shapes']:
                continue
            # TODO: Rh有正周期旋转的时候会有问题
            val = body_params[key]
            if key == 'Rh':
                pass
            if key in ['Th', 'poses']:
                velocity = torch.cat([val[1:2] - val[0:1], val[1:] - val[:-1]], dim=0)
                valnew = val[:, None] + off * velocity[:, None]
                # vel = velocity.detach().cpu().numpy()
                # import matplotlib.pyplot as plt
                # plt.plot(vel)
                # plt.show()
                # import ipdb;ipdb.set_trace()
            else:
                if len(val.shape) == 2:
                    valnew = val[:, None].repeat(1, nViews, 1)
                elif len(val.shape) == 3:
                    valnew = val[:, None].repeat(1, nViews, 1, 1)
                else:
                    print('[warn] Unknown {} shape {}'.format(key, valnew.shape))
                    import ipdb; ipdb.set_trace()
            valnew = valnew.reshape(-1, *val.shape[1:])
            body_params[key] = valnew
        return body_params
    
    def after(self,):
        pass

    def final(self, body_params):
        off = self.get_offset(body_params['sync_offset'])
        body_params = self.before(body_params)
        body_params['sync_offset'] = off
        return body_params