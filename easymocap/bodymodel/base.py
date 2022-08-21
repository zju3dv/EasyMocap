'''
  @ Date: 2022-03-17 19:23:59
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-15 12:15:46
  @ FilePath: /EasyMocapPublic/easymocap/bodymodel/base.py
'''
import numpy as np
import torch

from ..mytools.file_utils import myarray2string

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'custom'

    def forward(self):
        pass

    def vertices(self, params, **kwargs):
        return self.forward(return_verts=True, **kwargs, **params)

    def keypoints(self, params, **kwargs):
        return self.forward(return_verts=False, **kwargs, **params)
    
    def transform(self, params, **kwargs):
        raise NotImplementedError

class ComposedModel(torch.nn.Module):
    def __init__(self, config_dict):
        # 叠加多个模型的配置
        for name, config in config_dict.items():
            pass

class Params(dict):
    @classmethod
    def merge(self, params_list, share_shape=True, stack=np.vstack):
        output = {}
        for key in params_list[0].keys():
            if key == 'id':continue
            output[key] = stack([v[key] for v in params_list])
        if share_shape:
            output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
        return output

    def __len__(self):
        return len(self['poses'])
    
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __getitem__(self, index):
        if not isinstance(index, int):
            return super().__getitem__(index)
        if 'shapes' not in self.keys():
            # arbitray data
            ret = {}
            for key, val in self.items():
                if index >= 1 and val.shape[0] == 1:
                    ret[key] = val[0]
                else:
                    ret[key] = val[index]
            return Params(**ret)
        ret = {'id': 0}
        poses = self.poses
        shapes = self.shapes
        while len(shapes.shape) < len(poses.shape):
            shapes = shapes[None]
        if poses.shape[0] == shapes.shape[0]:
            if index >= 1 and shapes.shape[0] == 1:
                ret['shapes'] = shapes[0]
            else:
                ret['shapes'] = shapes[index]
        elif shapes.shape[0] == 1:
            ret['shapes'] = shapes[0]
        else:
            import ipdb; ipdb.set_trace()
        if index >= 1 and poses.shape[0] == 1:
            ret['poses'] = poses[0]
        else:
            ret['poses'] = poses[index]
        for key, val in self.items():
            if key == 'id':
                ret[key] = self[key]
                continue
            if key in ret.keys():continue
            if index >= 1 and val.shape[0] == 1:
                ret[key] = val[0]
            else:
                ret[key] = val[index]
        for key, val in ret.items():
            if key == 'id': continue
            if len(val.shape) == 1:
                ret[key] = val[None]
        return Params(**ret)
    
    def to_multiperson(self, pids):
        results = []
        for i, pid in enumerate(pids):
            param = self[i]
            # TODO: this class just implement getattr
            # param.id = pid # is wrong
            param['id'] = pid
            results.append(param)
        return results

    def __str__(self) -> str:
        ret = ''
        lastkey = list(self.keys())[-1]
        for key, val in self.items():
            if isinstance(val, np.ndarray):
                ret += '"{}": '.format(key) + myarray2string(val, indent=0)
            else:
                ret += '"{}": '.format(key) + str(val)
            if key != lastkey:
                ret += ',\n'
        return ret
    
    def shape(self):
        ret = ''
        lastkey = list(self.keys())[-1]
        for key, val in self.items():
            if isinstance(val, np.ndarray):
                ret += '"{}": {}'.format(key, val.shape)
            else:
                ret += '"{}": '.format(key) + str(val)
            if key != lastkey:
                ret += ',\n'
        print(ret)
        return ret