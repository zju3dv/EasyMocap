from ...config.baseconfig import load_object
import torch
import torch.nn as nn
from copy import deepcopy

class ComposedModel(nn.Module):
    def __init__(self, models) -> None:
        super().__init__()
        models = deepcopy(models)
        for key in ['human', 'ball']:
            if 'all' + key in models.keys():
                pids = models['all'+key].pop('pids')
                for pid in pids:
                    models['{}_{}'.format(key, pid)] = deepcopy(models['all'+key])
                    if 'pid' in models['{}_{}'.format(key, pid)].network_args.keys():
                        models['{}_{}'.format(key, pid)].network_args.pid = pid
                models.pop('all'+key)
        if 'allkeys' in models.keys():
            object_keys = models['allkeys'].pop('keys')
            for key in object_keys:
                models[key] = deepcopy(models['allkeys'])
            models.pop('allkeys')
        modules = {}
        for key, val in models.items():
            model = load_object(val['network_module'], val['network_args'])
            print('[model] {:15s}: {:4.1f}M'.format(key, sum([m.numel() for m in model.parameters()])/1000000))
            modules[key] = model
        self.models = nn.ModuleDict(modules)
        self.keys = list(self.models.keys())
        self.is_share = False
    
    def model(self, name):
        model = self.models[name]
        model.current = name
        return model

    def forward(self, pts):
        raise NotImplementedError

from .base import Base
class MultiLayer(Base):
    def __init__(self, sample_args, models):
        super().__init__(sample_args)
        modules = {}
        for key, val in models.items():
            model = load_object(val['network_module'], val['network_args'])
            print('[model] {:15s}: {:4.1f}M'.format(key, sum([m.numel() for m in model.parameters()])/1000000))
            modules[key] = model
        self.models = nn.ModuleDict(modules)
        self.keys = list(self.models.keys())
        self.num_layers = len(self.keys)
        self.name = None
        
    def model(self, name):
        self.current = name
        return self
    
    def clear_cache(self):
        pass

    def before(self, batch, key):
        for name, model in self.models.items():
            data = model.before(batch, key)
        return data
    
    def calculate_density_color(self, pts, viewdirs):
        map_semantic = {
            'human_0': 0,
            'upper': 1,
            'pant': 2,
        }
        outputs = []
        for name, model in self.models.items():
            raw_output_layer = model.calculate_density_color(pts, viewdirs)
            # if name in ['pant', 'human_0']:
            #     pass
            # # if name in ['human_0']:
            # if name in []:
            #     raw_output_layer['occupancy'] = torch.zeros_like(raw_output_layer['occupancy'])
            semantic = torch.zeros(*raw_output_layer['occupancy'].shape[:-1], self.num_layers, device=raw_output_layer['occupancy'].device)
            semantic[..., map_semantic[name]] = 1.
            raw_output_layer['semantic'] = semantic
            outputs.append(raw_output_layer)
        ret = {}
        for key in outputs[0].keys():
            ret[key] = torch.cat([output[key] for output in outputs], dim=1)
        return ret
    
    def calculate_density_color_from_ray(self, *kargs, **kwargs):
        z_vals, pts, raw_output = super().calculate_density_color_from_ray(*kargs, **kwargs)
        # TODO: add perturbation
        z_vals = torch.cat([z_vals for _ in range(self.num_layers)], dim=1)
        pts = torch.cat([pts for _ in range(self.num_layers)], dim=1)
        # sort multi layer
        z_vals_sorted, indices = torch.sort(z_vals[..., 0], dim=-1)
        # toc('sort')
        ind_0 = torch.zeros_like(indices, device=indices.device)
        ind_0 = ind_0 + torch.arange(0, indices.shape[0], device=indices.device).reshape(-1, 1)
        raw_sorted = {}
        for key, val in raw_output.items():
            val_sorted = val[ind_0, indices]
            raw_sorted[key] = val_sorted
        pts_sorted = pts[ind_0, indices]
        return z_vals_sorted[..., None], pts_sorted, raw_sorted