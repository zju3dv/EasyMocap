import numpy as np
import cv2
import torch.nn as nn
import torch
import time
import json
from ..model.base import augment_z_vals, concat

_time_ = 0
def tic():
    global _time_
    _time_ = time.time()

def toc(name):
    global _time_
    print('{:15s}: {:.1f}'.format(name, 1000*(time.time() - _time_)))
    _time_ = time.time()

def raw2acc(raw):
    alpha = raw[..., -1]
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    acc_map = torch.sum(weights, -1)
    return acc_map

def raw2outputs(outputs, z_vals, rays_d, bkgd=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        acc: [num_rays, num_samples along ray, 1]. Prediction from model.
        feature: [num_rays, num_samples along ray, N]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        feat_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    if 'occupancy' in outputs.keys():
        alpha = outputs['occupancy'][..., 0]
    elif 'density' in outputs.keys():
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists,
            torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)],
            -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d, dim=-1)
        noise = 0.
        # alpha = raw2alpha(raw[..., -1] + noise, dists)  # [N_rays, N_samples]
        alpha = 1 - torch.exp(-dists*torch.relu(outputs['density'][..., 0] + noise)) # (N_rays, N_samples_)
    else:
        raise NotImplementedError
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    acc_map = torch.sum(weights, -1)
    # ATTN: here depth must /||ray_d||
    depth_map = torch.sum(weights * z_vals, -1)/(1e-10 + acc_map)/torch.norm(rays_d, dim=-1).squeeze()
    results = {
        'acc_map': acc_map,
        'depth_map': depth_map,
    }
    for key, val in outputs.items():
        if key == 'occupancy':
            continue
        results[key+'_map'] = torch.sum(weights[..., None] * val, -2)  # [N_rays, 3]

    if bkgd is not None:
        results['rgb_map'] = results['rgb_map'] + bkgd[0] * (1 - acc_map[..., None])

    return results

class BaseRenderer(nn.Module):
    def __init__(self, net, chunk, white_bkgd, use_occupancy, N_samples, split,
        render_layer=False,
        return_raw=False, return_extra=False, use_canonical=False):
        super().__init__()
        self.net = net
        self.chunk = chunk
        self.white_bkgd = white_bkgd
        self.use_occupancy = use_occupancy
        self.N_samples = 64
        self.split = split
        self.return_extra = return_extra
        self.use_canonical = use_canonical
        self.render_layer = render_layer
        if use_canonical:
            self.net.use_canonical = use_canonical
    
    def forward_any(self, net, data, meta, bkgd):
        # give network and data, return the corresponding output
        raw, z_val_ = [], []
        ray_o = data['ray_o'][0].unsqueeze(1)
        ray_d = data['ray_d'][0].unsqueeze(1)
        # Sample depth points
        z_steps = torch.linspace(0, 1, self.N_samples, device=ray_o.device).reshape(1, -1)
        for bn in range(0, ray_o.shape[0], self.chunk):
            start, end = bn, bn + self.chunk
            # first sample points
            near, far = [data[key][0, start:end][:, None] for key in ['near', 'far']]
            if False:
                # z_vals: (nrays, N_samples)
                z_vals = near * (1-z_steps) + far * z_steps
                z_vals = z_vals.unsqueeze(2)
                if self.split == 'train':
                    z_vals = augment_z_vals(z_vals)
                pts = ray_o[mask] + ray_d[mask] * z_vals
                viewdir = viewdirs[mask].expand(-1, pts.shape[1], -1)
                raw_output = model.calculate_density_color(pts, viewdir)
            else:
                z_vals, pts, raw_output = net.calculate_density_color_from_ray(
                    ray_o[start:end], ray_d[start:end], near, far, self.split)
            # directly render
            if bkgd.shape[1] != 1:
                bkgd_ = bkgd[:, start:end]
            else:
                bkgd_ = bkgd
            results = raw2outputs(
                raw_output, z_vals[..., 0], ray_d[start:end], bkgd_)
            raw.append(results)
        return raw

    def compose(self, retlist, mask=None, bkgd=None):
        res = {}
        for key in retlist[0].keys():
            val = torch.cat([r[key] for r in retlist])
            if mask is not None and val.shape[0] != mask.shape[0]:
                val_ = torch.zeros((mask.shape[0], *val.shape[1:]), device=val.device, dtype=val.dtype)
                if key == 'rgb_map': # consider the background
                    if bkgd is None:
                        import ipdb; ipdb.set_trace()
                    elif bkgd is not None and bkgd.shape[0] > 1:
                        val_[~mask] = bkgd[~mask]
                    else:
                        val_[~mask] = bkgd[0]
                val_[mask] = val
                val = val_.unsqueeze(0)
            else:
                val = val.unsqueeze(0)
            res[key] = val
        return res

    def forward_single(self, batch, bkgd):
        keys = [d[0] for d in batch['meta']['keys']]
        assert len(keys) == 1, 'Only support one key'
        key = keys[0]
        model = self.net.model(key)
        model.clear_cache()
        data = model.before(batch, key)
        # get the background
        bkgd_ = bkgd
        if bkgd is not None and bkgd.shape[0] > 1:
            bkgd_ = bkgd[data['mask'][0]][None] # (1, nValid, 3)
        retlist = self.forward_any(model, data, batch['meta'], bkgd_)
        res = self.compose(retlist, data['mask'][0], bkgd)
        res['keys'] = keys
        return res

    def batch_forward(self, batch, viewdir, start, end, bkgd):
        ray_o = batch['ray_o'][0, start:end, None]
        ray_d = batch['ray_d'][0, start:end, None]
        viewdirs = batch['viewdirs'][0, start:end, None].expand(-1, 1, -1)
        keys_all = self.net.keys.copy()
        object_keys = [d[0] for d in batch['meta']['object_keys']]
        if len(object_keys) > len(keys_all) or True:
            mapkeys = {}
            operation = {}
            keys_all = object_keys
            for key in object_keys:
                mapkeys[key] = key.split('_@')[0]
                if '_@' in key:
                    params = json.loads(key.split('_@')[1].replace("'", '"'))
                    operation[key] = params
        else:
            mapkeys = {key:key for key in keys_all}
        # keys_all.sort(key=lambda x:0 if x=='back' else int(x.split('_@')[0].replace('human_', ''))+1 if x.startswith('human') else 9999)
        # print('render keys: ', keys_all)
        ret_all = []
        dimGroups = [0]
        for key in object_keys:
            if '@' in key:
                model = self.net.model(mapkeys[key])
                model.current = key
            else:
                model = self.net.model(key)
                # 这里手动设置一下key,因为在非share模式下,不会自动覆盖
                model.current = key
            
            mask = batch[key + '_mask'][0]
            start_ = mask[:start].sum()
            end_ = mask[:end].sum()
            near, far = [batch[key+'_'+nearfar][0, start_:end_][:, None] for nearfar in ['near', 'far']]
            mask = mask[start:end]
            if mask.sum() < 1:
                # print('Skip {} [{}, {}]'.format(key, start, end))
                continue
            if False:
                if key in self.net.N_samples.keys():
                    N_samples = self.net.N_samples[key]
                else:
                    N_samples = self.net.N_samples['default']
                dimGroups.append(dimGroups[-1]+N_samples)
                z_steps = torch.linspace(0, 1, N_samples, device=ray_d.device).reshape(1, -1)
                # z_vals: (nrays, N_samples)
                z_vals = near * (1-z_steps) + far * z_steps
                z_vals = z_vals.unsqueeze(2)
                if self.split == 'train':
                    z_vals = augment_z_vals(z_vals)
                pts = ray_o[mask] + ray_d[mask] * z_vals
                viewdir = viewdirs[mask].expand(-1, pts.shape[1], -1)
                raw_output = model.calculate_density_color(pts, viewdir)
            else:
                z_vals, pts, raw_output = model.calculate_density_color_from_ray(
                    ray_o[mask], ray_d[mask], near, far, self.split)
                dimGroups.append(dimGroups[-1]+z_vals.shape[-2])
                if not self.use_occupancy:
                    # set the density of last points to zero
                    raw_output['density'][:, -1] = 0.

            if '_@' in key:
                if 'scale_occ' in operation[key].keys():
                    raw_output['occupancy'] *= operation[key]['scale_occ']
            # TODO: remove bounds
            # if key.startswith('human') or key.startswith('ball'):
            #     notInBound = pts[..., -1] < 0.02
            #     raw_output['occupancy'][notInBound] = 0.

            raw_output['z_vals'] = z_vals[..., 0]
            # add instance
            instance_ = torch.zeros((*pts.shape[:-1], len(keys_all)), 
                dtype=pts.dtype, device=pts.device)
            instance_[..., keys_all.index(key)] = 1.
            raw_output['instance'] = instance_
            raw_padding = {}
            for key_out, val in raw_output.items():
                if len(val.shape) == 1: # for traj
                    raw_padding[key_out] = val
                    continue
                padding = torch.zeros([mask.shape[0], *val.shape[1:]], dtype=val.dtype, device=val.device)
                padding[mask] = val
                raw_padding[key_out] = padding
            ret_all.append(raw_padding)
            # toc(key)
            # if key.startswith('back') and (self.radius_max is not None or self.ranges is not None):
            #     if self.radius_max is not None:
            #         notInBound = torch.norm(wpts, dim=-1) > self.radius_max
            #     elif self.ranges is not None:
            #         bound_l = torch.FloatTensor(self.ranges[0]).to(wpts.device).reshape(1, 1, 3)
            #         bound_u = torch.FloatTensor(self.ranges[1]).to(wpts.device).reshape(1, 1, 3)
            #         self.bound_l = bound_l
            #         self.bound_u = bound_u
            #         notInBound = ((wpts < bound_l)|(wpts>bound_u)).any(dim=-1)
            #     mask_valid = mask & (~notInBound.all(dim=-1))
            #     # 注意:这里只在背景的时候正确,如果不是背景,数据量不一样的
            #     wpts_valid = wpts[mask_valid]
            #     # print('[back] forward {} points of {}'.format(mask_valid.sum(), mask.sum()))
            #     z_val_valid = z_val[mask_valid]
            #     viewdir_valid = viewdir_[mask_valid]
            #     viewdir_valid = viewdir_valid.expand(-1, wpts.shape[1], -1)
            #     raw_ = torch.zeros([wpts.shape[0], wpts.shape[1], 4], device=wpts.device, dtype=wpts.dtype)
            #     if mask_valid.sum() > 0:
            #         raw_valid = model.calculate_density_color(wpts_valid, viewdir_valid)
            #         raw_[mask_valid] = raw_valid
            # else:
            #     viewdir_ = viewdir_.expand(-1, wpts.shape[1], -1)
            #     raw_ = model.calculate_density_color(wpts, viewdir_)
            # if key+'_scale_occ' in batch.keys():
            #     raw_[..., -1] *= batch[key+'_scale_occ']
            #     if self.remove_fog:
            #         acc_ = raw2acc(raw_)
            #         raw_[acc_<self.thres_fog*batch[key+'_scale_occ']] = 0.
            # elif self.remove_fog and key != 'back' and key.startswith('human'):
            #     acc_ = raw2acc(raw_)
            #     raw_[acc_<self.thres_fog] = 0.
            # import ipdb; ipdb.set_trace()
            # if self.render_layer:
                # ret_layer = self.render_func(raw_now, zval_now, ray_d)
            #     for ret_name, val in ret_layer.items():
            #         ret_all[ret_name+'_'+key] = val
        if len(ret_all) == 0:
            # 补全0
            occupancy = torch.zeros([ray_d.shape[0], 1, 1], device=ray_d.device)
            color = torch.zeros([ray_d.shape[0], 1, 3], device=ray_d.device)
            instance = torch.zeros([ray_d.shape[0], 1, len(object_keys)], device=ray_d.device)
            z_vals_blank = torch.zeros([ray_d.shape[0], 1], device=ray_d.device)
            blank_output = {'occupancy': occupancy, 'rgb': color, 'instance': instance,
                'raw_alpha': occupancy}
            blank_output['raw_rgb'] = blank_output['rgb']
            ret = raw2outputs(blank_output, z_vals_blank, ray_d, bkgd)
            return ret
        raw_concat = concat(ret_all, dim=1, unsqueeze=False)
        z_vals = raw_concat.pop('z_vals')
        z_vals_sorted, indices = torch.sort(z_vals, dim=-1)
        # toc('sort')
        ind_0 = torch.zeros_like(indices, device=indices.device)
        ind_0 = ind_0 + torch.arange(0, indices.shape[0], device=indices.device).reshape(-1, 1)
        raw_sorted = {}
        for key, val in raw_concat.items():
            val_sorted = val[ind_0, indices]
            raw_sorted[key] = val_sorted
        ret = raw2outputs(raw_sorted, z_vals_sorted, ray_d, bkgd)
        if self.render_layer:
            for ikey, key in enumerate(object_keys):
                raw_key = {k:v[:, dimGroups[ikey]:dimGroups[ikey+1]] for k,v in raw_concat.items()}
                layer = raw2outputs(raw_key, z_vals[:, dimGroups[ikey]:dimGroups[ikey+1]], ray_d, bkgd)
                for k in ['acc_map', 'rgb_map']:
                    ret[key+'_'+k] = layer[k]
        # toc('render')
        return ret

    def forward_multi(self, batch, bkgd):
        keys = [d[0] for d in batch['meta']['keys']]
        # prepare each model
        res_cache = {}
        for key in self.net.keys:
            model = self.net.model(key)
            model.clear_cache()
        for key in keys:
            if '@' in key:
                key0 = key.split('_@')[0]
                model = self.net.model(key0)
                model.current = key
            else:
                model = self.net.model(key)
            model.before(batch, key)
            if key in model.cache.keys():
                res_cache[key+'_cache'] = model.cache[key]
        viewdir = batch['viewdirs'][0].unsqueeze(1)
        retlist = []
        for bn in range(0, viewdir.shape[0], self.chunk):
            start, end = bn, min(bn + self.chunk, viewdir.shape[0])
            ret = self.batch_forward(batch, viewdir, start, end, bkgd)
            if ret is not None:
                retlist.append(ret)
        res = self.compose(retlist)
        # add cache
        res.update(res_cache)
        res['keys'] = keys
        return res

    def forward(self, batch):
        keys = [d[0] for d in batch['meta']['keys']]
        rand_bkgd = None
        device = batch['rgb'].device
        if self.split == 'train':
            rand_bkgd = torch.rand(3, device=device).reshape(1, 1, 3)
        else:
            if self.white_bkgd:
                rand_bkgd = torch.ones(3, device=device).reshape(1, 1, 3)
        if len(keys) == 1:
            results = self.forward_single(batch, rand_bkgd)
        else:
            results = self.forward_multi(batch, rand_bkgd)

        # set the random background in target
        if self.split == 'train':
            idx = torch.nonzero(batch['rgb'][0, :, 0] < 0)
            if rand_bkgd is not None:
                batch['rgb'][0, idx] = rand_bkgd
            else:
                batch['rgb'][0, idx] = 0.
        return results

class RendererWithBkgd(BaseRenderer):
    def forward(self, batch):
        keys = [d[0] for d in batch['meta']['keys']]
        device = batch['rgb'].device
        nview = batch['meta']['nview'][0]
        if 'background' not in batch.keys():
            bkgd = torch.zeros((1, 1, 3), device=device)
            if self.white_bkgd:
                bkgd += 1.
        else:
            bkgd = batch['background'][0]
        if len(keys) == 1:
            results = self.forward_single(batch, bkgd)
        else:
            results = self.forward_multi(batch, bkgd)
        return results

class BackgroundRenderer(BaseRenderer):
    def forward(self, batch):
        keys = [d[0] for d in batch['meta']['keys']]
        coord = batch['coord'][0]
        background = self.net.model('background')
        background.before(batch)
        bkgd = background(coord, batch['meta'])
        if len(keys) == 1:
            results = self.forward_single(batch, bkgd)
        else:
            results = self.forward_multi(batch, bkgd)
        return results

class MirrorDemoRenderer(BackgroundRenderer):
    def forward(self, _batch):
        background = self.net.model('background').background
        background_init = self.net.model('background').background_init
        background = background.detach().cpu().numpy().copy()
        accmap = np.zeros_like(background[:, :, 0])
        for mirror_key in ['left', 'right']:
            batch = _batch[mirror_key]
            batch['meta']['index'] = batch['meta']['index']//2
            H, W = int(batch['meta']['H'][0]), int(batch['meta']['W'][0])
            keys = [d[0] for d in batch['meta']['keys']]
            coord = batch['coord'][0].cpu().numpy()
            if mirror_key == 'right':
                coord[:, 1] = W - 1 - coord[:, 1]
            bkgd = torch.zeros((1, 1, 3), device=batch['coord'].device)
            results = self.forward_single(batch, bkgd)
            rgb_map = results['rgb_map'][0].detach().cpu().numpy()
            acc_map = np.clip(results['acc_map'][0].detach().cpu().numpy(), 0., 1.).reshape(-1, 1)
            accmap[coord[:, 0], coord[:, 1]] = acc_map[..., 0]
            background[coord[:, 0], coord[:, 1]] = rgb_map * acc_map + background[coord[:, 0], coord[:, 1]] * (1-acc_map)
        return {
            'rgb_map': background, 
            'acc_map': accmap,
            'meta': batch['meta']}