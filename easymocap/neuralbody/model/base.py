'''
  @ Date: 2021-09-03 16:56:14
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 21:17:25
  @ FilePath: /EasyMocap/easymocap/neuralbody/model/base.py
'''
import torch
import torch.nn as nn
from torch import searchsorted

def augment_z_vals(z_vals, perturb=1):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    perturb_rand = perturb * torch.rand(z_vals.shape, device=z_vals.device)
    z_vals = lower + (upper - lower) * perturb_rand
    return z_vals

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))
    
    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(cdf.device)
    inds = searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    if False:
        import matplotlib as mpl
        mpl.use('TkAgg')
        import matplotlib.pyplot as plt
        pdf0 = pdf[0:1].detach().cpu().numpy()
        cdf0 = cdf[0:1].detach().cpu().numpy()
        plt.figure(1)
        plt.title('pdf')
        plt.plot(bins[0].detach().cpu().numpy()[:-1], pdf0.T)
        plt.vlines(samples[0].detach().cpu().numpy(), ymin=0, ymax=1, colors='r')
        plt.figure(2)
        plt.title('cdf')
        plt.plot(cdf0.T)        
        plt.show()
        import ipdb;ipdb.set_trace()
    return samples

    
def get_near_far(ray_o, ray_d, bounds):
    """ get near and far

    Args:
        ray_o (np): 
        ray_d ([type]): [description]
        bounds ([type]): [description]

    Returns:
        near, far, mask_at_box
        这里的near是实际物理空间中的深度
    """
    norm_d = torch.norm(ray_d, dim=-1, keepdim=True)
    viewdir = ray_d/norm_d
    viewdir[(viewdir<1e-10)&(viewdir>-1e-10)] = 1e-10
    viewdir[(viewdir>-1e-10)&(viewdir<1e-10)] = -1e-10
    inv_dir = 1.0/viewdir
    tmin = (bounds[:1] - ray_o[:1])*inv_dir
    tmax = (bounds[1:2] - ray_o[:1])*inv_dir
    # 限定时间是增加的
    t1 = torch.minimum(tmin, tmax)
    t2 = torch.maximum(tmin, tmax)

    near, _ = torch.max(t1, dim=-1)
    far, _ = torch.min(t2, dim=-1)
    mask_at_box = near < far
    # pts = ray_o[mask_at_box] + far[mask_at_box, None]/norm_d[mask_at_box] * ray_d[mask_at_box]
    # if (pts.max(0)[0] > bounds[1]).any():
    #     print(pts.max(0)[0] - bounds[1])
    #     import ipdb;ipdb.set_trace()
    return near, far, mask_at_box
    
def get_near_far_RTBBox(ray_o, ray_d, bounds, R, T):
    # sample the near far in canonical coordinate
    ray_o_rt = (ray_o - T) @ R
    ray_d_rt = ray_d @ R
    near, far, mask_at_box = get_near_far(ray_o_rt, ray_d_rt, bounds)
    return near, far, mask_at_box

def concat(retlist, dim=0, unsqueeze=True, mask=None):
    res = {}
    if len(retlist) == 0:
        return res
    for key in retlist[0].keys():
        val = torch.cat([r[key] for r in retlist], dim=dim)
        if mask is not None and val.shape[0] != mask.shape[0]:
            val_ = torch.zeros((mask.shape[0], *val.shape[1:]), device=val.device, dtype=val.dtype)
            val_[mask] = val
            val = val_
        if unsqueeze:
            val = val.unsqueeze(0)
        res[key] = val
    return res

class Base(nn.Module):
    def __init__(self, sample_args) -> None:
        super().__init__()
        self.cache = {}
        self.N_samples = sample_args.N_samples
        self.sample_method = sample_args.method
        self.sample_args = sample_args
        z_steps = torch.linspace(0, 1, self.N_samples).reshape(1, -1)
        self.register_buffer('z_steps', z_steps)
    
    def model(self, key):
        self.current = key
        return self
    
    def clear_cache(self):
        self.cache = {}

    def before(self, batch, name):
        """ The operation before each step
            - neuralbody: encode sparse voxel
            - aninerf: encode blending weight
        """
        datas = {key.replace(name+'_', ''):val for key,val in batch.items() if key.startswith(name)}
        return datas

    def calculate_density(self, wpts):
        raise NotImplementedError

    @staticmethod
    def sample(near, far, ray_o, ray_d, z_steps, split):
        # This function provides a uniform sample strategy
        # Override this function to implement more sampling, e.g. importance_sample, unisurf
        z_vals = near * (1 - z_steps) + far * z_steps
        z_vals = z_vals.unsqueeze(2)
        if split == 'train':
            z_vals = augment_z_vals(z_vals)
        pts = ray_o + ray_d * z_vals
        return pts, z_vals
    
    def sample_pdf(self, near, far, ray_o, ray_d, split):
        pts, z_vals = self.sample(near, far, ray_o, ray_d, self.z_steps, split)
        # forward
        viewdirs = ray_d / torch.norm(ray_d, keepdim=True, dim=-1)
        viewdirs_ = viewdirs.expand(-1, pts.shape[1], -1)
        raw_output = self.calculate_density_color(pts, viewdirs_)
        # resample
        alpha = raw_output['occupancy'][..., 0]
        weights = alpha * torch.cumprod(
            torch.cat(
                [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
                -1), -1)[:, :-1]
        z_vals_mid = .5 * (z_vals[...,1:, 0] + z_vals[...,:-1, 0])

        z_samples = sample_pdf(z_vals_mid, weights[:, 1:-1], self.sample_args.N_importance, det=(split!='train'))
        z_samples = z_samples.detach()
        if split == 'train':
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples[..., None]], -2), -2)
        else:
            z_vals = z_samples[..., None]
        pts = ray_o + ray_d * z_vals
        return pts, z_vals

    def calculate_density_color(self, wpts, viewdir):
        raise NotImplementedError
    
    def calculate_density_color_from_ray(self, ray_o, ray_d, near, far, split):
        if self.sample_method == 'uniform':
            pts, z_vals = self.sample(near, far, ray_o, ray_d, self.z_steps, split)
        elif self.sample_method == 'importance':
            pts, z_vals = self.sample_pdf(near, far, ray_o, ray_d, split)
        elif self.sample_method == 'raymarching':
            import ipdb;ipdb.set_trace()
        else:
            print('Please check the sample :', self.sample_method)
            raise NotImplementedError
        viewdirs = ray_d / torch.norm(ray_d, keepdim=True, dim=-1)
        viewdirs_ = viewdirs.expand(-1, pts.shape[1], -1)
        raw_output = self.calculate_density_color(pts, viewdirs_)
        return z_vals, pts, raw_output

    def after(self, batch):
        pass