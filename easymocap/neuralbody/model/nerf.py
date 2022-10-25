'''
  @ Date: 2021-09-03 17:12:29
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 19:58:54
  @ FilePath: /EasyMocap/easymocap/neuralbody/model/nerf.py
'''
from .base import Base
from .embedder import get_embedder
import torch.nn.functional as F
import torch
import torch.nn as nn

class Nerf(Base):
    def __init__(self, D, W, skips, init_bias=0.693,
        D_rgb=1, W_rgb=256,
        xyz_res=10, dim_pts=3,
        view_res=4, dim_dir=3,# embed
        ch_pts_extra=0, ch_dir_extra=0, # extra channels
        latent={}, # latent code
        pts_to_density=True, pts_to_rgb=False, latent_to_density=False,
        use_viewdirs=True, use_occupancy=True, # option
        act_fn='expsoftplus', linear_func='Linear',
        relu_fn='relu',
        sample_args=None,
        embed_pts='none', embed_dir='none',
        density_bias=True,
        ) -> None:
        super().__init__(sample_args=sample_args)
        # set the embed
        self.embed_pts_name = embed_pts
        self.embed_dir_name = embed_dir
        if embed_pts == 'hash':
            from .hashnerf import get_embedder
        else:
            from .embedder import get_embedder
        self.embed_pts, ch_pts = get_embedder(xyz_res, dim_pts)
        if embed_dir == 'hash':
            from .hashnerf import get_embedder
        else:
            from .embedder import get_embedder
        self.embed_dir, ch_dir = get_embedder(view_res, dim_dir)
        self.latent_keys = list(latent.keys())
        if len(self.latent_keys) > 0:
            latent_dim = sum([latent[key] for key in self.latent_keys])
        else:
            latent_dim = 0
        self.ch_pts = ch_pts
        self.ch_dir = ch_dir
        # set the input channels
        ch_pts_inp = ch_pts_extra
        if pts_to_density:
            ch_pts_inp += ch_pts
        if latent_to_density:
            ch_pts_inp += latent_dim
        self.pts_to_density = pts_to_density
        self.latent_to_density = latent_to_density
        self.pts_to_rgb = pts_to_rgb
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.use_occupancy = use_occupancy
        if linear_func == 'Linear':
            linear_func = lambda input_w, output_w, bias=True: nn.Linear(input_w, output_w, bias=bias)
            cat_dim = -1
        elif linear_func == 'Conv1d':
            linear_func = lambda input_w, output_w, bias=True: nn.Conv1d(input_w, output_w, 1, bias=bias)
            cat_dim = 1
        else:
            raise NotImplementedError
        if relu_fn == 'relu':
            self.relu = F.relu
        elif relu_fn == 'relu6':
            self.relu = F.relu6
        elif relu_fn == 'LeakyRelu':
            self.relu = nn.LeakyReLU(0.1)
        self.cat_dim = cat_dim

        self.pts_linears = nn.ModuleList(
            [linear_func(ch_pts_inp, W, density_bias)] + 
            [linear_func(W, W, density_bias) if i not in self.skips else 
            linear_func(W + ch_pts_inp, W, density_bias) for i in range(D - 1)
        ])

        self.alpha_linear = linear_func(W, 1)
        # following neuralbody structure
        # 1. net feature: (256) => feature_fc => (256, 256)
        self.feature_linear = linear_func(W, W)
        # 2. concat other feature (256+other)
        # 3. latent_fc: (feature+other) => 384 => 256 (feature)
        # 4. concat viewdir, light_pts (256 + viewdir /+ pts)
        # 5. view_fc: (346, 128)
        # 6. rgb_fc: (128, 3)

        self.latent_linear = linear_func(W+latent_dim, W)
        if self.use_viewdirs and not self.pts_to_rgb:
            ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            view_dim = ch_dir + W
            ### Implementation according to the paper
            # self.views_linears = nn.ModuleList(
            #     [linear_func(input_ch_views + W, W//2)] + [linear_func(W//2, W//2) for i in range(D//2)])
        elif self.use_viewdirs and self.pts_to_rgb:
            view_dim = ch_dir + ch_pts + W
        else:
            view_dim = W
        if D_rgb == 1:
            self.views_linears = nn.ModuleList(
                [linear_func(view_dim, W // 2)])
        else:
            self.views_linears = nn.ModuleList(
                [linear_func(view_dim, W_rgb)] + 
                [linear_func(W_rgb, W_rgb) for _ in range(D_rgb - 2)] + 
                [linear_func(W_rgb, W // 2)]
            )
        self.rgb_linear = linear_func(W // 2, 3)

        if self.use_occupancy:
            # initial value for (e^x - 1) / e^x to give 0.5
            self.alpha_linear.bias.data.fill_(init_bias)
        if act_fn == 'exprelu':
            self.act_alpha = lambda x:1 - torch.exp(-torch.relu(x))
        elif act_fn == 'expsoftplus':
            self.act_alpha = lambda x:1 - torch.exp(-F.softplus(x-1))
        elif act_fn == 'sigmoid':
            self.act_alpha = torch.sigmoid

    def calculate_density_color(self, wpts, viewdir, latents={}, **kwargs):
        # Linear mode
        # wpts: (..., 3) => (..., 63)
        # return: (..., 1)
        if self.embed_pts_name == 'hash':
            self.embed_pts.bound = self.datas_cache['bounds']
        # prepare latents
        latent_embeding = []
        for key in self.latent_keys:
            inp = latents[key]
            if self.cat_dim == 1: # with Conv1d
                inp = inp[..., None].expand(*inp.shape, wpts.shape[-1])
            else:
                inp = inp[None].expand(*wpts.shape[:-1], inp.shape[-1])
            latent_embeding.append(inp)
        if len(latent_embeding) > 0:
            latent_embeding = torch.cat(latent_embeding, dim=self.cat_dim)
        wpts = self.embed_pts(wpts)
        extra_density = kwargs.get('extra_density', None)
        inp = []
        if self.pts_to_density:
            inp.append(wpts)
        if extra_density is not None:
            inp.append(extra_density)
        if self.latent_to_density:
            inp.append(latent_embeding)
        inp = torch.cat(inp, dim=self.cat_dim)

        h = inp
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.relu(h)
            if i in self.skips:
                h = torch.cat([inp, h], self.cat_dim)
        alpha = self.alpha_linear(h)
        raw_alpha = alpha
        if self.use_occupancy:
            alpha = self.act_alpha(alpha)
        # rgb part:
        feature = self.feature_linear(h)
        # latent: 
        if len(self.latent_keys) > 0:
            features = [feature, latent_embeding]
            features = torch.cat(features, dim=self.cat_dim)
            feature = self.latent_linear(features)
        # append viewdir
        if self.use_viewdirs:
            input_views = self.embed_dir(viewdir)
            if self.pts_to_rgb:
                feature = torch.cat([feature, input_views, wpts], self.cat_dim)
            else:
                feature = torch.cat([feature, input_views], self.cat_dim)
        for i, l in enumerate(self.views_linears):
            feature = self.views_linears[i](feature)
            feature = self.relu(feature)
        rgb = self.rgb_linear(feature)
        rgb_01 = torch.sigmoid(rgb)
        outputs = {
            'occupancy': alpha,
            'raw_alpha': raw_alpha,
            'rgb': rgb_01,
            'raw_rgb': rgb
        }
        return outputs

class MultiLinear(nn.Module):
    def __init__(self, D, W, input_ch, output_ch, skips,
        init_bias=0.693,
        act_fn='none', linear_func='Linear',
        **kwargs) -> None:
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips
        if linear_func == 'Linear':
            linear_func = lambda input_w, output_w: nn.Linear(input_w, output_w)
            cat_dim = -1
        elif linear_func == 'Conv1d':
            linear_func = lambda input_w, output_w: nn.Conv1d(input_w, output_w, 1)
            cat_dim = 1
        else:
            raise NotImplementedError
        self.cat_dim = cat_dim
        if D > 0:
            self.linears = nn.ModuleList(
                [linear_func(input_ch, W)] + 
                [linear_func(W, W) if i not in self.skips else 
                linear_func(W + input_ch, W) for i in range(D - 1)])
            self.output_linear = linear_func(W, output_ch)
        else:
            self.linears = []
            self.output_linear = linear_func(input_ch, output_ch)

        if act_fn == 'softplus':
            act_fn = torch.nn.functional.softplus
        elif act_fn == 'none':
            act_fn = None
        elif act_fn == 'tanh':
            act_fn = torch.tanh
        self.act_fn = act_fn
        # initialize to 0.5
        self.output_linear.bias.data.fill_(init_bias)
    
    def forward(self, inputs):
        h = inputs
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([inputs, h], self.cat_dim)
        output = self.output_linear(h)
        if self.act_fn is not None:
            output = self.act_fn(output)
        return output

class EmbedMLP(nn.Module):
    def __init__(self, input_ch, output_ch, multi_res, W, D, bounds) -> None:
        super().__init__()
        self.embed, ch_time = get_embedder(multi_res, input_ch)
        self.bounds = bounds
        self.linear = MultiLinear(
            input_ch=ch_time,
            output_ch=output_ch, init_bias=0, act_fn='none',
            D=D, W=W, skips=[])

    def forward(self, time):
        embed = self.embed(time.reshape(1, -1).float()/self.bounds)
        output = self.linear(embed)
        return output

if __name__ == "__main__":
    cfg = {
        'D': 8,
        'W': 256,
        'dim_pts': 3,
        'dim_dir': 3,
        'xyz_res': 10,
        'view_res': 4,
        'skips': [4],
        'use_viewdirs': True,
        'use_occupancy': True,
        'linear_func': 'Linear',
    }
    network = Nerf(**cfg)
    pts = torch.rand((1, 4, 4, 64, 3))
    alpha = network.calculate_density(pts)
    print(alpha.shape)