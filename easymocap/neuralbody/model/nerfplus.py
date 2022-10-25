import torch
import torch.nn as nn
from .nerf import Nerf, EmbedMLP, MultiLinear
from os.path import join
from ...mytools.file_utils import read_json
import numpy as np

def create_dynamic_embedding(mode, embed):
    if mode == 'dense':
        embedding = nn.Embedding(embed.shape[0], embed.shape[1])
    elif mode == 'mlp':
        if 'D' not in embed.keys():
            embedding = EmbedMLP(
                input_ch=1,
                multi_res=32,
                W=128,
                D=2,
                bounds=embed.shape[0],
                output_ch=embed.shape[1])
        else:
            embedding = EmbedMLP(
                input_ch=1,
                multi_res=32,
                W=embed.W,
                D=embed.D,
                bounds=embed.shape[0],
                output_ch=embed.shape[1])
    else:
        raise NotImplementedError
    return embedding

class NeRFT(Nerf):
    def __init__(self, embed, nerf):
        nerf['latent'] = {'time': embed.shape[1]}
        super().__init__(**nerf)
        self.mode = embed.mode
        self.embedding = create_dynamic_embedding(self.mode, embed)
        self.cache = {}
    
    def clear_cache(self):
        self.cache = {}

    def before(self, batch, name):
        data = super().before(batch, name)
        nf, nv = batch['meta']['time'][0], batch['meta']['nview'][0]
        if 'frame' in name:
            nf = nf + batch[name+'_frame'] - batch['meta']['nframe']
        self.cache['embed'] = self.embedding(nf)
        return data

    def calculate_density_color(self, wpts, viewdir, **kwargs):
        latents = {'time': self.cache['embed']}
        return super().calculate_density_color(wpts, viewdir, latents, **kwargs)

class NeRFGroundShadow(Nerf):
    def __init__(self, embed, shadow, nerf):
        super().__init__(**nerf)
        self.shadow = MultiLinear(
            D=shadow.D,
            W=shadow.W,
            input_ch=self.ch_pts + embed.shape[1],
            output_ch=1, # 输出一维阴影
            init_bias=5,
            act_fn='none',
            skips=[]
        )
        nerf['latent'] = {'time': embed.shape[1]}
        self.mode = embed.mode
        self.embedding = create_dynamic_embedding(self.mode, embed)
        self.cache = {}
    
    def clear_cache(self):
        self.cache = {}

    def before(self, batch, name):
        data = super().before(batch, name)
        nf, nv = batch['meta']['time'][0], batch['meta']['nview'][0]
        if 'frame' in name:
            nf = nf + batch[name+'_frame'] - batch['meta']['nframe']
        self.cache['embed'] = self.embedding(nf)
        return data

    def calculate_density_color(self, wpts, viewdir, **kwargs):
        latents = self.cache['embed'][None]
        raw_output = super().calculate_density_color(wpts, viewdir, **kwargs)
        pts_embed = self.embed_pts(wpts)
        latents = latents.expand(pts_embed.shape[0], pts_embed.shape[1], -1)
        shadow = self.shadow(torch.cat([pts_embed, latents], dim=-1))
        shadow = torch.sigmoid(shadow)
        raw_output['rgb'] = shadow * raw_output['rgb']
        return raw_output

class NeRFT_pretrain(Nerf):
    def __init__(self, nerf, embed_time, pretrain, dcolor):
        super().__init__(**nerf)
        state_dict = torch.load(pretrain, map_location='cpu')['state_dict']
        state_dict_new = {}
        for key, val in state_dict.items():
            if key.startswith('train_renderer.'): continue
            state_dict_new[key.replace('network.', '')] = val
        self.load_state_dict(state_dict_new)
        for p in self.parameters():
            p.requires_grad = False
        self.mode = embed_time.mode
        self.embedding = create_dynamic_embedding(self.mode, embed_time)
        # create dynamic color layers:
        # input: embeding, pts, viewdirs => delta_color
        self.delta_color = MultiLinear(
            input_ch=self.ch_pts+self.ch_dir+embed_time.shape[1],
            output_ch=3,
            init_bias=0.,
            act_fn='none',
            **dcolor
        )
    
    def before(self, batch, name):
        data = super().before(batch, name)
        nf, nv = batch['meta']['time'][0], batch['meta']['nview'][0]
        if 'frame' in name:
            nf = nf + batch[name+'_frame'] - batch['meta']['nframe']
        self.cache['embed'] = self.embedding(nf)
        return data
    
    def calculate_density_color(self, wpts, viewdir, **kwargs):
        raw_output = super().calculate_density_color(wpts, viewdir, **kwargs)
        wpts = self.embed_pts(wpts)
        input_views = self.embed_dir(viewdir)
        embed = self.cache['embed'][None].expand(wpts.shape[0], wpts.shape[1], -1)
        inputs = torch.cat([wpts, input_views, embed], dim=-1)
        delta_color = self.delta_color(inputs) * 0.1 # avoid too much delta
        color = torch.sigmoid(delta_color + raw_output['raw_rgb'])
        raw_output['rgb'] = color
        return raw_output

class DynamicColorNerf(NeRFT):
    def __init__(self, pid, traj, embed, nerf, opt_traj_step, share_view):
        super().__init__(embed, nerf)
        trajs = []
        for nf in range(*traj.ranges):
            annname = join(traj.path, '{:06d}.json'.format(nf))
            annots = read_json(annname)
            annots = [a for a in annots if a['id'] == pid][0]
            center = annots['keypoints3d'][0][:3]
            trajs.append(center)
        if share_view:
            traj.nViews = 1
        trajs = np.array(trajs, dtype=np.float32)[None].repeat(traj.nViews, 0)
        trajs = torch.Tensor(trajs)
        self.register_buffer('init_t', trajs.clone())
        self.traj = nn.Parameter(trajs)
        self.opt_traj_step = opt_traj_step
        self.share_view = share_view
    
    def before(self, batch, name):
        data = super().before(batch, name)
        if False:
            results = []
            for nf in range(200):
                t = torch.tensor([nf/200*300], dtype=torch.float32).to(data['rgb'].device)
                embed = self.embedding(t)
                results.append(embed.detach().cpu().numpy())
            import numpy as np
            results = np.vstack(results)
            import matplotlib.pyplot as plt
            plt.imshow(results)
            plt.show()
            import ipdb;ipdb.set_trace()
        nf, nv = batch['meta']['nframe'][0], batch['meta']['nview'][0]
        if batch['meta']['sub'][0].startswith('novel'):
            nv = 0 # use view 0 for novel view
        if batch['meta']['split'][0] != 'train':
            nv = 0
        if self.share_view:
            nv = 0
        self.cache['T'] = self.traj[nv, nf]
        self.cache['init'] = self.init_t[nv, nf]
        if batch['step'] < self.opt_traj_step:
            self.cache['T'] = self.cache['T'].detach()
        reg_t = self.cache['T'] - self.cache['init']
        self.cache['reg_t'] = reg_t
    
    def calculate_density_color(self, wpts, viewdir, **kwargs):
        # unpose to canonical space
        wpts = wpts - self.cache['T'][None, None]
        output = super().calculate_density_color(wpts, viewdir, **kwargs)
        return output

if __name__ == '__main__':
    embedding = MultiResEmbedding()