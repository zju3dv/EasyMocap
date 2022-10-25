'''
  @ Date: 2021-09-03 16:52:42
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-03 22:41:50
  @ FilePath: /EasyMocap/easymocap/neuralbody/model/neuralbody.py
'''
from .nerf import Nerf, EmbedMLP
import torch
import spconv
try:
    if spconv.__version__.split('.')[0] == '2':
        import spconv.pytorch as spconv
except:
    pass
import torch.nn as nn
import torch.nn.functional as F

def pts_to_can_pts(pts, sp_input):
    """transform pts from the world coordinate to the smpl coordinate"""
    Th = sp_input['Th']
    pts = pts - Th
    R = sp_input['R']
    pts = torch.matmul(pts, R)
    if 'scale' in sp_input.keys():
        pts = pts / sp_input['scale'].float()
    return pts

def get_grid_coords(pts, sp_input, voxel_size):
    # convert xyz to the voxel coordinate dhw
    dhw = pts[..., [2, 1, 0]]
    # min_dhw = sp_input['bounds'][:, 0, [2, 1, 0]]
    min_dhw = sp_input['min_dhw']
    dhw = dhw - min_dhw[:, None]
    dhw = dhw / voxel_size
    # convert the voxel coordinate to [-1, 1]
    out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
    dhw = dhw / out_sh * 2 - 1
    # convert dhw to whd, since the occupancy is indexed by dhw
    grid_coords = dhw[..., [2, 1, 0]]
    if True: # clamp points
        grid_coords[grid_coords>1.] = 1.
        grid_coords[grid_coords<-1.] = -1
    return grid_coords

def encode_sparse_voxels(xyzc_net, sp_input, code):
    coord = sp_input['coord']
    out_sh = sp_input['out_sh']
    batch_size = sp_input['batch_size']

    xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)
    feature_volume = xyzc_net(xyzc)

    return feature_volume

def my_grid_sample(feat, grid, mode='bilinear', align_corners=True, padding_mode='border'):
    B, C, ID, IH, IW = feat.shape
    assert(B==1)
    feat = feat[0]
    grid = grid[0, 0, 0]
    N_g, _ = grid.shape
    ix, iy, iz = grid[..., 0], grid[..., 1], grid[..., 2]

    ix = ((ix+1)/2) * (IW-1)
    iy = ((iy+1)/2) * (IH-1)
    iz = ((iz+1)/2) * (ID-1)
    with torch.no_grad():
        ix_floor = torch.floor(ix).long()
        iy_floor = torch.floor(iy).long()
        iz_floor = torch.floor(iz).long()
        ix_ceil = ix_floor + 1
        iy_ceil = iy_floor + 1
        iz_ceil = iz_floor + 1
    # w_000: xyz
    w_111 = (ix-ix_floor) * (iy-iy_floor) * (iz-iz_floor)

    w_110 = (ix-ix_floor) * (iy-iy_floor) * (iz_ceil-iz)
    w_101 = (ix-ix_floor) * (iy_ceil-iy) * (iz-iz_floor)
    w_011 = (ix_ceil-ix) * (iy-iy_floor) * (iz-iz_floor)

    w_100 = (ix-ix_floor) * (iy_ceil-iy) * (iz_ceil-iz)
    w_010 = (ix_ceil-ix) * (iy-iy_floor) * (iz_ceil-iz)
    w_001 = (ix_ceil-ix) * (iy_ceil-iy) * (iz-iz_floor)

    w_000 = (ix_ceil-ix) * (iy_ceil-iy) * (iz_ceil-iz)

    weights = [w_000, w_001, w_010, w_100, w_011, w_101, w_110, w_111]
    with torch.no_grad():
        torch.clamp(ix_floor, 0, IW-1, out=ix_floor)
        torch.clamp(iy_floor, 0, IH-1, out=iy_floor)
        torch.clamp(iz_floor, 0, ID-1, out=iz_floor)
        torch.clamp(ix_ceil, 0, IW-1, out=ix_ceil)
        torch.clamp(iy_ceil, 0, IH-1, out=iy_ceil)
        torch.clamp(iz_ceil, 0, ID-1, out=iz_ceil)
    v_000 = feat[:, iz_floor, iy_floor, ix_floor]

    v_001 = feat[:, iz_ceil,  iy_floor, ix_floor]
    v_010 = feat[:, iz_floor, iy_ceil, ix_floor]
    v_100 = feat[:, iz_floor,  iy_floor, ix_ceil]

    v_011 = feat[:, iz_ceil,  iy_ceil, ix_floor]
    v_101 = feat[:, iz_ceil,  iy_floor, ix_ceil]
    v_110 = feat[:, iz_floor,  iy_ceil, ix_ceil]

    v_111 = feat[:, iz_ceil,  iy_ceil, ix_ceil]

    val = v_000 * w_000[None] + v_001 * w_001[None] + v_010 * w_010[None] + v_100 * w_100[None] + \
          v_011 * w_011[None] + v_101 * w_101[None] + v_110 * w_110[None] + v_111 * w_111[None]
    return val[None, :, None, None]

def interpolate_features(grid_coords, feature_volume, padding_mode):
    features = []
    for volume in feature_volume:
        feature = F.grid_sample(volume,
                                grid_coords,
                                padding_mode=padding_mode,
                                align_corners=True)
        # feature = my_grid_sample(volume, grid_coords)
        features.append(feature)
    features = torch.cat(features, dim=1)
    # features: (nFeatures, nPoints)
    features = features.view(-1, features.size(4))
    features = features.transpose(0, 1)
    return features

def prepare_sp_input(batch, voxel_pad, voxel_size):
    vertices = batch['vertices'][0]
    R, Th = batch['R'][0], batch['Th'][0]
    # Here: R^-1 @ (X - T) => (X - T) @ R^-1.T
    can_xyz = torch.matmul(vertices - Th, R.transpose(0, 1).transpose(0, 1))
    # construct the coordinate
    min_xyz, _ = torch.min(can_xyz - voxel_pad, dim=0)
    max_xyz, _ = torch.max(can_xyz + voxel_pad, dim=0)
    dhw = can_xyz[:, [2, 1, 0]]
    min_dhw = min_xyz[[2, 1, 0]]
    max_dhw = max_xyz[[2, 1, 0]]
    # coordinate in the canonical space
    coord = torch.round((dhw - min_dhw)/voxel_size).to(torch.int)
    # construct the output shape
    out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).to(torch.int)
    x = 32
    out_sh = (out_sh | (x - 1)) + 1
    
    # feature, coordinate, shape, batch size
    sp_input = {}
    # coordinate: [N, 4], batch_idx, z, y, x
    coord = coord[None]
    sh = coord.shape
    idx = [torch.full([sh[1]], i, dtype=torch.long) for i in range(sh[0])]
    idx = torch.cat(idx).to(coord)

    out_sh, _ = torch.max(out_sh, dim=0)
    sp_input = {
        'coord': torch.cat([idx[:, None], coord[0]], dim=1),
        'out_sh': out_sh.tolist(),
        'batch_size': sh[0],
        # used for feature interpolation
        'min_dhw': min_dhw[None],
        'max_dhw': max_dhw[None],
        'min_xyz': min_xyz[None],
        'max_xyz': max_xyz[None],        
        'R': R,
        'Th': Th,
        # 'scale': ,
    }
    return sp_input
    
class Network(Nerf):
    def __init__(self, nerf, embed_vert, embed_time, sparse, use_mlp_vert=False, start_embed_time=0,
        use_canonical_viewdirs=True, use_viewdirs=False,
        padding_mode='zeros',
        voxel_size = [0.005, 0.005, 0.005], voxel_pad = [0.05, 0.05, 0.05],
        pretrain=None) -> None:
        nerf['ch_pts_extra'] = sparse['dims'][-1]*2 + sparse['dims'][-2] + sparse['dims'][-3]
        nerf['latent'] = {'time': embed_time.shape[1]}
        if use_canonical_viewdirs and use_viewdirs:
            # 注意:这里不能写*2, 因为多个人的时候这个字典没有拷贝
            nerf['dim_dir'] = 6
        self.use_canonical_viewdirs = use_canonical_viewdirs
        print('- [Load Network](Neuralbody) use_viewdirs={}, use_canonical_viewdirs={}'.format(use_viewdirs, use_canonical_viewdirs))
        self.use_world_viewdirs = use_viewdirs
        super().__init__(**nerf)
        self.sp_input = None
        self.feature_volume = None
        # add embedding
        self.nVertices = embed_vert[0]
        self.nFrames = embed_time.shape[0]
        self.embed_vert = nn.Embedding(embed_vert[0], embed_vert[1])
        self.padding_mode = padding_mode
        if embed_time.mode == 'dense':
            self.embed_time = nn.Embedding(embed_time.shape[0], embed_time.shape[1])
        elif embed_time.mode == 'mlp':
            if 'res' not in embed_time.keys():
                self.embed_time = EmbedMLP(
                    input_ch=1,
                    multi_res=32,
                    W=128,
                    D=2,
                    bounds=embed_time.shape[0],
                    output_ch=embed_time.shape[1])
            else:
                self.embed_time = EmbedMLP(
                    input_ch=1,
                    multi_res=embed_time['res'],
                    W=128,
                    D=embed_time.D,
                    bounds=embed_time.shape[0],
                    output_ch=embed_time.shape[1])
        self.start_embed_time = start_embed_time
        vert_idx = torch.arange(0, embed_vert[0])
        self.xyzc_net = SparseConvNet(**sparse)
        self.register_buffer('vert_idx', vert_idx)
        self.register_buffer('voxel_size', torch.tensor(voxel_size).reshape(1, 3))
        self.register_buffer('voxel_pad', torch.tensor(voxel_pad).reshape(1, 3))
        if pretrain is not None:
            print('[nerf] load from {}'.format(pretrain))
            checkpoint = torch.load(pretrain)
            self.load_state_dict(checkpoint['net'], strict=True)
        self.current = None
        self.sparse_feature = {}
    
    def clear_cache(self):
        self.sparse_feature = {}

    def model(self, key):
        self.current = key
        return self

    def before(self, batch, name):
        self.current = name
        datas = {key.replace(name+'_', ''):val for key,val in batch.items() if key.startswith(name)}
        device = datas['ray_o'].device
        sp_input = prepare_sp_input(datas, self.voxel_pad, self.voxel_size)
        pid = int(name.split('_')[1])
        sp_input['latent_person'] = torch.IntTensor([pid]).to(device)
        frame = batch['meta']['time'].to(device)
        if 'frame' in name:
            frame = frame + batch[name+'_frame'] - batch['meta']['nframe']
        latent_time = self.embed_time(frame)
        self.latent_time = latent_time
        code = self.embed_vert(self.vert_idx)
        feature_volume = encode_sparse_voxels(self.xyzc_net, sp_input, code)
        self.sparse_feature[self.current] = {
            'pid': pid,
            'sp_input': sp_input,
            'feature_volume': feature_volume,
            'latent_time': latent_time
        }
        return datas
    
    def calculate_density(self, wpts, **kwargs):
        raise NotImplementedError

    def calculate_density_color(self, wpts, viewdir, **kwargs):
        # interpolate features
        wpts_flat = wpts.reshape(-1, 3)
        # convert viewdir to canonical space
        sparse_feature = self.sparse_feature[self.current]
        viewdir_canonical = torch.matmul(viewdir, sparse_feature['sp_input']['R'])
        if self.use_canonical_viewdirs and not self.use_world_viewdirs:
            viewdir = viewdir_canonical
        elif self.use_canonical_viewdirs and self.use_world_viewdirs:
            viewdir = torch.cat([viewdir, viewdir_canonical], dim=-1)
        viewdir_flat = viewdir.reshape(-1, viewdir.shape[-1])
        ppts = pts_to_can_pts(wpts_flat, sparse_feature['sp_input'])
        valid = (ppts>sparse_feature['sp_input']['min_xyz'])&(ppts<sparse_feature['sp_input']['max_xyz'])
        valid = valid.all(dim=-1)
        if valid.sum() == 0:
            outputs = {
                'occupancy': torch.zeros((*wpts.shape[:-1], 1), device=wpts.device, dtype=wpts.dtype),
                'rgb': torch.zeros((*wpts.shape[:-1], 3), device=wpts.device, dtype=wpts.dtype)
            }
            outputs['raw_rgb'] = outputs['rgb']
            outputs['raw_alpha'] = outputs['occupancy']
            return outputs
        ppts_inlier = ppts[valid]
        viewdir_inlier = viewdir_flat[valid]
        grid_coords = get_grid_coords(ppts_inlier, sparse_feature['sp_input'], self.voxel_size)
        grid_coords = grid_coords[:, None, None]
        # xyzc_features: (nPoints, nFeatures)
        xyzc_features = interpolate_features(grid_coords, sparse_feature['feature_volume'], self.padding_mode)
        # latent_time: (1, nTime)
        outputs = super().calculate_density_color(ppts_inlier, viewdir_inlier, 
            extra_density=xyzc_features, latents={'time': sparse_feature['latent_time'][0]})
        outputs_all = {}
        for key, val in outputs.items():
            padding = torch.zeros((wpts_flat.shape[0], val.shape[-1]), device=val.device, dtype=val.dtype)
            padding[valid] = val
            outputs_all[key] = padding.view(*wpts.shape[:-1], val.shape[-1])
        return outputs_all

class SparseConvNet(nn.Module):
    def __init__(self, dims=[16, 32, 64, 128]):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(dims[0], dims[0], 'subm0')
        self.down0 = stride_conv(dims[0], dims[1], 'down0')

        self.conv1 = double_conv(dims[1], dims[1], 'subm1')
        self.down1 = stride_conv(dims[1], dims[2], 'down1')

        self.conv2 = triple_conv(dims[2], dims[2], 'subm2')
        self.down2 = stride_conv(dims[2], dims[3], 'down2')

        self.conv3 = triple_conv(dims[3], dims[3], 'subm3')
        self.down3 = stride_conv(dims[3], dims[3], 'down3')

        self.conv4 = triple_conv(dims[3], dims[3], 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())