# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .layers import drop_path, to_2tuple, trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos
    
    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MoEMlp(nn.Module):
    def __init__(self, num_expert=1, in_features=1024, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., part_features=256):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.part_features = part_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features - part_features)
        self.drop = nn.Dropout(drop)
        
        self.num_expert = num_expert
        experts = []

        for i in range(num_expert):
            experts.append(
                        nn.Linear(hidden_features, part_features)
                        )
        self.experts = nn.ModuleList(experts)

    def forward(self, x, indices):

        expert_x = torch.zeros_like(x[:, :, -self.part_features:], device=x.device, dtype=x.dtype)

        x = self.fc1(x)
        x = self.act(x)
        shared_x = self.fc2(x)
        indices = indices.view(-1, 1, 1)

        # to support ddp training
        for i in range(self.num_expert):
            selectedIndex = (indices == i)
            current_x = self.experts[i](x) * selectedIndex
            expert_x = expert_x + current_x

        x = torch.cat([shared_x, expert_x], dim=-1)

        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attn_head_dim=None, num_expert=1, part_features=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MoEMlp(num_expert=num_expert, in_features=dim, hidden_features=mlp_hidden_dim, 
                            act_layer=act_layer, drop=drop, part_features=part_features)

    def forward(self, x, indices=None):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), indices))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ViTMoE(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 num_expert=1, part_features=None
                 ):
        # Protect mutable default arguments
        super(ViTMoE, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        self.part_features = part_features

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                num_expert=num_expert, part_features=part_features
                )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding, part_features=self.part_features)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, dataset_source=None):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, dataset_source)
            else:
                x = blk(x, dataset_source)

        x = self.last_norm(x)

        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp

    def forward(self, x, dataset_source=None):
        x = self.forward_features(x, dataset_source)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()

class Head(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),):
        super().__init__()
        self.in_channels = in_channels
        self.deconv_layers = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels)
        self.final_layer = nn.Conv2d(in_channels=num_deconv_filters[-1], out_channels=out_channels, 
                                      kernel_size=1, stride=1, padding=0)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)
    
    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def forward(self, x):
        """Forward function."""
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


class ComposeVit(nn.Module):
    def __init__(self):
        super().__init__()
        cfg_backbone = dict(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.3,
            num_expert=6,
            part_features=192
        )
        cfg_head = dict(
            in_channels=768,
            out_channels=17,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
        )
        cfg_head_133 = dict(
            in_channels=768,
            out_channels=133,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
        )
        self.backbone = ViTMoE(**cfg_backbone)
        self.keypoint_head = Head(**cfg_head)
        self.associate_head = Head(**cfg_head_133)
    
    def forward(self, x):
        indices = torch.zeros((x.shape[0]), dtype=torch.long, device=x.device)
        back_out = self.backbone(x, indices)
        out = self.keypoint_head(back_out)
        if True:
            indices += 5 # 最后一个是whole body dataset
            back_133 = self.backbone(x, indices)
            out_133 = self.associate_head(back_133)
            out_foot = out_133[:, 17:23]
            out = torch.cat([out, out_foot], dim=1)
            if False:
                import cv2
                vis = x[0].permute(1, 2, 0).cpu().numpy()
                mean= np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
                std=np.array([0.229, 0.224, 0.225]).reshape(1, 1 ,3)
                vis = np.clip(vis * std + mean, 0., 1.)
                vis = (vis[:,:,::-1] * 255).astype(np.uint8)
                value = out_133[0].detach().cpu().numpy()
                vis_all = []
                for i in range(value.shape[0]):
                    _val = np.clip(value[i], 0., 1.)
                    _val = (_val * 255).astype(np.uint8)
                    _val = cv2.resize(_val, None, fx=4, fy=4)
                    _val = cv2.applyColorMap(_val, cv2.COLORMAP_JET)
                    _vis = cv2.addWeighted(vis, 0.5, _val, 0.5, 0)
                    vis_all.append(_vis)
                from easymocap.mytools.vis_base import merge
                cv2.imwrite('debug.jpg', merge(vis_all))

                import ipdb; ipdb.set_trace()
        return {
            'output': out
        }

from ..basetopdown import BaseTopDownModelCache
from ..topdown_keypoints import BaseKeypoints

class MyViT(BaseTopDownModelCache, BaseKeypoints):
    def __init__(self, ckpt='data/models/vitpose+_base.pth', single_person=True, url='https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G', **kwargs):
        super().__init__(name='myvit', bbox_scale=1.25,
                         res_input=[192, 256], **kwargs)
        self.single_person = single_person
        model = ComposeVit()
        if not os.path.exists(ckpt):
            print('')
            print('{} not exists, please download it from {} and place it to {}'.format(ckpt, url, ckpt))
            print('')
            raise FileNotFoundError
        ckpt = torch.load(ckpt, map_location='cpu')['state_dict']
        ckpt_backbone = {key:val for key, val in ckpt.items() if key.startswith('backbone.')}
        ckpt_head = {key:val for key, val in ckpt.items() if key.startswith('keypoint_head.')}
        key_whole = 'associate_keypoint_heads.4.'
        ckpt_head_133 = {key.replace(key_whole, 'associate_head.'):val for key, val in ckpt.items() if key.startswith(key_whole)}
        ckpt_backbone.update(ckpt_head)
        ckpt_backbone.update(ckpt_head_133)
        state_dict = ckpt_backbone
        self.load_checkpoint(model, state_dict, prefix='', strict=True)
        model.eval()
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def dump(self, cachename, output):
        _output = output['output']
        kpts = self.get_max_preds(_output)
        kpts_ori = self.batch_affine_transform(kpts, output['inv_trans'])
        kpts = np.concatenate([kpts_ori, kpts[..., -1:]], axis=-1)
        output = {'keypoints': kpts}
        super().dump(cachename, output)
        return output

    def estimate_keypoints(self, bbox, images, imgnames):
        squeeze = False
        if not isinstance(images, list):
            images = [images]
            imgnames = [imgnames]
            bbox = [bbox]
            squeeze = True
        nViews = len(images)
        kpts_all = []
        for nv in range(nViews):
            _bbox = bbox[nv]
            if _bbox.shape[0] == 0:
                if self.single_person:
                    kpts = np.zeros((1, self.num_joints, 3))
                else:
                    kpts = np.zeros((_bbox.shape[0], self.num_joints, 3))
            else:
                img = images[nv]
                # TODO: add flip test
                out = super().__call__(_bbox, img, imgnames[nv])
                kpts = out['params']['keypoints']
            if kpts.shape[-2] == 23:
                kpts = self.coco23tobody25(kpts)
            elif kpts.shape[-2] == 17:
                kpts = self.coco17tobody25(kpts)
            else:
                raise NotImplementedError
            kpts_all.append(kpts)
        if self.single_person:
            kpts_all = [k[0] for k in kpts_all]
            kpts_all = np.stack(kpts_all)
        if squeeze:
            kpts_all = kpts_all[0]
        return {
            'keypoints': kpts_all
        }

    def __call__(self, bbox, images, imgnames):
        return self.estimate_keypoints(bbox, images, imgnames)

if __name__ == '__main__':
    # Load checkpoint
    rand_input = torch.rand(1, 3, 256, 192)
    model = MyViT()
