'''
  @ Date: 2021-09-05 20:24:24
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 21:34:16
  @ FilePath: /EasyMocap/easymocap/neuralbody/renderer/render_loss.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...config import load_object

class LossRGB(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
        self.norm = norm
    
    def forward(self, inp, out):
        diff = inp['rgb'] - out['rgb_map']
        if self.norm == 'l2':
            loss =  torch.mean(diff ** 2)
        elif self.norm == 'l1':
            loss =  torch.mean(torch.abs(diff))
        return loss

class LossDepth(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
        self.norm = norm
        self.depth_max = 15.
    
    def forward(self, inp, out):
        loss_sum = 0
        for key in out['keys']:
            depth_gt = inp[key+'_depth']
            depth_est = out['depth_map']
            valid = depth_gt > 0.
            depth_diff = (depth_gt[valid] - depth_est[valid])/self.depth_max
            loss = torch.sum((depth_diff**2)/(1e-5 + valid.sum()))
            loss_sum += loss
        return loss_sum

class AnyReg(nn.Module):
    def __init__(self, key, norm) -> None:
        super().__init__()
        self.key = key
        self.norm = norm
    
    def forward(self, inp, out):
        if self.key not in out.keys():
            return torch.tensor(0.).to(out['rgb_map'].device)
        diff = out[self.key]
        if self.norm == 'l2':
            loss = torch.mean(diff ** 2)
        elif self.norm == 'norm':
            loss = torch.norm(diff)
        else:
            raise NotImplementedError
        return loss

class LossMask(nn.Module):
    def __init__(self, norm='l1', key='human_0'):
        super().__init__()
        self.norm = norm
        self.key = key
    
    def forward(self, inp, out):
        pred = out['acc_map']
        gt = inp['{}_coord_mask'.format(self.key)]
        if self.norm == 'l1':
            loss_fore = torch.mean(torch.abs(1 - pred[gt]))
            loss_back = torch.mean(torch.abs(pred[~gt]))
            loss = loss_fore + loss_back
        elif self.norm == 'bce':
            target = gt.float()
            loss_fore = F.binary_cross_entropy(pred[gt].clip(1e-5, 1.0 - 1e-5), target[gt])
            loss_back = F.binary_cross_entropy(pred[~gt].clip(1e-5, 1.0 - 1e-5), target[~gt])
            loss = loss_fore + loss_back
        return loss

class LossStepWrapper(nn.Module):
    def __init__(self, weights, module, args):
        super().__init__()
        self.loss = load_object(module, args)
        self.weights = weights
    
    def forward(self, inp, out):
        step = inp['step']
        weight = 0.
        for (start, end, weight) in self.weights:
            if step >= start and (end == -1 or step < end):
                break
        if weight == 0.:
            loss = torch.tensor(0.).to(out['rgb_map'].device)
        else:
            loss = weight * self.loss(inp, out)
        return loss

class LossSemantic(nn.Module):
    def __init__(self, norm, start, end) -> None:
        super().__init__()
        self.norm = norm
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, inp, out):
        semantic = out['semantic_map']
        background = 1 - torch.sum(semantic, dim=-1, keepdim=True)
        semantic = torch.cat([background, semantic], dim=-1)
        loss = self.loss(semantic[0], inp['human_0_semantic'][0])
        return loss

class LossAcc(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
    
    def forward(self, inp, out):
        # TODO:暂时只考虑一个人的情况
        diff = inp['human_0_acc'] - out['fore_acc_map']
        loss = torch.mean(diff**2)
        return loss

class LossCollision(nn.Module):
    def __init__(self, norm, start) -> None:
        super().__init__()
        self.start = start
    
    def forward(self, model, batch, output):
        if len(batch['meta']['collision']) == 0 or batch['step'] < self.start:
            loss = torch.tensor(0.).to(output['rgb_map'].device)
            return loss
        key0s, key1s, ptss = batch['meta']['collision'][0]
        loss_all = []
        for key0, key1, pts in zip(key0s, key1s, ptss):
            pts = pts[None].to(output['rgb_map'].device)
            # occ: (nPoints, 1)
            occ0 = model.model(key0).calculate_density(pts)[0]
            occ1 = model.model(key1).calculate_density(pts)[0]
            occ01 = (occ0 + occ1 + 1e-5)
            occ0_ = torch.clamp(occ0/occ01, min=1e-5)
            occ1_ = torch.clamp(occ1/occ01, min=1e-5)
            loss = -occ01 * (occ0_ * torch.log(occ0_) + occ1_ * torch.log(occ1_))
            loss_all.append(loss)
        loss_all = torch.cat(loss_all, dim=0)
        loss = loss_all.mean()        
        return loss

class LossNormal(nn.Module):
    def __init__(self, norm, perturb_surface_pt=0.01) -> None:
        super().__init__()
        self.perturb_surface_pt = perturb_surface_pt
    
    @staticmethod
    def get_sampling_points(bounds):
        sh = bounds.shape
        min_xyz = bounds[:, 0]
        max_xyz = bounds[:, 1]
        N_samples = 1024 * 32
        x_vals = torch.rand([sh[0], N_samples], device=bounds.device)
        y_vals = torch.rand([sh[0], N_samples], device=bounds.device)
        z_vals = torch.rand([sh[0], N_samples], device=bounds.device)
        vals = torch.stack([x_vals, y_vals, z_vals], dim=2)
        vals = vals.to(bounds.device)
        pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
        return pts

    def forward(self, model, batch, output):
        # TODO:暂时只考虑一个人的情况
        key = 'human_0'
        model = model.model('human_0')
        # (1, 2, 3)
        bounds = batch[key+'_bounds']
        if False:
            pts = self.get_sampling_points(bounds)
        else:
            pts = batch[key+'_pts'].reshape(1, -1, 3)
            # 采样一些点
            N_sample = 1024*32
            idx = torch.randint(0, pts.shape[1], (N_sample,))
            pts = pts[:, idx]
        pts_neighbor = pts \
                + (torch.rand(pts.shape, device=pts.device) - 0.5) * 2. \
                    * self.perturb_surface_pt
        _, gradients = model.gradient(pts)
        _, gradients_nei = model.gradient(pts_neighbor)
        loss = F.mse_loss(F.normalize(gradients, dim=-1), F.normalize(gradients_nei, dim=-1))
        return loss

class LossOcc(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
    
    def forward(self, inp, out):
        loss = 0
        for key in out.keys():
            if not key.endswith('occ'):
                continue
            diff = inp[key] - out[key]
            loss +=  torch.mean(diff ** 2)
        return loss

class SmoothT(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
    
    def forward(self, model, batch, output):
        value = model.models['basketball'].tvec
        nframe = batch['meta']['nframe'].item()
        loss0, loss1 = 0, 0
        cnt = 0
        # 直接优化所有帧的话，会出现全都坍缩到一个点上去
        if nframe > 0:
            cnt += 1
            loss0 = torch.mean((value[nframe] - value[nframe-1].detach())**2)
        if nframe < value.shape[0]-1:
            cnt += 1
            loss1 = torch.mean((value[nframe] - value[nframe+1].detach())**2)
        loss = (loss0 + loss1)/cnt
        return loss

class LossDensity(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
    
    def forward(self, inp, out):
        flag = inp['flag'][0]
        inp['density']
        inpd = inp['density'][0]
        outd = out['density'][0]
        weight = inpd.sum()/inpd.shape[0]
        diff0 = torch.mean(inpd[flag] - outd[flag]) ** 2
        diff1 = torch.mean(inpd[~flag] - outd[~flag]) ** 2
        loss = diff0 + diff1
        return loss

class LossGround(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
    
    def forward(self, inp, out):
        pts = inp['back_pts'][0]
        mask = pts[..., 2] < 1e-5 # under the ground
        occ_dens = out['occ_back'][mask]
        loss = (1. - occ_dens).mean()
        return loss

class LossEntropy(nn.Module):
    def __init__(self, norm, start) -> None:
        super().__init__()
        self.start = start
    
    def forward(self, inp, out):
        occ = out['occ_object']
        if inp['step'] < self.start:
            loss = torch.tensor(0.).to(occ.device)
            return loss
        entropy = -occ * torch.log(torch.clamp(occ, min=1e-5))
        loss = entropy.mean()
        return loss

class LossEntropyInstance(nn.Module):
    def __init__(self, norm, start, end) -> None:
        super().__init__()
        self.norm = norm
        self.start = start
        self.end = end
    
    def forward(self, inp, out):
        instance = out['instance_map'][0]
        loss = torch.tensor(0.).to(instance.device)
        if inp['step'] < self.start or inp['step'] > self.end:
            loss = torch.tensor(0.).to(instance.device)
            return loss
        for ikey, key in enumerate(out['keys']):
            if key+'_label' in inp.keys():
                label = inp[key+'_label'][0]
                msk = (inp[key+'_mask'] & (label > 0))[0]
                if msk.sum() > 0:
                    loss_ = label[msk] * (1 - instance[msk, ikey])
                    loss += loss_.sum()/label[msk].sum()
        return loss

class LossACC(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
    
    def forward(self, inp, out):
        diff = 1. - out['acc_map']
        loss =  torch.mean(diff ** 2)
        return loss

class LossSparseEntropy(nn.Module):
    def __init__(self, norm, start, end) -> None:
        super().__init__()
        self.start = start
        self.end = end
    
    def forward(self, inp, out):
        instance = out['instance_map']
        if inp['step'] < self.start or inp['step'] > self.end:
            loss = torch.tensor(0.).to(instance.device)
            return loss
        entropy = -instance * torch.log(torch.clamp(instance, min=1e-5))
        return entropy.sum(dim=-1).mean()

class LossSemantic1(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inp, out):
        label_origin = inp['label'][0] # (N, 2)
        label_valid = label_origin[:, 0] != -1
        semantic_map = out['feat_map'][0][label_valid] # (N, nFeat)
        if semantic_map.shape[0] == 0:
            return torch.tensor(0.).to(semantic_map.device)
        weight = label_origin[:, 0][label_valid]
        label = label_origin[:, 1][label_valid].long()
        index0 = torch.arange(0, semantic_map.shape[0])
        est = semantic_map[index0, label]
        loss = torch.where(est>0.5, 
            -torch.log(torch.clamp(est, min=1e-5)), 
            1. - est) * weight
        # import ipdb;ipdb.set_trace()
        # loss = - (torch.log(torch.clamp(semantic_map[index0, label], min=1e-5)) * weight).mean()
        loss = loss.mean() / weight.sum()
        return loss

class LossLayer(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
    
    def forward(self, inp, out):
        # weights = {
        #     0: 0.1,
        #     1000: 0.05,
        #     2000: 0.01,
        #     3000: 0.
        # }
        weights = {
            0: 0.1,
            5000: 0.05,
            10000: 0.01,
            15000: 0.
        }
        weight = 0
        for key, val in weights.items():
            if inp['step'] > key:
                weight = val
        if weight == 0.:
            loss = torch.tensor(0.).to(out['rgb_map'].device)
            return loss
        loss = 0.
        cnt = 0.
        for key in out['keys']:
            if key + '_label' not in inp.keys():continue
            label = inp[key+'_label']
            acc = out[key+'_acc_map']
            loss_ = ((label>0)*(1-acc)).sum()
            loss += loss_
            cnt += (label>0).sum()
        loss = weight * loss.sum() / cnt
        # print('step: {}, valid {}, weight={:.2f}, loss = {:.4f}'.format(inp['step'], cnt, weight, loss.item()))
        return loss