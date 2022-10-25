import numpy as np
import cv2
import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class LossLPIPS(VGGPerceptualLoss):
    def forward(self, inp, out):
        W, H = 32, 32
        target = inp['rgb'].reshape(-1, W, H, 3)
        inputs = out['rgb_map'].reshape(-1, W, H, 3)
        if inp['step'] % 100 == 0:
            vis_all = []
            for i in range(inputs.shape[0]):
                target_ = target[i].detach().cpu().numpy()
                inputs_ = inputs[i].detach().cpu().numpy()
                vis = np.hstack([target_, inputs_])
                vis = (vis*255).astype(np.uint8)
                vis_all.append(vis)
            vis_all = np.vstack(vis_all)
            vis_all = vis_all[..., ::-1]
            cv2.imwrite('debug/vis_lpips_{:08d}.jpg'.format(inp['step']), vis_all)
        target = target.permute(0, 3, 1, 2)
        inputs = inputs.permute(0, 3, 1, 2)
        return super().forward(inputs, target)

if __name__ == '__main__':
    lpips = VGGPerceptualLoss()
