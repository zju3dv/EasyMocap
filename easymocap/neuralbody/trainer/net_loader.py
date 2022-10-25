'''
  @ Date: 2021-09-05 20:12:41
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 20:12:42
  @ FilePath: /EasyMocap/easymocap/neuralbody/trainer/net_load.py
'''
import os
from termcolor import colored
import torch

def load_model(net,
               optim,
               scheduler,
               recorder,
               model_dir,
               resume=True,
               epoch=-1):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, recorder, model_dir, epoch, last=False):
    os.system('mkdir -p {}'.format(model_dir))
    model = {
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    if epoch > 20 and (epoch+1) % 10 != 0 and not last:
        return 0
    if last:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))
    return 0
    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) <= 20:
        return
    os.system('rm {}'.format(
        os.path.join(model_dir, '{}.pth'.format(min(pths)))))

def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('pretrained model does not exist', 'red'))
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth'
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0
        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = max(epoch, -1)
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('load model: {}'.format(model_path))
    pretrained_model = torch.load(model_path)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1

