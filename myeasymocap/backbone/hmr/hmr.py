import os
import numpy as np
import torch
from ..basetopdown import BaseTopDownModelCache, gdown_models
import pickle
from .models import hmr

class MyHMR(BaseTopDownModelCache):
    def __init__(self, ckpt, url=None):
        super().__init__('handhmr', bbox_scale=1., res_input=224)
        self.model = hmr()
        self.model.eval()
        if not os.path.exists(ckpt) and url is not None:
            gdown_models(ckpt, url)
        assert os.path.exists(ckpt), f'{ckpt} not exists'
        checkpoint = torch.load(ckpt)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        state_dict = checkpoint['state_dict']
        prefix = 'model.'
        self.load_checkpoint(self.model, state_dict, prefix, strict=True)
        self.model.to(self.device)
    
    def __call__(self, bbox, images, imgnames):
        output = super().__call__(bbox, images, imgnames)
        Rh = output['params']['poses'][:3].copy()
        poses = output['params']['poses'][3:]
        Th = np.zeros_like(Rh)
        Th[2] = 1.
        output['params'] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'shapes': output['params']['shapes'],
        }
        return output