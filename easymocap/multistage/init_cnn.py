'''
  @ Date: 2022-04-26 17:54:28
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-30 19:47:04
  @ FilePath: /EasyMocapPublic/easymocap/multistage/init_cnn.py
'''
import os
import numpy as np
import cv2
from tqdm import tqdm
from os.path import join
import torch
from ..bodymodel.base import Params
from ..estimator.wrapper_base import bbox_from_keypoints
from ..mytools.writer import write_smpl
from ..mytools.reader import read_smpl

class InitSpin:
    # initialize the smpl results by spin
    def __init__(self, mean_params, ckpt_path, share_shape, 
        multi_person=False, compose_mp=False) -> None:
        from ..estimator.SPIN.spin_api import SPIN
        import torch
        self.share_shape = share_shape
        self.spin_model = SPIN(
            SMPL_MEAN_PARAMS=mean_params, 
            checkpoint=ckpt_path,
            device=torch.device('cpu'))
        self.distortMap = {}
        self.multi_person = multi_person
        self.compose_mp = compose_mp

    def undistort(self, image, K, dist, nv):
        if np.linalg.norm(dist) < 0.01:
            return image
        if nv not in self.distortMap.keys():
            h,  w = image.shape[:2]
            mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K, (w,h), 5)
            self.distortMap[nv] = (mapx, mapy)
        mapx, mapy = self.distortMap[nv]
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        return image

    def __call__(self, body_model, body_params, infos):
        self.spin_model.model.to(body_model.device)
        self.spin_model.device = body_model.device
        params_all = []
        for nf, imgname in enumerate(tqdm(infos['imgname'], desc='Run SPIN')):
            # 暂时不考虑多视角情况
            # TODO: 没有考虑多人的情况
            basename = os.sep.join(imgname.split(os.sep)[-2:]).split('.')[0] + '.json'
            sub = os.path.dirname(basename)
            cache_dir = os.path.abspath(join(os.sep.join(imgname.split(os.sep)[:-3]), 'cache_spin'))
            outname = join(cache_dir, basename)
            if os.path.exists(outname):
                params = read_smpl(outname)
                if self.multi_person:
                    params_all.append(params)
                else:
                    params_all.append(params[0])
                continue
            camera = {key: infos[key][nf].numpy() for key in ['K', 'Rc', 'Tc', 'dist']}
            camera['R'] = camera['Rc']
            camera['T'] = camera['Tc']
            image = cv2.imread(imgname)
            image = self.undistort(image, camera['K'], camera['dist'], sub)
            if len(infos['keypoints2d'].shape) == 3:
                k2d = infos['keypoints2d'][nf][None]
            else:
                k2d = infos['keypoints2d'][nf]
            params_current = []
            for pid in range(k2d.shape[0]):
                keypoints = k2d[pid].numpy()
                bbox = bbox_from_keypoints(keypoints)
                nValid = (keypoints[:, -1] > 0).sum()
                if nValid > 4:
                    result = self.spin_model(body_model, image, 
                        bbox, keypoints, camera, ret_vertices=False)
                elif len(params_all) == 0:
                    print('[WARN] not enough joints: {} in first frame'.format(imgname))
                else:
                    print('[WARN] not enough joints: {}'.format(imgname))
                    if self.multi_person:
                        result = {'body_params': params_all[-1][pid]}
                    else:
                        result = {'body_params': params_all[-1]}
                params = result['body_params']
                params['id'] = pid
                params_current.append(params)
            write_smpl(outname, params_current)
            if self.multi_person:
                params_all.append(params_current)
            else:
                params_all.append(params_current[0])
        if not self.multi_person:
            params_all = Params.merge(params_all, share_shape=self.share_shape)
            params_all = body_model.encode(params_all)
        elif self.compose_mp:
            params_all = Params.merge([Params.merge(p_, share_shape=False) for p_ in params_all], share_shape=False, stack=np.stack)
            params_all['id'] = 0
        return params_all