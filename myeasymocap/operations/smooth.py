from typing import Any
import numpy as np

class SmoothAny:
    def __init__(self, window_size) -> None:
        self.w = window_size

    def __call__(self, value, with_conf=True):
        wsize = self.w
        value = value.copy()
        if with_conf:
            pos_sum = np.zeros_like(value[:-wsize, ..., :-1])
            conf_sum = np.zeros_like(value[:-wsize, ..., -1:])
        else:
            pos_sum = np.zeros_like(value[:-wsize])
        for w in range(wsize):
            if with_conf:
                pos_sum += value[w:w-wsize, ..., :-1] * value[w:w-wsize, ..., -1:]
                conf_sum += value[w:w-wsize, ..., -1:]
            else:
                pos_sum += value[w:w-wsize]
        if with_conf:
            pos_smooth = pos_sum / (1e-5 + conf_sum)
            value[wsize//2:-wsize//2] = np.dstack([pos_smooth, conf_sum])
        else:
            pos_smooth = pos_sum / (wsize)
            value[wsize//2:-wsize//2] = pos_smooth
        return value

class Smooth(SmoothAny):
    def __call__(self, keypoints3d):
        return {'keypoints3d': super().__call__(keypoints3d, with_conf=True)}

class SmoothPoses:
    def __init__(self, window_size) -> None:
        self.W = window_size
    
    def __call__(self, params):
        # TODO: 这个是使用了padding的
        poses = params['poses']
        padding_before = poses[:1].copy().repeat(self.W, 0)
        padding_after = poses[-1:].copy().repeat(self.W, 0)
        mean = poses.copy()
        nFrames = mean.shape[0]
        poses_full = np.vstack([padding_before, poses, padding_after])
        for w in range(1, self.W+1):
            mean += poses_full[self.W-w:self.W-w+nFrames]
            mean += poses_full[self.W+w:self.W+w+nFrames]
        mean /= 2*self.W + 1
        params['poses'] = mean
        return {'params': params}
    
class SmoothRealtime:
    def __init__(self, opt_name, win_sizes) -> None:
        # import cv2
        self.size = {}
        self.opt_name = opt_name
        self.smdata={}
        for idx, name in enumerate(opt_name):
            self.smdata[name] = []
            self.size[name] = win_sizes[idx]
    def cvt_Rh_Rot(self, Rh):
        import cv2
        RotList = []
        Rh = Rh.reshape((-1,3))
        for i in range(Rh.shape[0]):
            RotList.append(cv2.Rodrigues(Rh[i])[0])
        return np.stack(RotList)

    def cvt_Rot_Rh(self, Rot):
        import cv2
        RhList = []
        for i in range(Rot.shape[0]):
            RhList.append(cv2.Rodrigues(Rot[i])[0].reshape(3))
        return np.stack(RhList).reshape((1,-1))

    def now_smplh(self):
        data={}
        for name in self.opt_name:
            # if name == 'Rh':
            if name in ['Rh','poses']:
                out = (sum(self.smdata[name])/len(self.smdata[name]))
                data[name] = self.cvt_Rot_Rh(out) 
            else:
                data[name] = (sum(self.smdata[name])/len(self.smdata[name])) 
        return data
    def __call__(self, data):
        # breakpoint()
        for name in self.opt_name:
            if name in ['Rh','poses']:
                self.smdata[name].append(self.cvt_Rh_Rot(data[name].copy()))
                if len(self.smdata[name])>self.size[name]:
                    self.smdata[name].pop(0)
                out = (sum(self.smdata[name])/len(self.smdata[name]))
                data[name] = self.cvt_Rot_Rh(out) #.reshape(1,self.smdata[key][0].shape[-1])
            else:
                self.smdata[name].append(data[name].copy())
                if len(self.smdata[name])>self.size[name]:
                    self.smdata[name].pop(0)
                data[name] = (sum(self.smdata[name])/len(self.smdata[name])) #.reshape(1,self.smdata[key][0].shape[-1])
        return data
class SmoothHandlr:
    def __init__(self, opt_name, win_sizes):
        self.smooth_handl = SmoothRealtime(opt_name, win_sizes)
        self.smooth_handr = SmoothRealtime(opt_name, win_sizes)
    def __call__(self, params_l, params_r) -> Any:
        params_l = self.smooth_handl(params_l)
        params_r = self.smooth_handr(params_r)
        return {'params_l': params_l, 'params_r': params_r}

class SmoothSmplh(SmoothRealtime):
    def __init__(self, opt_name, win_sizes):
        self.opt_name = opt_name
        self.win_sizes = win_sizes
        self.smooth_lists=[]
        # self.smooth_smplh = SmoothRealtime(opt_name, win_sizes)
    def __call__(self, params_smplh):
        #TODO 应该根据id， 放入到对应的smooth列表中， 长久不在的要删除或者清空,之后把id作为输入，然后smoothlists换成map
        bz = params_smplh['Rh'].shape[0]
        while (len(self.smooth_lists)<bz):
            self.smooth_lists.append(SmoothRealtime(self.opt_name, self.win_sizes))
        for i in range(bz):
            param={}
            for key in params_smplh.keys():
                param[key] = params_smplh[key][i].reshape(1,-1)
            out = self.smooth_lists[i](param)
            for key in params_smplh.keys():
                params_smplh[key][i] = out[key]
        # params_smplh = self.smooth_smplh(params_smplh)
        return {'params_smplh': params_smplh}

class Smoothkeypoints3d(SmoothRealtime):
    def __init__(self, opt_name, win_sizes):
        self.smooth_smplh = SmoothRealtime(opt_name, win_sizes)
    def __call__(self, keypoints3d):
        ret = self.smooth_smplh({'keypoints3d':keypoints3d})
        return ret
    