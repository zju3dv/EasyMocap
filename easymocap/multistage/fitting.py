'''
  @ Date: 2022-03-22 16:11:44
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-25 11:51:50
  @ FilePath: /EasyMocapPublic/easymocap/multistage/fitting.py
'''
# This function provides a realtime fitting interface
from collections import namedtuple
from time import time, sleep
import numpy as np
import cv2
import torch
import copy

from ..config.baseconfig import load_object_from_cmd
from ..mytools.debug_utils import log, mywarn
from ..mytools import Timer
from ..config import Config
from ..mytools.triangulator import iterative_triangulate
from ..bodymodel.base import Params
from .torchgeometry import axis_angle_to_euler, euler_to_axis_angle

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

from scipy.spatial.transform import Rotation
def aa2euler(aa):
    aa = np.array(aa)
    R = cv2.Rodrigues(aa)[0]
    # res = Rotation.from_dcm(R).as_euler('XYZ', degrees=True)
    res = Rotation.from_matrix(R).as_euler('XYZ', degrees=False)
    return np.round(res, 2).tolist()

def rotmat2euler(rot):
    res = Rotation.from_matrix(rot).as_euler('XYZ', degrees=True)
    return res

def euler2rotmat(euler):
    res = Rotation.from_euler('XYZ', euler, degrees=True)
    return res.as_matrix()

def batch_rodrigues_jacobi(rvec):
    shape = rvec.shape
    rvec = rvec.view(-1, 3)
    device = rvec.device
    dSkew = torch.zeros(3, 9, device=device)
    dSkew[0, 5] = -1
    dSkew[1, 6] = -1
    dSkew[2, 1] = -1
    dSkew[0, 7] =  1
    dSkew[1, 2] =  1
    dSkew[2, 3] =  1
    dSkew = dSkew[None]
    theta = torch.norm(rvec, dim=-1, keepdim=True) + 1e-5
    c = torch.cos(theta)
    s = torch.sin(theta)
    c1 = 1 - c
    itheta = 1 / theta
    r = rvec / theta
    zeros = torch.zeros_like(r[:, :1])
    rx, ry, rz = torch.split(r, 1, dim=1)
    rrt = torch.matmul(r[:, :, None], r[:, None, :])
    skew = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((r.shape[0], 3, 3))
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype)[None]
    rot_mat = I + s[:, None] * skew + c1[:, None] * torch.bmm(skew, skew)
    
    drrt = torch.stack([
        rx + rx, ry, rz, ry, zeros, zeros, rz, zeros, zeros,
        zeros, rx, zeros, rx, ry + ry, rz, zeros, rz, zeros,
        zeros, zeros, rx, zeros, zeros, ry, rx, ry, rz + rz
    ], dim=-1).view((r.shape[0], 3, 9))
    jacobi = torch.zeros((r.shape[0], 3, 9), device=rvec.device, dtype=rvec.dtype)
    for i in range(3):
        ri = r[:, i:i+1]
        a0 = -s * ri
        a1 = (s - 2*c1*itheta)*ri
        a2 = c1 * itheta
        a3 = (c-s*itheta)*ri
        a4 = s * itheta
        jaco = a0[:, None] * I + a1[:, None] * rrt + a2[:, None] * drrt[:, i].view(-1, 3, 3) + a3[:, None] * skew + a4[:, None] * dSkew[:, i].view(-1, 3, 3)
        jacobi[:, i] = jaco.view(-1, 9)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    jacobi = jacobi.view(*shape[:-1], 3, 9)
    return rot_mat, jacobi

def getJacobianOfRT(rvec, tvec, joints):
    # joints: (bn, nJ, 3)
    dtype, device = rvec.dtype, rvec.device
    bn, nJoints = joints.shape[:2]
    # jacobiToRvec: (bn, 3, 9) // tested by OpenCV and PyTorch 
    Rot, jacobiToRvec = batch_rodrigues_jacobi(rvec)
    I3 = torch.eye(3, dtype=dtype, device=device)[None]
    # jacobiJ_R: (bn, nJ, 3, 3+3+3) => (bn, nJ, 3, 9)
    # // flat by column:
    # // x, 0, 0 | y, 0, 0 | z, 0, 0
    # // 0, x, 0 | 0, y, 0 | 0, z, 0
    # // 0, 0, x | 0, 0, y | 0, 0, z
    jacobi_J_R = torch.zeros((bn, nJoints, 3, 9), dtype=dtype, device=device)
    jacobi_J_R[:, :, 0, :3] = joints
    jacobi_J_R[:, :, 1, 3:6] = joints
    jacobi_J_R[:, :, 2, 6:9] = joints
    # jacobi_J_rvec: (bn, nJ, 3, 3)
    jacobi_J_rvec = torch.matmul(jacobi_J_R, jacobiToRvec[:, None].transpose(-1, -2))
    # if True: # 测试自动梯度
    #     def test_func(rvec):
    #         Rot = batch_rodrigues(rvec[None])[0]
    #         joints_new = joints[0] @ Rot.t()
    #         return joints_new
    #     jac_J_rvec = torch.autograd.functional.jacobian(test_func, rvec[0])
    #     my_j = jacobi_joints_RT[0, ..., :3]
    # jacobi_J_tvec: (bn, nJx3, 3)
    jacobi_J_tvec = I3[None].expand(bn, nJoints, -1, -1)
    jacobi_J_rt = torch.cat([jacobi_J_rvec, jacobi_J_tvec], dim=-1)
    return Rot, jacobiToRvec, jacobi_J_rt

class Model:
    rootIdx = 0
    parents = []


INDEX_HALF = [0,1,2,3,4,5,6,7,15,16,17,18]

class LowPassFilter:
    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None
    
    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s

class OneEuroFilter:
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
    
    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def process(self, x):
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        return self.x_filter.process(x, self.compute_alpha(cutoff))

class BaseBody:
    def __init__(self, cfg_triangulator, cfg_model, cfg) -> None:
        self.triangulator = load_object_from_cmd(cfg_triangulator, [])
        self.body_model = load_object_from_cmd(cfg_model, ['args.use_pose_blending', False, 'args.device', 'cpu'])
        self.cfg = cfg
        self.register_from_lbs(self.body_model)

    def register_from_lbs(self, body_model):
        kintree_shape = np.array(self.cfg.shape.kintree)
        self.nJoints = body_model.J_regressor.shape[0]
        self.k_shapeBlend = body_model.j_shapedirs[self.nJoints:]
        self.j_shapeBlend = body_model.j_shapedirs[:self.nJoints]
        self.jacobian_limb_shapes = self.k_shapeBlend[kintree_shape[:, 1]] - self.k_shapeBlend[kintree_shape[:, 0]]
        self.k_template = body_model.j_v_template[self.nJoints:]
        self.j_template = body_model.j_v_template[:self.nJoints]
        self.k_weights = body_model.j_weights[self.nJoints:]
        self.j_weights = body_model.j_weights[:self.nJoints]
        parents = body_model.parents[1:].cpu().numpy()
        child = np.arange(1, parents.shape[0]+1, dtype=np.int64)
        self.kintree = np.stack([parents, child], axis=1)
        self.parents = np.zeros(parents.shape[0]+1, dtype=int) - 1
        self.parents[self.kintree[:, 1]] = self.kintree[:, 0]
        self.rootIdx = 0
        self.time = time()

def rotation_matrix_from_3x3(A):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    # does the current solution use a reflection?
    have_reflection = np.linalg.det(T) < 0

    # if that's not what was specified, force another reflection
    if have_reflection:
        V[:,-1] *= -1
        s[-1] *= -1
        T = np.dot(V, U.T)
    return T

def svd_rot(src, tgt, reflection=False, debug=True):
    # optimum rotation matrix of Y
    A = np.dot(src.T, tgt)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    # does the current solution use a reflection?
    have_reflection = np.linalg.det(T) < 0

    # if that's not what was specified, force another reflection
    if reflection != have_reflection:
        V[:,-1] *= -1
        s[-1] *= -1
        T = np.dot(V, U.T)
    if debug:
        err = np.linalg.norm(tgt - src @ T.T, axis=1)
        print('[svd] ', err)
    return T

def normalize(vector):
    return vector/np.linalg.norm(vector)

def rad_from_2vec(vec1, vec2):
    return np.arccos((normalize(vec1)*normalize(vec2)).sum())

def smoothing_factor(t_e, cutoff):
    r = 2 * 3.14 * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

FilterResult = namedtuple('FilterResult', ['x', 'dx', 'v', 't'])

class MyFilter:
    def __init__(self, key, filled, min_cutoff=1.0, d_cutoff=1.0,
        beta=0.1) -> None:
        self.key = key
        self.fill_result = filled
        self.min_cutoff = min_cutoff
        self.d_cutoff = d_cutoff
        self.beta = beta
        self.init = False
        self.records = []
        self.result = None
        self.counter = 0
        self.data = []
        self.conf = []
        self.smooth_record = []

    def fill(self, value, conf):
        filled = conf < 0.1
        if filled.sum() >= 1:
            value[filled] = self.fill_result[0][filled]
        if self.key == 'Rh':
            value = rotation_matrix_from_3x3(value.reshape(3, 3))
            value = cv2.Rodrigues(value)[0].reshape(3,)
            if (value < 0).all():
                value = -value
        return value[None]

    def __call__(self, value, conf):
        self.counter += 1
        x, v = value[0], conf[0]
        if self.key == 'Rh':
            x = cv2.Rodrigues(x)[0].reshape(-1)
            v = np.zeros((9,)) + v[0]
        t = np.zeros_like(x)
        t[v>0.1] = self.counter
        self.data.append(x)
        self.conf.append(v)
        if len(self.smooth_record) == 0:
            if self.key == 'Rh':
                start = x
            else:
                # start = self.fill_result[0]
                start = x
            smoothed = FilterResult(start, np.zeros_like(x), np.zeros_like(x), t)
            self.smooth_record.append(smoothed)
        if len(self.data) < 3:
            return self.fill(x, v)
        data = np.array(self.data)
        conf = np.array(self.conf)
        smoothed = self.smooth_record[-1]
        # 预计的速度
        dx_new = x - smoothed.x
        # 滤波器可见，当前可见
        flag_vis = (smoothed.v > 0.1) & (v > 0.1)
        # - 速度异常的，移除掉，认为当前帧不可见
        flag_outlier = (np.abs(smoothed.dx) > 0.05) & (np.abs(dx_new - smoothed.dx)/(1e-5 + smoothed.dx) > 2.)
        if self.key != 'Rh':
            v[flag_vis&flag_outlier] = 0.
        # 滤波器不可见，当前可见，速度打折，认为是新增的帧
        flag_new = (smoothed.v < 0.1)&(v>0.1)
        dx_new[flag_new] /= 3
        # 滤波器可见，当前不可见，速度使用滤波器的速度
        flag_unvis = (v<0.1) & (conf[-2] < 0.1)
        dx_new[flag_unvis] = smoothed.dx[flag_unvis]
        # 滤波器不可见，当前也不可见，速度清0
        dx_new[(v<0.1)&(smoothed.v<0.1)] = 0.
        # 实际估计出来的速度，这里要去掉不可见的地方
        # 混合的权重使用 0.7, 0.3默认全部使用新的一半
        weight_dx = np.zeros_like(dx_new) + 0.7
        dx_smoothed = smoothed.dx*(1-weight_dx) + dx_new*weight_dx
        smoothed_value = smoothed.x + dx_smoothed
        v_new = smoothed.v.copy()
        v_new = v_new * (1-weight_dx) + v*weight_dx
        t_new = smoothed.t.copy()
        t_new[v>0.1] = t[v>0.1]
        smooth_new = FilterResult(smoothed_value, dx_smoothed, v_new, t_new)
        self.smooth_record.append(smooth_new)
        if self.counter == 1000:
            if self.key == 'poses':
                import matplotlib.pyplot as plt
                xrange = np.arange(0, data.shape[0])
                smoothed = np.array([d.x for d in self.smooth_record])
                for nj in range(data.shape[1]):
                    valid = conf[:, nj] > 0.
                    plt.scatter(xrange[valid], data[valid, nj])
                    # yhat = savgol_filter(data[:, nj], data.shape[0], 3)
                    # plt.plot(yhat)
                plt.plot(smoothed)
                plt.show()
                import ipdb;ipdb.set_trace()
        # return self.fill(x, v)
        return self.fill(smooth_new.x, smooth_new.v)

    def __call__0(self, value, conf):
        self.counter += 1
        x, v = value[0], conf[0]
        if self.key == 'Rh':
            x = cv2.Rodrigues(x)[0].reshape(-1)
            v = np.zeros((9,)) + v[0]
        if self.result is None:
            result = FilterResult(x, np.zeros_like(x), v, (v>0)*self.counter)
            self.result = result
            self.records.append(result)
            return self.fill(result.x, result.v)
        # return self.fill(x, v)
        # 维护一个前一帧的，去除outlier
        prev = self.result
        t = prev.t.copy()
        t[v>0.] = self.counter
        dx = x - prev.x # 这里直接使用与之前的结果的差了，避免多帧不可见，然后速度过大
        MAX_DX = 1.
        WINDOW = 31
        not_valid = ((np.abs(dx) > MAX_DX) & (prev.v > 0.1))|\
            (t-prev.t > WINDOW)
        v[not_valid] = 0.
        x_all = np.stack([r.x for r in self.records[-WINDOW:]])
        v_all = np.stack([r.v for r in self.records[-WINDOW:]])
        dx_all = np.stack([r.dx for r in self.records[-WINDOW:]])
        v_sum = v_all.sum(axis=0)
        dx_mean = (dx_all * v_all).sum(axis=0)/(1e-5 + v_all.sum(axis=0))
        # if (x_all.shape[0] > 30) & (self.counter % 40 == 0):
        if True:
            x_mean = (x_all * v_all).sum(axis=0)/(1e-5 + v_all.sum(axis=0))
            x_pred = x_mean
            dx_pred = np.zeros_like(x_pred)
        elif x_all.shape[0] >= 5:
            # 进行smooth
            axt = np.zeros((2, x_all.shape[1]))
            xrange = np.arange(x_all.shape[0]).reshape(-1, 1)
            A0 = np.hstack([xrange, np.ones((x_all.shape[0], 1))])
            for nj in range(x_all.shape[1]):
                conf = v_all[:, nj:nj+1]
                if (conf>0.).sum() < 3:
                    continue
                A = conf * A0
                b = conf * (x_all[:, nj:nj+1])
                est = np.linalg.inv(A.T @ A) @ A.T @ b
                axt[:, nj] = est[:, 0]
            x_all_smoothed = xrange * axt[0:1] + axt[1:]
            x_pred = x_all_smoothed[x_all.shape[0]//2]
            dx_pred = axt[0]
        else:
            x_pred = x_all[x_all.shape[0]//2]
            dx_pred = dx_mean
        if x_all.shape[0] == 1:
            current = FilterResult(x, dx, v, t)
            self.records.append(current)
            self.result = current
        else: 
            # dx_hat = (dx * v + dx_mean * v_mean)/(v+v_mean+1e-5)
            # x_pred = x_mean + dx_hat
            # current = FilterResult(x_pred, dx_hat, v, t)
            current = FilterResult(x, dx, v, t)
            self.records.append(current)
            # 使用平均速度模型
            self.result = FilterResult(x_pred, dx_pred, v_sum, t)
        return self.fill(self.result.x, self.result.v)

    def __call__2(self, value, conf):
        self.counter += 1
        x, v = value[0], conf[0]
        if self.result is None:
            result = FilterResult(x, np.zeros_like(x), v, (v>0)*self.counter)
            self.result = result
            return self.fill(result.x, result.v)
        prev = self.result
        t = prev.t.copy()
        t[v>0.] = self.counter
        # update t
        # dx = (x - prev.x)/(np.maximum(t-prev.t, 1))
        dx = x - prev.x # 这里直接使用与之前的结果的差了，避免多帧不可见，然后速度过大
        dx_ = dx.copy()
        # 判断dx的大小
        large_dx = np.abs(dx) > 0.5
        if large_dx.sum() > 0:
            v[large_dx] = 0.
            t[large_dx] = prev.t[large_dx]
            dx[large_dx] = 0.
        missing_index = ((prev.v > 0.1) & (v < 0.1)) | (t - prev.t > 10)
        if missing_index.sum() > 0:
            print('missing', missing_index)
        new_index = (prev.v < 0.1) & (v > 0.1)
        if new_index.sum() > 0:
            print('new', new_index)
            dx[new_index] = 0.
        weight_dx = v/(1e-5+ 3*prev.v + 1*v)
        weight_x  = v/(1e-5+ 3*prev.v + 1*v)
        # 移除速度过大的点
        dx_hat = exponential_smoothing(weight_dx, dx, prev.dx)
        x_pred = prev.x + dx_hat
        x_hat = exponential_smoothing(weight_x, x, x_pred)
        dx_real = x_hat - prev.x
        # consider the unvisible v
        print_val = {
            't_pre': prev.t,
            'x_inp': x,
            'x_pre': prev.x,
            'x_new': x_hat,
            'dx_inp': dx_,
            'dx_pre': prev.dx,
            'dx_new': dx_hat,
            'dx_real': dx_real,
            'v': v,
            'v_pre': prev.v,
            'w_vel': weight_dx,
            'w_x': weight_x
        }
        for key in print_val.keys():
            print('{:7s}'.format(key), end='  ')
        print('')        
        for i in range(x.shape[0]):
            for key in print_val.keys():
                print('{:7.2f}'.format(print_val[key][i]), end='  ')
            print('')
        v[missing_index] = prev.v[missing_index] / 1.2 # 衰减系数
        result = FilterResult(x_hat, dx_hat, v, t)
        self.result = result
        return self.fill(result.x, result.v)
        if len(self.records) < 10:
            self.records.append([self.counter, value, conf])
            return self.fill(value[0], conf[0])
        if self.x is None:
            time = np.vstack([x[0] for x in self.records])
            value_pre = np.vstack([x[1] for x in self.records])
            conf_pre = np.vstack([x[2] for x in self.records])
            conf_sum = conf_pre.sum(axis=0)
            value_mean = (value_pre * conf_pre).sum(axis=0)/(conf_sum + 1e-5)
            self.x = value_mean
            self.x_conf = conf_sum
            t_prev = np.zeros_like(self.x, dtype=int) - 1
            t_prev[conf_sum>0] = self.counter
            self.t_prev = t_prev
            # 零速度初始化
            self.d_x = np.zeros_like(self.x)
            return self.fill(self.x, self.x_conf)
        # 假设每帧都传进来的吧
        return self.fill(self.x, self.x_conf)
        x_est, v_est, conf_est = self.x.copy(), self.d_x.copy(), self.x_conf.copy()
        value = value[0]
        conf = conf[0]
        d_x = value - self.x
        t_current = np.zeros_like(self.x, dtype=int) - 1
        t_current[conf>0.] = self.counter
        t_est = t_current - self.t_prev
        # 前一帧有观测，当前帧有观测，两帧之差在10帧以内。正常更新
        flag_vv = (t_current > 0) & (self.t_prev > 0) & \
            (t_current - self.t_prev < 10)
        # 前一帧无观测；当前帧有观测的；判断为新增的
        flag_iv = (self.t_prev < 0) & (t_current > 0)
        weight_vel = smoothing_factor(t_est, self.d_cutoff)
        # 将观测的速度权重置0
        weight_vel[flag_vv] = 0.
        vel_hat = exponential_smoothing(weight_vel, d_x, self.d_x)
        cutoff = self.min_cutoff + self.beta * np.abs(vel_hat)
        weight_value = smoothing_factor(t_est, cutoff)
        # 将观测的数值权重置0
        weight_value[flag_vv] = 0.
        weight_value[flag_iv] = 1. # 当前帧可见的，之前的帧不可见的，直接选择当前帧
        vel_hat[flag_iv] = 0.
        x_hat = exponential_smoothing(weight_value, value, self.x)
        flag_vi = (self.t_prev > 0) & (~flag_vv)
        flag_v = flag_vv | flag_vi | flag_iv
        # 前一帧有观测；当前帧无观测的；判断为丢失的
        x_est[flag_v] = x_hat[flag_v]
        v_est[flag_v] = vel_hat[flag_v]
        conf_est[flag_v] = (self.x_conf + conf)[flag_v]/2
        self.t_prev[flag_v] = self.counter
        self.x = x_est
        self.d_x = v_est
        self.x_conf = conf_est
        return self.fill(x_est, conf_est)

class IKBody(BaseBody):
    def __init__(self, key, cfg_triangulator, cfg_model, cfg, debug) -> None:
        super().__init__(cfg_triangulator, cfg_model, cfg)
        self.key = key
        self.frame_index = 0
        self.frame_latest = 0
        self.init = False
        self.records = []
        self.results = []
        self.blank_result = self.make_blank()
        self.fill_result = self.make_fill()
        self.results_newest = self.blank_result
        if True:
            self.lefthand = ManoFitterCPPCache('LEFT')
            self.righthand = ManoFitterCPPCache('RIGHT')
        self.up_vector = 'z'
        self.filter = {}
        for key in ['Rh', 'poses', 'handl', 'handr']:
            self.filter[key] = MyFilter(key, self.fill_result[key])

    def make_blank(self):
        raise NotImplementedError

    def make_fill(self):
        raise NotImplementedError

    def smooth_results(self, params=None):
        results = {'id': 0, 'type': 'smplh_half'}
        for key in ['Rh', 'poses', 'handl', 'handr']:
            value = self.filter[key](params[key], params[key+'_conf'])
            results[key] = value
        for key in ['shapes', 'Th']:
            results[key] = self.blank_result[key]
        return results

    def smooth_results_old(self, params=None):
        if params is not None:
            self.results.append(params)
            if len(self.results) < 10:
                return params
        else:
            if len(self.results) < 10:
                return self.fill_result
            else:
                params = self.fill_result
        results = {'id': 0}
        if False:
            for key in ['Rh', 'poses', 'handl', 'handr']:
                if not self.filter[key].init:
                    import ipdb;ipdb.set_trace()
                else:
                    value = self.filter[key](self.results[-1][key])
        if True:
            for key in ['Rh', 'poses', 'handl', 'handr']:
                find = False
                for WINDOW in [10, 20, 40]:
                    if WINDOW > len(self.results):
                        break
                    records = self.results[-WINDOW:]
                    value = np.vstack([r[key] for r in records])
                    conf = np.vstack([r[key+'_conf'] for r in records])
                    valid = conf[..., 0] > 0
                    if valid.sum() < WINDOW // 3:
                        import ipdb;ipdb.set_trace()
                    else:
                        value, conf = value[valid], conf[valid]
                        mean_value = value.mean(axis=0)
                        std_value = value.std(axis=0)
                        valid2 = (np.abs(value - mean_value) < WINDOW//3 * std_value).any(axis=-1)
                        if valid2.sum() < WINDOW // 4:
                            continue
                        find = True
                        value, conf = value[valid2], conf[valid2]
                        conf_sum = conf.sum(axis=0)
                        mean = (value*conf).sum(axis=0)/(conf_sum + 1e-5)
                        # 计算latest
                        break
                        if key in ['poses', 'handl', 'handr']:
                            conf_sum_p = conf.sum(axis=0)
                            mean_previous = (value*conf).sum(axis=0)/(conf_sum_p + 1e-5)
                            mean[conf_sum<0.01] = mean_previous[conf_sum<0.01]
                            conf_sum[conf_sum<0.01] = conf_sum_p[conf_sum<0.01]
                            # 使用fill的填值
                            mean[conf_sum<0.01] = self.fill_result[key][0][conf_sum<0.01]
                        break
                if find:
                    results[key] = mean[None]
                else:
                    results[key] = self.fill_result[key]
        if False: # 均值滤波
            for key in ['Rh', 'poses', 'handl', 'handr']:
                if key not in params.keys():
                    continue
                if key not in self.cfg.SMOOTH_SIZE.keys():
                    results[key] = params[key]
                records = self.results[-self.cfg.SMOOTH_SIZE[key]:]
                value = np.vstack([r[key] for r in records])
                conf = np.vstack([r[key+'_conf'] for r in records])
                conf_sum = conf.sum(axis=0)
                mean = (value*conf).sum(axis=0)/(conf_sum + 1e-5)
                # 计算latest
                if key in ['poses', 'handl', 'handr']:
                    records = self.results[-5*self.cfg.SMOOTH_SIZE[key]:]
                    value = np.vstack([r[key] for r in records])
                    conf = np.vstack([r[key+'_conf'] for r in records])
                    conf_sum_p = conf.sum(axis=0)
                    mean_previous = (value*conf).sum(axis=0)/(conf_sum_p + 1e-5)
                    mean[conf_sum<0.01] = mean_previous[conf_sum<0.01]
                    conf_sum[conf_sum<0.01] = conf_sum_p[conf_sum<0.01]
                    # 使用fill的填值
                    mean[conf_sum<0.01] = self.fill_result[key][0][conf_sum<0.01]
                results[key] = mean[None]
        results['Th'] = self.blank_result['Th']
        results['shapes'] = self.blank_result['shapes']
        return results

    def get_keypoints3d(self, records, key=None):
        if key is None:
            return np.stack([r[self.key] for r in records])
        else:
            return np.stack([r[key] for r in records])

    def check_keypoints(self, keypoints3d):
        flag = (keypoints3d[..., -1]>self.cfg.MIN_THRES).sum() > 5
        if len(self.records) > 1:
            pre = self.records[-1]
            k_pre = self.get_keypoints3d([pre])
            dist = np.linalg.norm(keypoints3d[..., :3] - k_pre[..., :3], axis=-1)
            conf = np.sqrt(keypoints3d[..., 3] * k_pre[..., 3])
            dist_mean = (dist * conf).sum()/conf.sum()
            flag = flag and dist_mean < 0.1
        return flag

    def __call__(self, data):
        self.frame_index += 1
        k3d = self.triangulator(data)[0]
        keypoints3d = self.get_keypoints3d([k3d])
        flag = self.check_keypoints(keypoints3d)
        if not flag:
            mywarn('Missing keypoints {} [{}->{}]'.format(keypoints3d[..., -1].sum(), self.frame_latest, self.frame_index))
            # 1. 初始化过了，但是超出帧数了，清零
            # 2. 没有初始化过，超出了，清零
            if (self.frame_index - self.frame_latest > 10 and self.init) or not self.init:
                mywarn('Missing keypoints, resetting...')
                self.init = False
                self.records = []
                self.results = []
                return [self.fill_result]
            else:
                return [self.smooth_results()]
        elif not self.init: # 暂时还没有初始化，先等待
            if len(self.records) < 10:
                self.records.append(k3d)
                return [self.fill_result]
        self.records.append(k3d)
        flag, params = self.fitting(keypoints3d, self.results_newest)
        if not flag:
            return [self.fill_result]
        self.frame_latest = self.frame_index
        # smooth results
        results = self.smooth_results(params)
        self.results_newest = results
        k3d['type'] = 'body25'
        return [results, k3d]

class HalfBodyIK(IKBody):
    def get_keypoints3d(self, records):
        THRES_WRIST = 0.2
        keypoints3d = super().get_keypoints3d(records)
        keypoints3d = keypoints3d[:, INDEX_HALF]
        handl = super().get_keypoints3d(records, key='handl3d')
        handr = super().get_keypoints3d(records, key='handr3d')
        dist_ll = np.linalg.norm(keypoints3d[:, 7, :3] - handl[:, 0, :3], axis=-1)
        dist_rr = np.linalg.norm(keypoints3d[:, 4, :3] - handr[:, 0, :3], axis=-1)
        log('Dist left = {}, right = {}'.format(dist_ll, dist_rr))
        handl[dist_ll>THRES_WRIST] = 0.
        handr[dist_rr>THRES_WRIST] = 0.
        keypoints3d = np.hstack([keypoints3d, handl, handr])
        conf = keypoints3d[..., 3:]
        keypoints3d = np.hstack([(keypoints3d[..., :3] * conf).sum(axis=0)/(1e-5 + conf.sum(axis=0)), conf.min(axis=0)])
        keypoints3d = keypoints3d[None]
        # if (keypoints3d.shape[0] == 10):
        return keypoints3d
    
    def _ik_shoulder(self, keypoints3d, params):
        SHOULDER_IDX = [2, 5]
        shoulder = keypoints3d[SHOULDER_IDX[1], :3] - keypoints3d[SHOULDER_IDX[0], :3]
        if self.up_vector == 'x':
            shoulder[..., 0] = 0.
            up_vector = np.array([1., 0., 0.], dtype=np.float32)
        elif self.up_vector == 'z':
            shoulder[..., 2] = 0.
            up_vector = np.array([0., 0., 1.], dtype=np.float32)        
        shoulder = shoulder/np.linalg.norm(shoulder, keepdims=True)
        # 限定一下角度范围
        theta = -np.rad2deg(np.arctan2(shoulder[1], shoulder[2]))
        if (theta < 30 or theta > 150) and False:
            return False, params
        front = np.cross(shoulder, up_vector)
        front = front/np.linalg.norm(front, keepdims=True)
        R = np.stack([shoulder, up_vector, front]).T
        Rh = cv2.Rodrigues(R)[0].reshape(1, 3)
        log('Shoulder:{}'.format(Rh))
        params['R'] = R
        params['Rh'] = Rh
        params['Rh_conf'] = np.zeros((1, 3)) + keypoints3d[SHOULDER_IDX, 3].min()
        return True, params

    def _ik_head(self, keypoints3d, params):
        HEAD_IDX = [0, 8, 9, 10, 11]
        HEAD_ROT_IDX = 0
        est_points = keypoints3d[HEAD_IDX, :3]
        valid = (keypoints3d[HEAD_IDX[0], 3] > self.cfg.MIN_THRES) and (keypoints3d[HEAD_IDX[1:], 3]>self.cfg.MIN_THRES).sum()>=2
        if not valid:
            params['poses_conf'][:, 3*HEAD_ROT_IDX:3*(HEAD_ROT_IDX+1)] = 0.
            return params
        params['poses_conf'][:, 3*HEAD_ROT_IDX:3*(HEAD_ROT_IDX+1)] = keypoints3d[HEAD_IDX, 3].sum()

        gt_points = self.k_template[HEAD_IDX].numpy()
        gt_points = gt_points - gt_points[:1]
        est_points = est_points - est_points[:1]
        # gt_points = gt_points / np.linalg.norm(gt_points, axis=-1, keepdims=True)
        # est_points = est_points / np.linalg.norm(est_points, axis=-1, keepdims=True)

        if True:
            R_global = svd_rot(gt_points, est_points)
            R_local = params['R'].T @ R_global
        elif False:
            est_points_inv = est_points @ params['R'].T.T
            R_local = svd_rot(gt_points, est_points_inv)
        else:
            gt_points = gt_points @ params['R'].T
            R_local = svd_rot(gt_points, est_points)
        euler = rotmat2euler(R_local)
        euler[0] = euler[0] - 25
        # log('euler before filter: {}'.format(euler))
        euler[0] = max(min(euler[0], 30), -30)
        euler[1] = max(min(euler[1], 60), -60)
        euler[2] = max(min(euler[2], 45), -45)
        # log('euler after filter: {}'.format(euler))
        R_local = euler2rotmat(euler)
        R_head = cv2.Rodrigues(R_local)[0].reshape(1, 3)
        params['poses'][:, 3*HEAD_ROT_IDX:3*(HEAD_ROT_IDX+1)] = R_head
        return params
    
    @staticmethod
    def _rad_from_twovec(keypoints3d, start, mid, end, MIN_THRES):
        start = keypoints3d[start]
        mid = keypoints3d[mid]
        end = keypoints3d[end]
        if isinstance(end, list):
            # dst is a list 
            if (end[:, 3] > MIN_THRES).sum() < 2:
                return 0, 0.
            end = np.sum(end * end[:, 3:], axis=0)/(end[:, 3:].sum())
            # use its mean to represent the points
        conf = [start[3], mid[3], end[3]]
        valid = (min(conf) > MIN_THRES).all()
        if not valid:
            return 0, 0.
        conf = sum(conf)
        dir_src = normalize(mid[:3] - start[:3])
        dir_dst = normalize(end[:3] - mid[:3])
        rad = rad_from_2vec(dir_src, dir_dst)
        return conf, rad

    def _ik_elbow(self, keypoints3d, params):
        for name, info in self.cfg.LEAF.items():
            conf, rad = self._rad_from_twovec(keypoints3d, info.start, info.mid, info.end, self.cfg.MIN_THRES)
            if conf <= 0.: continue
            params['poses_conf'][:, 3*info['index']:3*(info['index']+1)] = conf
            rad = np.clip(rad, *info['ranges'])
            rot = rad*np.array(info['axis']).reshape(1, 3)
            params['poses'][:, 3*info['index']:3*(info['index']+1)] = rot
        return params
    
    def _ik_arm(self, keypoints3d, params):
        # forward一遍获得关键点
        # 这里需要确保求解的关节点没有父节点了
        template = self.body_model.keypoints({'poses': params['poses'], 'shapes': params['shapes']}, return_tensor=False)[0]
        for name, info in self.cfg.NODE.items():
            idx = info['children']
            conf = keypoints3d[info['children'], 3]
            if not (conf >self.cfg.MIN_THRES).all():
                continue
            params['poses_conf'][:, 3*info['index']:3*(info['index']+1)] = conf.sum()
            est_points = keypoints3d[idx, :3]
            gt_points = template[idx]
            est_points = est_points - est_points[:1]
            gt_points = gt_points - gt_points[:1]
            R_children = svd_rot(gt_points, est_points)
            R_local = params['R'].T @ R_children
            euler = rotmat2euler(R_local)
            # log('euler {} before filter: {}'.format(name, euler))
            # euler[0] = max(min(euler[0], 90), -90)
            # euler[1] = max(min(euler[1], 90), -90)
            # euler[2] = max(min(euler[2], 90), -90)
            # log('euler {} after filter: {}'.format(name, euler))
            R_local = euler2rotmat(euler)
            params['poses'][:, 3*info['index']:3*(info['index']+1)] = cv2.Rodrigues(R_local)[0].reshape(1, 3)
        return params

    def _ik_palm(self, keypoints3d, params):
        # template = self.body_model.keypoints({'poses': params['poses'], 'shapes': params['shapes']}, return_tensor=False)[0]
        T_joints, _ = self.body_model.transform({'poses': params['poses'], 'shapes': params['shapes']}, return_vertices=False)
        T_joints = T_joints[0].cpu().numpy()
        for name, info in self.cfg.PALM.items():
            # 计算手掌的朝向
            est_points = keypoints3d[:, :3]
            est_conf = keypoints3d[:, 3]
            if est_conf[info.children].min() < self.cfg.MIN_THRES:
                continue
            # 计算朝向
            dir0 = normalize(est_points[info.children[1]] - est_points[info.children[0]])
            dir1 = normalize(est_points[info.children[-1]] - est_points[info.children[0]])
            normal = normalize(np.cross(dir0, dir1))
            dir_parent = normalize(est_points[info.parent[1]] - est_points[info.parent[0]])
            # 计算夹角
            rad = np.arccos((normal * dir_parent).sum()) - np.pi/2
            rad = np.clip(rad, *info['ranges'])
            rot = rad*np.array(info['axis']).reshape(1, 3)
            params['poses'][:, 3*info['index']:3*(info['index']+1)] = rot
            # 考虑手肘的朝向；这个时候还差一个绕手肘的朝向的方向的旋转；这个旋转是在手肘扭曲之前的
            # 先计算出这个朝向；再转化
            R_parents = params['R'] @ T_joints[info.index, :3, :3]
            normal_canonical = R_parents.T @ normal.reshape(3, 1)
            normal_canonical[0, 0] = 0
            normal_canonical = normalize(normal_canonical)
            # 在canonical下的投影
            # normal_T = np.array([0., -1., 0.])
            # trick: 旋转角度的正弦值等于在z轴上的坐标
            rad = np.arcsin(normal_canonical[2, 0])
            rot_x = np.array([-rad, 0., 0.])
            R_x = cv2.Rodrigues(rot_x)[0]
            R_elbow = cv2.Rodrigues(params['poses'][:, 3*info.index_elbow:3*info.index_elbow+3])[0]
            R_elbow = R_elbow @ R_x
            params['poses'][:, 3*info.index_elbow:3*info.index_elbow+3] = cv2.Rodrigues(R_elbow)[0].reshape(1, 3)
        return params

    def _ik_hand(self, template, keypoints3d, poses, conf, is_left):
        # 计算手的每一段的置信度
        poses_full = np.zeros((1, 45))
        conf_full = np.zeros((1, 45))
        y_axis = np.array([0., 1., 0.])
        log('_ik for left: {}'.format(is_left))
        for name, info in self.cfg.HAND.LEAF.items():
            conf, rad = self._rad_from_twovec(keypoints3d, *info.ranges, self.cfg.MIN_THRES)
            if conf <= 0.: 
                log('- skip: {}'.format(name))
                continue
            # trick: 手的朝向是反的
            rad = - rad
            if info.axis == 'auto':
                template_dir = template[info.ranges[2]] - template[info.ranges[1]]
                # y轴方向设成0
                template_dir[1] = 0.
                template_dir = normalize(template_dir)
                # 计算旋转轴，在与z轴的cross方向上
                rot_vec = normalize(np.cross(template_dir, y_axis)).reshape(1, 3)
            elif info.axis == 'root':
                template_dir0 = template[info.ranges[1]] - template[info.ranges[0]]
                template_dir1 = template[info.ranges[2]] - template[info.ranges[1]]
                template_dir0 = normalize(template_dir0)
                template_dir1 = normalize(template_dir1)
                costheta0 = (template_dir0 *template_dir1).sum()
                # 计算当前的夹角
                est_dir0 = keypoints3d[info.ranges[1], :3] - keypoints3d[info.ranges[0], :3]
                est_dir1 = keypoints3d[info.ranges[2], :3] - keypoints3d[info.ranges[1], :3]
                est_dir0 = normalize(est_dir0)
                est_dir1 = normalize(est_dir1)
                costheta1 = (est_dir0 * est_dir1).sum()
                # trick: 手的旋转角度都是相反的
                rad = - np.arccos(np.clip(costheta1/costheta0, 0., 1.))
                rot_vec = normalize(np.cross(template_dir1, y_axis)).reshape(1, 3)
            log('- get: {}: {:.1f}, {}'.format(name, np.rad2deg(rad), rot_vec))            
            poses_full[:, 3*info.index:3*info.index+3] = rot_vec * rad
            conf_full[:, 3*info.index:3*info.index+3] = conf
        # 求解
        usePCA = False
        if usePCA:
            ncomp = 24
            lamb = 0.05
            if is_left:
                A_full = self.body_model.components_full_l[:ncomp].T
                mean_full = self.body_model.mean_full_l
            else:
                A_full = self.body_model.components_full_r[:ncomp].T
                mean_full = self.body_model.mean_full_r
            valid = conf_full[0] > 0.
            A = A_full[valid, :]
            res = (poses_full[:, valid] - mean_full[:, valid]).T
            x = np.linalg.inv(A.T @ A + lamb * np.eye(ncomp)) @ A.T @ res
            poses_full = (A_full @ x).reshape(1, -1) + mean_full
            conf_full = np.zeros_like(poses_full) + valid.sum()
        return poses_full, conf_full

    def make_blank(self):
        params = self.body_model.init_params(1, ret_tensor=False)
        params['id'] = 0
        params['type'] = 'smplh_half'
        params['Th'][0, 0] = 1.
        params['Th'][0, 1] = -1
        params['Th'][0, 2] = 1.
        return params
    
    def make_fill(self):
        params = self.body_model.init_params(1, ret_tensor=False)
        params['id'] = 0
        params['type'] = 'smplh_half'
        params['Rh'][0, 2] = -np.pi/2
        params['handl'] = self.body_model.mean_full_l
        params['handr'] = self.body_model.mean_full_r
        params['poses'][0, 3*4+2] = -np.pi/4
        params['poses'][0, 3*5+2] =  np.pi/4
        return params    

    def fitting(self, keypoints3d, results_pre):
        # keypoints3d: (nFrames, nJoints, 4)
        # 根据肩膀计算身体朝向
        if len(keypoints3d.shape) == 3:
            keypoints3d = keypoints3d[0]
        params = self.body_model.init_params(1, ret_tensor=False)
        params['poses_conf'] = np.zeros_like(params['poses'])
        params['handl_conf'] = np.zeros_like(params['handl'])
        params['handr_conf'] = np.zeros_like(params['handr'])
        params['Rh_conf'] = 0.
        params['id'] = 0
        flag, params = self._ik_shoulder(keypoints3d, params)
        if (params['Rh_conf'] <= 0.01).all():
            return False, params
        params = self._ik_head(keypoints3d, params)
        params = self._ik_elbow(keypoints3d, params)
        params = self._ik_arm(keypoints3d, params)
        params = self._ik_palm(keypoints3d, params)
        if False:
            params['handl'], params['handl_conf'] = self._ik_hand(self.k_template[12:12+21].numpy(), keypoints3d[12:12+21], params['handl'], params['handl_conf'], is_left=True)
            params['handr'], params['handr_conf'] = self._ik_hand(self.k_template[12+21:12+21+21].numpy(), keypoints3d[12+21:12+21+21], params['handr'], params['handr_conf'], is_left=False)
        else:
            params_l = self.lefthand(keypoints3d[12:12+21])[0]
            params_r = self.righthand(keypoints3d[12+21:12+21+21])[0]
            # log('[{:06d}] {}'.format(self.frame_index, params_l['poses'][0]))
            # log('[{:06d}] {}'.format(self.frame_index, params_r['poses'][0]))
            ncomp = params_l['poses'].shape[1]
            A_full = self.body_model.components_full_l[:ncomp].T
            mean_full = self.body_model.mean_full_l
            poses_full = (A_full @ params_l['poses'].T).T + mean_full
            params['handl'] = poses_full            
            A_full = self.body_model.components_full_r[:ncomp].T
            mean_full = self.body_model.mean_full_r
            poses_full = (A_full @ params_r['poses'].T).T + mean_full
            params['handr'] = poses_full
            params['handl_conf'] = np.ones((1, 45))
            params['handr_conf'] = np.ones((1, 45))
        return True, params

class BaseFitter(BaseBody):
    def __init__(self, cfg_triangulator, cfg_model, 
        INIT_SIZE, WINDOW_SIZE, FITTING_SIZE, SMOOTH_SIZE,
        init_dict, fix_dict,
        cfg) -> None:
        super().__init__(cfg_triangulator, cfg_model, cfg)
        self.records = []
        self.results = []
        self.INIT_SIZE = INIT_SIZE
        self.WINDOW_SIZE = WINDOW_SIZE
        self.FITTING_SIZE = FITTING_SIZE
        self.SMOOTH_SIZE = SMOOTH_SIZE
        self.time = 0
        self.frame_latest = 0
        self.frame_index = 0
        self.init = False
        self.init_dict = init_dict
        self.fix_dict = fix_dict
        self.identity_cache = {}

    def get_keypoints3d(self, records):
        raise NotImplementedError
    
    def get_init_params(self, nFrames):
        params = self.body_model.init_params(nFrames, ret_tensor=True)
        for key, val in self.init_dict.items():
            if key == 'Rh':
                import cv2
                R = cv2.Rodrigues(params['Rh'][0].cpu().numpy())[0]
                for vec in self.init_dict['Rh']:
                    Rrel = cv2.Rodrigues(np.deg2rad(np.array(vec)))[0]
                    R = Rrel @ R
                Rh = cv2.Rodrigues(R)[0]
                params['Rh'] = torch.Tensor(Rh).reshape(-1, 3).repeat(nFrames, 1)
            else:
                params[key] = torch.Tensor(val).repeat(nFrames, 1)
        params['id'] = 0
        return params

    def add_any_reg(self, val, val0, JTJ, JTr, w):
        # loss: (val - val0)
        nVals = val.shape[0]
        if nVals not in self.identity_cache.keys():
            self.identity_cache[nVals] = torch.eye(nVals, device=val.device, dtype=val.dtype)
        identity = self.identity_cache[nVals]
        JTJ += w * identity
        JTr += -w*(val - val0).view(-1, 1)
        return 0
    
    def log(self, name, step, delta, res, keys_range=None):
        toc = (time() - self.time)*1000
        norm_d = torch.norm(delta).item()
        norm_f = torch.norm(res).item()
        text = '[{}:{:6.2f}]: step = {:3d}, ||delta|| = {:.4f}, ||res|| = {:.4f}'.format(name, toc, step, norm_d, norm_f)
        print(text)
        self.time = time()
        return norm_d, norm_f

    def fitShape(self, keypoints3d, params, weight, option):
        kintree = np.array(self.cfg.shape.kintree)
        nShapes = params['shapes'].shape[-1]
        # shapes: (1, 10)
        shapes = params['shapes'].T
        shapes0 = shapes.clone()
        device, dtype = shapes.device, shapes.dtype
        lengths3d_est = torch.norm(keypoints3d[:, kintree[:, 1], :3] - keypoints3d[:, kintree[:, 0], :3], dim=-1)
        conf = torch.sqrt(keypoints3d[:, kintree[:, 1], 3:] * keypoints3d[:, kintree[:, 0], 3:])
        conf = conf.repeat(1, 1, 3).reshape(-1, 1)
        nFrames = keypoints3d.shape[0]
        # jacobian: (nFrames, nLimbs, 3, nShapes)
        jacob_limb_shapes = self.jacobian_limb_shapes[None].repeat(nFrames, 1, 1, 1)
        jacob_limb_shapes = jacob_limb_shapes.reshape(-1, nShapes)
        # 注意：这里乘到雅克比的应该是 sqrt(conf)，这里把两个合并了
        JTJ_limb_shapes = jacob_limb_shapes.t() @ (jacob_limb_shapes * conf)
        lossnorm = 0
        self.time = time()
        for iter_ in range(option.max_iter):
            # perform shape blending
            shape_offset = self.k_shapeBlend @ shapes
            keyShaped = self.k_template + shape_offset[..., 0]
            JTJ = JTJ_limb_shapes
            JTr = torch.zeros((nShapes, 1), device=device, dtype=dtype)
            # 并行添加所有的骨架
            dir = keyShaped[kintree[:, 1]] - keyShaped[kintree[:, 0]]
            dir_normalized = dir / torch.norm(dir, dim=-1, keepdim=True)
            # res: (nFrames, nLimbs, 3)
            res = lengths3d_est[..., None] * dir_normalized[None] - dir[None]
            res = conf * res.reshape(-1, 1)
            JTr += jacob_limb_shapes.t() @ res
            self.add_any_reg(shapes, shapes0, JTJ, JTr, w=weight.init_shape)
            delta = torch.linalg.solve(JTJ, JTr)
            shapes += delta
            norm_d, norm_f = self.log('shape', iter_, delta, res)
            if torch.norm(delta) < option.gtol:
                break
            if iter_ > 0 and abs(norm_f - lossnorm)/norm_f < option.ftol:
                break
            lossnorm = norm_f
        shapes = shapes.t()
        params['shapes'] = shapes
        return params, keyShaped

    def fitRT(self, keypoints3d, params, weight, option, kpts_index=None):
        keys_optimized = ['Rh', 'Th']
        if kpts_index is not None:
            keypoints3d = keypoints3d[:, kpts_index]
        init_dict = {
            'Rh': params['Rh'].clone(),
            'Th': params['Th'].clone(),
        }
        init_dict['Rot'] = batch_rodrigues(init_dict['Rh'])
        params_dict = {
            'Rh': params['Rh'],
            'Th': params['Th'],
        }
        keys_range = {}
        for ikey, key in enumerate(keys_optimized):
            if ikey == 0:
                keys_range[key] = [0, init_dict[key].shape[-1]]
            else:
                start = keys_range[keys_optimized[ikey-1]][1]
                keys_range[key] = [start, start+init_dict[key].shape[-1]]
        NUM_FRAME = keys_range[keys_optimized[-1]][1]
        kpts = self.body_model.keypoints({'poses': params['poses'], 'shapes': params['shapes']})
        bn = keypoints3d.shape[0]
        conf = keypoints3d[..., -1:].repeat(1, 1, 3).reshape(bn, -1)
        dtype, device = kpts.dtype, kpts.device
        w_joints = 1./keypoints3d.shape[-2] * weight.joints
        self.time = time()
        for iter_ in range(option.max_iter):
            Rh, Th = params_dict['Rh'], params_dict['Th']
            rot, jacobi_R_rvec, jacobi_joints_RT = getJacobianOfRT(Rh, Th, kpts)
            kpts_rt = torch.matmul(kpts, rot.transpose(-1, -2)) + Th[:, None]
            # // loss: J_obs - (R @ jest + T) => -dR/drvec - dT/dtvec - Rx(djest/dtheta)
            jacobi_keypoints = -jacobi_joints_RT
            if kpts_index is not None:
                jacobi_keypoints = jacobi_keypoints[:, kpts_index]
                kpts_rt = kpts_rt[:, kpts_index]
            jacobi_keypoints_flat = jacobi_keypoints.view(bn, -1, jacobi_keypoints.shape[-1])
            JTJ_keypoints = jacobi_keypoints_flat.transpose(-1, -2) @ (jacobi_keypoints_flat * conf[..., None])
            res = conf[..., None] * ((keypoints3d[..., :3] - kpts_rt).view(bn, -1, 1))
            JTr_keypoints = jacobi_keypoints_flat.transpose(-1, -2) @ res
            # 
            JTJ = torch.eye(bn*NUM_FRAME, device=device, dtype=dtype) * 1e-4
            JTr = torch.zeros((bn*NUM_FRAME, 1), device=device, dtype=dtype)
            # accumulate loss
            for nf in range(bn):
                JTJ[nf*NUM_FRAME:(nf+1)*NUM_FRAME, nf*NUM_FRAME:(nf+1)*NUM_FRAME] += w_joints * JTJ_keypoints[nf]
            # add regularization for each parameter
            for nf in range(bn):
                for key in keys_optimized:
                    start, end = nf*NUM_FRAME + keys_range[key][0], nf*NUM_FRAME + keys_range[key][1]
                    if key == 'Rh':
                        # 增加初始化的loss
                        res_init = rot[nf] - init_dict['Rot'][nf]
                        JTJ[start:end, start:end] += weight['init_'+key] * jacobi_R_rvec[nf] @ jacobi_R_rvec[nf].T
                        JTr[start:end] += -weight.init_Rh * jacobi_R_rvec[nf] @ res_init.reshape(-1, 1)
                    else:
                        res_init = Th[nf] - init_dict['Th'][nf]
                        JTJ[start:end, start:end] += weight['init_'+key] * torch.eye(3)
                        JTr[start:end] += -weight['init_'+key] * res_init.reshape(-1, 1)
            JTr += - w_joints * JTr_keypoints.reshape(-1, 1)
            # solve
            delta = torch.linalg.solve(JTJ, JTr)
            norm_d, norm_f = self.log('pose', iter_, delta, res)
            if torch.norm(delta) < option.gtol:
                break
            if iter_ > 0 and abs(norm_f - lossnorm)/norm_f < option.ftol:
                break
            delta = delta.view(bn, NUM_FRAME)
            lossnorm = norm_f
            for key, _range in keys_range.items():
                if key not in params_dict.keys():continue
                params_dict[key] += delta[:, _range[0]:_range[1]]
                norm_key = torch.norm(delta[:, _range[0]:_range[1]])
        params.update(params_dict)
        return params

    @staticmethod    
    def localTransform(J_shaped, poses, rootIdx, kintree):
        bn = poses.shape[0]
        nThetas = poses.shape[1]//3
        localTrans = torch.eye(4, device=poses.device)[None, None].repeat(bn, nThetas, 1, 1)
        poses_flat = poses.reshape(-1, 3)
        rot_flat = batch_rodrigues(poses_flat)
        rot = rot_flat.view(bn, nThetas, 3, 3)
        localTrans[:, :, :3, :3] = rot
        # set the root
        localTrans[:, rootIdx, :3, 3] = J_shaped[rootIdx].view(1, 3)
        # relTrans: (nKintree, 3)
        relTrans = J_shaped[kintree[:, 1]] - J_shaped[kintree[:, 0]]
        localTrans[:, kintree[:, 1], :3, 3] = relTrans[None]
        return localTrans
    
    @staticmethod
    def globalTransform(localTrans, rootIdx, kintree):
        # localTrans: (bn, nJoints, 4, 4)
        globalTrans = localTrans.clone()
        # set the root
        for (parent, child) in kintree:
            globalTrans[:, child] = globalTrans[:, parent] @ localTrans[:, child]
        return globalTrans

    def jacobi_GlobalTrans_theta(self, poses, j_shaped, rootIdx, kintree,
        device, dtype):
        parents = self.parents
        start = time()
        tic = lambda x: print('-> [{:20s}]: {:.3f}ms'.format(x, 1000*(time()-start)))
        localTrans = self.localTransform(j_shaped, poses, rootIdx, kintree)
        # tic('local trans')
        globalTrans = self.globalTransform(localTrans, rootIdx, kintree)
        # tic('global trans')
        # 计算localTransformation
        poses_flat = poses.view(poses.shape[0], -1, 3)
        # jacobi_R_rvec: (bn, nJ, 3, 9)
        Rot, jacobi_R_rvec = batch_rodrigues_jacobi(poses_flat)
            
        bn, nJoints = localTrans.shape[:2]
        dGlobalTrans_template = torch.zeros((bn, nJoints, 4, 4), device=device, dtype=dtype)
        # compute global transformation
        # results: global transformation to each theta: (bn, nJ, 4, 4, nTheta)
        jacobi_GlobalTrans_theta = torch.zeros((bn, nJoints, 4, 4, nJoints*3), device=device, dtype=dtype)
        # tic('rodrigues')
        for djIdx in range(1, nJoints):
            if djIdx in self.cfg.IGNORE_JOINTS: continue
            # // 第djIdx个轴角的第dAxis个维度
            for dAxis in range(3):
                if dAxis in self.cfg.IGNORE_AXIS.get(str(djIdx), []): continue
                # if(model->frozenJoint[3*djIdx+dAxis])continue;
                # // 从上至下堆叠起来
                dGlobalTrans = dGlobalTrans_template.clone()
                # // 将local的映射过来
                dGlobalTrans[:, djIdx, :3, :3] = jacobi_R_rvec[:, djIdx, dAxis].view(bn, 3, 3)
                if djIdx != rootIdx:
                    # // d(R0 @ R1)/dt1 = R0 @ dR1/dt1, 这里的R0是上一级所有的累积，因此使用全局的
                    parent = parents[djIdx]
                    dGlobalTrans[:, djIdx] = globalTrans[:, parent] @ dGlobalTrans[:, djIdx]
                valid = np.zeros(nJoints, dtype=np.bool)
                valid[djIdx] = True
                # tic('current {}'.format(djIdx))
                # // 遍历骨架树: 将对当前theta的导数传递下去
                for (src, dst) in kintree:
                    # // 当前处理的关节为子节点的不用考虑
                    if dst == djIdx: continue
                    # if dst in self.cfg.IGNORE_JOINTS:continue
                    valid[dst] = valid[src]
                    if valid[src]:
                        # // 如果父节点是有效的: d(R0 @ R1)/dt0 = dR0/dt0 @ R1，这里的R1是当前的局部的，因此使用local的
                        dGlobalTrans[:, dst] = dGlobalTrans[:, src] @ localTrans[:, dst]
                # tic('forward {}'.format(djIdx))
                jacobi_GlobalTrans_theta[..., 3*djIdx+dAxis] = dGlobalTrans
        # tic('jacobia')
        return globalTrans, jacobi_GlobalTrans_theta

    def fitPose(self, keypoints3d, params, weight, option, kpts_index=None):
        # preprocess input data
        if kpts_index is not None:
            keypoints3d = keypoints3d[:, kpts_index]
        bn = keypoints3d.shape[0]
        conf = keypoints3d[..., -1:].repeat(1, 1, 3).reshape(bn, -1)
        if (conf > 0.3).sum() < 4: 
            print('skip')
            return params
        w_joints = 1./keypoints3d.shape[-2] * weight.joints
        # pre-calculate the shape
        Rh, Th, poses = params['Rh'], params['Th'], params['poses']
        init_dict = {
            'Rh': Rh.clone(),
            'Th': Th.clone(),
            'poses': poses.clone()
        }
        init_dict['Rot'] = batch_rodrigues(init_dict['Rh'])
        zero_dict = {key:torch.zeros_like(val) for key, val in init_dict.items()}
        keys_optimized = ['Rh', 'Th', 'poses']
        keys_range = {}
        for ikey, key in enumerate(keys_optimized):
            if ikey == 0:
                keys_range[key] = [0, init_dict[key].shape[-1]]
            else:
                start = keys_range[keys_optimized[ikey-1]][1]
                keys_range[key] = [start, start+init_dict[key].shape[-1]]
        NUM_FRAME = keys_range[keys_optimized[-1]][1]
        # calculate J_shaped
        shapes_t = params['shapes'].t()
        shape_offset = self.j_shapeBlend @ shapes_t
        # jshaped: (nJoints, 3)
        j_shaped = self.j_template + shape_offset[..., 0]
        shape_offset = self.k_shapeBlend @ shapes_t
        # kshaped: (nJoints, 3)
        k_shaped = self.k_template + shape_offset[..., 0]        
        # optimize parameters
        nJoints = j_shaped.shape[0]
        dtype, device = poses.dtype, poses.device
        lossnorm = 0
        self.time = time()
        for iter_ in range(option.max_iter):
            # forward the model
            # 0. poses => full poses
            def tic(name):
                print('[{:20}] {:.3f}ms'.format(name, 1000*(time()-self.time)))
            if 'handl' in params.keys():
                poses_full = self.body_model.extend_poses(poses, params['handl'], params['handr'])
                jacobi_posesful_poses = self.body_model.jacobian_posesfull_poses_
            else:
                poses_full = self.body_model.extend_poses(poses)
            jacobi_posesful_poses = self.body_model.jacobian_posesfull_poses(poses, poses_full)
            # tic('jacobian poses full')
            # 1. poses => local transformation  => global transformation(bn, nJ, 4, 4)
            globalTrans, jacobi_GlobalTrans_theta = self.jacobi_GlobalTrans_theta(poses_full, j_shaped, self.rootIdx, self.kintree, device, dtype)
            # tic('global transform')
            # 2. global transformation => relative transformation => final transformation
            relTrans = globalTrans.clone()
            relTrans[..., :3, 3:] -= torch.matmul(globalTrans[..., :3, :3], j_shaped[None, :, :, None])

            relTrans_weight = torch.einsum('kj,fjab->fkab', self.k_weights, relTrans)
            jacobi_relTrans_theta = jacobi_GlobalTrans_theta.clone()
            # // consider topRight: T - RT0: add -dRT0/dt
            # rot: (bn, nJoints, 3, 3, nThetas) @ (bn, nJoints, 1, 3, 1) => (bn, nJoints, 3, nThetas)
            rot = jacobi_GlobalTrans_theta[..., :3, :3, :]
            j0 = j_shaped.reshape(1, nJoints, 1, 3, 1).expand(bn, -1, -1, -1, -1)
            jacobi_relTrans_theta[..., :3, 3, :] -= torch.sum(rot*j0, dim=-2)
            jacobi_blendtrans_theta = torch.einsum('kj,fjabt->fkabt', self.k_weights, jacobi_relTrans_theta)
            kpts = torch.einsum('fkab,kb->fka', relTrans_weight[..., :3, :3], k_shaped) + relTrans_weight[..., :3, 3]
            # d(RJ0 + J1)/dtheta = d(R)/dtheta @ J0 + dJ1/dtheta
            rot = jacobi_blendtrans_theta[..., :3, :3, :]
            k0 = k_shaped.reshape(1, k_shaped.shape[0], 1, 3, 1).expand(bn, -1, -1, -1, -1)
            # jacobi_keypoints_theta: (bn, nKeypoints, 3, nThetas)
            jacobi_keypoints_theta = torch.sum(rot*k0, dim=-2) + jacobi_blendtrans_theta[..., :3, 3, :]
            # tic('keypoints')
            # // compute the jacobian of R T
            # // loss: J_obs - (R @ jest + T)
            rot, jacobi_R_rvec, jacobi_joints_RT = getJacobianOfRT(Rh, Th, kpts)
            kpts_rt = torch.matmul(kpts, rot.transpose(-1, -2)) + Th[:, None]
            rot_nk = rot[:, None].expand(-1, k_shaped.shape[0], -1, -1)
            jacobi_keypoints_theta = torch.matmul(rot_nk, jacobi_keypoints_theta)
            # compute jacobian of poses
            shape_jacobi = jacobi_keypoints_theta.shape[:-1]
            NUM_THETAS = jacobi_posesful_poses.shape[0]
            jacobi_keypoints_poses = (jacobi_keypoints_theta[..., :NUM_THETAS].view(-1, NUM_THETAS) @ jacobi_posesful_poses).reshape(*shape_jacobi, -1)
            # // loss: J_obs - (R @ jest + T) => -dR/drvec - dT/dtvec - Rx(djest/dtheta)
            jacobi_keypoints = torch.cat([-jacobi_joints_RT, -jacobi_keypoints_poses], dim=-1)
            if kpts_index is not None:
                jacobi_keypoints = jacobi_keypoints[:, kpts_index]
                kpts_rt = kpts_rt[:, kpts_index]
            jacobi_keypoints_flat = jacobi_keypoints.view(bn, -1, jacobi_keypoints.shape[-1])
            # tic('jacobian keypoints')
            JTJ_keypoints = jacobi_keypoints_flat.transpose(-1, -2) @ (jacobi_keypoints_flat * conf[..., None])
            res = conf[..., None] * ((keypoints3d[..., :3] - kpts_rt).view(bn, -1, 1))
            JTr_keypoints = jacobi_keypoints_flat.transpose(-1, -2) @ res
            cache_dict = {
                'Th': Th,
                'Rh': Rh,
                'poses': poses,
            }
            # 计算loss
            # JTJ = torch.zeros((bn*NUM_FRAME, bn*NUM_FRAME), device=device, dtype=dtype)
            JTJ = torch.eye(bn*NUM_FRAME, device=device, dtype=dtype) * 1e-4
            JTr = torch.zeros((bn*NUM_FRAME, 1), device=device, dtype=dtype)
            # add regularization for each parameter
            for nf in range(bn):
                for key in keys_optimized:
                    start, end = nf*NUM_FRAME + keys_range[key][0], nf*NUM_FRAME + keys_range[key][1]
                    JTJ[start:end, start:end] += weight['reg_{}'.format(key)] * torch.eye(end-start)
                    JTr[start:end] += -weight['reg_{}'.format(key)] * cache_dict[key][nf].view(-1, 1)
                    # add init for Rh
                    if key == 'Rh':
                        # 增加初始化的loss
                        res_init = rot[nf] - init_dict['Rot'][nf]
                        JTJ[start:end, start:end] += weight['init_'+key] * jacobi_R_rvec[nf] @ jacobi_R_rvec[nf].T
                        JTr[start:end] += -weight.init_Rh * jacobi_R_rvec[nf] @ res_init.reshape(-1, 1)
                    else:
                        res_init = cache_dict[key][nf] - init_dict[key][nf]
                        JTJ[start:end, start:end] += weight['init_'+key] * torch.eye(end-start)
                        JTr[start:end] += -weight['init_'+key] * res_init.reshape(-1, 1)
            # add keypoints loss
            for nf in range(bn):
                JTJ[nf*NUM_FRAME:(nf+1)*NUM_FRAME, nf*NUM_FRAME:(nf+1)*NUM_FRAME] += w_joints * JTJ_keypoints[nf]
            JTr += - w_joints * JTr_keypoints.reshape(-1, 1)
            # tic('add loss')
            delta = torch.linalg.solve(JTJ, JTr)
            # tic('solve')
            norm_d, norm_f = self.log('pose', iter_, delta, res, keys_range)
            if torch.norm(delta) < option.gtol:
                break
            if iter_ > 0 and abs(norm_f - lossnorm)/norm_f < option.ftol:
                break
            delta = delta.view(bn, NUM_FRAME)
            lossnorm = norm_f
            for key, _range in keys_range.items():
                if key not in cache_dict.keys():continue
                cache_dict[key] += delta[:, _range[0]:_range[1]]
        res = {
            'id': params['id'],
            'poses': poses,
            'shapes': params['shapes'],
            'Rh': Rh,
            'Th': Th,
        }
        for key, val in params.items():
            if key not in res.keys():
                res[key] = val
        return res

    def try_to_init(self, records):
        if self.init:
            return copy.deepcopy(self.params_newest)
        mywarn('>> Initialize')
        keypoints3d = self.get_keypoints3d(records)
        params, keypoints_template = self.fitShape(keypoints3d, self.get_init_params(keypoints3d.shape[0]), self.cfg.shape.weight, self.cfg.shape.option)
        Rot = batch_rodrigues(params['Rh'])
        keypoints_template = torch.matmul(keypoints_template, Rot[0].t())
        if self.cfg.initRT.mean_T:
            conf = keypoints3d[..., 3:]
            T = ((keypoints3d[..., :3] - keypoints_template[None])*conf).sum(dim=-2)/(conf.sum(dim=-2))
            params['Th'] = T
        params = self.fitRT(keypoints3d, params, self.cfg.initRT.weight, self.cfg.initRT.option,
            kpts_index=self.cfg.TORSO_INDEX)
        params = self.fitPose(keypoints3d, params, self.cfg.init_pose.weight, self.cfg.init_pose.option,
            kpts_index=self.cfg.TORSO_INDEX)
        params = self.fitPose(keypoints3d, params, self.cfg.init_pose.weight, self.cfg.init_pose.option,
            kpts_index=self.cfg.BODY_INDEX)
        mywarn('>> Initialize Rh = {}, Th = {}'.format(params['Rh'][0], params['Th'][0]))
        params = Params(params)[-1]
        self.init = True
        return params

    def fitting(self, params_init, records):
        keypoints3d = self.get_keypoints3d(records[-self.WINDOW_SIZE:])
        params = params_init
        params = self.fitRT(keypoints3d[-self.FITTING_SIZE:], params, self.cfg.RT.weight, self.cfg.RT.option)
        params = self.fitPose(keypoints3d[-self.FITTING_SIZE:], params, self.cfg.pose.weight, self.cfg.pose.option)
        return params

    def filter_result(self, result):
        poses = result['poses'].reshape(-1, 3)
        # TODO: 这里的xyz是scipy中的XYZ
        euler = axis_angle_to_euler(poses, order='xyz')
        # euler[euler>np.pi] = 0.
        poses = euler_to_axis_angle(euler, order='xyz')
        result['euler'] = euler
        return result

    def smooth_results(self):
        results_ = {}
        for key in self.results[0].keys():
            if key == 'id': continue
            if key not in self.SMOOTH_SIZE.keys():continue
            results_[key] = np.vstack([r[key] for r in self.results[-self.SMOOTH_SIZE[key]:]])
            results_[key] = np.mean(results_[key], axis=0, keepdims=True)
        results_['id'] = 0
        for key, val in self.fix_dict.items():
            results_[key] = np.array(val)
        # results_['Th'][:] = 0.
        return [results_]

    def check_keypoints(self, keypoints3d):
        flag = (keypoints3d[..., -1]>0.3).sum() > 5
        if len(self.records) > 1:
            pre = self.records[-1]
            k_pre = self.get_keypoints3d([pre])
            dist = torch.norm(keypoints3d[..., :3] - k_pre[..., :3], dim=-1)
            conf = torch.sqrt(keypoints3d[..., 3] * k_pre[..., 3])
            dist_mean = (dist * conf).sum()/conf.sum()
            flag = flag and dist_mean < 0.1
        return flag

    def __call__(self, data):
        self.frame_index += 1
        k3d = self.triangulator(data)[0]
        keypoints3d = self.get_keypoints3d([k3d])
        flag = self.check_keypoints(keypoints3d)
        if not flag:
            mywarn('Missing keypoints {} [{}->{}]'.format(keypoints3d[..., -1].sum(), self.frame_latest, self.frame_index))
            if self.frame_index - self.frame_latest > 10 and self.init:
                mywarn('Missing keypoints, resetting...')
                self.init = False
                self.records = []
                self.results = []
            return -1
        self.records.append(k3d)
        if len(self.records) < self.INIT_SIZE:
            return -1
        with Timer('fitting', True):
            params = self.try_to_init(self.records)
            params = self.fitting(params, self.records)
        params = self.filter_result(params)
        self.results.append(params)
        self.params_newest = params
        self.frame_latest = self.frame_index
        return self.smooth_results()

class FitterCPPCache:
    def __init__(self) -> None:
        self.init = False
        self.records = []
        self.frame_index = 0
        self.frame_latest = 0

    def try_to_init(self, keypoints3d):
        if self.init:
            return copy.deepcopy(self.params_newest)
        mywarn('>> Initialize')
        params = self.handmodel.init_params(1)
        if not (keypoints3d[..., -1] > 0.).all(): # 只有一半的点可见
            return params
        params = self.handmodel.fit3DShape(keypoints3d, params)
        params = self.handmodel.init3DRT(keypoints3d[-1:], params)
        params = self.handmodel.fit3DPose(keypoints3d[-1:], params)
        mywarn('>> Initialize Rh = {}, Th = {}'.format(params['Rh'][0], params['Th'][0]))
        self.init = True
        return params

    def check_keypoints(self, keypoints3d):
        flag = (keypoints3d[..., -1]>0.3).sum() > 5
        if len(self.records) > 1:
            k_pre = self.records[-1][None]
            dist = np.linalg.norm(keypoints3d[..., :3] - k_pre[..., :3], axis=-1)
            conf = np.sqrt(keypoints3d[..., 3] * k_pre[..., 3])
            dist_mean = (dist * conf).sum()/(1e-5+conf.sum())
            print(dist_mean)
            flag = flag and dist_mean < 0.1
        return flag

    def smooth_results(self, params=None):
        if params is None:
            params = self.params_newest.copy()
        params['poses'] = params['poses'][:, 3:]
        params['id'] = 0
        return [params]

    def __call__(self, keypoints3d):
        self.frame_index += 1
        flag = self.check_keypoints(keypoints3d)
        if not flag:
            mywarn('Missing keypoints {} [{}->{}]'.format(keypoints3d[..., -1].sum(), self.frame_latest, self.frame_index))
            if self.frame_index - self.frame_latest > 10:
                mywarn('Missing keypoints, resetting...')
                self.init = False
                self.records = []
            return self.smooth_results(self.handmodel.init_params(1))
        self.records.append(keypoints3d)
        with Timer('fitting'):
            params = self.try_to_init(keypoints3d[None])
            params = self.handmodel.fit3DPose(keypoints3d[None], params)
        self.params_newest = params
        self.frame_latest = self.frame_index
        return self.smooth_results()
class ManoFitterCPPCache(FitterCPPCache):
    def __init__(self, name='LEFT') -> None:
        super().__init__()
        self.handmodel = load_object_from_cmd('config/model/mano_full_cpp.yml', [
            'args.model_path', 'data/bodymodels/manov1.2/MANO_{}.pkl'.format(name),
            'args.regressor_path', 'data/smplx/J_regressor_mano_{}.txt'.format(name),
        ])

class SMPLFitterCPPCache(FitterCPPCache):
    def __init__(self, name='half') -> None:
        super().__init__()
        self.handmodel = load_object_from_cmd('config/model/mano_full_cpp.yml', [
            'args.model_path', 'data/bodymodels/manov1.2/MANO_{}.pkl'.format(name),
            'args.regressor_path', 'data/smplx/J_regressor_mano_{}.txt'.format(name),
        ])

class ManoFitterCPP:
    def __init__(self, cfg_triangulator, key) -> None:
        self.handmodel = load_object_from_cmd('config/model/mano_full_cpp.yml', [])
        self.triangulator = load_object_from_cmd(cfg_triangulator, [])
        self.time = 0
        self.frame_latest = 0
        self.frame_index = 0
        self.key = 'handl3d'
        self.init = False
        self.params_newest = None
        self.records, self.results = [], []
        self.INIT_SIZE = 10

    def get_keypoints3d(self, records, key=None):
        if key is None:
            return np.stack([r[self.key] for r in records])
        else:
            return np.stack([r[key] for r in records])
    
    def try_to_init(self, records):
        if self.init:
            return copy.deepcopy(self.params_newest)
        mywarn('>> Initialize')
        keypoints3d = self.get_keypoints3d(records)
        params = self.handmodel.init_params(1)
        params = self.handmodel.fit3DShape(keypoints3d, params)
        params = self.handmodel.init3DRT(keypoints3d[-1:], params)
        params = self.handmodel.fit3DPose(keypoints3d[-1:], params)
        mywarn('>> Initialize Rh = {}, Th = {}'.format(params['Rh'][0], params['Th'][0]))
        self.init = True
        return params

    def smooth_results(self):
        params = self.params_newest.copy()
        params['poses'] = params['poses'][:, 3:]
        params['id'] = 0
        return [params]

    def __call__(self, data):
        self.frame_index += 1
        k3d = self.triangulator(data)[0]
        keypoints3d = self.get_keypoints3d([k3d])
        # flag = self.check_keypoints(keypoints3d)
        flag = True
        if not flag:
            mywarn('Missing keypoints {} [{}->{}]'.format(keypoints3d[..., -1].sum(), self.frame_latest, self.frame_index))
            if self.frame_index - self.frame_latest > 10 and self.init:
                mywarn('Missing keypoints, resetting...')
                self.init = False
                self.records = []
                self.results = []
            return -1
        self.records.append(k3d)
        if len(self.records) < self.INIT_SIZE:
            return -1
        with Timer('fitting'):
            params = self.try_to_init(self.records)
            k3d = self.get_keypoints3d(self.records[-1:])
            params = self.handmodel.fit3DPose(k3d, params)
            # params['poses'] = torch.Tensor(params['poses'][:, 3:])
            # params['shapes'] = torch.Tensor(params['shapes'])
            # params['Rh'] = torch.Tensor(params['Rh'])
            # params['Th'] = torch.Tensor(params['Th'])
        # params = self.filter_result(params)
        self.results.append(params)
        self.params_newest = params
        self.frame_latest = self.frame_index
        return self.smooth_results()
        
class BodyFitter(BaseFitter):
    def __init__(self, key, **kwargs):
        super().__init__(**kwargs)
        self.key = key
    
    def get_keypoints3d(self, records, key=None):
        if key is None:
            return torch.Tensor(np.stack([r[self.key] for r in records]))
        else:
            return torch.Tensor(np.stack([r[key] for r in records]))

class ManoFitter(BodyFitter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class HalfFitter(BodyFitter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.INDEX_HALF = [0,1,2,3,4,5,6,7,15,16,17,18]

    def get_keypoints3d(self, records):
        THRES_WRIST = 0.05
        keypoints3d = super().get_keypoints3d(records)
        keypoints3d = keypoints3d[:, self.INDEX_HALF]
        handl = super().get_keypoints3d(records, key='handl3d')
        handr = super().get_keypoints3d(records, key='handr3d')
        dist_ll = torch.norm(keypoints3d[:, 7, :3] - handl[:, 0, :3], dim=-1)
        dist_rr = torch.norm(keypoints3d[:, 4, :3] - handr[:, 0, :3], dim=-1)
        handl[dist_ll>THRES_WRIST] = 0.
        handr[dist_rr>THRES_WRIST] = 0.
        keypoints3d = np.hstack([keypoints3d, handl, handr])
        conf = keypoints3d[..., 3:]
        keypoints3d = np.hstack([(keypoints3d[..., :3] * conf).sum(axis=0)/(1e-5 + conf.sum(axis=0)), conf.min(axis=0)])
        keypoints3d = keypoints3d[None]
        # if (keypoints3d.shape[0] == 10):
        return torch.Tensor(keypoints3d)
    
    def filter_result(self, result):
        result = super().filter_result(result)
        # 限定一下关节旋转
        # 手肘
        # result['poses'][:, 5*3+1] = np.clip(result['poses'][:, 5*3+1], -2.5, 0.1)
        # result['poses'][:, 6*3+1] = np.clip(result['poses'][:, 5*3+1], -0.1, 2.5)
        # 手腕
        return result

class HalfHandFitter(HalfFitter):
    def __init__(self, cfg_handl, cfg_handr, **kwargs) -> None:
        super().__init__(**kwargs)
        self.handl = load_object_from_cmd(cfg_handl, [])
        self.handr = load_object_from_cmd(cfg_handr, [])

    def get_init_params(self, nFrames):
        params = super().get_init_params(nFrames)
        params_ = self.handl.get_init_params(nFrames)
        params['shapes_handl'] = params_['shapes']
        params['shapes_handr'] = params_['shapes'].clone()
        params['Rh_handl'] = torch.zeros((nFrames, 3))
        params['Rh_handr'] = torch.zeros((nFrames, 3))
        params['Th_handl'] = torch.zeros((nFrames, 3))
        params['Th_handr'] = torch.zeros((nFrames, 3))
        return params

    def fitPose(self, keypoints3d, params, weight, option):
        keypoints = {
            'handl': keypoints3d[:, -21-21:-21, :],
            'handr': keypoints3d[:, -21:, :]
        }
        for key in ['handl', 'handr']:
            kpts = keypoints[key]
            params_ = {
                'id': 0,
                'Rh': params['Rh_'+key],
                'Th': params['Th_'+key],
                'shapes': params['shapes_'+key],
                'poses': params[key],
            }
            if key == 'handl':
                params_ = self.handl.fitPose(kpts, params_, self.handl.cfg.pose.weight, self.handl.cfg.pose.option)
            else:
                params_ = self.handr.fitPose(kpts, params_, self.handr.cfg.pose.weight, self.handr.cfg.pose.option)
            params['Rh_'+key] = params_['Rh']
            params['Th_'+key] = params_['Th']
            params['shapes_'+key] = params_['shapes']
            params[key] = params_['poses']
        return super().fitPose(keypoints3d, params, weight, option, 
            kpts_index=[0,1,2,3,4,5,6,7,8,9,10,11,
                12, 17, 21, 25, 29,
                24, 29, 33, 37, 41])
    
    def try_to_init(self, records):
        if self.init:
            return self.params_newest
        params = super().try_to_init(records)
        self.handl.init = False
        self.handr.init = False
        key = 'handl'
        params_ = self.handl.try_to_init(records)
        params['handl'] = params_['poses']
        params['Rh_'+key] = params_['Rh']
        params['Th_'+key] = params_['Th']
        params['shapes_'+key] = params_['shapes']
        key = 'handr'
        params_ = self.handr.try_to_init(records)
        params[key] = params_['poses']
        params['Rh_'+key] = params_['Rh']
        params['Th_'+key] = params_['Th']
        params['shapes_'+key] = params_['shapes']
        return params

if __name__ == '__main__':
    from glob import glob
    from os.path import join
    from tqdm import tqdm
    from ..mytools.file_utils import read_json
    from ..config.baseconfig import load_object_from_cmd

    data= '/nas/datasets/EasyMocap/302'
    # data = '/home/qing/Dataset/handl'
    mode = 'half'
    # data = '/home/qing/DGPU/home/shuaiqing/zju-mocap-mp/female-jump'
    data = '/home/qing/Dataset/desktop/0402/test3'
    k3dnames = sorted(glob(join(data, 'output-keypoints3d', 'keypoints3d', '*.json')))
    if mode == 'handl':
        fitter = load_object_from_cmd('config/recon/fit_manol.yml', [])
    elif mode == 'half':
        fitter = load_object_from_cmd('config/recon/fit_half.yml', [])
    elif mode == 'smpl':
        fitter = load_object_from_cmd('config/recon/fit_smpl.yml', [])
    from easymocap.socket.base_client import BaseSocketClient
    client = BaseSocketClient('0.0.0.0', 9999)

    for k3dname in tqdm(k3dnames):
        k3ds = read_json(k3dname)
        if mode == 'handl':
            k3ds = np.array(k3ds[0]['handl3d'])
            data = {fitter.key3d: k3ds}
        elif mode == 'half':
            data = {
                'keypoints3d': np.array(k3ds[0]['keypoints3d']), 
                'handl3d': np.array(k3ds[0]['handl3d']), 
                'handr3d': np.array(k3ds[0]['handr3d'])
            }
        elif mode == 'smpl':
            k3ds = np.array(k3ds[0]['keypoints3d'])
            data = {fitter.key3d: k3ds}
        results = fitter(data)
        if results != -1:
            client.send_smpl(results)
