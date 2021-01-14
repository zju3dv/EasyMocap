'''
  @ Date: 2020-11-18 14:04:10
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-14 20:14:34
  @ FilePath: /EasyMocap/code/smplmodel/body_model.py
'''
import torch
import torch.nn as nn
from .lbs import lbs, batch_rodrigues
import os.path as osp
import pickle
import numpy as np

def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

class SMPLlayer(nn.Module):
    def __init__(self, model_path, gender='neutral', device=None,
        regressor_path=None) -> None:
        super(SMPLlayer, self).__init__()
        dtype = torch.float32
        self.dtype = dtype
        self.device = device
        # create the SMPL model
        if osp.isdir(model_path):
            model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
            smpl_path = osp.join(model_path, model_fn)
        else:
            smpl_path = model_path
        assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
            smpl_path)

        with open(smpl_path, 'rb') as smpl_file:
            data = pickle.load(smpl_file, encoding='latin1')
        self.faces = data['f']
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))
        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = data['posedirs']
        data['posedirs'] = np.reshape(data['posedirs'], [-1, num_pose_basis]).T
        
        for key in ['J_regressor', 'v_template', 'weights', 'posedirs', 'shapedirs']:
            val = to_tensor(to_np(data[key]), dtype=dtype)
            self.register_buffer(key, val)
        # indices of parents for each joints
        parents = to_tensor(to_np(data['kintree_table'][0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        # joints regressor
        if regressor_path is not None:
            X_regressor = to_tensor(np.load(regressor_path))
            X_regressor = torch.cat((self.J_regressor, X_regressor), dim=0)

            j_J_regressor = torch.zeros(24, X_regressor.shape[0], device=device)
            for i in range(24):
                j_J_regressor[i, i] = 1
            j_v_template = X_regressor @ self.v_template
            # 
            j_shapedirs = torch.einsum('vij,kv->kij', [self.shapedirs, X_regressor])
            # (25, 24)
            j_weights = X_regressor @ self.weights
            j_posedirs = torch.einsum('ab, bde->ade', [X_regressor, torch.Tensor(posedirs)]).numpy()
            j_posedirs = np.reshape(j_posedirs, [-1, num_pose_basis]).T
            j_posedirs = to_tensor(j_posedirs)
            self.register_buffer('j_posedirs', j_posedirs)
            self.register_buffer('j_shapedirs', j_shapedirs)
            self.register_buffer('j_weights', j_weights)
            self.register_buffer('j_v_template', j_v_template)
            self.register_buffer('j_J_regressor', j_J_regressor)

    def forward(self, poses, shapes, Rh=None, Th=None, return_verts=True, return_tensor=True, only_shape=False, **kwargs):
        """ Forward pass for SMPL model

        Args:
            poses (n, 72)
            shapes (n, 10)
            Rh (n, 3): global orientation
            Th (n, 3): global translation
            return_verts (bool, optional): if True return (6890, 3). Defaults to False.
        """
        if 'torch' not in str(type(poses)):
            dtype, device = self.dtype, self.device
            poses = to_tensor(poses, dtype, device)
            shapes = to_tensor(shapes, dtype, device)
            Rh = to_tensor(Rh, dtype, device)
            Th = to_tensor(Th, dtype, device)
        bn = poses.shape[0]
        if Rh is None:
            Rh = torch.zeros(bn, 3, device=poses.device)
        rot = batch_rodrigues(Rh)
        transl = Th.unsqueeze(dim=1)
        if shapes.shape[0] < bn:
            shapes = shapes.expand(bn, -1)
        if return_verts:
            vertices, joints = lbs(shapes, poses, self.v_template,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.weights, pose2rot=True, dtype=self.dtype)
        else:
            vertices, joints = lbs(shapes, poses, self.j_v_template,
                                self.j_shapedirs, self.j_posedirs,
                                self.j_J_regressor, self.parents,
                                self.j_weights, pose2rot=True, dtype=self.dtype, only_shape=only_shape)
            vertices = vertices[:, 24:, :]
        vertices = torch.matmul(vertices, rot.transpose(1, 2)) + transl
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()
        return vertices