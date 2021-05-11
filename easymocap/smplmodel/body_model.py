'''
  @ Date: 2020-11-18 14:04:10
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-11 15:09:44
  @ FilePath: /EasyMocap/easymocap/smplmodel/body_model.py
'''
import torch
import torch.nn as nn
from .lbs import lbs, batch_rodrigues
import os.path as osp
import pickle
import numpy as np
import os

def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def load_regressor(regressor_path):
    if regressor_path.endswith('.npy'):
        X_regressor = to_tensor(np.load(regressor_path))
    elif regressor_path.endswith('.txt'):
        data = np.loadtxt(regressor_path)
        with open(regressor_path, 'r') as f:
            shape = f.readline().split()[1:]
        reg = np.zeros((int(shape[0]), int(shape[1])))
        for i, j, v in data:
            reg[int(i), int(j)] = v
        X_regressor = to_tensor(reg)
    else:
        import ipdb; ipdb.set_trace()
    return X_regressor

NUM_POSES = {'smpl': 72, 'smplh': 78, 'smplx': 66 + 12 + 9}
NUM_SHAPES = 10
NUM_EXPR = 10
class SMPLlayer(nn.Module):
    def __init__(self, model_path, model_type='smpl', gender='neutral', device=None,
        regressor_path=None) -> None:
        super(SMPLlayer, self).__init__()
        dtype = torch.float32
        self.dtype = dtype
        self.device = device
        self.model_type = model_type
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
        if self.model_type == 'smplx':
            # shape
            self.num_expression_coeffs = 10
            self.num_shapes = 10
            self.shapedirs = self.shapedirs[:, :, :self.num_shapes+self.num_expression_coeffs]
        elif self.model_type in ['smpl', 'smplh']:
            self.shapedirs = self.shapedirs[:, :, :NUM_SHAPES]
        # joints regressor
        if regressor_path is not None:
            X_regressor = load_regressor(regressor_path)
            X_regressor = torch.cat((self.J_regressor, X_regressor), dim=0)

            j_J_regressor = torch.zeros(self.J_regressor.shape[0], X_regressor.shape[0], device=device)
            for i in range(self.J_regressor.shape[0]):
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
        if self.model_type == 'smplh':
            # load smplh data
            self.num_pca_comps = 6
            from os.path import join
            for key in ['LEFT', 'RIGHT']:
                left_file = join(os.path.dirname(smpl_path), 'MANO_{}.pkl'.format(key))
                with open(left_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                val = to_tensor(to_np(data['hands_mean'].reshape(1, -1)), dtype=dtype)
                self.register_buffer('mHandsMean'+key[0], val)
                val = to_tensor(to_np(data['hands_components'][:self.num_pca_comps, :]), dtype=dtype)
                self.register_buffer('mHandsComponents'+key[0], val)
            self.use_pca = True
            self.use_flat_mean = True
        elif self.model_type == 'smplx':
            # hand pose
            self.num_pca_comps = 6
            from os.path import join
            for key in ['Ll', 'Rr']:
                val = to_tensor(to_np(data['hands_mean'+key[1]].reshape(1, -1)), dtype=dtype)
                self.register_buffer('mHandsMean'+key[0], val)
                val = to_tensor(to_np(data['hands_components'+key[1]][:self.num_pca_comps, :]), dtype=dtype)
                self.register_buffer('mHandsComponents'+key[0], val)
            self.use_pca = True
            self.use_flat_mean = True
            
    def extend_pose(self, poses):
        if self.model_type not in ['smplh', 'smplx']:
            return poses
        elif self.model_type == 'smplh' and poses.shape[-1] == 156:
            return poses
        elif self.model_type == 'smplx' and poses.shape[-1] == 165:
            return poses
        
        NUM_BODYJOINTS = 22 * 3
        if self.use_pca:
            NUM_HANDJOINTS = self.num_pca_comps
        else:
            NUM_HANDJOINTS = 15 * 3
        NUM_FACEJOINTS = 3 * 3
        poses_lh = poses[:, NUM_BODYJOINTS:NUM_BODYJOINTS + NUM_HANDJOINTS]
        poses_rh = poses[:, NUM_BODYJOINTS + NUM_HANDJOINTS:NUM_BODYJOINTS+NUM_HANDJOINTS*2]
        if self.use_pca:
            poses_lh = poses_lh @ self.mHandsComponentsL
            poses_rh = poses_rh @ self.mHandsComponentsR
        if self.use_flat_mean:
            poses_lh = poses_lh + self.mHandsMeanL
            poses_rh = poses_rh + self.mHandsMeanR
        if self.model_type == 'smplh':
            poses = torch.cat([poses[:, :NUM_BODYJOINTS], poses_lh, poses_rh], dim=1)
        elif self.model_type == 'smplx':
            # the head part have only three joints
            # poses_head: (N, 9), jaw_pose, leye_pose, reye_pose respectively
            poses_head = poses[:, NUM_BODYJOINTS+NUM_HANDJOINTS*2:]
            # body, head, left hand, right hand
            poses = torch.cat([poses[:, :NUM_BODYJOINTS], poses_head, poses_lh, poses_rh], dim=1)
        return poses

    def get_root(self, poses, shapes, return_tensor=False):
        if 'torch' not in str(type(poses)):
            dtype, device = self.dtype, self.device
            poses = to_tensor(poses, dtype, device)
            shapes = to_tensor(shapes, dtype, device)
        vertices, joints = lbs(shapes, poses, self.v_template,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.weights, pose2rot=True, dtype=self.dtype, only_shape=True)
        # N x 3
        j0 = joints[:, 0, :]
        if not return_tensor:
            j0 = j0.detach().cpu().numpy()
        return j0

    def convert_from_standard_smpl(self, poses, shapes, Rh=None, Th=None, expression=None):
        if 'torch' not in str(type(poses)):
            dtype, device = self.dtype, self.device
            poses = to_tensor(poses, dtype, device)
            shapes = to_tensor(shapes, dtype, device)
            Rh = to_tensor(Rh, dtype, device)
            Th = to_tensor(Th, dtype, device)
            if expression is not None:
                expression = to_tensor(expression, dtype, device)

        bn = poses.shape[0]
        # process shapes
        if shapes.shape[0] < bn:
            shapes = shapes.expand(bn, -1)
        vertices, joints = lbs(shapes, poses, self.v_template,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.weights, pose2rot=True, dtype=self.dtype, only_shape=True)
        # N x 3
        j0 = joints[:, 0, :]
        Rh = poses[:, :3].clone()
        # N x 3 x 3
        rot = batch_rodrigues(Rh)
        Tnew = Th + j0 - torch.einsum('bij,bj->bi', rot, j0)
        poses[:, :3] = 0
        res = dict(poses=poses.detach().cpu().numpy(),
            shapes=shapes.detach().cpu().numpy(),
            Rh=Rh.detach().cpu().numpy(),
            Th=Tnew.detach().cpu().numpy()
            )
        return res

    def forward(self, poses, shapes, Rh=None, Th=None, expression=None, return_verts=True, return_tensor=True, only_shape=False, **kwargs):
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
            if Rh is not None:
                Rh = to_tensor(Rh, dtype, device)
            if Th is not None:
                Th = to_tensor(Th, dtype, device)
            if expression is not None:
                expression = to_tensor(expression, dtype, device)

        bn = poses.shape[0]
        # process Rh, Th
        if Rh is None:
            Rh = torch.zeros(bn, 3, device=poses.device)
        if Th is None:
            Th = torch.zeros(bn, 3, device=poses.device)
        
        if len(Rh.shape) == 2: # angle-axis
            rot = batch_rodrigues(Rh)
        else:
            rot = Rh
        transl = Th.unsqueeze(dim=1)
        # process shapes
        if shapes.shape[0] < bn:
            shapes = shapes.expand(bn, -1)
        if expression is not None and self.model_type == 'smplx':
            shapes = torch.cat([shapes, expression], dim=1)
        # process poses
        if self.model_type == 'smplh' or self.model_type == 'smplx':
            poses = self.extend_pose(poses)
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
            vertices = vertices[:, self.J_regressor.shape[0]:, :]
        vertices = torch.matmul(vertices, rot.transpose(1, 2)) + transl
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()
        return vertices
    
    def check_params(self, body_params):
        model_type = self.model_type
        nFrames = body_params['poses'].shape[0]
        if body_params['poses'].shape[1] != NUM_POSES[model_type]:
            body_params['poses'] = np.hstack((body_params['poses'], np.zeros((nFrames, NUM_POSES[model_type] - body_params['poses'].shape[1]))))
        if model_type == 'smplx' and 'expression' not in body_params.keys():
            body_params['expression'] = np.zeros((nFrames, NUM_EXPR))
        return body_params

    @staticmethod    
    def merge_params(param_list, share_shape=True):
        output = {}
        for key in ['poses', 'shapes', 'Rh', 'Th', 'expression']:
            if key in param_list[0].keys():
                output[key] = np.vstack([v[key] for v in param_list])
        if share_shape:
            output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
        return output