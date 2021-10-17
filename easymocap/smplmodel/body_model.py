'''
  @ Date: 2020-11-18 14:04:10
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-08-28 16:37:55
  @ FilePath: /EasyMocap/easymocap/smplmodel/body_model.py
'''
import torch
import torch.nn as nn
from .lbs import batch_rodrigues
from .lbs import lbs, dqs
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

def load_bodydata(model_type, model_path, gender):
    if osp.isdir(model_path):
        model_fn = '{}_{}.{ext}'.format(model_type.upper(), gender.upper(), ext='pkl')
        smpl_path = osp.join(model_path, model_fn)
    else:
        smpl_path = model_path
    assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
        smpl_path)

    with open(smpl_path, 'rb') as smpl_file:
        data = pickle.load(smpl_file, encoding='latin1')
    return data

NUM_POSES = {'smpl': 72, 'smplh': 78, 'smplx': 66 + 12 + 9, 'mano': 9}
NUM_SHAPES = 10
NUM_EXPR = 10
class SMPLlayer(nn.Module):
    def __init__(self, model_path, model_type='smpl', gender='neutral', device=None,
        regressor_path=None,
        use_pose_blending=True, use_shape_blending=True, use_joints=True,
        with_color=False, use_lbs=True,
        **kwargs) -> None:
        super(SMPLlayer, self).__init__()
        dtype = torch.float32
        self.dtype = dtype
        self.use_pose_blending = use_pose_blending
        self.use_shape_blending = use_shape_blending
        self.use_joints = use_joints
        
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model_type = model_type
        self.NUM_POSES = NUM_POSES[model_type]
        # create the SMPL model
        if use_lbs:
            self.lbs = lbs
        else:
            self.lbs = dqs
        data = load_bodydata(model_type, model_path, gender)
        if with_color:
            self.color = data['vertex_colors']
        else:
            self.color = None
        self.faces = data['f']
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))
        for key in ['J_regressor', 'v_template', 'weights']:
            val = to_tensor(to_np(data[key]), dtype=dtype)
            self.register_buffer(key, val)
        # add poseblending
        if use_pose_blending:
            # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
            num_pose_basis = data['posedirs'].shape[-1]
            # 207 x 20670
            posedirs = data['posedirs']
            data['posedirs'] = np.reshape(data['posedirs'], [-1, num_pose_basis]).T
            val = to_tensor(to_np(data['posedirs']), dtype=dtype)
            self.register_buffer('posedirs', val)
        else:
            self.posedirs = None
        # add shape blending
        if use_shape_blending:
            val = to_tensor(to_np(data['shapedirs']), dtype=dtype)
            self.register_buffer('shapedirs', val)
        else:
            self.shapedirs = None
        if use_shape_blending:
            self.J_shaped = None
        else:
            val = to_tensor(to_np(data['J']), dtype=dtype)
            self.register_buffer('J_shaped', val)

        self.nVertices = self.v_template.shape[0]
        # indices of parents for each joints
        parents = to_tensor(to_np(data['kintree_table'][0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        
        if self.use_shape_blending:
            if self.model_type == 'smplx':
                # shape
                self.num_expression_coeffs = 10
                self.num_shapes = 10
                self.shapedirs = self.shapedirs[:, :, :self.num_shapes+self.num_expression_coeffs]
            elif self.model_type in ['smpl', 'smplh']:
                self.shapedirs = self.shapedirs[:, :, :NUM_SHAPES]
        # joints regressor
        if regressor_path is not None and use_joints:
            X_regressor = load_regressor(regressor_path)
            X_regressor = torch.cat((self.J_regressor, X_regressor), dim=0)

            j_J_regressor = torch.zeros(self.J_regressor.shape[0], X_regressor.shape[0], device=device)
            for i in range(self.J_regressor.shape[0]):
                j_J_regressor[i, i] = 1
            j_v_template = X_regressor @ self.v_template
            # 
            # (25, 24)
            j_weights = X_regressor @ self.weights
            if self.use_pose_blending:
                j_posedirs = torch.einsum('ab, bde->ade', [X_regressor, torch.Tensor(posedirs)]).numpy()
                j_posedirs = np.reshape(j_posedirs, [-1, num_pose_basis]).T
                j_posedirs = to_tensor(j_posedirs)
                self.register_buffer('j_posedirs', j_posedirs)
            else:
                self.j_posedirs = None
            if self.use_shape_blending:
                j_shapedirs = torch.einsum('vij,kv->kij', [self.shapedirs, X_regressor])
                self.register_buffer('j_shapedirs', j_shapedirs)
            else:
                self.j_shapedirs = None
            self.register_buffer('j_weights', j_weights)
            self.register_buffer('j_v_template', j_v_template)
            self.register_buffer('j_J_regressor', j_J_regressor)
        if self.model_type == 'smplh':
            # load smplh data
            self.num_pca_comps = kwargs['num_pca_comps']
            from os.path import join
            for key in ['LEFT', 'RIGHT']:
                left_file = join(kwargs['mano_path'], 'MANO_{}.pkl'.format(key))
                with open(left_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                val = to_tensor(to_np(data['hands_mean'].reshape(1, -1)), dtype=dtype)
                self.register_buffer('mHandsMean'+key[0], val)
                val = to_tensor(to_np(data['hands_components'][:self.num_pca_comps, :]), dtype=dtype)
                self.register_buffer('mHandsComponents'+key[0], val)
            self.use_pca = kwargs['use_pca']
            self.use_flat_mean = kwargs['use_flat_mean']
            if self.use_pca:
                self.NUM_POSES = 66 + self.num_pca_comps * 2
            else:
                self.NUM_POSES = 66 + 15 * 3 * 2
        elif self.model_type == 'mano':
            self.num_pca_comps = kwargs['num_pca_comps']
            self.use_pca = kwargs['use_pca']
            self.use_flat_mean = kwargs['use_flat_mean']
            if self.use_pca:
                self.NUM_POSES = self.num_pca_comps + 3
            else:
                self.NUM_POSES = 45 + 3

            val = to_tensor(to_np(data['hands_mean'].reshape(1, -1)), dtype=dtype)
            self.register_buffer('mHandsMean', val)
            val = to_tensor(to_np(data['hands_components'][:self.num_pca_comps, :]), dtype=dtype)
            self.register_buffer('mHandsComponents', val)
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
        self.to(self.device)
    
    @staticmethod
    def extend_hand(poses, use_pca, use_flat_mean, coeffs, mean):
        if use_pca:
            poses = poses @ coeffs
        if not use_flat_mean:
            poses = poses + mean
        return poses

    def extend_pose(self, poses):
        # skip SMPL or already extend
        if self.model_type not in ['smplh', 'smplx', 'mano']:
            return poses
        elif self.model_type == 'smplh' and poses.shape[-1] == 156 and self.use_flat_mean:
            return poses
        elif self.model_type == 'smplx' and poses.shape[-1] == 165 and self.use_flat_mean:
            return poses
        elif self.model_type == 'mano' and poses.shape[-1] == 48 and self.use_flat_mean:
            return poses
        # skip mano
        if self.model_type == 'mano':
            poses_hand = self.extend_hand(poses[..., 3:], self.use_pca, self.use_flat_mean,
                self.mHandsComponents, self.mHandsMean)
            poses = torch.cat([poses[..., :3], poses_hand], dim=-1)
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
        if not self.use_flat_mean:
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
    
    def full_poses(self, poses):
        if 'torch' not in str(type(poses)):
            dtype, device = self.dtype, self.device
            poses = to_tensor(poses, dtype, device)
        poses = self.extend_pose(poses)
        return poses.detach().cpu().numpy()

    def forward(self, poses, shapes, Rh=None, Th=None, expression=None, 
        v_template=None,
        return_verts=True, return_tensor=True, return_smpl_joints=False, 
        only_shape=False, pose2rot=True, **kwargs):
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
        if pose2rot: # if given rotation matrix, no need for this
            poses = self.extend_pose(poses)
        if return_verts or not self.use_joints:
            if v_template is None:
                v_template = self.v_template
            vertices, joints = self.lbs(shapes, poses, v_template,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.weights, pose2rot=pose2rot, dtype=self.dtype,
                                use_pose_blending=self.use_pose_blending, use_shape_blending=self.use_shape_blending, J_shaped=self.J_shaped)
            if not self.use_joints and not return_verts:
                vertices = joints
        else:
            vertices, joints = self.lbs(shapes, poses, self.j_v_template,
                                self.j_shapedirs, self.j_posedirs,
                                self.j_J_regressor, self.parents,
                                self.j_weights, pose2rot=pose2rot, dtype=self.dtype, only_shape=only_shape,
                                use_pose_blending=self.use_pose_blending, use_shape_blending=self.use_shape_blending, J_shaped=self.J_shaped)
            if return_smpl_joints:
                vertices = vertices[:, :self.J_regressor.shape[0], :]
            else:
                vertices = vertices[:, self.J_regressor.shape[0]:, :]
        vertices = torch.matmul(vertices, rot.transpose(1, 2)) + transl
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()
        return vertices
    
    def init_params(self, nFrames=1, nShapes=1, ret_tensor=False):
        params = {
            'poses': np.zeros((nFrames, self.NUM_POSES)),
            'shapes': np.zeros((nShapes, NUM_SHAPES)),
            'Rh': np.zeros((nFrames, 3)),
            'Th': np.zeros((nFrames, 3)),
        }
        if self.model_type == 'smplx':
            params['expression'] = np.zeros((nFrames, NUM_EXPR))
        if ret_tensor:
            for key in params.keys():
                params[key] = to_tensor(params[key], self.dtype, self.device)
        return params

    def check_params(self, body_params):
        model_type = self.model_type
        nFrames = body_params['poses'].shape[0]
        if body_params['poses'].shape[1] != self.NUM_POSES:
            body_params['poses'] = np.hstack((body_params['poses'], np.zeros((nFrames, self.NUM_POSES - body_params['poses'].shape[1]))))
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
        # add other keys
        for key in param_list[0].keys():
            if key in output.keys():
                continue
            output[key] = np.stack([v[key] for v in param_list])
        return output
    
    @staticmethod
    def select_nf(params_all, nf):
        output = {}
        for key in ['poses', 'Rh', 'Th']:
            output[key] = params_all[key][nf:nf+1, :]
        if 'expression' in params_all.keys():
            output['expression'] = params_all['expression'][nf:nf+1, :]
        if params_all['shapes'].shape[0] == 1:
            output['shapes'] = params_all['shapes']
        else:
            output['shapes'] = params_all['shapes'][nf:nf+1, :]
        return output