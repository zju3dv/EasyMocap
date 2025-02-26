from .base import Model, Params
from .lbs import lbs, batch_rodrigues
import os
import numpy as np
import torch

def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def read_pickle(name):
    import pickle
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def load_model_data(model_path):
    model_path = os.path.abspath(model_path)
    assert os.path.exists(model_path), 'Path {} does not exist!'.format(
        model_path)
    if model_path.endswith('.npz'):
        data = np.load(model_path)
        data = dict(data)
    elif model_path.endswith('.pkl'):
        data = read_pickle(model_path)
    return data

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

def save_regressor(fname, data):
    with open(fname, 'w') as f:
        f.writelines('{} {} {}\r\n'.format('#', data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if(data[i, j] > 0):
                    f.writelines('{} {} {}\r\n'.format(i, j, data[i, j]))


class SMPLModel(Model):
    def __init__(self, model_path, regressor_path=None,
        device='cpu',    
        use_pose_blending=True, use_shape_blending=True, use_joints=True,
        NUM_SHAPES=-1, NUM_POSES=-1,
        use_lbs=True,
        use_root_rot=False,
        **kwargs) -> None:
        super().__init__()
        self.name = 'lbs'
        self.dtype = torch.float32 # not support fp16 now
        self.use_pose_blending = use_pose_blending
        self.use_shape_blending = use_shape_blending
        self.use_root_rot = use_root_rot
        self.NUM_SHAPES = NUM_SHAPES
        self.NUM_POSES = NUM_POSES
        self.NUM_POSES_FULL = NUM_POSES
        self.use_joints = use_joints
        
        if isinstance(device, str):
            device = torch.device(device)
        if not torch.torch.cuda.is_available():
            device = torch.device('cpu')
        self.device = device
        self.model_type = 'smpl'
        # create the SMPL model
        self.lbs = lbs
        self.data = load_model_data(model_path)
        self.register_any_lbs(self.data)
        # keypoints regressor
        if regressor_path is not None and use_joints:
            X_regressor = load_regressor(regressor_path)
            X_regressor = torch.cat((self.J_regressor, X_regressor), dim=0)
            self.register_any_keypoints(X_regressor)
        elif regressor_path is None:
            self.register_any_keypoints(self.J_regressor)
        if not self.use_root_rot:
            self.NUM_POSES -= 3 # remove first 3 dims
        self.to(self.device)
    
    def register_any_lbs(self, data):
        self.faces = to_np(self.data['f'], dtype=np.int64)
        self.register_buffer('faces_tensor',
                             to_tensor(self.faces, dtype=torch.long))
        for key in ['J_regressor', 'v_template', 'weights']:
            if key not in data.keys():
                print('Warning: {} not in data'.format(key))
                self.__setattr__(key, None)
                continue
            val = to_tensor(to_np(data[key]), dtype=self.dtype)
            self.register_buffer(key, val)
        self.NUM_POSES = self.weights.shape[-1] * 3
        self.NUM_POSES_FULL = self.NUM_POSES
        # add poseblending
        if self.use_pose_blending:
            # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
            num_pose_basis = data['posedirs'].shape[-1]
            # 207 x 20670
            data['posedirs_origin'] = data['posedirs']
            data['posedirs'] = np.reshape(data['posedirs'], [-1, num_pose_basis]).T
            val = to_tensor(to_np(data['posedirs']), dtype=self.dtype)
            self.register_buffer('posedirs', val)
        else:
            self.posedirs = None
        # add shape blending
        if self.use_shape_blending:
            val = to_tensor(to_np(data['shapedirs']), dtype=self.dtype)
            if self.NUM_SHAPES != -1:
                val = val[..., :self.NUM_SHAPES]
            self.register_buffer('shapedirs', val)
            self.NUM_SHAPES = val.shape[-1]
        else:
            self.shapedirs = None
        if self.use_shape_blending:
            self.J_shaped = None
        else:
            val = to_tensor(to_np(data['J']), dtype=self.dtype)
            self.register_buffer('J_shaped', val)

        self.nVertices = self.v_template.shape[0]
        # indices of parents for each joints
        kintree_table = data['kintree_table']
        if len(kintree_table.shape) == 2:
            kintree_table = kintree_table[0]
        parents = to_tensor(to_np(kintree_table)).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

    def register_any_keypoints(self, X_regressor):
        # set the parameter of keypoints level
        j_J_regressor = torch.zeros(self.J_regressor.shape[0], X_regressor.shape[0], device=self.device)
        for i in range(self.J_regressor.shape[0]):
            j_J_regressor[i, i] = 1
        j_v_template = X_regressor @ self.v_template
        j_weights = X_regressor @ self.weights
        if self.use_pose_blending:
            posedirs = self.data['posedirs_origin']
            j_posedirs = torch.einsum('ab, bde->ade', X_regressor, torch.Tensor(posedirs)).numpy()
            j_posedirs = np.reshape(j_posedirs, [-1, posedirs.shape[-1]]).T
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

    def forward(self, return_verts=True, return_tensor=True, 
        return_smpl_joints=False, 
        only_shape=False, pose2rot=True,
        v_template=None,
        **params):
        params = self.check_params(params)
        poses, shapes = params['poses'], params['shapes']
        poses = self.extend_poses(pose2rot=pose2rot, **params)
        Rh, Th = params['Rh'], params['Th']
        # check if there are multiple person
        if len(shapes.shape) == 3:
            reshape = poses.shape[:2]
            Rh = Rh.reshape(-1, *Rh.shape[2:])
            Th = Th.reshape(-1, *Th.shape[2:])
            poses = poses.reshape(-1, *poses.shape[2:])
            shapes = shapes.reshape(-1, *shapes.shape[2:])
        else:
            reshape = None
        if len(Rh.shape) == 2: # angle-axis
            Rh = batch_rodrigues(Rh)
        Th = Th.unsqueeze(dim=1)
        if return_verts or not self.use_joints:
            if v_template is None:
                v_template = self.v_template
            if 'scale' in params.keys():
                v_template = v_template * params['scale'][0]
            vertices, joints, T_joints, T_vertices = self.lbs(shapes, poses, v_template,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.weights, pose2rot=pose2rot, dtype=self.dtype, only_shape=only_shape,
                                use_pose_blending=self.use_pose_blending, use_shape_blending=self.use_shape_blending, J_shaped=self.J_shaped)
            if not self.use_joints and not return_verts:
                vertices = joints
        else:
            # only forward joints
            if v_template is None:
                v_template = self.j_v_template
            if 'scale' in params.keys():
                v_template = v_template * params['scale'][0]
            vertices, joints, _, _ = self.lbs(shapes, poses, v_template,
                                self.j_shapedirs, self.j_posedirs,
                                self.j_J_regressor, self.parents,
                                self.j_weights, pose2rot=pose2rot, dtype=self.dtype, only_shape=only_shape,
                                use_pose_blending=self.use_pose_blending, use_shape_blending=self.use_shape_blending, J_shaped=self.J_shaped)
            if return_smpl_joints:
                # vertices = vertices[:, :self.J_regressor.shape[0], :]
                vertices = joints
            else:
                vertices = vertices[:, self.J_regressor.shape[0]:, :]
        vertices = torch.matmul(vertices, Rh.transpose(1, 2)) + Th
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()
        if reshape is not None:
            vertices = vertices.reshape(*reshape, *vertices.shape[1:])
        return vertices
    
    def transform(self, params, pose2rot=True, return_vertices=True):
        v_template = self.v_template
        params = self.check_params(params)
        shapes = params['shapes']
        poses = self.extend_poses(**params)
        vertices, joints, T_joints, T_vertices = self.lbs(shapes, poses, v_template,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.weights, pose2rot=pose2rot, dtype=self.dtype,
                                use_pose_blending=self.use_pose_blending, use_shape_blending=self.use_shape_blending, J_shaped=self.J_shaped,
                                return_vertices=return_vertices)
        return T_joints, T_vertices

    def merge_params(self, params, **kwargs):
        return Params.merge(params, **kwargs)


    def convert_to_standard_smpl(self, params):
        params = self.check_params(params)
        poses, shapes = params['poses'], params['shapes']
        Th = params['Th']
        vertices, joints, _, _ = lbs(shapes, poses, self.v_template,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.weights, pose2rot=True, dtype=self.dtype, only_shape=True)
        # N x 3
        j0 = joints[:, 0, :]
        Rh = params['Rh']
        # N x 3 x 3
        rot = batch_rodrigues(Rh)
        # change the rotate center
        # min_xyz, _ = self.v_template.min(dim=0, keepdim=True)
        # X' = X + delta_center
        # delta_center = torch.tensor([0., -min_xyz[0, 1], 0.]).reshape(1, 3).to(j0.device)
        # J' = J + delta_center
        j0new = j0 + delta_center
        # Tnew = T - (R(d - J0) + J0)
        # Tnew = Th - (torch.einsum('bij,bj->bi', rot, delta_center-j0new) + j0new)
        Tnew = Th - (torch.einsum('bij,bj->bi', rot, j0) + j0new)
        if poses.shape[1] == 69:
            poses = torch.cat([Rh, poses], dim=1)
        elif poses.shape[1] == 72:
            poses[:, :3] = Rh
        else:
            import ipdb;ipdb.set_trace()
        res = dict(poses=poses.detach().cpu().numpy(),
            shapes=shapes.detach().cpu().numpy(),
            Th=Tnew.detach().cpu().numpy()
            )
        return res
    
    def convert_from_standard_smpl(self, params):
        params = self.check_params(params)
        poses, shapes = params['poses'], params['shapes']
        Th = params['Th']
        vertices, joints, _, _ = lbs(shapes, poses, self.v_template,
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

    def init_params(self, nFrames=1, nShapes=1, nPerson=1, ret_tensor=False, add_scale=False):
        params = {
            'poses': np.zeros((nFrames, self.NUM_POSES)),
            'shapes': np.zeros((nShapes, self.NUM_SHAPES)),
            'Rh': np.zeros((nFrames, 3)),
            'Th': np.zeros((nFrames, 3)),
        }
        if add_scale:
            params['scale'] = np.ones((1, 1))
        if nPerson > 1:
            for key in params.keys():
                params[key] = params[key][:, None].repeat(nPerson, axis=1)
        if ret_tensor:
            for key in params.keys():
                params[key] = to_tensor(params[key], self.dtype, self.device)
        return params

    def check_params(self, body_params):
        # 预先拷贝一下，不要修改到原始数据了
        body_params = body_params.copy()
        poses = body_params['poses']
        nFrames = poses.shape[0]
        # convert to torch
        if 'torch' not in str(type(poses)):
            dtype, device = self.dtype, self.device
            for key in ['poses', 'handl', 'handr', 'shapes', 'expression', 'Rh', 'Th', 'scale']:
                if key not in body_params.keys():
                    continue
                body_params[key] = to_tensor(body_params[key], dtype, device)
        poses = body_params['poses']
        # check Rh and Th
        for key in ['Rh', 'Th']:
            if key not in body_params.keys():
                body_params[key] = torch.zeros((nFrames, 3), dtype=poses.dtype, device=poses.device)
        # process shapes
        for key in ['shapes']:
            if body_params[key].shape[0] < nFrames and len(body_params[key].shape) == 2:
                body_params[key] = body_params[key].expand(nFrames, -1)
            elif body_params[key].shape[0] < nFrames and len(body_params[key].shape) == 3:
                body_params[key] = body_params[key].expand(*body_params['poses'].shape[:2], -1)
        return body_params
    
    def __str__(self) -> str:
        res  = '- Model: {}\n'.format(self.model_type)
        res += '  poses: {}\n'.format(self.NUM_POSES)
        res += '  shapes: {}\n'.format(self.NUM_SHAPES)
        res += '  vertices: {}\n'.format(self.v_template.shape)
        res += '  faces: {}\n'.format(self.faces.shape)
        res += '  posedirs: {}\n'.format(self.posedirs.shape)
        res += '  shapedirs: {}\n'.format(self.shapedirs.shape)
        return res
    
    def extend_poses(self, poses, **kwargs):
        if poses.shape[-1] == self.NUM_POSES_FULL:
            return poses
        if not self.use_root_rot:
            if kwargs.get('pose2rot', True):
                zero_rot = torch.zeros((*poses.shape[:-1], 3), dtype=poses.dtype, device=poses.device)
                poses = torch.cat([zero_rot, poses], dim=-1)
            elif poses.shape[-3] != self.NUM_POSES_FULL // 3:
                # insert a blank rotation
                zero_rot = torch.zeros((*poses.shape[:-3], 1, 3), dtype=poses.dtype, device=poses.device)
                zero_rot = batch_rodrigues(zero_rot)
                poses = torch.cat([zero_rot, poses], dim=-3)
        return poses

    def jacobian_posesfull_poses(self, poses, poses_full):
        # TODO: cache this 
        if self.use_root_rot:
            jacobian = torch.eye(poses.shape[-1], dtype=poses.dtype, device=poses.device)
        else:
            zero_root = torch.zeros((3, poses.shape[-1]), dtype=poses.dtype, device=poses.device)
            eye_right = torch.eye(poses.shape[-1], dtype=poses.dtype, device=poses.device)
            jacobian = torch.cat([zero_root, eye_right], dim=0)
        return jacobian

    def export_full_poses(self, poses, **kwargs):
        if not self.use_root_rot:
            poses = np.hstack([np.zeros((poses.shape[0], 3)), poses])
        return poses
    
    def encode(self, body_params):
        # This function provide standard SMPL parameters to this model
        poses = body_params['poses']
        if 'Rh' not in body_params.keys():
            body_params['Rh'] = poses[:, :3].copy()
        if 'Th' not in body_params.keys():
            if 'trans' in body_params.keys():
                body_params['Th'] = body_params.pop('trans')
            else:
                body_params['Th'] = np.zeros((poses.shape[0], 3), dtype=poses.dtype)
        if not self.use_root_rot and poses.shape[1] == 72:
            body_params['poses'] = poses[:, 3:].copy()
        return body_params

class SMPLLayerEmbedding(SMPLModel):
    def __init__(self, vposer_ckpt='data/body_models/vposer_v02', **kwargs):
        super().__init__(**kwargs)
        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser
        vposer, _ = load_model(vposer_ckpt, 
            model_code=VPoser,
            remove_words_in_model_weights='vp_model.',
            disable_grad=True)
        vposer.eval()
        vposer.to(self.device)
        self.vposer = vposer
        self.vposer_dim = 32
        self.NUM_POSES = self.vposer_dim

    def encode(self, body_params):
        # This function provide standard SMPL parameters to this model
        poses = body_params['poses']
        if poses.shape[1] == self.vposer_dim:
            return body_params
        poses_tensor = torch.Tensor(poses).to(self.device)
        ret = self.vposer.encode(poses_tensor[:, :63]).mean
        body_params = super().encode(body_params)
        body_params['poses'] = ret.detach().cpu().numpy()
        return body_params

    def extend_poses(self, poses, **kwargs):
        if poses.shape[-1] == self.vposer_dim:
            ret = self.vposer.decode(poses)
            poses_body = ret['pose_body'].reshape(poses.shape[0], -1)
        elif poses.shape[-1] == self.NUM_POSES_FULL:
            return poses
        elif poses.shape[-1] == self.NUM_POSES_FULL - 3:
            poses_zero = torch.zeros((poses.shape[0], 3), dtype=poses.dtype, device=poses.device)
            poses = torch.cat([poses_zero, poses], dim=-1)
            return poses
        poses_zero = torch.zeros((poses_body.shape[0], 3), dtype=poses_body.dtype, device=poses_body.device)
        poses = torch.cat([poses_zero, poses_body, poses_zero, poses_zero], dim=1)
        return poses

    def export_full_poses(self, poses, **kwargs):
        poses = torch.Tensor(poses).to(self.device)
        poses = self.extend_poses(poses)
        return poses.detach().cpu().numpy()

if __name__ == '__main__':
    vis = True
    test_config = {
        'smpl':{
            'model_path': 'data/bodymodels/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl',
            'regressor_path': 'data/smplx/J_regressor_body25.npy',
        },
        'smplh':{
            'model_path': 'data/bodymodels/smplhv1.2/male/model.npz',
            'regressor_path': None,
        },
        'mano':{
            'model_path': 'data/bodymodels/manov1.2/MANO_LEFT.pkl',
            'regressor_path': None,
        },
        'flame':{
            'model_path': 'data/bodymodels/FLAME2020/FLAME_MALE.pkl',
            'regressor_path': None,
        }
    }
    for name, cfg in test_config.items():
        print('Testing {}...'.format(name))
        model = SMPLModel(**cfg)
        print(model)
        params = model.init_params()
        for key in params.keys():
            params[key] = (np.random.rand(*params[key].shape) - 0.5)*0.5
        vertices = model.vertices(params, return_tensor=True)[0]
        if cfg['regressor_path'] is not None:
            keypoints = model.keypoints(params, return_tensor=True)[0]
            print(keypoints.shape)
        if vis:
            import open3d as o3d
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.reshape(-1, 3))
            mesh.triangles = o3d.utility.Vector3iVector(model.faces.reshape(-1, 3))
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])
