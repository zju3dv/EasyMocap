import os
import torch
import torch.nn as nn
from .config import update_hparams
# from .head import PareHead, SMPLHead, SMPLCamHead
from .head import PareHead
from .backbone.utils import get_backbone_info
from .backbone.hrnet import hrnet_w32
from os.path import join
from easymocap.multistage.torchgeometry import rotation_matrix_to_axis_angle
import cv2

def try_to_download():
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'pare')
    cmd = 'wget https://www.dropbox.com/s/aeulffqzb3zmh8x/pare-github-data.zip'
    os.system(cmd)
    os.makedirs(model_dir, exist_ok=True)
    cmd = 'unzip pare-github-data.zip -d {}'.format(model_dir)
    os.system(cmd)

CFG = 'models/pare/data/pare/checkpoints/pare_w_3dpw_config.yaml'
CKPT = 'models/pare/data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt'

class PARE(nn.Module):
    def __init__(
            self,
            num_joints=24,
            softmax_temp=1.0,
            num_features_smpl=64,
            backbone='resnet50',
            focal_length=5000.,
            img_res=224,
            pretrained=None,
            iterative_regression=False,
            iter_residual=False,
            num_iterations=3,
            shape_input_type='feats',  # 'feats.all_pose.shape.cam',
            pose_input_type='feats', # 'feats.neighbor_pose_feats.all_pose.self_pose.neighbor_pose.shape.cam'
            pose_mlp_num_layers=1,
            shape_mlp_num_layers=1,
            pose_mlp_hidden_size=256,
            shape_mlp_hidden_size=256,
            use_keypoint_features_for_smpl_regression=False,
            use_heatmaps='',
            use_keypoint_attention=False,
            keypoint_attention_act='softmax',
            use_postconv_keypoint_attention=False,
            use_scale_keypoint_attention=False,
            use_final_nonlocal=None,
            use_branch_nonlocal=None,
            use_hmr_regression=False,
            use_coattention=False,
            num_coattention_iter=1,
            coattention_conv='simple',
            deconv_conv_kernel_size=4,
            use_upsampling=False,
            use_soft_attention=False,
            num_branch_iteration=0,
            branch_deeper=False,
            num_deconv_layers=3,
            num_deconv_filters=256,
            use_resnet_conv_hrnet=False,
            use_position_encodings=None,
            use_mean_camshape=False,
            use_mean_pose=False,
            init_xavier=False,
            use_cam=False,
    ):
        super(PARE, self).__init__()
        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            # hrnet_w32-conv, hrnet_w32-interp
            self.backbone = eval(backbone)(
                pretrained=True,
                downsample=False,
                use_conv=(use_conv == 'conv')
            )
        else:
            self.backbone = eval(backbone)(pretrained=True)

        # self.backbone = eval(backbone)(pretrained=True)
        self.head = PareHead(
            num_joints=num_joints,
            num_input_features=get_backbone_info(backbone)['n_output_channels'],
            softmax_temp=softmax_temp,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=[num_deconv_filters] * num_deconv_layers,
            num_deconv_kernels=[deconv_conv_kernel_size] * num_deconv_layers,
            num_features_smpl=num_features_smpl,
            final_conv_kernel=1,
            iterative_regression=iterative_regression,
            iter_residual=iter_residual,
            num_iterations=num_iterations,
            shape_input_type=shape_input_type,
            pose_input_type=pose_input_type,
            pose_mlp_num_layers=pose_mlp_num_layers,
            shape_mlp_num_layers=shape_mlp_num_layers,
            pose_mlp_hidden_size=pose_mlp_hidden_size,
            shape_mlp_hidden_size=shape_mlp_hidden_size,
            use_keypoint_features_for_smpl_regression=use_keypoint_features_for_smpl_regression,
            use_heatmaps=use_heatmaps,
            use_keypoint_attention=use_keypoint_attention,
            use_postconv_keypoint_attention=use_postconv_keypoint_attention,
            keypoint_attention_act=keypoint_attention_act,
            use_scale_keypoint_attention=use_scale_keypoint_attention,
            use_branch_nonlocal=use_branch_nonlocal, # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
            use_final_nonlocal=use_final_nonlocal, # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
            backbone=backbone,
            use_hmr_regression=use_hmr_regression,
            use_coattention=use_coattention,
            num_coattention_iter=num_coattention_iter,
            coattention_conv=coattention_conv,
            use_upsampling=use_upsampling,
            use_soft_attention=use_soft_attention,
            num_branch_iteration=num_branch_iteration,
            branch_deeper=branch_deeper,
            use_resnet_conv_hrnet=use_resnet_conv_hrnet,
            use_position_encodings=use_position_encodings,
            use_mean_camshape=use_mean_camshape,
            use_mean_pose=use_mean_pose,
            init_xavier=init_xavier,
        )

        self.use_cam = use_cam
        # if self.use_cam:
        #     self.smpl = SMPLCamHead(
        #         img_res=img_res,
        #     )
        # else:
        #     self.smpl = SMPLHead(
        #         focal_length=focal_length,
        #         img_res=img_res
        #     )

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def forward(
            self,
            images,
            gt_segm=None,
    ):
        features = self.backbone(images)
        hmr_output = self.head(features, gt_segm=gt_segm)
        rotmat = hmr_output['pred_pose']
        shape = hmr_output['pred_shape']
        rotmat_flat = rotmat.reshape(-1, 3, 3)
        rvec_flat = rotation_matrix_to_axis_angle(rotmat_flat)
        rvec = rvec_flat.reshape(*rotmat.shape[:-2], 3)
        rvec = rvec.reshape(*rvec.shape[:-2], -1)
        return {
            'Rh': rvec[..., :3],
            'Th': torch.zeros_like(rvec[..., :3]),
            'poses': rvec[..., 3:],
            'shapes': shape,
        }

from ..basetopdown import BaseTopDownModelCache
import pickle

class NullSPIN:
    def __init__(self, ckpt) -> None:
        self.name = 'spin'

    def __call__(self, bbox, images, imgname):
        from easymocap.mytools.reader import read_smpl
        basename = os.path.basename(imgname)
        cachename = join(self.output, self.name, basename.replace('.jpg', '.json'))
        if os.path.exists(cachename):
            params = read_smpl(cachename)
            params = params[0]
            params = {key:val[0] for key, val in params.items() if key != 'id'}
            ret = {
                'params': params
            }
            return ret
        else:
            import ipdb; ipdb.set_trace()

class MyPARE(BaseTopDownModelCache):
    def __init__(self, ckpt) -> None:
        super().__init__('pare', bbox_scale=1.1, res_input=224)
        if not os.path.exists(CFG):
            from ...io.model import try_to_download_SMPL
            try_to_download_SMPL('models/pare')
        self.model_cfg = update_hparams(CFG)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self._build_model()
        self._load_pretrained_model(CKPT)
        self.model.eval()
        self.model.to(self.device)
    
    def __call__(self, bbox, images, imgnames):
        return super().__call__(bbox[0], images, imgnames)

    def _build_model(self):
        # ========= Define PARE model ========= #
        model_cfg = self.model_cfg

        if model_cfg.METHOD == 'pare':
            model = PARE(
                backbone=model_cfg.PARE.BACKBONE,
                num_joints=model_cfg.PARE.NUM_JOINTS,
                softmax_temp=model_cfg.PARE.SOFTMAX_TEMP,
                num_features_smpl=model_cfg.PARE.NUM_FEATURES_SMPL,
                focal_length=model_cfg.DATASET.FOCAL_LENGTH,
                img_res=model_cfg.DATASET.IMG_RES,
                pretrained=model_cfg.TRAINING.PRETRAINED,
                iterative_regression=model_cfg.PARE.ITERATIVE_REGRESSION,
                num_iterations=model_cfg.PARE.NUM_ITERATIONS,
                iter_residual=model_cfg.PARE.ITER_RESIDUAL,
                shape_input_type=model_cfg.PARE.SHAPE_INPUT_TYPE,
                pose_input_type=model_cfg.PARE.POSE_INPUT_TYPE,
                pose_mlp_num_layers=model_cfg.PARE.POSE_MLP_NUM_LAYERS,
                shape_mlp_num_layers=model_cfg.PARE.SHAPE_MLP_NUM_LAYERS,
                pose_mlp_hidden_size=model_cfg.PARE.POSE_MLP_HIDDEN_SIZE,
                shape_mlp_hidden_size=model_cfg.PARE.SHAPE_MLP_HIDDEN_SIZE,
                use_keypoint_features_for_smpl_regression=model_cfg.PARE.USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
                use_heatmaps=model_cfg.DATASET.USE_HEATMAPS,
                use_keypoint_attention=model_cfg.PARE.USE_KEYPOINT_ATTENTION,
                use_postconv_keypoint_attention=model_cfg.PARE.USE_POSTCONV_KEYPOINT_ATTENTION,
                use_scale_keypoint_attention=model_cfg.PARE.USE_SCALE_KEYPOINT_ATTENTION,
                keypoint_attention_act=model_cfg.PARE.KEYPOINT_ATTENTION_ACT,
                use_final_nonlocal=model_cfg.PARE.USE_FINAL_NONLOCAL,
                use_branch_nonlocal=model_cfg.PARE.USE_BRANCH_NONLOCAL,
                use_hmr_regression=model_cfg.PARE.USE_HMR_REGRESSION,
                use_coattention=model_cfg.PARE.USE_COATTENTION,
                num_coattention_iter=model_cfg.PARE.NUM_COATTENTION_ITER,
                coattention_conv=model_cfg.PARE.COATTENTION_CONV,
                use_upsampling=model_cfg.PARE.USE_UPSAMPLING,
                deconv_conv_kernel_size=model_cfg.PARE.DECONV_CONV_KERNEL_SIZE,
                use_soft_attention=model_cfg.PARE.USE_SOFT_ATTENTION,
                num_branch_iteration=model_cfg.PARE.NUM_BRANCH_ITERATION,
                branch_deeper=model_cfg.PARE.BRANCH_DEEPER,
                num_deconv_layers=model_cfg.PARE.NUM_DECONV_LAYERS,
                num_deconv_filters=model_cfg.PARE.NUM_DECONV_FILTERS,
                use_resnet_conv_hrnet=model_cfg.PARE.USE_RESNET_CONV_HRNET,
                use_position_encodings=model_cfg.PARE.USE_POS_ENC,
                use_mean_camshape=model_cfg.PARE.USE_MEAN_CAMSHAPE,
                use_mean_pose=model_cfg.PARE.USE_MEAN_POSE,
                init_xavier=model_cfg.PARE.INIT_XAVIER,
            ).to(self.device)
        else:
            exit()

        return model

    def _load_pretrained_model(self, ckpt):
        # ========= Load pretrained weights ========= #
        state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        pretrained_keys = state_dict.keys()
        new_state_dict = {}
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = state_dict[pk]
            else:
                new_state_dict[pk] = state_dict[pk]

        self.model.load_state_dict(new_state_dict, strict=False)

if __name__ == '__main__':
    pass
