from .config import Config
from ..mytools import Timer
from ..pyfitting import optimizeMirrorSoft, optimizeMirrorDirect

def load_weight_mirror(model, opts):
    if model == 'smpl':
        weight = {
            'k2d': 2e-4, 
            'init_poses': 1e-3, 'init_shapes': 1e-2,
            'smooth_body': 5e-1, 'smooth_poses': 1e-1,
            'par_self': 5e-2, 'ver_self': 2e-2,
            'par_mirror': 5e-2
        }
    elif model == 'smplh':
        weight = {'repro': 1, 'repro_hand': 0.1,
                'init_poses': 10., 'init_shapes': 10., 'init_Th': 0., 
                'reg_poses': 0., 'reg_shapes':10., 'reg_poses_zero': 10.,
                # 'smooth_poses': 100., 'smooth_Rh': 1000., 'smooth_Th': 1000.,
                'parallel_self': 10., 'vertical_self': 10., 'parallel_mirror': 0.
        }
    elif model == 'smplx':
        weight = {'repro': 1, 'repro_hand': 0.2, 'repro_face': 1.,
                'init_poses': 1., 'init_shapes': 0., 'init_Th': 0., 
                'reg_poses': 0., 'reg_shapes': 10., 'reg_poses_zero': 10., 'reg_head': 1., 'reg_expression': 1.,
                # 'smooth_body': 1., 'smooth_hand': 10.,
                # 'smooth_poses_l1': 1.,
                'parallel_self': 1., 'vertical_self': 1., 'parallel_mirror': 0.}
    else:
        weight = {}
    for key in opts.keys():
        if key in weight.keys():
            weight[key] = opts[key]
    return weight

def multi_stage_optimize(body_model, body_params, bboxes, keypoints2d, Pall, normal, args):
    weight = load_weight_mirror(args.model, args.opts)
    config = Config()
    config.device = body_model.device
    config.verbose = args.verbose
    config.OPT_R = True
    config.OPT_T = True
    config.OPT_SHAPE = True
    with Timer('Optimize 2D Pose/{} frames'.format(keypoints2d.shape[1]), not args.verbose):
        if args.direct:
            config.OPT_POSE = False
            body_params = optimizeMirrorDirect(body_model, body_params, bboxes, keypoints2d, Pall, normal, weight, config)
            config.OPT_POSE = True
            body_params = optimizeMirrorDirect(body_model, body_params, bboxes, keypoints2d, Pall, normal, weight, config)
        else:
            config.OPT_POSE = False
            body_params = optimizeMirrorSoft(body_model, body_params, bboxes, keypoints2d, Pall, normal, weight, config)
            config.OPT_POSE = True
            body_params = optimizeMirrorSoft(body_model, body_params, bboxes, keypoints2d, Pall, normal, weight, config)
    return body_params