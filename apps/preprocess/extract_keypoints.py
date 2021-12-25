'''
  @ Date: 2021-08-19 22:06:22
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-12-02 21:19:41
  @ FilePath: /EasyMocap/apps/preprocess/extract_keypoints.py
'''
import os
from os.path import join
from tqdm import tqdm
import numpy as np

def load_subs(path, subs):
    if len(subs) == 0:
        subs = sorted(os.listdir(join(path, 'images')))
    subs = [sub for sub in subs if os.path.isdir(join(path, 'images', sub))]
    if len(subs) == 0:
        subs = ['']
    return subs

config = {
    'openpose':{
        'root': '',
        'res': 1,
        'hand': False,
        'face': False,
        'vis': False,
        'ext': '.jpg'
    },
    'feet':{
        'root': '',
        'res': 1,
        'hand': False,
        'face': False,
        'vis': False,
        'ext': '.jpg'
    },
    'feetcrop':{
        'root': '',
        'res': 1,
        'hand': False,
        'face': False,
        'vis': False,
        'ext': '.jpg'
    },
    'yolo':{
        'ckpt_path': 'data/models/yolov4.weights',
        'conf_thres': 0.3,
        'box_nms_thres': 0.5, # means keeping the bboxes that IOU<0.5
        'ext': '.jpg',
        'isWild': False,
    },
    'hrnet':{
        'nof_joints': 17,
        'c': 48,
        'checkpoint_path': 'data/models/pose_hrnet_w48_384x288.pth'
    },
    'yolo-hrnet':{},
    'mp-pose':{
        'model_complexity': 2,
        'min_detection_confidence':0.5,
        'min_tracking_confidence': 0.5
    },
    'mp-holistic':{
        'model_complexity': 2,
        # 'refine_face_landmarks': True,
        'min_detection_confidence':0.5,
        'min_tracking_confidence': 0.5
    },
    'mp-handl':{
        'model_complexity': 1,
        'min_detection_confidence':0.3,
        'min_tracking_confidence': 0.1,
        'static_image_mode': False,
    },
    'mp-handr':{
        'model_complexity': 1,
        'min_detection_confidence':0.3,
        'min_tracking_confidence': 0.1,
        'static_image_mode': False,
    }
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default=None, help="the path of data")
    parser.add_argument('--subs', type=str, nargs='+', default=[], help="the path of data")
    # Output Control
    parser.add_argument('--annot', type=str, default='annots', 
        help="sub directory name to store the generated annotation files, default to be annots")
    # Detection Control
    parser.add_argument('--mode', type=str, default='openpose', choices=[
        'openpose', 'feet', 'feetcrop', 'yolo-hrnet', 'yolo', 'hrnet', 
        'mp-pose', 'mp-holistic', 'mp-handl', 'mp-handr', 'mp-face'], 
        help="model to extract joints from image")
    # Openpose
    parser.add_argument('--openpose', type=str, 
        default='/media/qing/Project/openpose')
    parser.add_argument('--openpose_res', type=float, default=1)
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--hand', action='store_true')
    parser.add_argument('--face', action='store_true')
    parser.add_argument('--wild', action='store_true',
        help='remove crowd class of yolo')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    config['yolo']['isWild'] = args.wild
    mode = args.mode
    subs = load_subs(args.path, args.subs)
    global_tasks = []
    for sub in subs:
        config[mode]['force'] = args.force
        image_root = join(args.path, 'images', sub)
        annot_root = join(args.path, args.annot, sub)
        tmp_root = join(args.path, mode, sub)
        if os.path.exists(annot_root) and not args.force:
            # check the number of annots and images
            if len(os.listdir(image_root)) == len(os.listdir(annot_root)):
                print('[Skip] detection {}'.format(annot_root))
                continue
        if mode == 'openpose':
            from easymocap.estimator.openpose_wrapper import extract_2d
            config[mode]['root'] = args.openpose
            config[mode]['hand'] = args.hand
            config[mode]['face'] = args.face
            config[mode]['res'] = args.openpose_res
            config[mode]['ext'] = args.ext
            global_tasks = extract_2d(image_root, annot_root, tmp_root, config[mode])
        elif mode == 'feet':
            from easymocap.estimator.openpose_wrapper import FeetEstimator
            config[mode]['openpose'] = args.openpose
            estimator = FeetEstimator(openpose=args.openpose)
            estimator.detect_foot(image_root, annot_root, args.ext)
        elif mode == 'yolo':
            from easymocap.estimator.yolohrnet_wrapper import extract_bbox
            config[mode]['ext'] = args.ext
            extract_bbox(image_root, annot_root, **config[mode])
        elif mode == 'hrnet':
            from easymocap.estimator.yolohrnet_wrapper import extract_hrnet
            config[mode]['ext'] = args.ext
            extract_hrnet(image_root, annot_root, **config[mode])
        elif mode == 'yolo-hrnet':
            from easymocap.estimator.yolohrnet_wrapper import extract_yolo_hrnet
            extract_yolo_hrnet(image_root, annot_root, args.ext, config['yolo'], config['hrnet'])
        elif mode in ['mp-pose', 'mp-holistic', 'mp-handl', 'mp-handr', 'mp-face']:
            from easymocap.estimator.mediapipe_wrapper import extract_2d
            config[mode]['ext'] = args.ext
            extract_2d(image_root, annot_root, config[mode], mode=mode.replace('mp-', ''))
        if mode == 'feetcrop':
            from easymocap.estimator.openpose_wrapper import FeetEstimatorByCrop
            config[mode]['openpose'] = args.openpose
            estimator = FeetEstimatorByCrop(openpose=args.openpose)
            estimator.detect_foot(image_root, annot_root, args.ext)
    for task in global_tasks:
        task.join()