'''
  @ Date: 2021-03-28 21:23:34
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-05-24 14:27:46
  @ FilePath: /EasyMocapPublic/apps/annotation/annot_keypoints.py
'''
from easymocap.annotator.basic_visualize import capture_screen, plot_skeleton_factory, resize_to_screen
import os
from os.path import join
import numpy as np
from easymocap.annotator import ImageFolder
from easymocap.annotator import AnnotBase
from easymocap.annotator import callback_select_bbox_corner, callback_select_bbox_center
from easymocap.annotator import plot_text, plot_bbox_body, plot_bbox_factory, vis_active_bbox, vis_bbox, plot_skeleton
from easymocap.annotator.keypoints_callback import callback_select_joints
from easymocap.annotator.keypoints_keyboard import set_unvisible, set_unvisible_according_previous, set_face_unvisible, check_track, mirror_keypoints2d, mirror_keypoints2d_leg

class Estimator:
    def __init__(self) -> None:
        import torch
        device = torch.device('cuda')
        from easymocap.estimator.HRNet import SimpleHRNet
        config = {
            'nof_joints': 17,
            'c': 48,
            'checkpoint_path': 'data/models/pose_hrnet_w48_384x288.pth'
        }
        self.pose_estimator = SimpleHRNet(device=device, **config)

    def _detect_with_bbox(self, param, rot):
        select = param['select']['bbox']
        if select == -1:
            return 0
        img = param['img0'].copy()
        annots = param['annots']['annots'][select]
        bboxes = [annots['bbox']]
        res = self.pose_estimator(img, bboxes, rot=rot)[0]
        # annots['keypoints'][:19] = res[:19].tolist()
        annots['keypoints'] = res.tolist()
        return res
    
    def _detect_with_previous(self, annotator, param, sigma):
        select = param['select']['bbox']
        if select == -1:
            return 0
        annots = param['annots']['annots'][select]
        pid = annots['personID']
        previous = annotator.previous()
        found = [d for d in previous['annots'] if d['personID'] == pid]
        if len(found) == 0:
            print('[Info] Not found {} in previous frame'.format(pid))
            return 0
        keypoints = np.array(found[0]['keypoints'])[None]
        bboxes = [annots['bbox']]
        img = param['img0'].copy()
        res = self.pose_estimator.predict_with_previous(img, bboxes, keypoints, sigma)[0]
        # annots['keypoints'][:19] = res[:19].tolist()
        annots['keypoints'] = res.tolist()
        return res
    
    def detect_with_previous_slow(self, annotator, param):
        "detect_with_previous_slow"
        self._detect_with_previous(annotator, param, sigma=1)
    
    def detect_with_previous_mid(self, annotator, param):
        "detect_with_previous_mid"
        self._detect_with_previous(annotator, param, sigma=3)

    def detect_with_previous_fast(self, annotator, param):
        self._detect_with_previous(annotator, param, sigma=5)

    def detect_with_bbox(self, annotator, param):
        "detect"
        self._detect_with_bbox(param, 0)
    
    def detect_with_bbox90(self, annotator, param):
        "detect 90"
        self._detect_with_bbox(param, 90)
    
    def detect_with_bbox180(self, annotator, param):
        "detect 90"
        self._detect_with_bbox(param, 180)
    
    def detect_with_bbox270(self, annotator, param):
        "detect 90"
        self._detect_with_bbox(param, -90)

def annot_example(path, sub, image, annot, step, args):
    # define datasets
    dataset = ImageFolder(path, sub=sub, image=image, annot=annot)
    key_funcs = {
        'v': set_unvisible,
        'V': set_unvisible_according_previous,
        # 'f': set_face_unvisible,
        'c': check_track,
        'm': mirror_keypoints2d,
        'M': mirror_keypoints2d_leg,
    }
    if args.hrnet:
        estimator = Estimator()
        key_funcs['e'] = estimator.detect_with_bbox
        key_funcs['r'] = estimator.detect_with_bbox90
        key_funcs['t'] = estimator.detect_with_bbox180
        key_funcs['y'] = estimator.detect_with_bbox270
        # key_funcs['g'] = estimator.detect_with_previous_slow
        key_funcs['j'] = estimator.detect_with_previous_mid
    # callback of bounding box
    callbacks = [callback_select_bbox_corner, callback_select_bbox_center, callback_select_joints]
    # callback of keypoints

    # define visualize
    vis_funcs = [plot_skeleton_factory('body25'), vis_bbox, vis_active_bbox]
    if args.hand:
        vis_funcs += [plot_bbox_factory('bbox_handl2d'), plot_bbox_factory('bbox_handr2d'), plot_bbox_factory('bbox_face2d')]
        vis_funcs += [plot_skeleton_factory('handl'), plot_skeleton_factory('handr'), plot_skeleton_factory('face')]
    vis_funcs += [resize_to_screen, plot_text, capture_screen]
    # construct annotations
    annotator = AnnotBase(
        dataset=dataset, 
        key_funcs=key_funcs,
        vis_funcs=vis_funcs,
        callbacks=callbacks,
        name=sub,
        step=step)
    while annotator.isOpen:
        annotator.run()

if __name__ == "__main__":
    from easymocap.annotator import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--hand', action='store_true')
    parser.add_argument('--hrnet', action='store_true', 
        help='loading HRNet model to detect human pose')
    args = parse_parser(parser)
    for sub in args.sub:
        annot_example(args.path, image=args.image, annot=args.annot, sub=sub, step=args.step, args=args)