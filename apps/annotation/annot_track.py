'''
  @ Date: 2021-03-28 21:22:38
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-28 21:23:19
  @ FilePath: /EasyMocap/annotation/annot_track.py
'''
import os
from os.path import join
from easymocap.annotator import ImageFolder
from easymocap.annotator import plot_text, plot_bbox_body, vis_active_bbox, vis_line
from easymocap.annotator import AnnotBase
from easymocap.annotator import callback_select_bbox_corner, callback_select_bbox_center, auto_pose_track

def annot_example(path, subs, annot, step):
    for sub in subs:
        # define datasets
        dataset = ImageFolder(path, sub=sub, annot=annot)
        key_funcs = {
            't': auto_pose_track
        }
        callbacks = [callback_select_bbox_corner, callback_select_bbox_center]
        # define visualize
        vis_funcs = [vis_line, plot_bbox_body, vis_active_bbox]
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the sub folder lists when in video mode')
    parser.add_argument('--annot', type=str, default='annots')
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if len(args.sub) == 0:
        args.sub = sorted(os.listdir(join(args.path, 'images')))
    annot_example(args.path, annot=args.annot, subs=args.sub, step=args.step)