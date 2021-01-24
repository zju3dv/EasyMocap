'''
  @ Date: 2021-01-15 12:09:27
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-24 20:57:22
  @ FilePath: /EasyMocapRelease/code/mytools/cmd_loader.py
'''

import argparse

def load_parser():
    parser = argparse.ArgumentParser('EasyMocap commond line tools')
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--annot', type=str, default=None)
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the sub folder lists when in video mode')
    parser.add_argument('--start', type=int, default=0,
        help='frame start')
    parser.add_argument('--end', type=int, default=10000,
        help='frame end')    
    parser.add_argument('--step', type=int, default=1,
        help='frame step')
    # 
    # keypoints and body model
    # 
    parser.add_argument('--body', type=str, default='body25', choices=['body15', 'body25', 'bodyhand', 'bodyhandface', 'total'])
    parser.add_argument('--model', type=str, default='smpl', choices=['smpl', 'smplh', 'smplx', 'mano'])
    parser.add_argument('--gender', type=str, default='neutral', 
        choices=['neutral', 'male', 'female'])
    # 
    # visualization part
    # 
    parser.add_argument('--vis_det', action='store_true')
    parser.add_argument('--vis_repro', action='store_true')
    parser.add_argument('--undis', action='store_true')
    parser.add_argument('--sub_vis', type=str, nargs='+', default=[],
        help='the sub folder lists for visualization')
    # 
    # debug
    # 
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser