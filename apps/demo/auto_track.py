'''
  @ Date: 2021-06-25 15:59:35
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-28 10:32:24
  @ FilePath: /EasyMocapRelease/apps/demo/auto_track.py
'''
from easymocap.assignment.track import Track2D, Track3D

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--track3d', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    cfg = {
        'path': args.path,
        'out': args.out,
        'WINDOW_SIZE': 10,
        'MIN_FRAMES': 10,
        'SMOOTH_SIZE': 5
    }
    if args.track3d:
        tracker = Track3D(with2d=False, **cfg)
    else:
        tracker = Track2D(**cfg)
    tracker.auto_track()