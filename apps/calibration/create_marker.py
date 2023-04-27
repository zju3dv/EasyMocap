'''
  @ Date: 2021-10-07 15:04:23
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-05-27 16:50:55
  @ FilePath: /EasyMocapPublic/apps/calibration/create_marker.py
'''
# create multiple markers
from easymocap.annotator.file_utils import getFileList, read_json, save_json
from os.path import join
import os
from tqdm import tqdm

def create_markers(path, name, N, N_group):
    outname = join(path, name)
    if os.path.exists(outname):
        results = read_json(outname)
        N_ = len(results['keypoints3d'])
        if N == N_:
            return 0
    results = {
        'keypoints3d': [[0., 0., 0.] for _ in range(N)],
        'lines': [[i, i+1] if (i+1)%N_group!=0 else [i, i-N_group+1] for i in range(N-1) ]
    }
    if N < 5:
        results['lines'].append([args.N-1, 0])
    save_json(outname, results)

def create_corners(path, grid, image='images', ext='.jpg', overwrite=True):
    imgnames = getFileList(join(path, image), ext=ext)
    keypoints3d = [
        [0., 0., 0.],
        [grid[0], 0., 0.],
        [grid[0], grid[1], 0.],
        [0., grid[1], 0.],
    ]
    template = {
        'keypoints3d': keypoints3d,
        'keypoints2d': [[0.,0.,0.] for _ in range(4)],
        'pattern': (2, 2),
        'grid_size': grid,
        'visited': False
    }
    for imgname in tqdm(imgnames, desc='create template chessboard'):
        annname = imgname.replace(ext, '.json')
        annname = join(path, args.annot, annname)
        if os.path.exists(annname) and overwrite:
            # 覆盖keypoints3d
            data = read_json(annname)
            data['keypoints3d'] = template['keypoints3d']
            data['grid_size'] = grid
            save_json(annname, data)
        elif os.path.exists(annname) and not overwrite:
            continue
        else:
            save_json(annname, template)
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--name', type=str, default='calib.json')
    parser.add_argument('--annot', type=str, default='chessboard')
    parser.add_argument('--N', type=int)
    parser.add_argument('--N_group', type=int, default=5)
    parser.add_argument('--grid', type=float, nargs=2, required=True, help='set the length of the grid')
    parser.add_argument('--corner', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.corner:
        create_corners(args.path, args.grid, overwrite=args.overwrite)
    else:
        create_markers(args.path, args.name, args.N, args.N_group)