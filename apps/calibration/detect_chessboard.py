'''
  @ Date: 2021-07-16 20:13:57
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-07-21 19:56:38
  @ FilePath: /EasyMocap/apps/calibration/detect_chessboard.py
'''
# detect the corner of chessboard
from easymocap.annotator.file_utils import getFileList, read_json, save_json
from tqdm import tqdm
from easymocap.annotator import ImageFolder
from easymocap.annotator.chessboard import getChessboard3d, findChessboardCorners
import numpy as np
from os.path import join
import cv2
import os

def create_chessboard(path, pattern, gridSize, ext):
    print('Create chessboard {}'.format(pattern))
    keypoints3d = getChessboard3d(pattern, gridSize=gridSize)
    keypoints2d = np.zeros((keypoints3d.shape[0], 3))
    imgnames = getFileList(path, ext=ext)
    template = {
        'keypoints3d': keypoints3d.tolist(),
        'keypoints2d': keypoints2d.tolist(),
        'visited': False
    }
    for imgname in tqdm(imgnames, desc='create template chessboard'):
        annname = imgname.replace('images', 'chessboard').replace(ext, '.json')
        annname = join(path, annname)
        if os.path.exists(annname):
            # 覆盖keypoints3d
            data = read_json(annname)
            data['keypoints3d'] = template['keypoints3d']
            save_json(annname, data)
        else:
            save_json(annname, template)

def detect_chessboard(path, out, pattern, gridSize, args):
    create_chessboard(path, pattern, gridSize, ext=args.ext)
    dataset = ImageFolder(path, annot='chessboard', ext=args.ext)
    dataset.isTmp = False
    if args.silent:
        trange = range(len(dataset))
    else:
        trange = tqdm(range(len(dataset)))
    for i in trange:
        imgname, annotname = dataset[i]
        # detect the 2d chessboard
        img = cv2.imread(imgname)
        annots = read_json(annotname)
        show = findChessboardCorners(img, annots, pattern)
        save_json(annotname, annots)
        if show is None:
            if args.debug:
                print('Cannot find {}'.format(imgname))
            continue
        outname = join(out, imgname.replace(path + '/images/', ''))
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, show)

def detect_chessboard_sequence(path, out, pattern, gridSize, args):
    create_chessboard(path, pattern, gridSize, ext=args.ext)
    subs = sorted(os.listdir(join(path, 'images')))
    for sub in subs:
        dataset = ImageFolder(path, sub=sub, annot='chessboard', ext=args.ext)
        dataset.isTmp = False
        nFrames = len(dataset)
        found = np.zeros(nFrames, dtype=np.bool)
        visited = np.zeros(nFrames, dtype=np.bool)
        proposals = []
        init_step = args.max_step
        min_step = args.min_step
        for nf in range(0, nFrames, init_step):
            if nf + init_step < len(dataset):
                proposals.append([nf, nf+init_step])
        while len(proposals) > 0:
            left, right = proposals.pop(0)
            print('Check [{}, {}]'.format(left, right))
            for nf in [left, right]:
                if not visited[nf]:
                    visited[nf] = True
                    imgname, annotname = dataset[nf]
                    # detect the 2d chessboard
                    img = cv2.imread(imgname)
                    annots = read_json(annotname)
                    show = findChessboardCorners(img, annots, pattern)
                    save_json(annotname, annots)
                    if show is None:
                        if args.debug:
                            print('Cannot find {}'.format(imgname))
                        found[nf] = False
                        continue
                    found[nf] = True
                    outname = join(out, imgname.replace(path + '/images/', ''))
                    os.makedirs(os.path.dirname(outname), exist_ok=True)
                    cv2.imwrite(outname, show)
            if not found[left] and not found[right]:
                continue
            mid = (left+right)//2
            if mid == left or mid == right:
                continue
            if mid - left > min_step:
                proposals.append((left, mid))
            if right - mid > min_step:
                proposals.append((mid, right))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
    parser.add_argument('--pattern', type=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])),
        help='The pattern of the chessboard', default=(9, 6))
    parser.add_argument('--grid', type=float, default=0.1, 
        help='The length of the grid size (unit: meter)')
    parser.add_argument('--max_step', type=int, default=50)
    parser.add_argument('--min_step', type=int, default=0)
    
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seq', action='store_true')
    args = parser.parse_args()
    if args.seq:
        detect_chessboard_sequence(args.path, args.out, pattern=args.pattern, gridSize=args.grid, args=args)
    else:
        detect_chessboard(args.path, args.out, pattern=args.pattern, gridSize=args.grid, args=args)