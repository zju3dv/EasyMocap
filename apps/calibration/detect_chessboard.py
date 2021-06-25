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

def detect_chessboard(path, out, pattern, gridSize):
    create_chessboard(path, pattern, gridSize, ext=args.ext)
    dataset = ImageFolder(path, annot='chessboard', ext=args.ext)
    dataset.isTmp = False
    for i in tqdm(range(len(dataset))):
        imgname, annotname = dataset[i]
        # detect the 2d chessboard
        img = cv2.imread(imgname)
        annots = read_json(annotname)
        show = findChessboardCorners(img, annots, pattern)
        save_json(annotname, annots)
        if show is None:
            continue
        outname = join(out, imgname.replace(path + '/images/', ''))
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, show)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
    parser.add_argument('--pattern', type=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])),
        help='The pattern of the chessboard', default=(9, 6))
    parser.add_argument('--grid', type=float, default=0.1, 
        help='The length of the grid size (unit: meter)')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    detect_chessboard(args.path, args.out, pattern=args.pattern, gridSize=args.grid)