'''
  @ Date: 2021-01-13 20:38:33
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-14 16:59:06
  @ FilePath: /EasyMocapRelease/scripts/preprocess/extract_video.py
'''
import os
import cv2
from os.path import join
from tqdm import tqdm
from glob import glob
import numpy as np

mkdir = lambda x: os.makedirs(x, exist_ok=True)

def extract_video(videoname, path, start=0, end=10000, step=1):
    base = os.path.basename(videoname).replace('.mp4', '')
    if not os.path.exists(videoname):
        return base
    video = cv2.VideoCapture(videoname)
    outpath = join(path, 'images', base)
    if os.path.exists(outpath) and len(os.listdir(outpath)) > 0:
        return base
    else:
        os.makedirs(outpath)
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for cnt in tqdm(range(totalFrames)):
        ret, frame = video.read()
        if cnt < start:continue
        if cnt > end:break
        if not ret:break
        cv2.imwrite(join(outpath, '{:06d}.jpg'.format(cnt)), frame)
    video.release()
    return base

def extract_2d(openpose, image, keypoints, render):
    if not os.path.exists(keypoints):
        cmd = './build/examples/openpose/openpose.bin --image_dir {} --write_json {} --display 0'.format(image, keypoints)
        if args.handface:
            cmd = cmd + ' --hand --face'
        if args.render:
            cmd = cmd + ' --write_images {}'.format(render)
        else:
            cmd = cmd + ' --render_pose 0'
        os.chdir(openpose)
        os.system(cmd)

import json
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def create_annot_file(annotname, imgname):
    assert os.path.exists(imgname), imgname
    img = cv2.imread(imgname)
    height, width = img.shape[0], img.shape[1]
    imgnamesep = imgname.split(os.sep)
    filename = os.sep.join(imgnamesep[imgnamesep.index('images'):])
    annot = {
        'filename':filename,
        'height':height,
        'width':width,
        'annots': [],
        'isKeyframe': False
    }
    save_json(annotname, annot)
    return annot

def bbox_from_openpose(keypoints, rescale=1.2, detection_thresh=0.01):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    bbox_size = bbox_size * rescale
    bbox = [
        center[0] - bbox_size[0]/2, 
        center[1] - bbox_size[1]/2,
        center[0] + bbox_size[0]/2, 
        center[1] + bbox_size[1]/2,
        keypoints[valid, :2].mean()
    ]
    return bbox

def load_openpose(opname):
    mapname = {'face_keypoints_2d':'face2d', 'hand_left_keypoints_2d':'handl2d', 'hand_right_keypoints_2d':'handr2d'}
    assert os.path.exists(opname), opname
    data = read_json(opname)
    out = []
    pid = 0
    for i, d in enumerate(data['people']):
        keypoints = d['pose_keypoints_2d']
        keypoints = np.array(keypoints).reshape(-1, 3)
        annot = {
            'bbox': bbox_from_openpose(keypoints),
            'personID': pid + i,
            'keypoints': keypoints.tolist(),
            'isKeyframe': False
        }
        for key in ['face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            if len(d[key]) == 0:
                continue
            kpts = np.array(d[key]).reshape(-1, 3)
            annot[mapname[key]] = kpts.tolist()
        out.append(annot)
    return out
    
def convert_from_openpose(src, dst):
    # convert the 2d pose from openpose
    inputlist = sorted(os.listdir(src))
    for inp in tqdm(inputlist):
        annots = load_openpose(join(src, inp))
        base = inp.replace('_keypoints.json', '')
        annotname = join(dst, base+'.json')
        imgname = annotname.replace('annots', 'images').replace('.json', '.jpg')
        if not os.path.exists(imgname):
            os.remove(join(src, inp))
            continue
        annot = create_annot_file(annotname, imgname)
        annot['annots'] = annots
        save_json(annotname, annot)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default=None)
    parser.add_argument('--handface', action='store_true')
    parser.add_argument('--openpose', type=str, 
        default='/media/qing/Project/openpose')
    parser.add_argument('--render', action='store_true', help='use to render the openpose 2d')
    parser.add_argument('--no2d', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if os.path.isdir(args.path):
        videos = sorted(glob(join(args.path, 'videos', '*.mp4')))
        subs = []
        for video in videos:
            basename = extract_video(video, args.path)
            subs.append(basename)
        if not args.no2d:
            os.makedirs(join(args.path, 'openpose'), exist_ok=True)
            for sub in subs:
                annot_root = join(args.path, 'annots', sub)
                if os.path.exists(annot_root):
                    continue
                extract_2d(args.openpose, join(args.path, 'images', sub), 
                    join(args.path, 'openpose', sub), 
                    join(args.path, 'openpose_render', sub))
                convert_from_openpose(
                    src=join(args.path, 'openpose', sub),
                    dst=annot_root
                )
    else:
        print(args.path, ' not exists')
