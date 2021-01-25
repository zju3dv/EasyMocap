'''
  @ Date: 2021-01-13 20:38:33
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-25 14:41:56
  @ FilePath: /EasyMocap/scripts/preprocess/extract_video.py
'''
import os, sys
import cv2
from os.path import join
from tqdm import tqdm
from glob import glob
import numpy as np
code_path = join(os.path.dirname(__file__), '..', '..', 'code')
sys.path.append(code_path)

mkdir = lambda x: os.makedirs(x, exist_ok=True)

def extract_video(videoname, path, start, end, step):
    base = os.path.basename(videoname).replace('.mp4', '')
    if not os.path.exists(videoname):
        return base
    outpath = join(path, 'images', base)
    if os.path.exists(outpath) and len(os.listdir(outpath)) > 0:
        return base
    else:
        os.makedirs(outpath)
    video = cv2.VideoCapture(videoname)
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
        os.makedirs(keypoints, exist_ok=True)
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
        keypoints[valid, 2].mean()
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

def detect_frame(detector, img, pid=0):
    lDetections = detector.detect([img])[0]
    annots = []
    for i in range(len(lDetections)):
        annot = {
            'bbox': [float(d) for d in lDetections[i]['bbox']],
            'personID': pid + i,
            'keypoints': lDetections[i]['keypoints'].tolist(),
            'isKeyframe': True
        }
        annots.append(annot)
    return annots
    
def extract_yolo_hrnet(image_root, annot_root):
    imgnames = sorted(glob(join(image_root, '*.jpg')))
    import torch
    device = torch.device('cuda')
    from estimator.detector import Detector
    config = {
        'yolov4': {
              'ckpt_path': 'data/models/yolov4.weights',
              'conf_thres': 0.3,
              'box_nms_thres': 0.5 # 阈值=0.9，表示IOU 0.9的不会被筛掉
        },
        'hrnet':{
            'nof_joints': 17,
            'c': 48,
            'checkpoint_path': 'data/models/pose_hrnet_w48_384x288.pth'
        },
        'detect':{
            'MIN_PERSON_JOINTS': 10,
            'MIN_BBOX_AREA': 5000,
            'MIN_JOINTS_CONF': 0.3,
            'MIN_BBOX_LEN': 150
        }
    }
    detector = Detector('yolo', 'hrnet', device, config)
    for nf, imgname in enumerate(tqdm(imgnames)):
        annotname = join(annot_root, os.path.basename(imgname).replace('.jpg', '.json'))
        annot = create_annot_file(annotname, imgname)
        img0 = cv2.imread(imgname)
        annot['annots'] = detect_frame(detector, img0, 0)
        for i in range(len(annot['annots'])):
            x = annot['annots'][i]
            x['area'] = max(x['bbox'][2] - x['bbox'][0], x['bbox'][3] - x['bbox'][1])**2
        annot['annots'].sort(key=lambda x:-x['area'])
        # 重新赋值人的ID
        for i in range(len(annot['annots'])):
            annot['annots'][i]['personID'] = i
        save_json(annotname, annot)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default=None)
    parser.add_argument('--mode', type=str, default='openpose', choices=['openpose', 'yolo-hrnet'])
    parser.add_argument('--handface', action='store_true')
    parser.add_argument('--openpose', type=str, 
        default='/media/qing/Project/openpose')
    parser.add_argument('--render', action='store_true', help='use to render the openpose 2d')
    parser.add_argument('--no2d', action='store_true')
    parser.add_argument('--start', type=int, default=0,
        help='frame start')
    parser.add_argument('--end', type=int, default=10000,
        help='frame end')    
    parser.add_argument('--step', type=int, default=1,
        help='frame step')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    mode = args.mode
    
    if os.path.isdir(args.path):
        videos = sorted(glob(join(args.path, 'videos', '*.mp4')))
        subs = []
        for video in videos:
            basename = extract_video(video, args.path, start=args.start, end=args.end, step=args.step)
            subs.append(basename)
        print('cameras: ', ' '.join(subs))
        if not args.no2d:
            for sub in subs:
                image_root = join(args.path, 'images', sub)
                annot_root = join(args.path, 'annots', sub)
                if os.path.exists(annot_root):
                    print('skip ', annot_root)
                    continue
                if mode == 'openpose':
                    extract_2d(args.openpose, image_root, 
                        join(args.path, 'openpose', sub), 
                        join(args.path, 'openpose_render', sub))
                    convert_from_openpose(
                        src=join(args.path, 'openpose', sub),
                        dst=annot_root
                    )
                elif mode == 'yolo-hrnet':
                    extract_yolo_hrnet(image_root, annot_root)
    else:
        print(args.path, ' not exists')
