'''
  @ Date: 2021-04-02 20:33:14
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-06 13:10:53
  @ FilePath: /EasyMocap/scripts/dataset/pre_panoptic.py
'''
# process script for CMU-Panoptic data
import numpy as np
import json
from glob import glob
from os.path import join
import os
from easymocap.mytools import write_camera, read_json, save_json
from easymocap.dataset import CONFIG

import shutil
from tqdm import tqdm, trange

SCALE = 100
def convert_camera(inp, out):
    camnames = glob(join(inp, '*.json'))
    assert len(camnames) == 1, camnames
    # Load camera calibration parameters
    with open(camnames[0]) as cfile:
        calib = json.load(cfile)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras_ = {cam['name']:cam for cam in calib['cameras']}
    cameras = {}
    # Convert data into numpy arrays for convenience
    for k, cam in cameras_.items():    
        if cam['type'] != 'hd':
            continue
        cam['K'] = np.array(cam['K'])
        cam['dist'] = np.array(cam['distCoef']).reshape(1, -1)
        cam['R'] = np.array(cam['R'])
        cam['T'] = np.array(cam['t']).reshape((3,1))/SCALE
        cam = {key:cam[key] for key in ['K', 'dist', 'R', 'T']}
        cameras[k] = cam
    write_camera(cameras, out)

def copy_videos(inp, out):
    outdir = join(out, 'videos')
    os.makedirs(outdir, exist_ok=True)
    hdnames = os.listdir(join(inp, 'hdVideos'))
    for hdname in tqdm(hdnames):
        outname = join(outdir, hdname.replace('hd_', ''))
        shutil.copy(join(inp, 'hdVideos', hdname), outname)

def convert_keypoints3d(inp, out):
    bodynames = join(inp, 'hdPose3d_stage1_coco19', 'body3DScene_{:08d}.json')
    handnames = join(inp, 'hdHand3d', 'handRecon3D_hd{:08d}.json')
    out = join(out, 'keypoints3d')
    os.makedirs(out, exist_ok=True)
    names_i = CONFIG['panoptic']['joint_names']
    names_o = CONFIG['body25']['joint_names']
    commons = [i for i in names_o if i in names_i]
    idx_i = [names_i.index(i) for i in commons]
    idx_o = [names_o.index(i) for i in commons]
    use_hand = True
    if use_hand:
        zero_body = np.zeros((25 + 21 + 21, 4))
    else:
        zero_body = np.zeros((25, 4))
    for i in trange(10000):
        bodyname = bodynames.format(i)
        if not os.path.exists(bodyname):
            continue
        bodies = read_json(bodyname)
        results = []
        for data in bodies['bodies']:
            pid = data['id']
            joints19 = np.array(data['joints19']).reshape(-1, 4)
            joints19[:, :3] /= SCALE
            keypoints3d = zero_body.copy()
            keypoints3d[idx_o, :] = joints19[idx_i, :]
            results.append({'id': pid, 'keypoints3d': keypoints3d})
        handname = handnames.format(i)
        hands = read_json(handname)
        lwrists = np.stack([res['keypoints3d'][7] for res in results])
        left_valid = np.zeros(len(results)) + 0.2
        rwrists = np.stack([res['keypoints3d'][4] for res in results])
        right_valid = np.zeros(len(results)) + 0.2
        for data in hands['people']:
            pid = data['id']
            if 'left_hand' in data.keys():
                left_p = np.array(data['left_hand']['landmarks']).reshape((-1,3))
                left_v = np.array(data['left_hand']['averageScore']).reshape((-1,1))
                left = np.hstack((left_p/SCALE, left_v))
                if left[0, -1] > 0 and (left_v > 0).sum() > 10:
                    dist = np.linalg.norm(left[:1, :3] - lwrists[:, :3], axis=1)
                    dist_min, pid = dist.min(), dist.argmin()
                    if left_valid[pid] > dist_min:
                        left_valid[pid] = dist_min
                        results[pid]['keypoints3d'][25:25+21, :] = left
            if 'right_hand' in data.keys():
                right_p = np.array(data['right_hand']['landmarks']).reshape((-1,3))
                right_v = np.array(data['right_hand']['averageScore']).reshape((-1,1))
                right = np.hstack((right_p/SCALE, right_v))
                if right[0, -1] > 0 and (right_v > 0).sum() > 10:
                    dist = np.linalg.norm(right[:1, :3] - rwrists[:, :3], axis=1)
                    dist_min, pid = dist.min(), dist.argmin()
                    if right_valid[pid] > dist_min:
                        right_valid[pid] = dist_min
                        results[pid]['keypoints3d'][25+21:25+21+21, :] = right
            # find the correspondent people
        outname = join(out, '{:06d}.json'.format(i))
        # results = [val for key, val in results.items()]
        for res in results:
            res['keypoints3d'] = res['keypoints3d'].tolist()
        save_json(outname, results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inp', type=str)
    parser.add_argument('--camera', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.camera:
        convert_camera(args.inp, args.inp)