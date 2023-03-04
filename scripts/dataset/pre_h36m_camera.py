import os
from os.path import join
from glob import glob
import numpy as np
import cv2
from easymocap.mytools.camera_utils import write_camera

def rotation_matrix(args):

    (x, y, z) = args

    X = np.vstack([[1, 0, 0], [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    Y = np.vstack([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Z = np.vstack([[np.cos(z), -np.sin(z), 0], [np.sin(z),
                                                np.cos(z), 0], [0, 0, 1]])

    return (X.dot(Y)).dot(Z)

def read_cam_parameters(xml_path, sbj_id, cam_id):
    import xml.etree.ElementTree

    # use the notation from 0 -- more practical to access array
    sbj_id = sbj_id - 1
    cam_id = cam_id - 1

    n_sbjs = 11
    n_cams = 4

    root = xml.etree.ElementTree.parse(xml_path).getroot()

    for child in root:
        if child.tag == 'w0':
            all_cameras = child.text
            tokens = all_cameras.split(' ')
            tokens[0] = tokens[0].replace('[', '')
            tokens[-1] = tokens[-1].replace(']', '')

            start = (cam_id * n_sbjs) * 6 + sbj_id * 6
            extrs = tokens[start:start + 6]

            start = (n_cams * n_sbjs * 6) + cam_id * 9
            intrs = tokens[start:start + 9]

            rot = rotation_matrix(np.array(extrs[:3], dtype=float))

            rt = rot
            t = np.array(extrs[3:], dtype=float)

            f = np.array(intrs[:2], dtype=float)
            c = np.array(intrs[2:4], dtype=float)

            distortion = np.array(intrs[4:], dtype=float)

            k = np.hstack((distortion[:2], distortion[3:5], distortion[2:3]))
            return (rt, t, f, c, k)

def process_camera(xml_path, seq, act, cams):
    cameras = {}
    for i, cam in enumerate(cams, 1):
        (rt, t, f, c, k) = read_cam_parameters(xml_path, int(seq.replace('S', '')), i)
        K = np.eye(3)
        K[0, 0] = f[0]
        K[1, 1] = f[1]
        K[0, 2] = c[0]
        K[1, 2] = c[1]
        # camera center
        T = t.reshape(3, 1) / 1000
        T = -np.dot(rt, T)
        cameras[cam] = {
            'K': K,
            'R': rt,
            'T': T,
            'dist': k.reshape(1, 5)
        }
    return cameras

def convert_h36m_easymocap(H36M_ROOT, OUT_ROOT, seqs, cams):
    xml_path = join(H36M_ROOT, 'metadata.xml')
    for seq in seqs:
        print('convert {}'.format(seq))
        # path with GT 3D pose
        pose_path = join(H36M_ROOT, seq, 'MyPoseFeatures', 'D3_Positions_mono')
        action_list = glob(join(pose_path, '*.cdf'))
        action_list = list(set([os.path.basename(seq).split('.')[0] for seq in action_list]))
        action_list.sort()

        for action in action_list:
            print('  ', action)
            outdir = join(OUT_ROOT, seq, action)
            # conver cameras
            cameras = process_camera(xml_path, seq, action, cams)
            write_camera(cameras, outdir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to h36m dataset')
    parser.add_argument('out', type=str, help='path to output')
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    seqs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    cams = ['54138969', '55011271', '58860488', '60457274']
    XML_PATH = join(args.path, 'metadata.xml')
    convert_h36m_easymocap(args.path, args.out, seqs, cams)