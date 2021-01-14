'''
  @ Date: 2021-01-12 17:08:25
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-14 17:08:05
  @ FilePath: /EasyMocap/code/demo_mv1pmf_skel.py
'''
# show skeleton and reprojection
from dataset.mv1pmf import MV1PMF
from dataset.config import CONFIG
from mytools.reconstruction import simple_recon_person, projectN3
from tqdm import tqdm
import numpy as np

def smooth_skeleton(skeleton):
    # nFrames, nJoints, 4: [[(x, y, z, c)]]
    nFrames = skeleton.shape[0]
    span = 2
    # results = np.zeros((nFrames-2*span, skeleton.shape[1], skeleton.shape[2]))
    origin = skeleton[span:nFrames-span, :, :].copy()
    conf = origin[:, :, 3:4].copy()
    skel = origin[:, :, :3] * conf
    base_start = span
    for i in range(-span, span+1):
        sample = skeleton[base_start+i:base_start+i+skel.shape[0], :, :]
        skel += sample[:, :, :3] * sample[:, :, 3:]
        conf += sample[:, :, 3:]
    not_f, not_j, _ = np.where(conf<0.1)
    skel[not_f, not_j, :] = 0.
    conf[not_f, not_j, :] = 1.
    skel = skel/conf
    skeleton[span:nFrames-span, :, :3] = skel
    return skeleton

def mv1pmf_skel(path, sub, out, mode, args):
    MIN_CONF_THRES = 0.5
    no_img = not (args.vis_det or args.vis_repro)
    dataset = MV1PMF(path, cams=sub, config=CONFIG[mode], add_hand_face=args.add_hand_face,
        undis=args.undis, no_img=no_img, out=out)
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    for nf in tqdm(range(start, end), desc='triangulation'):
        images, annots = dataset[nf]
        conf = annots['keypoints'][..., -1]
        conf[conf < MIN_CONF_THRES] = 0
        keypoints3d, _, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall, ret_repro=True)
        kp3ds.append(keypoints3d)
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, annots, kpts_repro, nf, sub_vis=args.sub_vis)
    # smooth the skeleton
    kp3ds = np.stack(kp3ds)
    if args.smooth:
        kp3ds = smooth_skeleton(kp3ds)
    for nf in tqdm(range(kp3ds.shape[0]), desc='dump'):
        dataset.write_keypoints3d(kp3ds[nf], nf + start)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('multi_view one_person multi_frame skel')
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the sub folder lists when in video mode')
    parser.add_argument('--start', type=int, default=0,
        help='frame start')
    parser.add_argument('--end', type=int, default=10000,
        help='frame end')    
    parser.add_argument('--step', type=int, default=1,
        help='frame step')
    parser.add_argument('--body', type=str, default='body25', choices=['body15', 'body25', 'total'])
    parser.add_argument('--undis', action='store_true')
    parser.add_argument('--add_hand_face', action='store_true')
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--vis_det', action='store_true')
    parser.add_argument('--vis_repro', action='store_true')
    parser.add_argument('--sub_vis', type=str, nargs='+', default=[],
        help='the sub folder lists for visualization')

    args = parser.parse_args()
    mv1pmf_skel(args.path, args.sub, args.out, args.body, args)