'''
  @ Date: 2021-01-12 17:08:25
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-24 20:57:35
  @ FilePath: /EasyMocapRelease/code/demo_mv1pmf_skel.py
'''
# show skeleton and reprojection
from dataset.mv1pmf import MV1PMF
from dataset.config import CONFIG
from mytools.reconstruction import simple_recon_person, projectN3
# from mytools.robust_triangulate import robust_triangulate
from tqdm import tqdm
import numpy as np
from smplmodel import check_keypoints

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

def get_limb_length(config, keypoints):
    skeleton = {}
    for i, j_ in config['kintree']:
        if j_ == 25:
            j = 7
        elif j_ == 46:
            j = 4
        else:
            j = j_
        key = tuple(sorted([i, j]))
        length, confs = 0, 0
        for nf in range(keypoints.shape[0]):
            limb_length = np.linalg.norm(keypoints[nf, i, :3] - keypoints[nf, j, :3])
            conf = keypoints[nf, [i, j], -1].min()
            length += limb_length * conf
            confs += conf
        limb_length = length/confs
        skeleton[key] = {'mean': limb_length, 'std': limb_length*0.2}
    print('{')
    for key, val in skeleton.items():
        res = '    ({:2d}, {:2d}): {{\'mean\': {:.3f}, \'std\': {:.3f}}}, '.format(*key, val['mean'], val['std'])
        if 'joint_names' in config.keys():
            res += '# {:9s}->{:9s}'.format(config['joint_names'][key[0]], config['joint_names'][key[1]])
        print(res)
    print('}')

def mv1pmf_skel(path, sub, out, mode, args):
    MIN_CONF_THRES = 0.3
    no_img = not (args.vis_det or args.vis_repro)
    config = CONFIG[mode]
    dataset = MV1PMF(path, cams=sub, config=config, mode=mode,
        undis=args.undis, no_img=no_img, out=out)
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    for nf in tqdm(range(start, end), desc='triangulation'):
        images, annots = dataset[nf]
        conf = annots['keypoints'][..., -1]
        conf[conf < MIN_CONF_THRES] = 0
        annots['keypoints'] = check_keypoints(annots['keypoints'], WEIGHT_DEBUFF=1)
        keypoints3d, _, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall, config=config, ret_repro=True)
        # keypoints3d, _, kpts_repro = robust_triangulate(annots['keypoints'], dataset.Pall, config=config, ret_repro=True)
        kp3ds.append(keypoints3d)
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, annots, kpts_repro, nf, sub_vis=args.sub_vis)
    # smooth the skeleton
    kp3ds = np.stack(kp3ds)
    # 计算一下骨长
    # get_limb_length(config, kp3ds)
    # if args.smooth:
    #     kp3ds = smooth_skeleton(kp3ds)
    for nf in tqdm(range(kp3ds.shape[0]), desc='dump'):
        dataset.write_keypoints3d(kp3ds[nf], nf + start)

if __name__ == "__main__":
    from mytools.cmd_loader import load_parser
    parser = load_parser()

    args = parser.parse_args()
    mv1pmf_skel(args.path, args.sub, args.out, args.body, args)