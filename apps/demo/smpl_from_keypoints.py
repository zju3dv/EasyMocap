'''
  @ Date: 2021-06-14 22:27:05
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-28 10:33:26
  @ FilePath: /EasyMocapRelease/apps/demo/smpl_from_keypoints.py
'''
# This is the script of fitting SMPL to 3d(+2d) keypoints
from easymocap.dataset import CONFIG
from easymocap.mytools import Timer
from easymocap.smplmodel import load_model, select_nf
from easymocap.mytools.reader import read_keypoints3d_all
from easymocap.mytools.file_utils import write_smpl
from easymocap.pipeline.weight import load_weight_pose, load_weight_shape
from easymocap.pipeline import smpl_from_keypoints3d
import os
from os.path import join
from tqdm import tqdm

def smpl_from_skel(path, sub, out, skel3d, args):
    config = CONFIG[args.body]
    results3d, filenames = read_keypoints3d_all(skel3d)
    pids = list(results3d.keys())
    weight_shape = load_weight_shape(args.model, args.opts)
    weight_pose = load_weight_pose(args.model, args.opts)
    with Timer('Loading {}, {}'.format(args.model, args.gender)):
        body_model = load_model(args.gender, model_type=args.model)
    for pid, result in results3d.items():
        body_params = smpl_from_keypoints3d(body_model, result['keypoints3d'], config, args,
            weight_shape=weight_shape, weight_pose=weight_pose)
        result['body_params'] = body_params
    
    # write for each frame
    for nf, skelname in enumerate(tqdm(filenames, desc='writing')):
        basename = os.path.basename(skelname)
        outname = join(out, basename)
        res = []
        for pid, result in results3d.items():
            frames = result['frames']
            if nf in frames:
                nnf = frames.index(nf)
                val = {'id': pid}
                params = select_nf(result['body_params'], nnf)
                val.update(params)
                res.append(val)
        write_smpl(outname, res)

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--skel3d', type=str, required=True)
    args = parse_parser(parser)
    help="""
  Demo code for fitting SMPL to 3d(+2d) skeletons:

    - Input : {} => {}
    - Output: {}
    - Body  : {}=>{}, {}
""".format(args.path, args.skel3d, args.out, 
    args.model, args.gender, args.body)
    print(help)
    smpl_from_skel(args.path, args.sub, args.out, args.skel3d, args)