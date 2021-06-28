'''
  @ Date: 2021-04-13 22:21:39
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-14 15:31:48
  @ FilePath: /EasyMocap/apps/demo/mv1p_mirror.py
'''
import os
from os.path import join
from mv1p import mv1pmf_skel, mv1pmf_smpl
from easymocap.dataset import CONFIG

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    args = parse_parser(parser)
    help="""
  Demo code for multiple views and one person with mirror:

    - Input : {} => {}
    - Output: {}
    - Body  : {}=>{}, {}
""".format(args.path, ', '.join(args.sub), args.out, 
    args.model, args.gender, args.body)
    print(help)
    from easymocap.dataset import MV1PMF_Mirror as MV1PMF
    dataset = MV1PMF(args.path, annot_root=args.annot, cams=args.sub, out=args.out,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=False, verbose=args.verbose)
    dataset.writer.save_origin = args.save_origin
    skel_path = join(args.out, 'keypoints3d')
    if args.skel or not os.path.exists(skel_path):
        mv1pmf_skel(dataset, check_repro=False, args=args)
    from easymocap.pipeline.weight import load_weight_pose, load_weight_shape
    weight_shape = load_weight_shape(args.model, args.opts)
    weight_pose = load_weight_pose(args.model, args.opts)
    mv1pmf_smpl(dataset, args=args, weight_pose=weight_pose, weight_shape=weight_shape)