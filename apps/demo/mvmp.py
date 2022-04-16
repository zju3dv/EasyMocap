'''
  @ Date: 2021-06-23 16:13:53
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 11:52:49
  @ FilePath: /EasyMocapRelease/apps/demo/mvmp.py
'''
from easymocap.dataset import CONFIG
from easymocap.dataset import CONFIG
from easymocap.affinity.affinity import ComposedAffinity
from easymocap.assignment.associate import simple_associate
from easymocap.assignment.group import PeopleGroup

from easymocap.mytools import Timer
from tqdm import tqdm

def mvposev1(dataset, args, cfg):
    dataset.no_img = not (args.vis_det or args.vis_match or args.vis_repro or args.ret_crop)
    start, end = args.start, min(args.end, len(dataset))
    affinity_model = ComposedAffinity(cameras=dataset.cameras, basenames=dataset.cams, cfg=cfg.affinity)
    group = PeopleGroup(Pall=dataset.Pall, cfg=cfg.group)

    if args.vis3d:
        from easymocap.socket.base_client import BaseSocketClient
        vis3d = BaseSocketClient(args.host, args.port)
    for nf in tqdm(range(start, end), desc='reconstruction'):
        group.clear()
        with Timer('load data', not args.time):
            images, annots = dataset[nf]
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        # 计算不同视角的检测结果的affinity
        with Timer('compute affinity', not args.time):
            affinity, dimGroups = affinity_model(annots, images=images)
        with Timer('associate', not args.time):
            group = simple_associate(annots, affinity, dimGroups, dataset.Pall, group, cfg=cfg.associate)
            results = group
        if args.vis_match:
            dataset.vis_detections(images, annots, nf, mode='match', sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, results, nf, sub_vis=args.sub_vis)
        dataset.write_keypoints2d(annots, nf)
        dataset.write_keypoints3d(results, nf)
        if args.vis3d:
            vis3d.send(group.results)
    Timer.report()

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--vis_match', action='store_true')
    parser.add_argument('--time', action='store_true')
    parser.add_argument('--vis3d', action='store_true')
    parser.add_argument('--ret_crop', action='store_true')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument("--host", type=str, default='none')  # cn0314000675l
    parser.add_argument("--port", type=int, default=9999)
    args = parse_parser(parser)
    from easymocap.config.mvmp1f import Config
    cfg = Config.load(args.cfg, args.cfg_opts)
    # Define dataset
    help="""
  Demo code for multiple views and multiple persons:

    - Input : {} => {}
    - Output: {}
    - Body  : {}
""".format(args.path, ', '.join(args.sub), args.out, 
    args.body)
    print(help)
    from easymocap.dataset import MVMPMF
    dataset = MVMPMF(args.path, cams=args.sub, annot_root=args.annot,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=True, out=args.out, filter2d=cfg.dataset)
    mvposev1(dataset, args, cfg)