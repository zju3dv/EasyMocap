# 这个脚本提供mocap的基本运行接口
import os
from easymocap.config import Config, load_object
from tqdm import tqdm

def process(dataset, model, args):
    ret_all = []
    print('[Run] dataset has {} samples'.format(len(dataset)))
    if args.num_workers == -1:
        for i in tqdm(range(len(dataset)), desc='[Run]'):
            data = dataset[i]
            ret = model.at_step(data, i)
            ret_all.append(ret)
    else:
        import torch
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=1, num_workers=args.num_workers, shuffle=False, collate_fn=lambda x:x, drop_last=False)
        index = 0
        for data in tqdm(dataloader, desc='[Run]'):
            data = data[0]
            ret = model.at_step(data, index)
            if not args.skip_final:
                ret_all.append(ret)
            index += 1
    if not args.skip_final:
        ret_all = model.at_final(ret_all)

def update_data_by_args(cfg_data, args):
    if args.root is not None:
        cfg_data.args.root = args.root
    if args.subs is not None:
        cfg_data.args.subs = args.subs
    if args.subs_vis is not None:
        cfg_data.args.subs_vis = args.subs_vis
    if args.ranges is not None:
        cfg_data.args.ranges = args.ranges
    if args.cameras is not None:
        cfg_data.args.reader.cameras.root = args.cameras
    if args.skip_vis or args.skip_vis_step:
        cfg_data.args.subs_vis = []
    return cfg_data

def update_exp_by_args(cfg_exp, args):
    opts_alias = []
    if 'alias' in cfg_exp.keys():
        for i in range(len(args.opt_exp)//2):
            if args.opt_exp[i*2] in cfg_exp.alias.keys():
                opts_alias.append(cfg_exp.alias[args.opt_exp[i*2]])
                opts_alias.append(args.opt_exp[i*2+1])
        cfg_exp.merge_from_list(opts_alias)
    if args.skip_vis or args.skip_vis_step:
        for key, val in cfg_exp.args.at_step.items():
            if key.startswith('vis'):
                val.skip = True
    if args.skip_vis or args.skip_vis_final:
        for key, val in cfg_exp.args.at_final.items():
            if key.startswith('vis') or key == 'make_video':
                val.skip = True    

def load_cfg_from_file(cfg, args):
    cfg = Config.load(cfg)
    cfg_data = Config.load(cfg.data)
    cfg_data.args.merge_from_other_cfg(cfg.data_opts)
    cfg_data = update_data_by_args(cfg_data, args)
    cfg_exp = Config.load(cfg.exp)
    cfg_exp.args.merge_from_other_cfg(cfg.exp_opts)
    update_exp_by_args(cfg_exp, args)
    return cfg_data, cfg_exp

def load_cfg_from_cmd(args):
    cfg_data = Config.load(args.data, args.opt_data)
    cfg_data = update_data_by_args(cfg_data, args)
    cfg_exp = Config.load(args.exp, args.opt_exp)
    update_exp_by_args(cfg_exp, args)
    return cfg_data, cfg_exp

def main_entrypoint():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    for name in ['data', 'exp']:
        parser.add_argument('--{}'.format(name), type=str, required=False)
        parser.add_argument('--opt_{}'.format(name), type=str, nargs='+', default=[])
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--subs', type=str, default=None, nargs='+')
    parser.add_argument('--subs_vis', type=str, default=None, nargs='+')
    parser.add_argument('--ranges', type=int, default=None, nargs=3)
    parser.add_argument('--cameras', type=str, default=None, help='Camera file path')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--skip_vis', action='store_true')
    parser.add_argument('--skip_vis_step', action='store_true')
    parser.add_argument('--skip_vis_final', action='store_true')
    parser.add_argument('--skip_final', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.cfg is not None:
        cfg_data, cfg_exp = load_cfg_from_file(args.cfg, args)
    else:
        cfg_data, cfg_exp = load_cfg_from_cmd(args)

    if args.out is not None:
        cfg_exp.args.output = args.out
    out = cfg_exp.args.output
    os.makedirs(out, exist_ok=True)
    print(cfg_data, file=open(os.path.join(out, 'cfg_data.yml'), 'w'))
    print(cfg_exp, file=open(os.path.join(out, 'cfg_exp.yml'), 'w'))
    
    dataset = load_object(cfg_data.module, cfg_data.args)
    print(dataset)

    model = load_object(cfg_exp.module, cfg_exp.args)
    process(dataset, model, args)

if __name__ == '__main__':
    main_entrypoint()