from os.path import join
from easymocap.mytools.debug_utils import log, run_cmd
from easymocap.config.baseconfig import Config, CN
import os
from glob import glob
from copy import deepcopy

def reload_config(config, data, outdir):
    # generate config file
    os.makedirs(outdir, exist_ok=True)
    _cfg = CN()
    if 'exp' in config.keys():
        config_ = config_dict[config.pop('exp')]
        opts = config_.get('opts', []) + config.get('opts', [])
        for key in ['base', 'data', 'model', 'trainer', 'visualizer']:
            if key in config.keys():
                config_[key] = config[key]
        config = config_
        config.opts = opts
    _cfg['parents'] = []
    for key in ['base', 'data', 'model', 'trainer', 'visualizer']:
        _cfg['parents'].append(config[key])
    tmp_name = join(outdir, '_config.yml')
    print(_cfg, file=open(tmp_name, 'w'))
    opts_cfg = config.get('opts', [])
    opts_cfg_dict = config.get('opts_dict', CN({}))
    if len(data)>0:
        for i in range(len(opts_cfg)):
            if isinstance(opts_cfg[i], str) and r'${data}' in opts_cfg[i]:
                opts_cfg[i] = opts_cfg[i].replace(r'${data}', data)
    config = Config.load(tmp_name, opts=opts_cfg)
    config.merge_from_other_cfg(opts_cfg_dict)
    config.merge_from_list(args.opts)
    data_share = config.pop('data_share_args')
    if len(data)>0:
        data_share.root = data
    if args.V100:
        data_share.sample_args.nrays *= 4
    for split in ['train', 'val', 'demo']:
        data = deepcopy(data_share)
        data.merge_from_other_cfg(config['data_{}_args'.format(split)])
        config['data_{}_args'.format(split)] = data
    cfg_name = join(outdir, 'config.yml')
    print(config, file=open(cfg_name, 'w'))

def neuralbody_train(data, config, mode, exp=None):
    # run reconstruction
    exp = mode if exp is None else exp
    outdir = join(args.out, exp)
    cfg_name = join(outdir, 'config.yml')
    cmd =  f'python3 apps/neuralbody/train_pl.py --cfg {cfg_name} gpus {args.gpus} distributed True exp {exp}'
    if args.recfg or (not args.test and not args.demo and not args.eval):
        reload_config(config, data, outdir)
    if args.eval or args.demo or args.test or args.trainvis or args.canonical or args.poses is not None:
        if args.test:
            cmd += ' split test'
        elif args.eval:
            cmd += ' split eval'        
        elif args.trainvis:
            cmd += ' split trainvis'
        elif args.canonical is not None:
            cmd += ' split canonical data_canonical_args.root {}'.format(args.canonical)
        elif args.poses is not None:
            cmd += ' split novelposes data_novelposes_args.root {}'.format(args.poses)
        elif args.demo:
            cmd += ' split demo'
        print(cmd)
        run_cmd(cmd)
        # generate videos
        split = cmd.split()[cmd.split().index('split')+1]
        find_epoch = lambda x:os.path.basename(x).replace(split+'_', '')
        demolists = [i for i in glob(join(outdir, split+'*')) if os.path.isdir(i) and find_epoch(i).isdigit()]
        demolists = sorted(demolists, key=lambda x:int(find_epoch(x)))
        if len(demolists) == 0:
            log('No demo results found')
        else:
            newest = demolists[-1]
            for key in ['rgb_map', 'acc_map', 'feat_map']:
                cmd = f'ffmpeg -y -i {newest}/{key}_%06d.jpg -vcodec libx264 -pix_fmt yuv420p {newest}_{key}.mp4 -loglevel quiet'
                run_cmd(cmd)
        return 0
    if args.debug:
        cmd += ' --debug'
    print(cmd)
    run_cmd(cmd)

if __name__ == '__main__':
    config_dict = Config.load('config/neuralbody_index.yml')
    filelists = glob(join('config', 'neuralbody_index_*.yml'))
    for filename in filelists:
        log(filename)
        config_ = Config.load(filename)
        config_dict.update(config_)
    modes = list(config_dict.keys())
    usage = '''This script helps you to motion capture from multiple views.

    The following modes are supported:
'''
    for mode, config in config_dict.items():
        usage += '\t{:20s}: {}\n'.format(mode, config.comment)
    import argparse
    parser = argparse.ArgumentParser(
        usage=usage)
    parser.add_argument('path', type=str, nargs='+', default=[])
    parser.add_argument('--mode', type=str, default=modes[0],
        choices=modes)
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--gpus', type=str, default='0,')
    parser.add_argument('--out', type=str, default='neuralbody')
    parser.add_argument('--opts', type=str, default=[], nargs='+')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--trainvis', action='store_true')
    parser.add_argument('--canonical', type=str, default=None)
    parser.add_argument('--poses', type=str, default=None)
    parser.add_argument('--recfg', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--V100', action='store_true')
    args = parser.parse_args()

    config = config_dict[args.mode]
    if len(args.path) == 1:
        args.path = args.path[0]

    neuralbody_train(args.path, config, mode=args.mode, exp=args.exp)