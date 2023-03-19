'''
  @ Date: 2022-04-14 14:05:50
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-05-19 23:09:57
  @ FilePath: /EasyMocapPublic/apps/postprocess/render.py
'''
from os.path import join
from easymocap.config import Config, load_object
from easymocap.config.baseconfig import load_config_from_index, load_object_from_cmd
from easymocap.mytools.debug_utils import mywarn, log, myerror
from tqdm import tqdm
from easymocap.mytools import Timer

index = Config.load('config/render_index.yml', [])

def vis(cfg):
    # 读入模型
    body_model = load_object_from_cmd(args.model, [])
    # # 读入参数
    results = load_object(cfg.result_module, cfg.result_args, body_model=body_model)
    inputs = load_object(cfg.input_module, cfg.input_args)
    outputs = load_object(cfg.output_module, cfg.output_args)
    silent = True
    for nf in tqdm(range(cfg.ranges[0], min(cfg.ranges[1], len(results)), cfg.ranges[2]), desc='vis'):
        with Timer('result', silent):
            basename, result = results[nf]
        with Timer('inputs', silent):
            images, cameras = inputs(basename)
        with Timer('outputs', silent):
            outputs(images, result, cameras, basename)
    if cfg.make_video:
        video = load_object(cfg.video_module, cfg.video_args)
        video.make_video(cfg.output_args.out)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--mode', type=str, default='mesh')
    parser.add_argument('--ranges', type=int, default=[], nargs=3)
    parser.add_argument('--subs', type=str, default=[], nargs="+")
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    config, ori = load_config_from_index(index, args.mode)
    mode = ori.alias
    if args.model is None:
        args.model = join(args.path, args.exp, 'cfg_model.yml')
        mywarn('[vis] args.model is not specified, use {}'.format(args.model))
    if args.result is None:
        args.result = join(args.path, args.exp, 'smpl')
        mywarn('[vis] args.result is not specified, use {}'.format(args.result))
    if args.output is None:
        args.output = join(args.path, args.exp, mode)
        mywarn('[vis] args.output is not specified, use {}'.format(args.output))
    if len(args.subs) != 0:
        config.input_args.subs = args.subs
    if len(args.ranges) != 0:
        config.ranges = args.ranges
    config.input_args.path = args.path
    config.result_args.path = args.result
    config.output_args.out = args.output
    vis(config)