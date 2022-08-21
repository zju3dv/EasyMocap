'''
  @ Date: 2021-05-28 14:18:20
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-21 14:38:18
  @ FilePath: /EasyMocapPublic/easymocap/config/baseconfig.py
'''
from .yacs import CfgNode as CN

class Config:
    @classmethod
    def load_from_args(cls, default_cfg='config/vis/base.yml'):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default=default_cfg)
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument("--opts", default=[], nargs='+')
        args = parser.parse_args()
        return cls.load(filename=args.cfg, opts=args.opts, debug=args.debug)
    
    @classmethod
    def load_args(cls, usage=None):
        import argparse
        parser = argparse.ArgumentParser(usage=usage)
        parser.add_argument('--cfg', type=str, default='config/vis/base.yml')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--slurm', action='store_true')
        parser.add_argument("opts", default=None, nargs='+')
        args = parser.parse_args()
        return args, cls.load(filename=args.cfg, opts=args.opts, debug=args.debug)

    @classmethod
    def load(cls, filename=None, opts=[], debug=False) -> CN:
        cfg = CN()
        cfg = cls.init(cfg)
        if filename is not None:
            cfg.merge_from_file(filename)
        if len(opts) > 0:
            cfg.merge_from_list(opts)
        cls.parse(cfg)
        if debug:
            cls.print(cfg)
        return cfg
    
    @staticmethod
    def init(cfg):
        return cfg
    
    @staticmethod
    def parse(cfg):
        pass

    @staticmethod
    def print(cfg):
        print('[Info] --------------')
        print('[Info] Configuration:')
        print('[Info] --------------')
        print(cfg)

import importlib
def load_object(module_name, module_args, **extra_args):
    module_path = '.'.join(module_name.split('.')[:-1])
    module = importlib.import_module(module_path)
    name = module_name.split('.')[-1]
    obj = getattr(module, name)(**extra_args, **module_args)
    return obj

def load_object_from_cmd(cfg, opt):
    cfg = Config.load(cfg, opt)
    model = load_object(cfg.module, cfg.args)
    return model

def load_renderer(cfg, network):
    if cfg.split == 'mesh':
        return load_object(cfg.renderer_mesh_module, cfg.renderer_mesh_args, net=network)
    else:
        return load_object(cfg.renderer_module, cfg.renderer_args, net=network)

def load_visualizer(cfg):
    if cfg.split == 'mesh':
        return load_object(cfg.visualizer_mesh_module, cfg.visualizer_mesh_args)
    else:
        return load_object(cfg.visualizer_module, cfg.visualizer_args)

def load_evaluator(cfg):
    if cfg.evaluator_args.skip_eval:
        return None
    else:
        return load_object(cfg.evaluator_module, cfg.evaluator_args)

def load_config_from_index(config_dict, mode):
    if isinstance(config_dict, str):
        config_dict = Config.load(config_dict, [])
    config_ori = config_dict[mode]
    _cfg = CN()
    if 'exp' in config_ori.keys():
        config_ = config_dict[config_ori.pop('exp')]
        opts = config_.get('opts', []) + config_ori.get('opts', [])
        config = config_
        config.opts = opts
    else:
        config = config_ori
    _cfg['parents'] = []
    opts = config.pop('opts', [])
    for key in list(config.keys()):
        if config[key].endswith('.yml'):
            _cfg['parents'].append(config[key])
    tmp_name = 'tmp_config.yml'
    print(_cfg, file=open(tmp_name, 'w'))
    print(config)
    config['alias'] = config_ori['alias']
    return Config.load(tmp_name, opts=opts), config