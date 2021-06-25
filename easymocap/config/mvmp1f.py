'''
  @ Date: 2021-05-28 14:29:24
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 11:39:11
  @ FilePath: /EasyMocapRelease/easymocap/config/mvmp1f.py
'''
from .baseconfig import CN
from .baseconfig import Config as BaseConfig

class Config(BaseConfig):
    @staticmethod
    def init(cfg):
        cfg.width = -1
        cfg.height = -1
        # dataset part
        cfg.dataset = CN()
        cfg.dataset.log = False
        cfg.dataset.debug = False
        cfg.dataset.height = -1
        cfg.dataset.width = -1
        cfg.dataset.min_conf = 0.1
        cfg.dataset.filter = CN()
        # affinity part
        cfg.affinity = CN()
        # SVT part
        cfg.affinity.aff_min = 0.2
        cfg.affinity.svt_py = True
        aff_funcs = CN()
        cfg.affinity.aff_funcs = aff_funcs
        svt_args = CN()
        svt_args.debug = 0
        svt_args.log = 0
        svt_args.maxIter = 10
        svt_args.w_sparse = 0.1
        svt_args.w_rank = 50
        svt_args.tol = 1e-4
        cfg.affinity.svt_args = svt_args
        # affinity debug
        cfg.affinity.vis_aff = False
        cfg.affinity.vis_res = False
        cfg.affinity.vis_pair = False
        # associate
        associate = CN()
        associate.debug = False
        associate.log = False
        associate.body = 'body25'
        associate.max_repro_error = 0.1
        associate.min_views = 2
        associate.criterions = CN()

        cfg.associate = associate
        cfg.group  = CN()
        return cfg
    
    @staticmethod
    def parse(cfg):
        for globalkey in ['height', 'width']:
            for key, val in cfg.items():
                if isinstance(val, CN) and globalkey in val.keys():
                    val[globalkey] = cfg[globalkey]