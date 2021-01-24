'''
  @ Date: 2020-11-18 14:33:20
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-20 16:33:02
  @ FilePath: /EasyMocap/code/smplmodel/__init__.py
'''
from .body_model import SMPLlayer
from .body_param import load_model
from .body_param import merge_params, select_nf, init_params, Config, check_params, check_keypoints