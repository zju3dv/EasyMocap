'''
  @ Date: 2021-04-15 16:56:18
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-09 10:16:29
  @ FilePath: /EasyMocap/easymocap/annotator/__init__.py
'''
from .basic_annotator import load_parser, parse_parser
from .basic_dataset import ImageFolder, MVBase
from .basic_visualize import vis_point, vis_line, vis_bbox
from .basic_visualize import plot_bbox_body, plot_skeleton, plot_text, vis_active_bbox, plot_bbox_factory
from .basic_annotator import AnnotBase, AnnotMV
# bbox callbacks
# create, delete, copy
from .bbox_callback import create_bbox, delete_bbox, delete_all_bbox, copy_previous_bbox, copy_previous_missing
# track
from .bbox_callback import get_auto_track
from .bbox_callback import callback_select_bbox_center, callback_select_bbox_corner
