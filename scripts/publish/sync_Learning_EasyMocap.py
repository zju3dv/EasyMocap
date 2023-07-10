filetree_l = '''
apps:
    - mocap:
        - run.py
myeasymocap:
    - datasets:
        - basedata.py
        - mv1p.py
        - sv1p.py
      io:
        - model.py
        - video.py
        - vis.py
        - vis3d.py
        - write.py
      operations:
        - init.py
        - loss.py
        - match.py
        - merge.py
        - optimizer.py
        - select.py
        - smooth.py
        - triangulate.py
        - match_base.py
        - iterative_triangulate.py
      stages:
        - basestage.py
        - collect.py
      backbone:
        - basetopdown.py
        - topdown_keypoints.py
        - yolo:
            - yolo.py
          vitpose:
            - layers.py
            - vit_moe.py
          hrnet:
            - __init__.py
            - modules.py
            - hrnet.py
            - myhrnet.py
          pare:
            - pare.py
            - config.py
            - constants.py
            - backbone:
                - __init__.py
                - resnet.py
                - mobilenet.py
                - hrnet.py
                - utils.py
              head:
                - __init__.py
                - pare_head.py
                - hmr_head.py
                - smpl_head.py
                - smpl_cam_head.py
              layers:
                - __init__.py
                - locallyconnected2d.py
                - interpolate.py
                - nonlocalattention.py
                - keypoint_attention.py
                - coattention.py
                - softargmax.py
                - non_local:
                  - __init__.py
                  - dot_product.py
              utils:
                - geometry.py
                - kp_utils.py
          mediapipe:
            - hand.py
          hand2d:
            - __init__.py
            - hand2d.py
            - resnet.py
          hmr:
            - __init__.py
            - hmr.py
            - models.py
            - hmr_api.py        
config:
    - datasets:
        - mvimage.yml
        - svimage.yml
      mv1p:
        - detect_triangulate.yml
        - detect_triangulate_fitSMPL.yml
        - detect_hand_triangulate.yml
        - detect_hand_triangulate_fitMANO.yml
      1v1p:
        - hrnet_pare_finetune.yml
        - hand_detect_finetune.yml
        - fixhand.yml
'''

filetree_e = '''
3rdparty:
  - eigen-3.3.7
  - pybind11
library:
    - pymatch:
        - setup.py
        - CMakeLists.txt
        - python:
            - __init__.py
            - CMakeLists.txt
            - pymatchlr.cpp
        - include:
            - base.h
            - projfunc.hpp
            - Timer.hpp
            - visualize.hpp
            - matchSVT.hpp
        - pymatchlr:
            - __init__.py
apps:
    - calibration:
        - vis_camera_by_open3d.py
easymocap:
    - multistage:
        - gmm.py
        - gmm_08.pkl
      mytools:
        - vis_base.py
        - camera_utils.py
'''

sync_config = {
    'EasyMocap': {
        'src': '../EasyMocapPublic',
        'dst': '.',
        'filetree': filetree_e
    },
    'Learn': {
        'src': '../Learning_EasyMocap',
        'dst': '.',
        'filetree': filetree_l
    }
}
import os
import shutil
import yaml
from os.path import join
from easymocap.mytools.debug_utils import log, mywarn

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        content1 = f1.read()
        content2 = f2.read()
        if content1 == content2:
            return True
        else:
            return False

def copy_node(dir, nodes):
    for node in nodes:
        if isinstance(node, str):
            srcname = join(SRC, dir, node)
            dstname = join(DST, dir, node)
            if not os.path.exists(srcname):
                mywarn('Not exists {}'.format(srcname))
                continue
            if os.path.exists(dstname):
                if os.path.isdir(srcname):
                    mywarn('Current not support overwrite folders: {}'.format(dstname))
                else:
                    if compare_files(srcname, dstname):
                        pass
                    else:
                        mywarn('Overwrite file: {}'.format(dstname))
                        shutil.copyfile(srcname, dstname)
            else:
                if os.path.isdir(srcname):
                    log('Copy dir: {}'.format(dstname))
                    shutil.copytree(srcname, dstname)
                else:
                    os.makedirs(join(DST, dir), exist_ok=True)
                    log('Copy file: {}'.format(dstname))
                    shutil.copyfile(srcname, dstname)
        elif isinstance(node, dict):
            for subdir, subnode in node.items():
                copy_node(join(dir, subdir), subnode)
    
if __name__ == '__main__':
    for cfg_name, cfg in sync_config.items():
        SRC = cfg['src']
        DST = cfg['dst']
        filetree = cfg['filetree']
        filetree = yaml.safe_load(filetree)
        SRC = os.path.abspath(SRC)
        DST = os.path.abspath(DST)
        os.chdir(SRC)
        cmd = 'git pull origin master'
        os.system(cmd)
        os.chdir(DST)
        for dir, files in filetree.items():
            copy_node(dir, files)
