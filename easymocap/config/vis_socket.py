'''
  @ Date: 2021-05-30 11:17:18
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-28 13:03:37
  @ FilePath: /EasyMocapRelease/easymocap/config/vis_socket.py
'''
from .baseconfig import CN
from .baseconfig import Config as BaseConfig
import socket
import numpy as np

class Config(BaseConfig):
    @staticmethod
    def init(cfg):
        # input and output
        cfg.host = 'auto'
        cfg.port = 9999
        cfg.width = 1920
        cfg.height = 1080
        
        cfg.max_human = 5
        cfg.track = True
        cfg.block = True # block visualization or not, True for visualize each frame, False in realtime applications
        cfg.rotate = False
        cfg.debug = False
        cfg.write = False
        cfg.out = '/'
        # scene:
        cfg.scene_module = "easymocap.visualize.o3dwrapper"
        cfg.scene = CN()
        cfg.extra = CN()
        cfg.range = CN()
        cfg.new_frames = 0

        # skel
        cfg.skel = CN()
        cfg.skel.joint_radius = 0.02
        cfg.body_model_template = "none"
        # camera
        cfg.camera = CN()
        cfg.camera.phi = 0
        cfg.camera.theta = -90 + 60
        cfg.camera.cx = 0.
        cfg.camera.cy = 0.
        cfg.camera.cz = 6.
        cfg.camera.set_camera = False
        cfg.camera.camera_pose = []
        # range
        cfg.range = CN()
        cfg.range.minr = [-100, -100, -10]
        cfg.range.maxr = [ 100,  100,  10]
        cfg.range.rate_inlier = 0.8
        cfg.range.min_conf = 0.1
        return cfg
    
    @staticmethod
    def parse(cfg):
        if cfg.host == 'auto':
            cfg.host = socket.gethostname()
        if cfg.camera.set_camera:
            pass
        else:# use default camera
            # theta, phi = cfg.camera.theta, cfg.camera.phi
            theta, phi = np.deg2rad(cfg.camera.theta), np.deg2rad(cfg.camera.phi)
            cx, cy, cz = cfg.camera.cx, cfg.camera.cy, cfg.camera.cz
            st, ct = np.sin(theta), np.cos(theta)
            sp, cp = np.sin(phi), np.cos(phi)
            dist = 6
            camera_pose = np.array([
                    [cp, -st*sp, ct*sp, cx],
                    [sp, st*cp, -ct*cp, cy],
                    [0., ct, st, cz],
                    [0.0, 0.0, 0.0, 1.0]])
            cfg.camera.camera_pose = camera_pose.tolist()