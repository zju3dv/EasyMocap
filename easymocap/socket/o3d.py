'''
  @ Date: 2021-05-25 11:15:53
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-12 14:54:43
  @ FilePath: /EasyMocapRelease/easymocap/socket/o3d.py
'''
import open3d as o3d
from ..config import load_object
from ..visualize.o3dwrapper import Vector3dVector, create_mesh, load_mesh
from ..mytools import Timer
from ..mytools.vis_base import get_rgb_01
from .base import BaseSocket, log
import json
import numpy as np
from os.path import join
import os
from ..assignment.criterion import CritRange

class VisOpen3DSocket(BaseSocket):
    def __init__(self, host, port, cfg) -> None:
        # output
        self.write = cfg.write
        self.out = cfg.out
        if self.write:
            print('[Info] capture the screen to {}'.format(self.out))
            os.makedirs(self.out, exist_ok=True)
        # scene
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Visualizer', width=cfg.width, height=cfg.height)
        self.vis = vis
        # load the scene
        for key, mesh_args in cfg.scene.items():
            mesh = load_object(key, mesh_args)
            self.vis.add_geometry(mesh)
        for key, val in cfg.extra.items():
            mesh = load_mesh(val["path"])
            trans = np.array(val['transform']).reshape(4, 4)
            mesh.transform(trans)
            self.vis.add_geometry(mesh)
        # create vis => update => super() init
        super().__init__(host, port, debug=cfg.debug)
        self.block = cfg.block
        self.body_model = load_object(cfg.body_model.module, cfg.body_model.args)
        zero_params = self.body_model.init_params(1)
        self.max_human = cfg.max_human
        self.track = cfg.track
        self.camera_pose = cfg.camera.camera_pose
        self.init_camera(cfg.camera.camera_pose)
        self.zero_vertices = Vector3dVector(np.zeros((self.body_model.nVertices, 3)))

        self.vertices, self.meshes = [], []
        for i in range(self.max_human):
            self.add_human(zero_params)

        self.count = 0
        self.previous = {}
        self.critrange = CritRange(**cfg.range)
        self.new_frames  = cfg.new_frames
    
    def add_human(self, zero_params):
        vertices = self.body_model(zero_params)[0]
        self.vertices.append(vertices)
        mesh = create_mesh(vertices=vertices, faces=self.body_model.faces)
        self.meshes.append(mesh)
        self.vis.add_geometry(mesh)
        self.init_camera(self.camera_pose)

    def init_camera(self, camera_pose):
        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        # init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
        init_param.extrinsic = np.array(camera_pose)
        ctr.convert_from_pinhole_camera_parameters(init_param) 

    def filter_human(self, datas):
        datas_new = []
        for data in datas:
            kpts3d = np.array(data['keypoints3d'])
            data['keypoints3d'] = kpts3d
            pid = data['id']
            if pid not in self.previous.keys():
                if not self.critrange(kpts3d):
                    continue
                self.previous[pid] = 0
            self.previous[pid] += 1
            if self.previous[pid] > self.new_frames:
                datas_new.append(data)
        return datas_new

    def main(self, datas):
        if self.debug:log('[Info] Load data {}'.format(self.count))
        datas = json.loads(datas)
        datas = self.filter_human(datas)
        with Timer('forward'):
            for i, data in enumerate(datas):
                if i >= len(self.meshes):
                    print('[Error] the number of human exceeds!')
                    self.add_human(np.array(data['keypoints3d']))
                vertices = self.body_model(data['keypoints3d'])
                self.vertices[i] = Vector3dVector(vertices[0])
            for i in range(len(datas), len(self.meshes)):
                self.vertices[i] = self.zero_vertices
        # Open3D will lock the thread here
        with Timer('set vertices'):
            for i in range(len(self.vertices)):
                self.meshes[i].vertices = self.vertices[i]
                if i < len(datas) and self.track:
                    col = get_rgb_01(datas[i]['id'])
                    self.meshes[i].paint_uniform_color(col)

    def update(self):
        if self.disconnect and not self.block:
            self.previous.clear()
        if not self.queue.empty():
            if self.debug:log('Update' + str(self.queue.qsize()))
            datas = self.queue.get()
            if not self.block:
                while self.queue.qsize() > 0:
                    datas = self.queue.get()
            self.main(datas)
            with Timer('update geometry'):
                for mesh in self.meshes:
                    mesh.compute_triangle_normals()
                    self.vis.update_geometry(mesh)
                self.vis.poll_events()
                self.vis.update_renderer()
            if self.write:
                outname = join(self.out, '{:06d}.jpg'.format(self.count))
                with Timer('capture'):
                    self.vis.capture_screen_image(outname)
            self.count += 1
        else:
            with Timer('update renderer', True):
                self.vis.poll_events()
                self.vis.update_renderer()