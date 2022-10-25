'''
  @ Date: 2021-05-13 14:20:13
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:24:20
  @ FilePath: /EasyMocapPublic/easymocap/visualize/pyrender_wrapper.py
'''
import pyrender
import numpy as np
import trimesh
import cv2
from .pyrender_flags import get_flags
from ..mytools.vis_base import get_rgb

def offscree_render(renderer, scene, img, flags):
    rend_rgba, rend_depth = renderer.render(scene, flags=flags)
    assert rend_depth.max() < 65, 'depth should less than 65.536: {}'.format(rend_depth.max())
    rend_depth = (rend_depth * 1000).astype(np.uint16)
    if rend_rgba.shape[2] == 3: # fail to generate transparent channel
        valid_mask = (rend_depth > 0)[:, :, None]
        rend_rgba = np.dstack((rend_rgba, (valid_mask*255).astype(np.uint8)))
    rend_rgba = rend_rgba[..., [2, 1, 0, 3]]
    if False:
        rend_cat = cv2.addWeighted(
            cv2.bitwise_and(img, 255 - rend_rgba[:, :, 3:4].repeat(3, 2)), 1, 
            cv2.bitwise_and(rend_rgba[:, :, :3], rend_rgba[:, :, 3:4].repeat(3, 2)), 1, 0)
    else:
        rend_cat = img.copy()
        rend_cat[rend_rgba[:,:,-1]==255] = rend_rgba[:,:,:3][rend_rgba[:,:,-1]==255]
    return rend_rgba, rend_depth, rend_cat

class Renderer:
    def __init__(self, bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.5, 0.5, 0.5], flags={}) -> None:
        self.bg_color = bg_color
        self.ambient_light = ambient_light
        self.renderer = pyrender.OffscreenRenderer(1024, 1024)
        self.flags = get_flags(flags)
    
    @staticmethod
    def add_light(scene, camera=None):
        # Use 3 directional lights
        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3)
        light_forward = np.eye(4)
        # here the location of the light is set to be the origin
        # and this location doesn't affect the render results
        scene.add(light, pose=light_forward)
        light_z = np.eye(4)
        light_z[:3, :3] = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
        # if camera is not None:
        #     light_z[:3, :3] = camera['R'] @ light_z[:3, :3]
        scene.add(light, pose=light_z)

    def __call__(self, render_data, images, cameras, extra_mesh=[],
        ret_image=False, ret_depth=False, ret_color=False, ret_mask=False, ret_all=True):
        if isinstance(images, np.ndarray) and isinstance(cameras, dict):
            images, cameras = [images], [cameras]
        assert isinstance(cameras, list)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        output_images, output_colors, output_depths = [], [], []
        for nv, img in enumerate(images):
            cam = cameras[nv]
            K, R, T = cam['K'], cam['R'], cam['T']
            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            scene = pyrender.Scene(bg_color=self.bg_color,
                                   ambient_light=self.ambient_light)
            for iextra, _mesh in enumerate(extra_mesh):
                mesh = _mesh.copy()
                trans_cam = np.eye(4)
                trans_cam[:3, :3] = R
                trans_cam[:3, 3:] = T
                mesh.apply_transform(trans_cam)
                mesh.apply_transform(rot)
                # mesh.vertices = np.asarray(mesh.vertices) @ R.T + T.T
                mesh_ = pyrender.Mesh.from_trimesh(mesh)
                scene.add(mesh_, 'extra{}'.format(iextra))
            for trackId, data in render_data.items():
                vert = data['vertices'].copy()
                faces = data['faces']
                vert = vert @ R.T + T.T
                if 'colors' not in data.keys():
                    # 如果使用了vid这个键，那么可视化的颜色使用vid的颜色
                    if False:
                        col = get_rgb(data.get('vid', trackId))
                    else:
                        col = get_colors(data.get('vid', trackId))
                    mesh = trimesh.Trimesh(vert, faces, process=False)
                    mesh.apply_transform(rot)
                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0,
                        roughnessFactor=0.0,
                        alphaMode='OPAQUE',
                        baseColorFactor=col)
                    # material = pyrender.material.SpecularGlossinessMaterial(
                    #     diffuseFactor=1.0, glossinessFactor=0.0
                    # )
                    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=data.get('smooth', True))
                else:
                    mesh = trimesh.Trimesh(vert, faces, vertex_colors=data['colors'], process=False)
                    mesh.apply_transform(rot)
                    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                scene.add(mesh, data['name'])
            camera_pose = np.eye(4)
            camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
            scene.add(camera, pose=camera_pose)
            self.add_light(scene, camera=cam)
            # pyrender.Viewer(scene, use_raymond_lighting=True)
            rend_rgba, rend_depth, rend_cat = offscree_render(self.renderer, scene, img, self.flags)
            
            output_colors.append(rend_rgba)
            output_depths.append(rend_depth)
            output_images.append(rend_cat)
        res = None
        if ret_depth:
            res = output_depths
        elif ret_color:
            res = output_colors
        elif ret_mask:
            res = [val[:, :, 3] for val in output_colors]
        elif ret_image:
            res = output_images
        else:
            res = output_colors, output_depths, output_images
        return res
    
    def render_image(self, render_data, images, cameras, extra_mesh, 
        **kwargs):
        return self.__call__(render_data, images, cameras, extra_mesh,
            ret_all=True, **kwargs)

def plot_meshes(img, meshes, K, R, T, mode='image'):
    renderer = Renderer()
    out = renderer.render_image(meshes, img, {'K': K, 'R': R, 'T': T}, [])
    if mode == 'image':
        return out[2][0]
    elif mode == 'mask':
        return out[0][0][..., -1]    
    elif mode == 'hstack':
        return np.hstack([img, out[0][0][:, :, :3]])
    elif mode == 'left':
        out = out[0][0]
        rend_rgba = np.roll(out, out.shape[1]//10, axis=1)
        rend_cat = img.copy()
        rend_cat[rend_rgba[:,:,-1]==255] = rend_rgba[:,:,:3][rend_rgba[:,:,-1]==255]
        return rend_cat

# 这个顺序是BGR的。虽然render的使用的是RGB的，但是由于和图像拼接了，所以又变成BGR的了
colors = [
    (94/255, 124/255, 226/255), # 青色
    (255/255, 200/255, 87/255), # yellow
    (74/255.,  189/255.,  172/255.), # green
    (8/255, 76/255, 97/255), # blue
    (219/255, 58/255, 52/255), # red
    (77/255, 40/255, 49/255), # brown
]

colors_table = {
    # colorblind/print/copy safe:
    '_blue': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [ 166/255.,  229/255.,  204/255.],
    '_mint2': [ 202/255.,  229/255.,  223/255.],
    '_green': [ 153/255.,  216/255.,  201/255.],
    '_green2': [ 171/255.,  221/255.,  164/255.],
    '_red': [ 251/255.,  128/255.,  114/255.],
    '_orange': [ 253/255.,  174/255.,  97/255.],
    '_yellow': [ 250/255.,  230/255.,  154/255.],
    'r':[255/255,0,0],
    'g':[0,255/255,0],
    'b':[0,0,255/255],
    'k':[0,0,0],
    'y':[255/255,255/255,0],
    'purple':[128/255,0,128/255]
}

def get_colors(pid):
    if isinstance(pid, int):
        return colors[pid % len(colors)]
    elif isinstance(pid, str):
        return colors_table[pid]
    elif isinstance(pid, list) or isinstance(pid, tuple):
        if len(pid) == 3:
            pid = (pid[0], pid[1], pid[2], 1.)
        assert len(pid) == 4
        return pid