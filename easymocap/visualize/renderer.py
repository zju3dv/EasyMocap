import os
import numpy as np
import cv2
import pyrender
import trimesh
import copy
# 这个顺序是BGR的。虽然render的使用的是RGB的，但是由于和图像拼接了，所以又变成BGR的了
colors = [
    # (0.5, 0.2, 0.2, 1.),  # Defalut BGR
    (.5, .5, .7, 1.),  # Pink BGR
    (.44, .50, .98, 1.), # Red
    (.7, .7, .6, 1.),  # Neutral
    (.5, .5, .7, 1.),  # Blue
    (.5, .55, .3, 1.),  # capsule
    (.3, .5, .55, 1.),  # Yellow
    # (.6, .6, .6, 1.), # gray
    (.9, 1., 1., 1.),
    (0.95, 0.74, 0.65, 1.),
    (.9, .7, .7, 1.)
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

from pyrender import RenderFlags
render_flags =  {
    'flip_wireframe': False,
    'all_wireframe': False,
    'all_solid': True,
    'shadows': False, # TODO:bug exists in shadow mode
    'vertex_normals': False,
    'face_normals': False,
    'cull_faces': False, # set to False
    'point_size': 1.0,
    'rgba':True
}

flags = RenderFlags.NONE
if render_flags['flip_wireframe']:
    flags |= RenderFlags.FLIP_WIREFRAME
elif render_flags['all_wireframe']:
    flags |= RenderFlags.ALL_WIREFRAME
elif render_flags['all_solid']:
    flags |= RenderFlags.ALL_SOLID

if render_flags['shadows']:
    flags |= RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT
if render_flags['vertex_normals']:
    flags |= RenderFlags.VERTEX_NORMALS
if render_flags['face_normals']:
    flags |= RenderFlags.FACE_NORMALS
if not render_flags['cull_faces']:
    flags |= RenderFlags.SKIP_CULL_FACES
if render_flags['rgba']:
    flags |= RenderFlags.RGBA


class Renderer(object):
    def __init__(self, focal_length=1000, height=512, width=512, faces=None,
        bg_color=[1.0, 1.0, 1.0, 0.0], down_scale=1,   # render 配置
        extra_mesh=[]
    ): 
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.faces = faces
        self.focal_length = focal_length
        self.bg_color = bg_color
        self.ambient_light = (0.5, 0.5, 0.5)
        self.down_scale = down_scale
        self.extra_mesh = extra_mesh

    def add_light(self, scene):
        trans = [0, 0, 0]
        # Use 3 directional lights
        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3)
        light_forward = np.eye(4)
        light_forward[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_forward)
        light_z = np.eye(4)
        light_z[:3, :3] = cv2.Rodrigues(np.array([-np.pi/2, 0, 0]))[0]
        scene.add(light, pose=light_z)

    def render(self, render_data, cameras, images,
        use_white=False, add_back=True,
        ret_depth=False, ret_color=False):
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        output_images, output_colors, output_depths = [], [], []
        for nv, img_ in enumerate(images):
            if use_white:
                img = np.zeros_like(img_, dtype=np.uint8) + 255
            else:
                img = img_.copy()
            K, R, T = cameras['K'][nv].copy(), cameras['R'][nv], cameras['T'][nv]
            # down scale the image to speed up rendering
            img = cv2.resize(img, None, fx=1/self.down_scale, fy=1/self.down_scale)
            K[:2, :] /= self.down_scale

            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            scene = pyrender.Scene(bg_color=self.bg_color,
                                   ambient_light=self.ambient_light)
            for iextra, _mesh in enumerate(self.extra_mesh):
                if True:
                    mesh = _mesh.copy()
                    trans_cam = np.eye(4)
                    trans_cam[:3, :3] = R
                    trans_cam[:3, 3:] = T
                    mesh.apply_transform(trans_cam)
                    mesh.apply_transform(rot)
                    # mesh.vertices = np.asarray(mesh.vertices) @ R.T + T.T
                    mesh_ = pyrender.Mesh.from_trimesh(mesh)
                    scene.add(mesh_, 'extra{}'.format(iextra))
                else:
                    vert = np.asarray(_mesh.vertices).copy()
                    faces = np.asarray(_mesh.faces)
                    vert = vert @ R.T + T.T
                    mesh = trimesh.Trimesh(vert, faces, process=False)
                    mesh.apply_transform(rot)
                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0,
                        alphaMode='OPAQUE',
                        baseColorFactor=(0., 0., 0., 1.))
                    mesh = pyrender.Mesh.from_trimesh(
                        mesh,
                        material=material)
                    scene.add(mesh, 'extra{}'.format(iextra))
            for trackId, data in render_data.items():
                vert = data['vertices'].copy()
                faces = data['faces']
                vert = vert @ R.T + T.T
                if 'colors' not in data.keys():
                    # 如果使用了vid这个键，那么可视化的颜色使用vid的颜色
                    col = get_colors(data.get('vid', trackId))
                    mesh = trimesh.Trimesh(vert, faces, process=False)
                    mesh.apply_transform(rot)
                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0,
                        alphaMode='OPAQUE',
                        baseColorFactor=col)
                    mesh = pyrender.Mesh.from_trimesh(
                        mesh,
                        material=material)
                    scene.add(mesh, data['name'])
                else:
                    mesh = trimesh.Trimesh(vert, faces, vertex_colors=data['colors'], process=False)
                    mesh.apply_transform(rot)
                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0,
                        alphaMode='OPAQUE',
                        baseColorFactor=(1., 1., 1.))
                    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
                    scene.add(mesh, data['name'])
            camera_pose = np.eye(4)
            camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
            scene.add(camera, pose=camera_pose)
            self.add_light(scene)
            # pyrender.Viewer(scene, use_raymond_lighting=True)
            # Alpha channel was not working previously need to check again
            # Until this is fixed use hack with depth image to get the opacity
            rend_rgba, rend_depth = self.renderer.render(scene, flags=flags)
            if rend_rgba.shape[2] == 3: # fail to generate transparent channel
                valid_mask = (rend_depth > 0)[:, :, None]
                rend_rgba = np.dstack((rend_rgba, (valid_mask*255).astype(np.uint8)))
            rend_rgba = rend_rgba[..., [2, 1, 0, 3]]
            if add_back:
                rend_cat = cv2.addWeighted(
                    cv2.bitwise_and(img, 255 - rend_rgba[:, :, 3:4].repeat(3, 2)), 1, 
                    cv2.bitwise_and(rend_rgba[:, :, :3], rend_rgba[:, :, 3:4].repeat(3, 2)), 1, 0)
            else:
                rend_cat = rend_rgba
            
            output_colors.append(rend_rgba)
            output_depths.append(rend_depth)
            output_images.append(rend_cat)
        if ret_depth:
            return output_images, output_depths
        elif ret_color:
            return output_colors
        else:
            return output_images

    def _render_multiview(self, vertices, K, R, T, imglist, trackId=0, return_depth=False, return_color=False,
        bg_color=[0.0, 0.0, 0.0, 0.0], camera=None):
        # List to store rendered scenes
        output_images, output_colors, output_depths = [], [], []
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        nViews = len(imglist)
        for nv in range(nViews):
            img = imglist[nv]
            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            # Create a scene for each image and render all meshes
            scene = pyrender.Scene(bg_color=bg_color,
                                   ambient_light=(0.3, 0.3, 0.3))
            camera_pose = np.eye(4)

            # for every person in the scene
            if isinstance(vertices, dict):
                for trackId, data in vertices.items():
                    vert = data['vertices'].copy()
                    faces = data['faces']
                    col = data.get('col', trackId)
                    vert = vert @ R[nv].T + T[nv]
                    mesh = trimesh.Trimesh(vert, faces)
                    mesh.apply_transform(rot)
                    trans = [0, 0, 0]

                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0,
                        alphaMode='OPAQUE',
                        baseColorFactor=colors[col % len(colors)])
                    mesh = pyrender.Mesh.from_trimesh(
                        mesh,
                        material=material)
                    scene.add(mesh, 'mesh')
            else:
                verts = vertices @ R[nv].T + T[nv]
                mesh = trimesh.Trimesh(verts, self.faces)
                mesh.apply_transform(rot)
                trans = [0, 0, 0]

                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    alphaMode='OPAQUE',
                    baseColorFactor=colors[trackId % len(colors)])
                mesh = pyrender.Mesh.from_trimesh(
                    mesh,
                    material=material)
                scene.add(mesh, 'mesh')

            if camera is not None:
                light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=70)
                light_pose = np.eye(4)
                light_pose[:3, 3] = [0, 0, 4.5]
                scene.add(light, pose=light_pose)

                light_pose[:3, 3] = [0, 1, 4]
                scene.add(light, pose=light_pose)

                light_pose[:3, 3] = [0, -1, 4]
                scene.add(light, pose=light_pose)
            else:
                trans = [0, 0, 0]
                # Use 3 directional lights
                # Create light source
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
                light_pose = np.eye(4)
                light_pose[:3, 3] = np.array([0, -1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([0, 1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([1, 1, 2]) + trans
                scene.add(light, pose=light_pose)
            if camera is None:
                if K is None:
                    camera_center = np.array([img.shape[1] / 2., img.shape[0] / 2.])
                    camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length, cx=camera_center[0], cy=camera_center[1])
                else:
                    camera = pyrender.camera.IntrinsicsCamera(fx=K[nv][0, 0], fy=K[nv][1, 1], cx=K[nv][0, 2], cy=K[nv][1, 2])
            scene.add(camera, pose=camera_pose)
            # Alpha channel was not working previously need to check again
            # Until this is fixed use hack with depth image to get the opacity
            color, rend_depth = self.renderer.render(scene, flags=flags)
            # color = color[::-1,::-1]
            # rend_depth = rend_depth[::-1,::-1]
            output_depths.append(rend_depth)
            color = color.astype(np.uint8)
            valid_mask = (rend_depth > 0)[:, :, None]
            if color.shape[2] == 3: # 在服务器上透明通道失败
                color = np.dstack((color, (valid_mask*255).astype(np.uint8)))
            output_colors.append(color)
            output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * img)
            
            output_img = output_img.astype(np.uint8)
            output_images.append(output_img)
        if return_depth:
            return output_images, output_depths
        elif return_color:
            return output_colors
        else:
            return output_images

def render_results(img, render_data, cam_params, outname=None, rotate=False, degree=90, axis=[1.,0.,0],
    fix_center=None):
    render_data = copy.deepcopy(render_data)
    render = Renderer(height=1024, width=1024, faces=None)
    Ks, Rs, Ts = [cam_params['K']], [cam_params['Rc']], [cam_params['Tc']]
    imgsrender = render.render_multiview(render_data, Ks, Rs, Ts, [img], return_color=True)[0]
    render0 = cv2.addWeighted(cv2.bitwise_and(img, 255 - imgsrender[:, :, 3:4].repeat(3, 2)), 1, imgsrender[:, :, :3], 1, 0.0)
    if rotate:
        # simple rotate the vertices
        if fix_center is None:
            center = np.mean(np.vstack([v['vertices'] for i, v in render_data.items()]), axis=0, keepdims=True)
            new_center = center.copy()
            new_center[:, 0:2] = 0
        else:
            center = fix_center.copy()
            new_center = fix_center.copy()
            new_center[:, 2] *= 1.5
        direc = np.array(axis)
        rot, _ = cv2.Rodrigues(direc*degree/90*np.pi/2)
        for key in render_data.keys():
            vertices = render_data[key]['vertices']
            vert = (vertices - center) @ rot.T + new_center
            render_data[key]['vertices'] = vert
        blank = np.zeros(())
        blank = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype) + 255
        imgsrender = render.render_multiview(render_data, Ks, Rs, Ts, [blank], return_color=True)[0]
        render1 = cv2.addWeighted(cv2.bitwise_and(blank, 255- imgsrender[:, :, 3:4].repeat(3, 2)), 1, imgsrender[:, :, :3], 1, 0.0)
        render0 = np.vstack([render0, render1])
    if outname is not None:
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, render0)
    return render0