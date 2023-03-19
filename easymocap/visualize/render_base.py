'''
  @ Date: 2021-11-22 15:16:14
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-28 21:40:17
  @ FilePath: /EasyMocapPublic/easymocap/visualize/render_base.py
'''
from glob import glob
from os.path import join
import numpy as np

from ..mytools.file_utils import read_json
from ..mytools.debug_utils import log
from ..mytools.reader import read_keypoints3d, read_smpl
import os
from ..mytools.camera_utils import read_cameras, Undistort
import cv2
from ..mytools.vis_base import merge, plot_keypoints_auto
from ..config.baseconfig import load_object
from .geometry import load_sphere

def imwrite(imgname, img):
    if not os.path.exists(os.path.dirname(imgname)):
        os.makedirs(os.path.dirname(imgname))
    if img.shape[0] % 2 == 1 or img.shape[1] % 2 == 1:
        img = cv2.resize(img, (img.shape[1]//2*2, img.shape[0]//2*2))
    cv2.imwrite(imgname, img)

def compute_normals(vertices, faces):
    normal = np.zeros_like(vertices)

    # compute normal per triangle
    normal_faces = np.cross(vertices[faces[:,1]] - vertices[faces[:,0]], vertices[faces[:,2]] - vertices[faces[:,0]])

    # sum normals at vtx
    normal[faces[:, 0]] += normal_faces[:]
    normal[faces[:, 1]] += normal_faces[:]
    normal[faces[:, 2]] += normal_faces[:]

    # compute norms
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    
    return normal

def get_dilation_of_mesh(delta):
    def func(info):
        vertices = info['vertices']
        normals = compute_normals(info['vertices'], info['faces'])
        vertices += delta * normals
        return info
    return func

class Results:
    def __init__(self, body_model, path, rend_type,
        operation='none') -> None:
        self.path = path
        self.body_model = body_model
        self.skelnames = sorted(glob(join(path, '*.json')))
        self.ismulti = False
        if len(self.skelnames) == 0:
            # 尝试找多视角的结果
            subs = sorted(os.listdir(path))
            assert len(subs) > 0, path
            self.ismulti = True
            self.subs = subs
            self.skelnames = {}
            for sub in subs:
                skelnames = sorted(glob(join(path, sub, '*.json')))
                self.skelnames[sub] = skelnames
        self.rend_type = rend_type
        self.read_func = {'skel': read_keypoints3d, 'mesh': read_smpl}[rend_type]
        if operation.startswith('dilation'):
            # TODO: 暂时直接解析
            opfunc = get_dilation_of_mesh(float(operation.replace('dilation:', '')))
        else:
            opfunc = lambda x:x
        self.operation = opfunc

    def process(self, info):
        return info

    def read(self, skelname):
        results = self.read_func(skelname)
        render_data = {}
        trans = np.array([[0., 0., 0., 0.]])
        for info in results:
            info['vertices'] = self.body_model(return_verts=True, return_tensor=False, **info)[0]
            info['keypoints3d'] = self.body_model(return_verts=False, return_tensor=False, **info)[0]
            info['vertices'] += trans[:, :3]
            info['faces'] = self.body_model.faces
            d = self.operation(info)
            render_data[d['id']] = {
                'vertices': d['vertices'], 'keypoints3d': d['keypoints3d'], 
                'faces': info['faces'], 
                'vid': d['id'], 'name': 'human_{}'.format(d['id'])}
            if self.rend_type == 'skel':
                render_data[d['id']]['smooth'] = False
        return render_data

    def get_multi(self, index):
        datas = {}
        for sub in self.subs:
            skelname = self.skelnames[sub][index]
            basename = os.path.basename(skelname).replace('.json', '')
            render_data = self.read(skelname)
            datas[sub] = render_data
        return basename, datas

    def __getitem__(self, index):
        if self.ismulti:
            return self.get_multi(index)
        else:
            skelname = self.skelnames[index]
            basename = os.path.basename(skelname).replace('.json', '')
            return basename, self.read(skelname)

    def __len__(self):
        if self.ismulti:
            return len(self.skelnames[self.subs[0]])
        else:
            return len(self.skelnames)

class ResultsObjects(Results):
    def __init__(self, object,**kwargs) -> None:
        super().__init__(**kwargs)
        self.object = object
    
    def __getitem__(self, index):
        basename, datas = super().__getitem__(index)
        objectname = join(self.path, '..', '..', self.object, basename+'.json')
        objects = read_json(objectname)
        ret_objects = {}
        for obj in objects:
            pid = obj['id']
            vertices, faces = load_sphere()
            vertices *= 0.12
            vertices += np.array(obj['keypoints3d'])[:, :3]
            ret_objects[1000+pid] = {
                'vertices': vertices,
                'faces': faces,
                'vid': pid,
                'name': 'object_{}'.format(pid)
            }
        if self.ismulti:
            for key, data in datas.items():
                data.update(ret_objects)
        else:
            datas.update(ret_objects)
        return basename, datas

class Images:
    def __init__(self, path, subs, image_args) -> None:
        if path == 'none':
            self.images = path
            # no need for images
            assert len(subs) > 0, '{} must non-empty'.format(subs)
        else:
            self.images = join(path, 'images')
            if len(subs) == 0:
                subs = sorted(os.listdir(self.images))
                if subs[0].isdigit():
                    subs.sort(key=lambda x:int(x))
            self.cameras = read_cameras(path)
        self.subs = subs
        self.image_args = image_args
        self.cameras_vis = {sub:cam.copy() for sub, cam in self.cameras.items()}
        # rescale the camera
        for cam in self.cameras_vis.values():
            K = cam['K'].copy()
            cam['K'][:2, :] *= image_args.scale
        self.distortMap = {}

    def __call__(self, basename):
        if self.images == 'none':
            # 返回空的图像
            imgs = {sub: self.blank.copy() for sub in self.subs}
            import ipdb; ipdb.set_trace()
        else:
            imgs = {}
            for sub in self.subs:
                imgname = join(self.images, sub, basename+self.image_args.ext)
                if not os.path.exists(imgname):
                    for ext in ['.jpg', '.png']:
                        imgname_ = imgname.replace(self.image_args.ext, ext)
                        if os.path.exists(imgname_):
                            self.image_args.ext = ext
                            imgname = imgname_
                assert os.path.exists(imgname), imgname
                img = cv2.imread(imgname)
                if self.image_args.scale != 1:
                    img = cv2.resize(img, None, fx=self.image_args.scale, fy=self.image_args.scale)
                if self.image_args.undis:
                    camera = self.cameras_vis[sub]
                    K, D = camera['K'], camera['dist']
                    if sub not in self.distortMap.keys():
                        h,  w = img.shape[:2]
                        mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, camera['K'], (w,h), 5)
                        self.distortMap[sub] = (mapx, mapy)
                    mapx, mapy = self.distortMap[sub]
                    img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                imgs[sub] = img
        return imgs, self.cameras_vis

class Repro:
    def __init__(self, out, merge='none') -> None:
        self.out = out
        os.makedirs(self.out, exist_ok=True)
        self.merge = merge
    
    @staticmethod
    def repro(P, keypoints3d, img):
        if keypoints3d.shape[1] == 3:
            keypoints3d = np.hstack((keypoints3d, np.ones((keypoints3d.shape[0], 1))))
        kcam = np.hstack([keypoints3d[:, :3], np.ones((keypoints3d.shape[0], 1))]) @ P.T
        kcam = kcam[:, :]/kcam[:, 2:]
        k2d = np.hstack((kcam, (keypoints3d[:, 3:]>0.)&(kcam[:, 2:] >0.1)))
        from ..estimator.wrapper_base import bbox_from_keypoints
        bbox = bbox_from_keypoints(k2d)
        return k2d, bbox

    def __call__(self, images, results, cameras, basename):
        imgsout = {}
        cams = list(images.keys())
        for nv, cam in enumerate(cams):
            img = images[cam]
            outname = join(self.out, basename+'_' + cam +'.jpg')
            # K可能缩放过了，所以需要重新计算
            P = cameras[cam]['K'] @ cameras[cam]['RT']
            for pid, info in results.items():
                keypoints3d = info['keypoints3d']
                k2d, bbox = self.repro(P, keypoints3d, img)
                lw = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1])/50)
                # plot_bbox(img, bbox, pid=pid, vis_id=pid)
                plot_keypoints_auto(img, k2d, pid=pid, use_limb_color=False)
            imgsout[outname] = img
        if self.merge == 'none':
            for outname, img in imgsout.items():
                cv2.imwrite(outname, img)
        else:
            outname = join(self.out, basename+'.jpg')
            out = merge(list(imgsout.values()), square=True)
            cv2.imwrite(outname, out)

class Outputs:
    def __init__(self, out, mode, backend, scene={}) -> None:
        self.out = out
        os.makedirs(self.out, exist_ok=True)
        from .render_func import get_render_func, get_ext
        self.render_func = get_render_func(mode, backend)
        self.mode = mode
        self.ext = get_ext(mode)
        self.extra_mesh = {}
        self.scene = []
        for key, val in scene.items():
            mesh = load_object(val.module, val.args)
            log('[vis] Load extra mesh {}'.format(key))
            self.scene.append(mesh)

    def __call__(self, images, results, cameras, basename):
        for i, mesh in enumerate(self.scene):
            results[10000+i] = mesh
        render_results = self.render_func(images, results, cameras, self.extra_mesh)
        output = merge(list(render_results.values()), square=True)
        if output.shape[0] > 10000:
            scale = 5000./output.shape[0]
            output = cv2.resize(output, None, fx=scale, fy=scale)
        outname = join(self.out, basename+self.ext)
        imwrite(outname, output)
        return 0

class MIOutputs(Outputs):
    def __init__(self, out, mode, backend, merge=True, scene={}) -> None:
        super().__init__(out, mode, backend, scene=scene)
        self.merge = merge

    def __call__(self, images, results, cameras, basename):
        # 传个subs进来
        # save the results to individual folders
        subs = list(images.keys())
        outputs = {}
        for nv, sub in enumerate(subs):
            if sub in results.keys():
                result = results[sub]
            # consider the case that the result is not in the results
            # especially in multiview results and render novel view
            else:
                result = results
                value0 = list(result.values())[0]
                if 'vertices' not in value0.keys():
                    continue
            for i, mesh in enumerate(self.scene):
                result[10000+i] = mesh
            if self.mode == 'instance-mask' or self.mode == 'instance-depth':
                # TODO: use depth to render instance mask and consider occlusion
                for pid, value in result.items():
                    if 'vertices' not in value.keys():
                        print('[ERROR] vis view {}, pid {}'.format(sub, pid))
                    output = self.render_func({sub:images[sub]}, 
                    {pid:value}, 
                    {sub:cameras[sub]}, self.extra_mesh)
                    outname = join(self.out, sub, basename + '_{}'.format(pid) + self.ext)
                    imwrite(outname, output[sub])
                    continue
            else:
                output = self.render_func({sub:images[sub]}, 
                    result, 
                    {sub:cameras[sub]}, self.extra_mesh)
                outputs[sub] = output[sub]
        if len(outputs.keys()) == 0:
            return 0
        if self.merge:
            outname = join(self.out, basename + self.ext)
            outputs = merge(list(outputs.values()), square=True)
            imwrite(outname, outputs)
        else:
            for nv, sub in enumerate(subs):
                if sub not in outputs.keys():continue
                outname = join(self.out, sub, basename + self.ext)
                imwrite(outname, outputs[sub])
        return 0