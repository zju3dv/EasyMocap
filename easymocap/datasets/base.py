# This file provides the base class for dataset
from os.path import join
import os
from glob import glob
import numpy as np

from easymocap.dataset.config import coco17tobody25

from ..mytools.vis_base import merge, plot_keypoints_auto, plot_keypoints_total
from ..mytools.camera_utils import Undistort, unproj, read_cameras
from ..mytools.file_utils import read_json, write_keypoints3d, save_json
import cv2
from tqdm import tqdm
from ..estimator.wrapper_base import bbox_from_keypoints
from ..annotator.file_utils import save_annot
from ..mytools.debug_utils import log_time, myerror, mywarn, log
import time

smooth_bbox_cache = {}
def smooth_bbox(bbox, name, W=5):
    if name not in smooth_bbox_cache.keys():
        smooth_bbox_cache[name] = [bbox] * W
    smooth_bbox_cache[name].append(bbox)
    bbox_ = np.stack(smooth_bbox_cache[name][-W:] + [bbox])
    bbox_mean = np.sum(bbox_[:, :4] * bbox_[:, 4:], axis=0)/(1e-5 + np.sum(bbox_[:, 4:], axis=0))
    vel_mean = (bbox_[1:, :4] - bbox_[:-1, :4]).mean()
    bbox_pred = bbox_mean[:4] + vel_mean * (W-1)//2
    conf_mean = bbox_[:, 4].mean()
    bbox_ = list(bbox_pred[:4]) + [conf_mean]
    return bbox_

def get_allname(root0, subs, ranges, root, ext, **kwargs):
    image_names = []
    count = 0
    for sub in subs:
        imgnames = sorted(glob(join(root0, root, sub, '*'+ext)))
        if len(imgnames) == 0:
            myerror('No image found in {}'.format(join(root0, root, sub)))
            continue
        if ranges[1] == -1:
            _ranges = [ranges[0], len(imgnames), ranges[-1]]
        else:
            _ranges = ranges
        nv = subs.index(sub)

        if len(imgnames) < _ranges[1]:
            raise ValueError('The number of images in {} is less than the range: {} vs {}'.format(join(root0, root, sub), len(imgnames), _ranges[1]))

        for nnf, nf in enumerate(range(*_ranges)):
            image_names.append({
                'sub': sub,
                'index': count,
                'frame': int(os.path.basename(imgnames[nf]).split('.')[0]),
                'nv': subs.index(sub),
                'nf': nnf,
                'imgname': imgnames[nf],
            })
            count += 1
    return image_names

def crop_image(img, bbox, crop_square=True):
    l, t, r, b, c = bbox
    if c <0.001: # consider the failed bbox
        l, t = 0, 0
        r, b = img.shape[1], img.shape[0]
    if crop_square:
        if b - t > r - l:
            diff = (b - t) - (r - l)
            l -= diff//2
            r += diff//2
        else:
            diff = (r - l) - (b - t)
            t -= diff//2
            b += diff//2
    l = max(0, int(l+0.5))
    t = max(0, int(t+0.5))
    r = min(img.shape[1], int(r+0.5))
    b = min(img.shape[0], int(b+0.5))
    crop_img = img[t:b, l:r, :]
    if crop_square:
        # 先padding黑边
        if crop_img.shape[0] < crop_img.shape[1] - 1:
            length = crop_img.shape[1] - crop_img.shape[0]
            padding0 = np.zeros((length//2, crop_img.shape[1], 3), dtype=np.uint8)
            padding1 = np.zeros((length - length//2, crop_img.shape[1], 3), dtype=np.uint8)
            crop_img = np.concatenate([padding0, crop_img, padding1], axis=0)
        elif crop_img.shape[0] > crop_img.shape[1] + 1:
            length = crop_img.shape[0] - crop_img.shape[1]
            padding0 = np.zeros((crop_img.shape[0], length//2, 3), dtype=np.uint8)
            padding1 = np.zeros((crop_img.shape[0], length - length//2, 3), dtype=np.uint8)
            crop_img = np.concatenate([padding0, crop_img, padding1], axis=1)
        crop_img = cv2.resize(crop_img, (256, 256))
    return crop_img

logo = cv2.imread(join(os.path.dirname(__file__), '..', '..', 'logo.png'), cv2.IMREAD_UNCHANGED)

def add_logo(img, logo_size=0.1):
    H, W = img.shape[:2]
    scale = H*logo_size / logo.shape[0]
    logo_ = cv2.resize(logo, (int(logo.shape[1]*scale), int(logo.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    local = img[:logo_.shape[0], :logo_.shape[1], :]
    mask = logo_[..., 3:]/255.
    local = 1.*logo_[..., :3]*mask + local*(1.-mask)
    local = local.astype(np.uint8)
    img[:logo_.shape[0], :logo_.shape[1], :] = local
    return img

class BaseData:
    def __init__(self) -> None:
        self.cache_shape = {}

    @staticmethod
    def annots_to_numpy(annots, filter):
        filters = []
        height, width = annots['height'], annots['width']
        for data in annots['annots']:
            for key, val in data.items():
                if isinstance(val, list) and len(val) == 0:
                    raise NotImplementedError
                if isinstance(val, list) and isinstance(val[0], list):
                    data[key] = np.array(val, dtype=np.float32)
            if 'bound' in filter.keys():
                thres = filter['bound']
                kpts = data['keypoints']
                valid = (kpts[:, 0] > thres * width) & (kpts[:, 0] < (1-thres)*width) & (kpts[:, 1] > thres * height) & (kpts[:, 1] < (1-thres)*height)
                kpts[~valid] = 0
            if 'min_conf' in filter.keys():
                conf = data['keypoints'][:, -1]
                data['keypoints'][conf<filter['min_conf']] = 0
            if 'min_joint' in filter.keys():
                valid = data['keypoints'][:, -1] > 0
                if valid.sum() < filter['min_joint']:
                    continue
            if 'coco17tobody25' in filter.keys() and 'keypoints' in data.keys() and data['keypoints'].shape[0] == 17:
                data['keypoints'] = coco17tobody25(data['keypoints'])
            filters.append(data)
        annots['annots'] = filters
        return annots

    @staticmethod
    def read_image_with_scale(imgname, scale=1):
        assert os.path.exists(imgname), '{} not exists'.format(imgname)
        img = cv2.imread(imgname)
        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        return img

    # TODO: 多视角多人的拆分成单独的数据
    def collect_data(self, data_all):
        keys = list(set(sum([list(d.keys()) for d in data_all], [])))
        if len(keys) < len(list(self.cache_shape.keys())):
            mywarn(keys)
            mywarn('Not enough key, {}'.format(data_all[0]['imgname']))
            keys = list(set(keys + list(self.cache_shape.keys())))
        ret = {}
        for key in keys:
            ret[key] = []
            for d in data_all:
                if key not in d:
                    mywarn('Not enough key {}: {}'.format(d['imgname'], key))
                    if self.loadmp:
                        if key not in self.cache_shape.keys():
                            if key.endswith('_distort') or key.endswith('_unproj'):
                                self.cache_shape[key] = self.cache_shape[key.replace('_distort', '').replace('_unproj', '')]
                            else:
                                mywarn('{} not in {}'.format(key, self.cache_shape.keys()))
                        ret[key].append(np.zeros((0, *self.cache_shape[key].shape[1:])))
                    else:
                        # 单人的情况，读入一个全是0的
                        ret[key].append(np.zeros_like(self.cache_shape[key]))
                else:
                    ret[key].append(d[key])
                    if key not in self.cache_shape.keys():
                        if key in ['dist', 'K', 'RT', 'Tc', 'Rc', 'annname', 'imgname', 'KRT', 'pid', 'annots']:
                            continue
                        self.cache_shape[key] = np.zeros_like(ret[key][0][0])
                        log('[Info] Load {} with shape {}'.format(key, self.cache_shape[key].shape))
        if self.loadmp: # TODO: compose datasets in multi view
            for key in ['K', 'RT', 'Rc', 'Tc', 'KRT']:
                if key not in ret.keys(): continue
                ret[key] = np.stack(ret[key], axis=0)
            return ret
        for key in keys:
            if isinstance(ret[key][0], np.ndarray):
                ret[key] = np.stack(ret[key])
                if key not in self.cache_shape.keys():
                    self.cache_shape[key] = np.zeros_like(ret[key][0])
                    log('[Info] Load {} with shape {}'.format(key, self.cache_shape[key].shape))
        return ret

    @staticmethod
    def write_image(outname, img):
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, add_logo(img))

    @staticmethod
    def write_params(outname, params, camera=None):
        out = ['{']
        if camera is not None:
            for val in str(camera).split('\n'):
                out.append(' '*4+val)
            out[-1] += ','
        out.append('    "annots": [')
        for i, human in enumerate(params):
            out.append(' '*8+'{')
            for val in str(human).split('\n'):
                out.append(' '*12+val)
            out.append(' '*8+'}')
            if i != len(params) - 1:
                out[-1] += ','
        out.append('    ]')
        out.append('}')
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        print('\n'.join(out), file=open(outname, 'w'))

class Base(BaseData):
    def __init__(self, path, subs, ranges,
        reader, writer, subs_vis,
        camera='none',
        filter={},
        out=None) -> None:
        super().__init__()
        self.root = path
        self.image_args = reader.image
        self.annot_args = reader.annot
        self.subs = self.check_subs(path, subs)
        self.subs_vis = subs_vis if len(subs_vis) > 0 else self.subs
        self.ranges = self.check_ranges(path, ranges)
        self.image_names = get_allname(self.root, self.subs, self.ranges, **reader.image)
        if len(self.image_names) == 0 and reader.image.ext == 'jpg':
            mywarn('Try to find png images')
            reader.image.ext = 'png'
            self.image_names = get_allname(self.root, self.subs, self.ranges, **reader.image)
        self.reader = reader
        self.writer = writer
        if camera != 'none':
            if not os.path.exists(camera) and not os.path.isabs(camera):
                camera = join(self.root, camera)
            if os.path.exists(camera):
                cameras = read_cameras(camera)
            else:
                cameras = None
        else:
            cameras = None
        self.cameras = cameras
        self.distortMap = {}
        self.out = out
        self.cache_shape = {}
        self.filter = filter

    def __str__(self) -> str:
        return '''dataset {} has {} items
  - in {}\n  - views: {}\n  - ranges: {}'''.format(self.__class__.__name__, len(self), self.root, self.subs, self.ranges)

    def __len__(self):
        return len(self.image_names)

    def check_subs(self, path, subs):
        if len(subs) == 0:
            subs = sorted(os.listdir(join(path, self.image_args['root'])))
        log('[Info] Load {} folders: {}...'.format(len(subs), ' '.join(subs[:4])))
        return subs
    
    def check_ranges(self, path, ranges):
        log('[Info] Load {} frames'.format(ranges))
        return ranges

    def get_view(self, index):
        data = self.image_names[index]
        return data['nv'], data['sub']
    
    def get_frame(self, index):
        data = self.image_names[index]
        return data['nf'], data['frame']
    
    def get_anyname(self, root, sub, frame, ext):
        name = join(self.root, root, sub, '{:06d}.{}'.format(frame, ext))
        # assert os.path.exists(name), name
        return name
    
    def get_imgname(self, sub, frame):
        return self.get_anyname(self.image_args['root'], sub, frame, self.image_args['ext'])

    def get_annname(self, sub, frame):
        return self.get_anyname(self.annot_args['root'], sub, frame, self.annot_args['ext'])

    def add_dimension(self, data):
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                data[key] = np.expand_dims(val, axis=0)
        return data

    def show(self, vis):
        cv2.imshow('vis', vis)
        cv2.waitKey(10)

    def __getitem__(self, index):
        # just return data in one image
        # Note: nv, nf represent the index of view andn frame
        # not the actual view and frame
        nv, sub = self.get_view(index)
        nf, frame = self.get_frame(index)
        imgname = self.get_imgname(sub, frame)
        annname = self.get_annname(sub, frame)
        if not os.path.exists(annname) and not self.annot_args.read:
            annname = annname
            annots = {}
        else:
            annots = read_json(annname)
            annots = self.annots_to_numpy(annots, self.filter)
        ret = {
            'imgname': imgname,
            'annname': annname,
            'annots': annots
        }
        if self.image_args.read_image:
            ret['img'] = cv2.imread(ret['imgname'])
        if self.cameras is not None:
            ret['K'] = self.cameras[sub]['K'].astype(np.float32)
            ret['dist'] = self.cameras[sub]['dist'].astype(np.float32)
            ret['Rc'] = self.cameras[sub]['R'].astype(np.float32)
            ret['Tc'] = self.cameras[sub]['T'].astype(np.float32)
            ret['RT'] = np.hstack([ret['Rc'], ret['Tc']])
            ret['KRT'] = ret['K'] @ ret['RT']
        return ret
    
    def vis_body(self, body_model, params, img, camera, scale=1, mode='image'):
        vis = img.copy()
        K = camera.K.copy()
        if scale != 1:
            vis = cv2.resize(vis, None, fx=scale, fy=scale)
            K[:2, :] *= scale

        meshes = {}
        from ..visualize.pyrender_wrapper import plot_meshes
        for param in params:
            vertices = body_model.vertices(param, return_tensor=False)[0]
            meshes[param.id+1] = {
                'vertices': vertices,
                'faces': body_model.faces,
                'id': param.id,
                'name': 'human_{}'.format(param.id)
            }
        ret = plot_meshes(vis, meshes, K, camera.R, camera.T, mode=mode)
        return ret

    def reshape_data(self, infos):
        return infos

    def read_image(self, imgname):
        assert os.path.exists(imgname), "image {} not exists".format(imgname)
        sub = os.path.basename(os.path.dirname(imgname))
        img = cv2.imread(imgname)
        if self.cameras is None:
            return img
        K, D = self.cameras[sub]['K'], self.cameras[sub]['dist']
        if np.linalg.norm(D) < 1e-3:
            return img
        if sub not in self.distortMap.keys():
            h,  w = img.shape[:2]
            mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (w,h), 5)
            self.distortMap[sub] = (mapx, mapy)
        mapx, mapy = self.distortMap[sub]
        img = cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST)
        return img

    def write(self, body_model, body_params, data, cameras):
        annnames = data['annname']
        if len(annnames) > 1:
            annnames = tqdm(annnames, desc='writing')
        for index, annname in enumerate(annnames):
            # get output name
            splitname = annname.split(os.sep)
            splitname = splitname[splitname.index(self.annot_args.root)+1:]
            splitname[-1] = splitname[-1].replace('.json', '')

            camera = cameras[index]
            imgname = data['imgname'][index]
            params = body_params[index]
            if params.poses.shape[0] == 1:
                params = [params]
                params[0]['id'] = 0
            else:
                params = params.to_multiperson(self.pids)

            if 'render' in self.writer.keys() and self.writer.render.enable:
                img = self.read_image(imgname)
                vis_mesh = self.vis_body(body_model, params, img, camera, scale=self.writer.render.scale, mode=self.writer.render.mode)
                meshname = join(self.out, self.writer['render']['root'], *splitname) + self.writer['render']['ext']
                os.makedirs(os.path.dirname(meshname), exist_ok=True)
                self.write_image(meshname, vis_mesh)
            # write json
            outname = join(self.out, 'smpl', *splitname) +'.json'
            self.write_params(outname, params, camera)
            # write full pose
            fullname = join(self.out, 'smplfull', *splitname) +'.json'
            for i, _param in enumerate(params):
                _param['poses'] = body_model.export_full_poses(**_param)
            self.write_params(fullname, params, camera)

class ImageFolder(Base):
    def __init__(self, keys, pid=0, loadmp=False, compose_mp=False, **kwargs):
        super().__init__(**kwargs)
        self.keys = keys
        self.loadmp = loadmp
        self.compose_mp = compose_mp
        self.pid = pid
        self.read_flag = {k:True for k in keys.keys()}
        self.bboxmap = {'keypoints2d': 'bbox', 'keypoints': 'bbox',
            'handl2d': 'bbox_handl2d', 'handr2d': 'bbox_handr2d', 'face2d': 'bbox_face2d'}

    def __getitem__(self, index, K=None, dist=None):
        data = super().__getitem__(index)
        if 'K' in data.keys():
            K, dist = data['K'], data['dist']
        if K is not None:
            invK = np.linalg.inv(K)
        for key, keyname in self.keys.items():
            if not self.read_flag[key]: continue
            if len(data['annots']['annots']) < 1:
                mywarn("no annotations in {}".format(data['imgname']))
            if index == 0 and len(data['annots']['annots']) >= 1 and keyname not in data['annots']['annots'][0].keys():
                # self.read_flag[key] = False
                mywarn("[data] Disable loading {}".format(key))
                # continue
            if self.loadmp:
                for i, data_ in enumerate(data['annots']['annots']):
                    if keyname not in data_.keys():
                        if key not in self.cache_shape.keys() and self.read_flag[keyname]:
                            cache_shape = {
                                'handl2d': np.zeros((21, 3)),
                                'handr2d': np.zeros((21, 3)),
                                'face2d': np.zeros((21, 3))
                            }
                            mywarn("no {} in {}".format(keyname, data['imgname']))
                            data_[key] = cache_shape[key].copy()
                        else:
                            data_[key] = self.cache_shape[key].copy()
                    else:
                        data_[key] = data_[keyname]
                        if isinstance(data_[key], np.ndarray):
                            self.cache_shape[key] = np.zeros_like(data_[key])
                    data_[key+'_distort'] = data_[key].copy()
                    if K is not None:
                        data_[key] = Undistort.points(data_[key], K, dist)
                        data_[key+'_unproj'] = unproj(data_[key], invK)
            else:
                if len(data['annots']['annots']) > 0:
                    annots = [d for d in data['annots']['annots'] if d['personID']==self.pid]
                    if len(annots) == 0:
                        data[key] = self.cache_shape[key].copy()
                    elif keyname not in annots[0].keys():
                        mywarn("no {} in {}".format(keyname, data['imgname']))
                        if key not in self.cache_shape.keys():
                            mywarn("no {} in cache_shape".format(key))
                            cache_shape = {
                                'handl2d': np.zeros((21, 3)),
                                'handr2d': np.zeros((21, 3)),
                                'face2d': np.zeros((21, 3))
                            }
                            data[key] = cache_shape[key].copy()
                        else:
                            data[key] = self.cache_shape[key].copy()
                    else:
                        data[key] = annots[0][keyname]
                    data[key+'_distort'] = data[key].copy()
                    if K is not None:
                        data[key] = Undistort.points(data[key], K, dist)
                        data[key+'_unproj'] = unproj(data[key], invK)
                    for _key in [key, key+'_distort', key+'_unproj']:
                        try:
                            self.cache_shape[_key] = np.zeros_like(data[_key])
                        except KeyError:
                            print(f"missed key: {_key}")
        if self.loadmp:
            data['annots'] = data['annots']['annots']
            # compose the data
            data['pid'] = [d['personID'] for d in data['annots']]
            for key, keyname in self.keys.items():
                if not self.read_flag[key]: continue
                if len(data['annots']) == 0: continue
                data[key] = np.stack([d[key] for d in data['annots']])
                if len(data['annots']) > 0 and key+'_unproj' in data['annots'][0].keys():
                    data[key+'_unproj'] = np.stack([d[key+'_unproj'] for d in data['annots']])
                data[key+'_distort'] = np.stack([d[key+'_distort'] for d in data['annots']])
        else:
            data.pop('annots')
        if not self.loadmp:
            if 'depth' in self.reader.keys():
                depthname = join(self.root, self.reader['depth']['root'], self.get_view(index)[1], '{}.png'.format(os.path.basename(data['annname']).replace('.json', '')))
                depthmap = cv2.imread(depthname, cv2.IMREAD_UNCHANGED)
                depthmap = depthmap.astype(np.float32)/1000.
                depths = np.zeros_like(data['keypoints2d'][:, :2])
                for i, (x, y, c) in enumerate(data['keypoints2d']):
                    if c < 0.3:continue
                    if i >= 15:continue
                    x, y = int(x+0.5), int(y+0.5)
                    if x > depthmap.shape[0] or y > depthmap.shape[1] or x < 0 or y < 0:
                        continue
                    d_value = depthmap[y, x]
                    if d_value < 0.1:continue
                    depths[i, 0] = d_value
                    depths[i, 1] = c
                data['depth'] = depths
        return data
    
    def vis_data(self, data, img=None):
        if img is None:
            img = self.read_image(data['imgname'])
        from easymocap.mytools.vis_base import plot_keypoints_auto
        plot_keypoints_auto(img, data['keypoints2d'], pid=0)
        return img
    
class MultiVideo(ImageFolder):
    def __init__(self, pids=[0], **kwargs):
        if 'camera' in kwargs['reader'].keys():
            kwargs['camera'] = 'none'
        super().__init__(**kwargs)
        # get image names for each video
        self.image_dict = {}
        self.pids = pids
        for sub in self.subs:
            self.image_dict[sub] = [d for d in self.image_names if d['sub']==sub]

    def __len__(self):
        return len(self.subs)
    
    def __getitem__(self, index):
        # collect the data for each video
        data_all = []
        sub = self.subs[index]
        camera_for_each_image = False
        if 'camera' in self.reader.keys():
            camera_for_each_image = True
            if os.path.exists(join(self.root, self.reader['camera'], sub)):
                cameras = read_cameras(join(self.root, self.reader['camera'], sub))
            elif os.path.exists(join(self.root, self.reader['camera'])):
                cameras = read_cameras(join(self.root, self.reader['camera']))
            else:
                myerror("You must give a valid camera path")
                raise NotImplementedError
        for info in tqdm(self.image_dict[sub], 'Loading {}'.format(sub)):
            basename = os.path.basename(info['imgname']).split('.')[0]
            if camera_for_each_image:
                if basename in cameras.keys():
                    camera = cameras[basename]
                elif sub+'/'+basename in cameras.keys():
                    camera = cameras[sub+'/'+basename]
                else:
                    myerror("You must give a valid camera")
                    raise NotImplementedError
                K, dist = camera['K'], camera['dist']
                data = super().__getitem__(info['index'], K=K, dist=dist)
                for oldkey, newkey in [('K', 'K'), ('dist', 'dist'), ('R', 'Rc'), ('T', 'Tc')]:
                    data[newkey] = camera[oldkey].astype(np.float32)
            else:
                data = super().__getitem__(info['index'])
            data_all.append(data)
        # load camera for each sub
        ret = self.collect_data(data_all)
        if self.loadmp and self.compose_mp: # 针对镜子的情况，需要load多人的数据
            for key in ['keypoints2d', 'keypoints2d_distort', 'keypoints2d_unproj']:
                if len(self.pids) > 0:
                    for i in range(len(ret[key])):
                        ret[key][i] = ret[key][i][:len(self.pids)]
                shapes = set([v.shape for v in ret[key]])
                if len(shapes) > 1:
                    myerror('The shape is not the same!')
                ret[key] = np.stack(ret[key])
            ret['pid'] = self.pids
        return ret
    
    def reshape_data(self, infos):
        for key in ['imgname', 'annname']:
            infos[key] = [d[0] for d in infos[key]]
        for key, val in infos.items():
            if 'torch' in str(type(val)):
                infos[key] = val[0]
        infos['nFrames'] = infos['keypoints2d'].shape[0]
        return infos

class MultiView(ImageFolder):
    def __init__(self, pids=[], *args, **kwargs):
        if kwargs['camera'] == 'none':
            kwargs['camera'] = kwargs['path']
        super().__init__(*args, **kwargs)
        self.frames = list(range(*self.ranges))
        self.pids = pids
        if 'keypoints3d' in self.reader.keys():
            k3ddir = join(self.root, self.reader.keypoints3d.root)
            if not os.path.exists(k3ddir):
                self.reader.keypoints3d.read = False
            self.cache_3dshape = {}
        # from ..mytools.camera_utils import get_fundamental_matrix
        # F = get_fundamental_matrix(self.cameras, self.subs)

    def check_ranges(self, path, ranges):
        if ranges[1] == -1:
            subs = self.subs
            maxlength = 999999
            for sub in subs:
                length = len(os.listdir(join(path, self.image_args['root'], sub)))
                if self.annot_args['check_length']:
                    length_ = len(os.listdir(join(path, self.annot_args['root'], sub)))
                    length = min(length, length_)
                if length < maxlength:
                    maxlength = length
            ranges = [ranges[0], maxlength, ranges[-1]]
        return ranges
    
    def collect_data(self, data_all):
        ret = super().collect_data(data_all)
        pids = self.pids
        if self.loadmp and self.compose_mp:
            nViews = ret['K'].shape[0]
            ret_compose = {'pid': pids}
            for key, zero_shape in self.cache_shape.items():
                # for fix in ['', '_unproj', '_distort']:
                if True:
                    fix = ''
                    # output shape: (nPerson, nViews, nJoints, 3)
                    val = np.zeros((nViews, len(pids), *zero_shape.shape), dtype=np.float32)
                    for nv in range(nViews):
                        pids_now = ret['pid']
                        if isinstance(pids_now, list):
                            pids_now = ret['pid'][nv]
                        for npid, pid, in enumerate(pids_now):
                            if pid not in pids:continue
                            val[nv][pids.index(pid)] = ret[key+fix][nv][npid]
                    ret_compose[key+fix] = val
            ret.update(ret_compose)
            ret.pop('annots')
        return ret

    def padding_keypoints3d(self, k3ds):
        ret = {}
        for key in self.reader.keypoints3d.key:
            if key not in k3ds.keys() and key in self.cache_3dshape.keys():
                ret[key] = self.cache_3dshape[key].copy()
                continue
            elif key not in k3ds.keys():
                continue
            ret[key] = np.array(k3ds[key], dtype=np.float32)
            if key not in self.cache_3dshape.keys():
                self.cache_3dshape[key] = np.zeros_like(ret[key])
        return ret

    def reshape_data(self, infos):
        infos['nFrames'] = infos['K'].shape[0]
        if self.compose_mp:
            infos['nPerson'] = len(self.pids)
            log('[Info] Load person {}'.format(self.pids))

        return super().reshape_data(infos)

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        data_all = []
        for nv, sub in enumerate(self.subs):
            idx = nv * len(self.frames) + index
            data = super().__getitem__(idx)
            data_all.append(data)
        ret = self.collect_data(data_all)
        if 'keypoints3d' in self.reader.keys() and self.reader.keypoints3d.read:
            k3ddir = join(self.root, self.reader.keypoints3d.root)
            if os.path.exists(k3ddir):
                basename = os.path.basename(data_all[0]['annname'])
                k3ds = read_json(join(k3ddir, basename))
                if self.loadmp:
                    if len(self.pids) == 0:
                        myerror('You do not input pids in the multi-person mode!')
                        raise NotImplementedError
                    else:
                        k3ds = {k['id']:k for k in k3ds}
                        k3ds = [self.padding_keypoints3d(k3ds[pid]) for pid in self.pids]
                    for key in self.reader.keypoints3d.key:
                        if key not in k3ds[0].keys():continue
                        ret[key] = np.stack([k[key] for k in k3ds], axis=0)
                else:
                    k3ds = k3ds[0]
                    ret.update(self.padding_keypoints3d(k3ds))
        # for key, val in ret.items():
        #     try:
        #         print(key, val.shape)
        #     except:
        #         pass
        # import ipdb;ipdb.set_trace()
        return ret
    
    def project(self, res3ds, data):
        reproj_allviews = []
        nViews = data['K'].shape[0]
        for nv in range(nViews):
            K, dist = data['K'][nv], data['dist'][nv]
            R, T = data['Rc'][nv], data['Tc'][nv]
            Rvec = cv2.Rodrigues(R)[0]
            annots = []
            for i3d, res3d in enumerate(res3ds):
                res2d = {'personID': res3d['id']}
                for key in ['keypoints3d', 'handl3d', 'handr3d', 'face3d']:
                    if key not in res3d.keys():continue
                    k3d = res3d[key]
                    k3d0 = np.ascontiguousarray(k3d[:, :3])
                    points2d_repro, _ = cv2.projectPoints(
                        k3d0, Rvec, T, K, dist)
                    k2d = np.hstack([points2d_repro[:, 0], k3d[:, -1:]])
                    bbox = bbox_from_keypoints(k2d)
                    k = key.replace('3d', '2d').replace('keypoints2d', 'keypoints')
                    res2d[self.bboxmap[k]] = bbox
                    res2d[k] = k2d
                annots.append(res2d)
            reproj_allviews.append(annots)
        return reproj_allviews

    def write_keypoints3d(self, results, data):
        basename = os.path.basename(data['annname'][0])
        write_ = {'id': 0}
        write_.update(results)
        write_ = [write_]
        outname = join(self.out, self.writer.keypoints3d.root, basename)
        write_keypoints3d(outname, write_, keys=['keypoints3d', 
        'handl3d', 'handr3d', 'face3d'])

    def write_all(self, results, data):
        images = {}
        results_all = []
        for res_ in results:
            res3d = {'id': res_['id']}
            for key in ['keypoints3d', 'handl3d', 'handr3d', 'face3d']:
                if key in res_.keys():
                    res3d[key] = res_[key]
            results_all.append(res3d)
        annnames = data['annname']
        if 'keypoints2d' in data.keys():
            data['keypoints'] = data['keypoints2d'] # for compatibility
        basename = os.path.basename(annnames[0])
        outname = join(self.out, self.writer.keypoints3d.root, basename)
        write_keypoints3d(outname, results_all, keys=['keypoints3d', 
            'handl3d', 'handr3d', 'face3d'])
        # make compatible for old name
        annots_origin, annots_match = [], []
        for nv, annname in enumerate(annnames):
            annots_ori = read_json(annname)
            annots_origin.append(annots_ori.copy())
            res2d_all = []
            for res_ in results:
                if 'keypoints2d' in res_.keys():
                    res_['keypoints'] = res_['keypoints2d']
                res2d = {'personID': res_['id']}
                for key in ['keypoints', 'handl2d', 'handr2d', 'face2d']:
                    if key not in data.keys() or key not in res_.keys():continue
                    if len(res_[key][nv]) == 1:
                        # only have 1 points
                        # caclulate the bbox from size
                        k3d = res_['keypoints3d'].copy()
                        k3d[:, -1] = 1.
                        k3drt = data['RT'][nv] @ k3d.T
                        k2d = data['K'][nv] @ k3drt
                        k2d /= k2d[2,0]
                        radius = 0.5 # 0.1meter
                        scale = data['K'][nv][0,0] * radius/k3drt[2, 0]
                        res2d[self.bboxmap[key]] = [k2d[0,0]-scale, k2d[1,0]-scale, k2d[0,0]+scale, k2d[1,0]+scale, 1.]
                    else:
                        res2d[self.bboxmap[key]] = bbox_from_keypoints(res_[key][nv])
                    res2d[key] = res_[key][nv]
                res2d_all.append(res2d)
            annots_ori['annots'] = res2d_all
            outname = join(self.out, self.writer.keypoints2d.root, os.sep.join(annname.split(os.sep)[-2:]))
            save_annot(outname, annots_ori)
            annots_match.append(annots_ori)
        reproj_allviews = self.project(results_all, data)
        images_cache = {}
        for key, params in self.writer.items():
            if key == 'keypoints3d':
                continue
            elif key == 'vismatch':
                if not params['enable'] and not self.writer.visrepro['enable']:
                    continue
                if params.crop: 
                    params.scale = 1
                visall = []
                for sub in self.subs_vis:
                    nv = self.subs.index(sub)
                    imgname = data['imgname'][nv]
                    if imgname not in images_cache.keys():
                        images_cache[imgname] = self.read_image_with_scale(imgname, params.scale)                    
                    img = images_cache[imgname].copy()
                    if self.writer.vismatch.enable:
                        vis = plot_keypoints_total(img, annots_match[nv]['annots'], params.scale)
                    if self.writer.visrepro.enable:
                        vis = plot_keypoints_total(img, reproj_allviews[nv], params.scale)
                    # if params.crop and len(annots_match[nv]['annots']) == 1:
                    if params.crop and 'bbox' in reproj_allviews[nv][0]:
                        bbox = reproj_allviews[nv][0]['bbox']
                        bbox = smooth_bbox(bbox, name='{}_{}_body'.format(key, nv))
                        vis = crop_image(vis, bbox)
                    # elif params.crop and len(annots_match[nv]['annots']) > 1:
                    #     bbox = smooth_bbox(annots_match[nv]['annots'][0]['bbox'], name='{}_{}_body'.format(key, nv))
                    #     vis = crop_image(vis, bbox)
                    visall.append(vis)
                visall = merge(visall, resize=False)
            elif key == 'visdetect':
                if not params['enable']:
                    continue
                if params.crop: 
                    params.scale = 1
                visall = []
                for sub in self.subs_vis:
                    nv = self.subs.index(sub)
                    imgname = data['imgname'][nv]
                    if imgname not in images_cache.keys():
                        images_cache[imgname] = self.read_image_with_scale(imgname, params.scale)
                    img = images_cache[imgname].copy()
                    vis = plot_keypoints_total(img, annots_origin[nv]['annots'], params.scale)
                    if params.crop and 'bbox' in annots_match[nv]['annots'][0].keys():
                        bbox = smooth_bbox(annots_match[nv]['annots'][0]['bbox'], name='{}_{}_body'.format(key, nv))
                        vis = crop_image(vis, bbox)
                    visall.append(vis)
                visall = merge(visall, resize=False)
            elif key == 'visrepro':
                continue
            else:
                continue
            outname = join(self.out, params.root, os.path.basename(data['imgname'][0]))
            self.write_image(outname, visall)

    def vis_data(self, data):
        from easymocap.mytools.vis_base import merge
        imgnames = data['imgname']
        out = []
        for nv, imgname in enumerate(imgnames):
            img = self.read_image(imgname)
            vis = super().vis_data({'keypoints2d': data['keypoints2d'][nv]}, img)
            for key in ['handl3d', 'handr3d']:
                vis2 = img.copy()
                if key in data.keys():
                    k3d = data[key]
                    k2d_proj = k3d @ data['KRT'][nv].T
                    k2d_proj = k2d_proj[:, :2] / k2d_proj[:, 2:]
                    k2d = np.hstack([k2d_proj, k3d[:, 3:]])
                    plot_keypoints_auto(vis2, k2d, pid=0)
                    # vis = np.hstack([vis, vis2])
                    vis = cv2.addWeighted(vis, 0.5, vis2, 0.5, 0)
            out.append(vis)
        out = merge(out, resize=True)
        return out
    
    def write_offset(self, offset):
        output = {}
        for nv, sub in enumerate(self.subs):
            output[sub] = float(offset[nv])
        if os.path.exists(join(self.root, 'offset.json')):
            gt = read_json(join(self.root, 'offset.json'))
            sub0 = self.subs[0]
            output = {sub:output[sub] - output[sub0] for sub in self.subs}
            gt = {sub:gt[sub] - gt[sub0] for sub in self.subs}
            mean_gt = sum([abs(gt[sub]) for sub in self.subs])/len(self.subs)
            mean_err = sum([abs(gt[sub] - output[sub]) for sub in self.subs])/len(self.subs)
            output['error'] = [mean_gt, mean_err]
            for nv, sub in enumerate(self.subs):
                print('{:s}: est = {:5.2f}, gt = {:5.2f}'.format(sub, output[sub], gt[sub]))
        save_json(join(self.out, 'time_offset.json'), output)
        
    def write(self, body_model, body_params, data, cameras):
        nFrames = body_params['poses'].shape[0]
        if nFrames > 10:
            iterator = tqdm(range(nFrames), desc='writing')
        else:
            iterator = range(nFrames)
        for index in iterator:
            annname0 = data['annname'][0][index]
            basename = os.path.basename(annname0).replace('.json', '')
            outname = join(self.out, 'smpl', basename) +'.json'
            params = body_params[index]
            # 单个视角 + 单个人
            # 多个视角 + 单个人
            # 单个视角 + 多个人
            # 多个视角 + 多个人
            if self.loadmp:
                for nv, sub in enumerate(self.subs):
                    if len(params.poses.shape) == 3: # with different views
                        param_v = params[nv]
                    else:
                        param_v = params
                    outname = join(self.out, 'smpl', sub, basename+'.json')
                    _params = param_v.to_multiperson(self.pids)
                    self.write_params(outname, _params)
                    fullname = join(self.out, self.writer.fullpose.root, sub, basename+'.json')
                    for i, _param in enumerate(_params):
                        _param['poses'] = body_model.export_full_poses(**_param)
                    self.write_params(fullname, _param)
            else:
                if params.poses.shape[0] == 1:
                    # if the model is 'manolr', the shape is (2, ndim)
                    self.write_params(outname, [params])
                    # write full poses
                    outname = join(self.out, self.writer.fullpose.root, basename) +'.json'
                    params['poses'] = body_model.export_full_poses(**params)
                    self.write_params(outname, [params])
                else: # 单人的；每个视角都不相同的情况
                    for nv, sub in enumerate(self.subs):
                        param_v = params[nv]
                        outname = join(self.out, 'smpl', sub, basename+'.json')
                        self.write_params(outname, [param_v])
                        fullname = join(self.out, self.writer.fullpose.root, sub, basename+'.json')
                        param_v['poses'] = body_model.export_full_poses(**param_v)
                        self.write_params(fullname, [param_v])
            # 可视化
            if not self.writer.render.enable:
                continue
            vis = []
            subs_all = self.subs
            if 'subs' in self.writer.render.keys():
                _subs = self.writer.render.subs
                subs = [s for s in _subs if s in self.subs]
                if len(subs) == 0: 
                    subs = self.subs
                    subs = subs[::len(subs)//len(_subs)]
            else:
                subs = self.subs_vis
            if len(subs) > 100:
                subs = subs[::len(subs)//25]
            for sub in subs:
                nv = subs_all.index(sub)
                img = self.read_image(data['imgname'][nv][index])
                camera = cameras[index][nv]
                if self.loadmp:
                    params = body_params[index]
                    if len(params.poses.shape) == 3:
                        params = params[nv]
                    params = params.to_multiperson(self.pids)
                else:
                    params = body_params[index]
                    if len(params.poses.shape) == 3 or (params.poses.shape[0] != 1):
                        params = params[nv]
                    params = [params]
                vis_mesh = self.vis_body(body_model, params, img, camera, scale=self.writer.render.scale, mode=self.writer.render.mode)
                vis.append(vis_mesh)
            from ..mytools.vis_base import merge
            vis = merge(vis, resize=False)
            outname = join(self.out, self.writer['render']['root'], basename) + self.writer['render']['ext']
            self.write_image(outname, vis)

def create_skeleton_model(nJoints):
    config = {
        'body_type': 'body25vis',
        'joint_radius': 0.02,
        'vis_type': 'cone',
        'res': 20,
    }
    from ..visualize.skelmodel import SkelModelFast
    if nJoints == 21:
        config['body_type'] = 'handvis'
        config['joint_radius'] = 0.005
    return SkelModelFast(**config)
        
class Keypoints3D(BaseData):
    def __init__(self, path, out, ranges, reader, writer):
        self.root = path
        self.out = out
        self.ranges = ranges
        self.reader = reader
        self.writer = writer
        self.data_all = self.cache_all_data(ranges)
    
    def cache_all_data(self, ranges):
        cfg = self.reader.keypoints3d
        names = sorted(glob(join(self.root, cfg.root, '*'+cfg.ext)))
        ranges[1] = len(names) if ranges[1] == -1 else ranges[1]
        names = [names[i] for i in range(ranges[0], ranges[1], ranges[2])]
        data_all = []
        for name in tqdm(names, desc='cache keypoints3d'):
            annots = read_json(name)
            data_all.append({
                'annname': name,
                'annots': annots,
            })
        return data_all

    def __len__(self):
        return len(self.data_all)
    
    def __getitem__(self, index):
        data = self.data_all[index]
        annots = data['annots'][0]
        ret = {
            'annname': data['annname'],
        }
        for outname, inpname in self.reader.keypoints3d.map.items():
            inp = np.array(annots[inpname], dtype=np.float32)
            ret[outname] = inp
        return ret
    
    def reshape_data(self, data):
        data['nFrames'] = data['keypoints3d'].shape[0]
        return data
    
    def write(self, body_model, body_params, data):
        config = {
            'align_first': True,
            'vis_skel': True,
            'vis_size': 1024,
            'z_offset': 1.5,
            'x_offset': 0.15,
            'rot_axis': [0., 1., 0.],
            'rot_angle': np.pi/2
        }
        if config['vis_skel']:
            vis_W = config['vis_size'] * 2
            vis_H = config['vis_size']
        else:
            vis_W, vis_H = config['vis_size'], config['vis_size']
        vis = np.zeros((vis_H, vis_W, 3), dtype=np.uint8) + 255

        K = np.eye(3)
        K[0, 0] = 5*config['vis_size']
        K[1, 1] = 5*config['vis_size']
        K[0, 2] = vis_W//2
        K[1, 2] = vis_H//2
        R = np.eye(3)
        T = np.zeros((3, 1))
        T[2, 0] = config['z_offset']
        x_offset = config['x_offset']
        invRot = cv2.Rodrigues(np.array(config['rot_axis'])*config['rot_angle'])[0]
        left = np.array([[-x_offset, 0., 0.]]).reshape(1, 3)
        right = np.array([[x_offset, 0., 0.]]).reshape(1, 3)
        keypoints3d_gt = data['keypoints3d'].cpu().numpy()
        traj_gt = keypoints3d_gt.mean(axis=1)
        keypoints3d = body_model.keypoints(body_params, return_tensor=False)
        skel_model = create_skeleton_model(keypoints3d.shape[-2])
        traj = keypoints3d.mean(axis=1)
        annnames = data['annname']
        if len(annnames) > 1:
            annnames = tqdm(annnames, desc='writing')
        from ..visualize.pyrender_wrapper import plot_meshes
        for nf, annname in enumerate(annnames):
            basename = os.path.basename(annname).replace('.json', '')
            if config['align_first']:
                invT = - traj[:1]
            else:
                invT = - traj_gt[nf:nf+1]
            params = body_params[nf]
            params['id'] = 0
            outname = join(self.out, 'smpl', basename) +'.json'
            self.write_params(outname, [params])
            outname = join(self.out, self.writer.fullpose.root, basename) +'.json'
            params['poses'] = body_model.export_full_poses(**params)
            self.write_params(outname, [params])
            # write image
            vertices = body_model.vertices(params, return_tensor=False)[0] + invT
            vertices = vertices @ invRot.T + right
            keypoints = keypoints3d[nf]
            mesh = {
                'id': 0,
                'name': 'human_0',
                'vertices': vertices,
                'faces': body_model.faces,
            }
            kpts_gt = keypoints3d_gt[nf]
            vertices_skel = skel_model(kpts_gt)[0] + invT
            vertices_skel = vertices_skel @ invRot.T + left
            mesh_skel = {
                'id': 1,
                'name': 'skel_0',
                'vertices': vertices_skel,
                'faces': skel_model.faces,
            }
            ret = plot_meshes(vis.copy(), {0:mesh, 1:mesh_skel}, K, R, T)
            outname = join(self.out, self.writer['render']['root'], basename) + self.writer['render']['ext']
            self.write_image(outname, ret)

class MultiViewSocket(BaseData):
    def __init__(self, host, nFrames, keys, camera) -> None:
        super().__init__()
        self.loadmp = False
        from ..socket.detect_client import BaseSocketClient
        self.socket = BaseSocketClient(host)
        self.nFrames = nFrames
        if camera == 'none':
            self.cameras = self.socket.cameras
        else:
            self.cameras = read_cameras(camera)
        self.subs = list(self.cameras.keys())
        self.keys = keys
        self.read_flag = {k:True for k in keys.keys()}
        self.nViews = len(self.subs)
    
    def __len__(self):
        return self.nFrames
    
    def getitem_asbase(self, data):
        if 'K' in data.keys():
            K, dist = data['K'], data['dist']
        if K is not None:
            invK = np.linalg.inv(K)
        for key, keyname in self.keys.items():
            if not self.read_flag[key]: continue
            if keyname not in data['annots']['annots'][0].keys():
                self.read_flag[key] = False
                continue
            data[key] = data['annots']['annots'][0][keyname]
            data[key+'_distort'] = data[key].copy()
            if K is not None:
                data[key] = Undistort.points(data[key], K, dist)
                data[key+'_unproj'] = unproj(data[key], invK)
        data.pop('annots')
        return data

    def __getitem__(self, index):
        while len(self.socket.results_all) < 1:
            time.sleep(0.001)
            # mywarn('Waiting for camera')
        if len(self.socket.results_all) > 3:
            self.socket.results_all = self.socket.results_all[-3:]
        result = self.socket.results_all.pop(0)
        # log_time("pop data from queue {}".format(len(self.socket.results_all)))
        mapkeys = {
            '0':'0', 
            '2':'1', 
            '4':'2'}
        if '4' in result.keys():
            result = {mapkeys[k]:result[k] for k in result.keys()}

        # TODO
        # self.cameras = self.socket.cameras
        data_all = []
        frame = index
        # log('frame: {}: {}'.format(frame, result.keys()))
        for nv, sub in enumerate(self.subs):
            annots = result[sub]
            # log('Load {} {} {}'.format(frame, sub, len(annots['annots'])))
            # log('{}'.format(annots['annots'][0].keys()))
            # print(annots)
            annots = self.annots_to_numpy(annots, filter={'coco17tobody25': True})
            ret = {
                'imgname': 'images/{}/{:06d}.jpg'.format(sub, frame),
                'annname': 'annots/{}/{:06d}.jpg'.format(sub, frame),
                'annots': annots
            }
            if False:
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                print(annots['annots'][0]['keypoints'].shape)
                plot_keypoints_auto(blank, annots['annots'][0]['keypoints'], 0)
                plot_keypoints_auto(blank, annots['annots'][0]['handl2d'], 1)
                plot_keypoints_auto(blank, annots['annots'][0]['handr2d'], 2)
                cv2.imshow(sub, blank)
                cv2.waitKey(10)
            if self.cameras is not None:
                ret['K'] = self.cameras[sub]['K'].astype(np.float32)
                ret['dist'] = self.cameras[sub]['dist'].astype(np.float32)
                ret['Rc'] = self.cameras[sub]['R'].astype(np.float32)
                ret['Tc'] = self.cameras[sub]['T'].astype(np.float32)
                ret['RT'] = np.hstack([ret['Rc'], ret['Tc']])
                ret['KRT'] = ret['K'] @ ret['RT']
            ret = self.getitem_asbase(ret)
            data_all.append(ret)
        data_all = self.collect_data(data_all)
        return data_all

class MV1Psocket(MultiViewSocket):
    def __init__(self, pid=0, **cfg) -> None:
        super().__init__(**cfg)
        self.pid = pid
    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        return data
    
    def vis_keypoints2d_mv(self, images, detections, outname=None,
        vis_id=True):
        return super().vis_keypoints2d_mv(images, {self.pid:detections}, 
            outname=outname, vis_id=vis_id, use_limb_color=True)
    
    def write_keypoints3d(self, keypoints3d, outname):
        return super().write_keypoints3d([{'id': self.pid, 'keypoints3d':keypoints3d}], outname)

    def write_all(self, results, data):
        pass

if __name__ == '__main__':
    pass
