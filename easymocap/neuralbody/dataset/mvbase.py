import os
from os.path import join
import json
import cv2
import copy
from ...mytools.reader import read_smpl, read_json
from ...mytools.camera_utils import Undistort, read_cameras
from ...mytools.debug_utils import myerror, mywarn, oncewarn
from ...mytools.timer import Timer
from .utils_sample import get_bounds, get_rays, sample_rays_rate, sample_rays
import numpy as np
from .utils_reader import img_to_numpy, numpy_to_img, read_json_with_cache, parse_semantic

class BaseBase:
    def __init__(self, split):
        self.split = split
        self.infos = []
        self.file_cache = {}
        self.back_mask_cache = {}
        self.cache_ray_o_d = {}
        self.timer = False

    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def scale_and_undistort(self, img, info, undis=True):
        img = img_to_numpy(img)
        if self.image_args.scale != 1:
            H, W = int(img.shape[0] * self.image_args['scale']), int(img.shape[1] * self.image_args['scale'])
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        K, D = info['camera']['K'], info['camera']['dist']
        sub = info['sub']
        if self.image_args.undis and np.linalg.norm(D) > 0. and undis:
            img = Undistort.image(img, K, D, sub=sub)
        
        return img

    def parse_object_args(self, object_keys, object_args, ignore_keys):
        for _class in ['human', 'ball']:
            if 'all'+_class in object_args.keys():
                object_args = object_args.copy()
                _allargs = object_args.pop('all'+_class)
                pids = _allargs.pop('pids')
                for pid in pids:
                    _args = copy.deepcopy(_allargs)
                    _args['args'].pid = pid
                    object_args[_class+'_{}'.format(pid)] = _args
        self.ignore_keys = ignore_keys
        self.object_args = object_args
        self.object_keys = object_keys

    def object_factory(self, root, obj_type, obj_args, info, params, file_cache):
        # 通用的一些参数：例如SMPL
        pid = obj_args.get('pid', -1) # pid only available in human or ball
        sub = info['sub']
        if 'frame' in params:
            frame = params['frame']
        else:
            frame = info['frame']
        feat = {}
        reader = obj_args.get('reader', {}) # reader only available in human
        use_param_foreachview = obj_args.get('use_param_foreachview', False)
        for key, args in reader.items():
            if key == 'smpl':
                if use_param_foreachview and sub.startswith('novel'):
                    sub_ = sorted(os.listdir(join(root, args.root)))[0]
                    smplname = join(root, args.root, sub_, '{:06d}.json'.format(frame))
                elif use_param_foreachview and not sub.startswith('novel'):
                    smplname = join(root, args.root, sub, '{:06d}.json'.format(frame))
                else:
                    smplname = join(root, args.root, '{:06d}.json'.format(frame))
                data = read_json_with_cache(smplname, file_cache)
                if isinstance(data, dict):
                    data = data['annots']
                data = [d for d in data if d['id'] == pid]
                assert len(data) > 0, "{} don't have {}".format(smplname, pid)
                data = data[0]
                for key in ['poses', 'shapes', 'Rh', 'Th']:
                    feat[key] = np.array(data[key], dtype=np.float32)
                feat['R'] = cv2.Rodrigues(feat['Rh'])[0]
            elif key == 'vertices':
                if use_param_foreachview and sub.startswith('novel'):
                    sub_ = sorted(os.listdir(join(root, args.root)))[0]
                    smplname = join(root, args.root, sub_, '{:06d}.json'.format(frame))
                elif use_param_foreachview and not sub.startswith('novel'):
                    smplname = join(root, args.root, sub, '{:06d}.json'.format(frame))
                else:
                    smplname = join(root, args.root, '{:06d}.json'.format(frame))
                data = read_json_with_cache(smplname, file_cache)
                if isinstance(data, dict):
                    data = data['annots']
                data = [d for d in data if d['id'] == pid][0]
                if 'keypoints3d' in data.keys() and 'vertices' not in data.keys():
                    # use keypoints3d instead
                    k3d = np.array(data['keypoints3d'], dtype=np.float32)
                    conf = k3d[:, 3]
                    k3d = k3d[:, :3]
                    data['vertices'] = k3d[conf>0.1]
                for key in ['vertices']:
                    feat[key] = np.array(data[key], dtype=np.float32)
                feat['bounds'] = get_bounds(feat['vertices'], delta=args.padding)
            elif key in ['depth']:
                depthname = join(root, args.root, sub, '{:06d}.png'.format(frame))
                if not os.path.exists(depthname):
                    depth = None
                else:
                    depth = cv2.imread(depthname, cv2.IMREAD_ANYDEPTH)
                    depth = depth.astype(np.float32)/1000
                feat['depth'] = depth
            elif key in ['mask', 'instance', 'label', 'semantic']:
                if args.root == 'none': continue
                mskname = join(root, args.root, sub, '{:06d}_{}.png'.format(frame, pid))
                if not os.path.exists(mskname):
                    mskname0 = join(root, args.root, sub, '{:06d}.png'.format(frame))
                    if pid == 0 and os.path.exists(mskname0):
                        mskname = mskname0
                        oncewarn('Using null pid to read mask')
                    else:
                        if not 'novel_' in mskname:
                            print('!!!{} not exists'.format(mskname))
                        feat[key] = None
                        continue
                assert os.path.exists(mskname), mskname
                if key == 'semantic':
                    msk = cv2.imread(mskname)
                else:
                    msk = cv2.imread(mskname, 0)
                msk = self.scale_and_undistort(msk, info, undis=args.undis)
                if key == 'mask':
                    feat[key] = msk>0
                elif key == 'label':
                    feat[key] = msk
                elif key == 'semantic':
                    feat[key] = parse_semantic(msk)
                    feat['mask'] = feat[key] > 0
                else:
                    raise NotImplementedError
        if obj_type in ['nearfar', 'nearfardepth']: 
            from .utils_sample import NearFarSampler
            obj = NearFarSampler(split=self.split, near=obj_args.near, far=obj_args.far, depth=feat.get('depth', None))
        elif obj_type == 'bbox':
            from .utils_sample import AABBSampler
            obj = AABBSampler(split=self.split, bounds=obj_args.bounds)
        elif obj_type == 'twobbox':
            from .utils_sample import TwoAABBSampler
            obj = TwoAABBSampler(split=self.split, bbox_inter=obj_args.bbox_inter, bbox_outer=obj_args.bbox_outer)
        elif obj_type == 'compose':
            objlist = []
            for key, obj_args_ in obj_args.items():
                obj_ = self.object_factory(root, obj_args_.model, obj_args_, info, params, file_cache)
                objlist.append(obj_)
            from .utils_sample import ComposeSampler
            obj = ComposeSampler(split=self.split, objlist=objlist)
        elif obj_type == 'bodybbox':
            from .utils_sample import AABBwMask, AABBSampler
            if 'vertices' in feat.keys():
                vertices = feat['vertices']
                if 'label' in feat.keys() and feat['label'] is not None:
                    obj = AABBwMask.from_vertices(
                        label=feat['label'].astype(np.float32),
                        mask=feat['label'], # use label to represents mask
                        rate_body=obj_args.rate_body,
                        dilate=True,
                        split=self.split, vertices=vertices, delta=obj_args.reader.vertices.padding)
                else:
                    obj = AABBSampler.from_vertices(
                        split=self.split, vertices=vertices, delta=obj_args.reader.vertices.padding)
            else:
                center = feat['Th']
                obj = AABBSampler(self.split, center=center, scale=obj_args.scale)
        elif obj_type == 'maskbbox':
            from .utils_sample import AABBwMask
            center = feat['Th']
            obj = AABBwMask(split=self.split, center=center, scale=obj_args.scale, mask=feat['mask'])
        elif obj_type == 'neuralbody' or obj_type == 'neuralbody-smplmask':
            from .utils_sample import AABBwMask
            # ATTN
            dilate = obj_type == 'neuralbody'

            if 'rotate_axis' in params.keys():
                axis = {
                    'y': np.array([0., 1., 0.], dtype=np.float32)
                }[params['rotate_axis']]
                Rrel = cv2.Rodrigues(params['rotate_angle']*axis)[0]
                feat['vertices'] = (feat['vertices'] - feat['Th']) @ Rrel.T + feat['Th']
                feat['R'] = Rrel @ feat['R']
                feat['Rh'] = cv2.Rodrigues(feat['R'])[0].reshape(1, 3)
                feat['bounds'] = get_bounds(feat['vertices'], delta=obj_args.reader.vertices.padding)
            if 'rotation' in params.keys() or 'translation' in params.keys():
                # support rotation and translation
                rotation = params.get('rotation', [0., 0., 0.])
                translation = params.get('translation', [0., 0., 0.])
                Rrel = cv2.Rodrigues(np.array(rotation))[0].astype(np.float32)
                Trel = np.array(translation).reshape(1, 3).astype(np.float32)
                # update the original Rh, Th
                feat['R'] = Rrel @ feat['R']
                feat['Rh'] = cv2.Rodrigues(feat['R'])[0].reshape(1, 3)
                feat['Th'] = (Rrel @ feat['Th'].T + Trel.T).T
                feat['vertices'] = feat['vertices'] @ Rrel.T + Trel
                feat['bounds'] = get_bounds(feat['vertices'], delta=obj_args.reader.vertices.padding)
            # calculate canonical vertices and bounds
            feat['vertices_canonical'] = (feat['vertices'] - feat['Th']) @ feat['R'].T.T
            feat['bounds_canonical'] = get_bounds(feat['vertices_canonical'], delta=obj_args.reader.vertices.padding)
            obj = AABBwMask(split=self.split, bounds=feat['bounds'], 
                mask=feat.get('mask', None), 
                label=feat.get('label', None),
                dilate=dilate,
                rate_body=obj_args.rate_body)
            for key in ['R', 'Rh', 'Th', 'vertices', 'poses', 'shapes',
                'vertices_canonical', 'bounds_canonical']:
                obj.feature[key] = feat[key]
            # extra keys
            for key in ['semantic']:
                if key in feat.keys():
                    obj.feature_input[key] = feat[key]
        elif obj_type == 'neuralbody-womask':
            from .utils_sample import AABBSampler
            obj = AABBSampler(split=self.split, bounds=feat['bounds'])
            for key in ['R', 'Rh', 'Th', 'vertices', 'poses', 'shapes']:
                obj.feature[key] = feat[key]
        elif obj_type == 'trajbbox':
            from .utils_sample import AABBSampler
            annname = join(self.root, obj_args.root, '{:06d}.json'.format(frame))
            annots = read_json_with_cache(annname, file_cache)
            annots = [a for a in annots if a['id'] == pid]
            assert len(annots) == 1, annname
            annots = annots[0]
            center = np.array(annots['keypoints3d'], dtype=np.float32)[:, :3]
            obj = AABBSampler(self.split, center=center, scale=obj_args.scale)
            obj.feature['center'] = center
        elif obj_type == 'cylinder':
            from .utils_sample import CylinderSampler
            obj = CylinderSampler(split=self.split, **obj_args)
        elif obj_type == 'plane':
            from .utils_sample import PlaneSampler
            obj = PlaneSampler(split=self.split, **obj_args)
        else:
            myerror('[Error] Unknown object type: {}'.format(obj_type))
            raise NotImplementedError
        for key, val in params.items():
            obj.feature[key] = val
        return obj

    def get_objects(self, root, info, object_keys, object_args):
        objects = {}
        current_keys = object_keys.copy()
        file_cache = {}
        if len(current_keys) == 0:
            current_keys = list(object_args.keys())
        for oid, wrapkey in enumerate(current_keys):
            key, params = wrapkey, {}
            if key in self.ignore_keys: continue
            if '@' in wrapkey:
                key_ = wrapkey.split('_@')[0]
                params = json.loads(wrapkey.split('_@')[1].replace("'", '"'))
                val = object_args[key_]
            else:
                val = object_args[key]
            with Timer(wrapkey, not self.timer):
                model = self.object_factory(root, val['model'], val['args'], info, params, file_cache)
            objects[key] = model
        return objects

    def sample_ray(self, img, back_mask, info, objects, debug=False):
        H, W = img.shape[:2]
        K, R, T = info['camera']['K'].copy(), info['camera']['R'], info['camera']['T']
        sub = info['sub']
        if sub not in self.cache_ray_o_d:
            # cache this ray as it takes 0.2s
            ray_o, ray_d = get_rays(H, W, K, R, T)
            self.cache_ray_o_d[sub] = (ray_o, ray_d)
        ray_o, ray_d = self.cache_ray_o_d[sub]
        # sample rays according to the rate
        bounds, rates = {}, {}
        for key, obj in objects.items():
            with Timer('mask '+key, not self.timer):
                ret = obj.mask(K, R, T, H, W, ray_o=ray_o, ray_d=ray_d)
            if isinstance(ret, np.ndarray):
                bounds[key] = ret
                if self.split == 'train':
                    rates[key] = self.object_args[key]['rate']
            elif isinstance(ret, dict):
                for key_, val_ in ret.items():
                    bounds[key+key_] = val_['mask']
                    if self.split == 'train':
                        rates[key+key_] = self.object_args[key]['rate'] * val_['rate']
            else:
                raise NotImplementedError
        if self.split == 'train':
        # if self.sample_args['method'] == 'rate' and self.split == 'train':
            with Timer('sample ray', not self.timer):
                coord = sample_rays_rate(bounds, rates, back_mask, **self.sample_args)
        else:
            bounds = np.dstack(list(bounds.values()))
            bound_sum = np.sum(bounds, axis=-1)
            coord = sample_rays(bound_sum, back_mask, self.split, **self.sample_args)
        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.scatter(coord[:, 1], coord[:, 0])
            plt.figure()
            bounds = np.dstack(bounds.values()).copy()
            for i in range(bounds.shape[-1]):
                bounds[..., i] *= 2**i
            bound_sum = np.sum(bounds, axis=-1)
            plt.imshow(bound_sum)
            plt.show()
            import ipdb;ipdb.set_trace()
        ray_o = ray_o[coord[:, 0], coord[:, 1]]
        ray_d = ray_d[coord[:, 0], coord[:, 1]]
        rgb = img[coord[:, 0], coord[:, 1]]
        return ray_o, ray_d, rgb, coord
    
    def sample_near_far(self, rgb, ray_o, ray_d, coord, objects):
        ret = {
            'rgb': rgb, 'coord': coord,
            'ray_o': ray_o, 'ray_d': ray_d, 'viewdirs': ray_d/np.linalg.norm(ray_d, axis=-1, keepdims=True),
        }
        # TODO:这里是每个物体分别进行采样，没有统一调度
        # sample the ray in the fore
        for key, model in objects.items():
            with Timer('sample '+key, not self.timer):
                near, far, mask = model(ray_o, ray_d, coord)
            # 这里把单个物体的相关信息直接返回了，用于兼容单人的情况，单人就不用考虑如何取mask了
            for k in ['rgb', 'coord', 'ray_o', 'ray_d', 'viewdirs']:
                ret[key+'_'+k] = ret[k][mask]
            ret.update({
                key+'_near': near, 
                key+'_far': far, 
                key+'_mask': mask,
                key+'_bounds': model.bounds})
            # update other features
            for k, v in model.feature.items():
                if 'coord' in k:
                    ret[key+'_'+k] = v[coord[:, 0], coord[:, 1]]
                else:
                    ret[key+'_'+k] = v
        return ret
    
    def create_cameras(self, camera_args):
        from .utils_sample import create_center_radius
        RT = create_center_radius(
            center=camera_args.center, 
            radius=camera_args.radius, up=camera_args.up, 
            angle_x=camera_args.angle_x,
            ranges=camera_args.ranges)
        focal = camera_args.focal
        K = np.array([
            focal, 0, camera_args.H/2, 0, focal, camera_args.W/2, 0, 0, 1
        ]).reshape(3, 3)
        cameras = [{'K': K, 'R': RT[i, :3, :3], 'T': RT[i, :3, 3:]} for i in range(RT.shape[0])]
        return cameras

class BaseDataset(BaseBase):
    # This class is mainly for multiview or single view training dataset
    # contains the operation for image
    def __init__(self, root, subs, ranges, split, 
        image_args, 
        object_keys, object_args, ignore_keys,
        sample_args) -> None:
        super().__init__(split=split)
        self.root = root
        self.cameras = read_cameras(root)
        self.subs = self.check_subs(root, subs, image_args.root)
        self.ranges = ranges
        self.infos = self.get_allnames(root, self.subs, ranges, image_args)
        self.split = split
        self.image_args = image_args
        self.sample_args = sample_args
        self.parse_object_args(object_keys, object_args, ignore_keys)
        self.debug = False
        self.check = False
        self.check_data()

    def check_data(self):
        from tqdm import tqdm
        visited = set()
        for i in tqdm(range(len(self)), desc='check all the data'):
            info = self.infos[i]
            sub = info['sub']
            if sub in visited:continue
            visited.add(sub)
            data = self[i]

    def get_allnames(self, root, subs, ranges, image_args):
        infos = []
        index = 0
        unsync = {}
        if image_args.get('unsync', 'none') != 'none':
            unsync = read_json(join(root, image_args['unsync']))
        for nv, sub in enumerate(subs):
            camera = self.cameras[sub].copy()
            K = camera['K'].copy()
            K[:2] *= image_args.scale
            camera['K'] = K
            for nnf, nf in enumerate(range(*ranges)):
                imgname = join(root, image_args.root, sub, '{:06d}{}'.format(nf, image_args.ext))
                info = {
                    'imgname': imgname,
                    'sub': sub,
                    'frame': nf,
                    'nf': nnf,
                    'nv': nv,
                    'index': index,
                    'camera': camera
                }
                if sub in unsync.keys():
                    info['time'] = nf + unsync[sub]
                infos.append(info)
        return infos

    @staticmethod
    def check_subs(root, subs, image_root):
        if len(subs) == 0:
            subs = sorted(os.listdir(join(root, image_root)))
            if subs[0].isdigit():
                subs.sort(key=lambda x:int(x))
        return subs

    def read_image(self, imgname, image_args, info, isgray=False, skip_mask=False, mask_global='_0.png'):
        if isgray:
            img = cv2.imread(imgname, 0)
        else:
            img = cv2.imread(imgname)
        img = img_to_numpy(img)
        if image_args.mask_bkgd and not skip_mask:
            # TODO: polish mask name
            mskname = imgname.replace(image_args.root, image_args.mask).replace(image_args.ext, mask_global)
            if not os.path.exists(mskname):
                mskname = mskname.replace(mask_global, '.png')
            assert os.path.exists(mskname), mskname
            msk = cv2.imread(mskname, 0)
            msk_rand = np.random.rand(msk.shape[0], msk.shape[1], 3)
            msk_rand = (msk_rand * 255).astype(np.uint8)
            img[msk==0] = -1.
            # img[msk==0] = msk_rand[msk==0]
        # 这里选择了先进行畸变矫正，再进行scale，这样相机参数就不需要修改
        # 由于预先进行了map，所以畸变矫正不会很慢
        if self.image_args.scale != 1:
            H, W = int(img.shape[0] * self.image_args['scale']), int(img.shape[1] * self.image_args['scale'])
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        K, D = info['camera']['K'], info['camera']['dist']
        sub = info['sub']
        if self.image_args.undis and np.linalg.norm(D) > 0.:
            img = Undistort.image(img, K, D, sub=sub)
        if info.get('flip_lr', False):
            img = cv2.flip(img, 1)
        return img

    def read_bkgd(self, imgname, image_args, info):
        backname = join(self.root, 'background', info['sub'], '000000.jpg')
        if not os.path.exists(backname) or not image_args.get('read_background'):
            return None
        back = self.read_image(backname, image_args, info, skip_mask=True)
        return back

    def read_backmask(self, imgname, image_args, info, blankshape):
        sub = info['sub']
        if self.split != 'demo' and sub in self.back_mask_cache.keys():
            back_mask = self.back_mask_cache[sub].copy()
        elif self.split != 'demo':
            mskname = join(self.root, 'mask-background', sub, '000000.png')
            # if not exists, should read and undistort
            back_mask = np.ones(blankshape, dtype=np.float32)
            K, D = info['camera']['K'], info['camera']['dist']
            if self.image_args.undis and np.linalg.norm(D) > 0.:
                back_mask = Undistort.image(back_mask, K, D, sub=sub)
            if os.path.exists(mskname) and self.split == 'train' and self.image_args.ignore_back:
                oncewarn('using mask of background')
                _back_mask = self.read_image(mskname, image_args, info, isgray=True)
                # 畸变矫正后边缘上填充的值是0，这里取反之后边缘上的填充值又变成了1
                _back_mask[_back_mask>0] = 1.
                back_mask = (1. - _back_mask) * back_mask
            self.back_mask_cache[sub] = back_mask
            back_mask = self.back_mask_cache[sub].copy()
        else:
            # in demo mode sample all pixels
            back_mask = np.ones(blankshape, dtype=np.float32)
        # get the framewise mask
        mskname = imgname.replace('images', 'mask-ignore').replace('.jpg', '.png')
        if os.path.exists(mskname):
            # oncewarn('using mask of frame {}'.format(mskname))
            mask_frame = self.read_image(mskname, image_args, info, isgray=True)
            mask_frame[mask_frame>0] = 1.
            mask_frame = 1. - mask_frame
            back_mask = back_mask * mask_frame
        return back_mask

    def __getitem__(self, index):
        info = self.infos[index]
        imgname = info['imgname']
        with Timer('read image', not self.timer):
            img = self.read_image(imgname, self.image_args, info)
        with Timer('read background', not self.timer):
            back = self.read_bkgd(imgname, self.image_args, info)        
        with Timer('read back', not self.timer):
            back_mask = self.read_backmask(imgname, self.image_args, info, blankshape=img.shape[:2])
        object_keys = info.get('object_keys', self.object_keys)
        objects = self.get_objects(self.root, info, object_keys, self.object_args)
        # sample the ray from image
        ray_o, ray_d, rgb, coord = self.sample_ray(img, back_mask, info, objects, debug=self.debug)
        ret = self.sample_near_far(rgb, ray_o, ray_d, coord, objects)
        # append the background
        if back is not None:            
            ret['background'] = back[coord[:, 0], coord[:, 1]]
        meta = {
            'split': self.split,
            'H': img.shape[0], 'W': img.shape[1],
            'index': index,
            'nframe': info['nf'], 'nview': info['nv'],
            'time': info.get('time', info['nf']),
            'sub': info['sub'],
            'keys': list(objects.keys()),
        }
        if self.check:
            meta['img'] = img
        meta['object_keys'] = meta['keys']
        ret['meta'] = meta
        return ret

class BaseDatasetDemo(BaseDataset):
    def __init__(self, camera_args, demo_args=None, **kwargs):
        self.camera_args = camera_args
        super().__init__(**kwargs)
        scale = self.image_args.scale
        self.blank = np.zeros((int(camera_args.H*scale), int(camera_args.W*scale), 3), dtype=np.uint8)
        self.infos = self.create_demo_cameras(self.image_args.scale, camera_args, demo_args)
        # self.camera_args = camera_args
        # K, RTs, D = self.create_demo_camera(camera_method, camera_args, self.cameras)

    @staticmethod
    def _demo_play_frames(ranges, K, R, T):
        infos = []
        index = 0
        frames = [i for i in range(*ranges)]
        for nv in range(K.shape[0]):
            nnf = nv % (len(frames))
            nf = frames[nnf]
            info = {
                'imgname': 'none',
                'sub': 'novel_'+str(nv),
                'frame': nf,
                'nf': nnf,
                'nv': nv,
                'index': index,
                'camera': {
                    'K': K[nv],
                    'dist': np.zeros((1, 5)),
                    'R': R[nv],
                    'T': T[nv]
                }
            }

            infos.append(info)
            index += 1
        return infos
    
    @staticmethod
    def _demo_keyframe(ranges, K, R, T, frame, nFrames):
        infos = []
        index = 0
        frames = [i for i in range(*ranges)]
        for nnf, nf in enumerate(frames):
            info = {
                'imgname': 'none',
                'sub': 'novel_'+str(0),
                'frame': nf,
                'nf': nnf,
                'nv': 0,
                'index': index,
                'camera': {
                    'K': K[0],
                    'dist': np.zeros((1, 5)),
                    'R': R[0],
                    'T': T[0]
                }
            }
            if nf == frame:
                for i in range(nFrames):
                    angle = i/nFrames * 2 * np.pi
                    object_keys = ["human_0_@{{'rotate_angle': {}, 'rotate_axis': 'y'}}".format(angle)]
                    _info = info.copy()
                    _info['object_keys'] = object_keys
                    infos.append(_info)
                    info['index'] += 1
                index = info['index'] + 1
                continue
            infos.append(info)
            index += 1
        return infos

    def _demo_script(self, ranges, K, R, T, stages):
        infos = []
        index = 0
        frames = [i for i in range(*ranges)]
        for name, stage in stages.items():
            _infos = []
            _frames = list(range(*stage.frame))
            _views = list(range(*stage.view))
            if len(_frames) == 1 and len(_views) != 1:
                _frames = _frames * len(_views)
            elif len(_views) == 1 and len(_frames) != 1:
                _views = _views * len(_frames)
            elif len(_views) == 1 and len(_frames) == 1 and 'steps' in stage.keys():
                _views = _views * stage.steps
                _frames = _frames * stage.steps
            elif len(_views) != 1 and len(_frames) != 1 and len(_views) != len(_frames):
                raise NotImplementedError
            _index = [i for i in range(len(_frames))]
            for _i in _index:
                nv, nf = _views[_i], _frames[_i]
                nv = nv % (K.shape[0])
                info = {
                    'imgname': 'none',
                    'sub': 'novel_'+str(nv),
                    'frame': nf,
                    'nf': frames.index(nf),
                    'nv': nv,
                    'index': _i + index,
                    'camera': {
                        'K': K[nv],
                        'dist': np.zeros((1, 5)),
                        'R': R[nv],
                        'T': T[nv]
                    }
                }
                # create object
                float_i = _i*1./(len(_index) - 1)
                object_keys = stage.object_keys.copy()
                if len(object_keys) == 0:
                    object_keys = list(self.object_args.keys()).copy()
                if 'effect' in stage.keys():
                    if stage.effect in ['disappear', 'appear']:
                        for _obj in stage.effect_args.key:
                            object_keys.remove(_obj)
                            if stage.effect == 'disappear':
                                occ = (1 - float_i)**3
                            elif stage.effect == 'appear':
                                occ = float_i**3
                            object_keys.append(_obj+"_@{{'scale_occ': {}, 'min_acc': 0.5}}".format(occ))
                    if stage.effect in ['zoom']:
                        scale = float_i * stage.effect_args.scale[1] + (1-float_i) * stage.effect_args.scale[0]
                        cx = float_i * stage.effect_args.cx[1] + (1-float_i) * stage.effect_args.cx[0]
                        cy = float_i * stage.effect_args.cy[1] + (1-float_i) * stage.effect_args.cy[0]
                        _K = info['camera']['K'].copy()
                        _K[:2, :2] *= scale
                        _K[0, 2] *= cx
                        _K[1, 2] *= cy
                        info['camera']['K'] = _K
                        info['camera']['K'] = _K
                        info['sub'] = info['sub'] + '_scale_{}'.format(scale)
                    if stage.effect_args.get('use_previous_K', False):
                        info['camera']['K'] = infos[-1]['camera']['K']
                info['object_keys'] = object_keys
                _infos.append(info)
            index += len(_index)
            infos.extend(_infos)
        return infos

    def create_demo_cameras(self, scale, camera_args, demo_args=None):
        if camera_args.method == 'none':
            from .utils_sample import create_center_radius
            RTs = create_center_radius(**camera_args)
            K = np.array([
                camera_args.focal, 0, camera_args.W/2,
                0, camera_args.focal, camera_args.H/2,
                0, 0, 1], dtype=np.float32).reshape(3, 3)[None].repeat(RTs.shape[0], 0)
            R = RTs[:, :3, :3]
            T= RTs[:, :3, 3:]
        elif camera_args.method == 'mean':
            from .utils_sample import create_cameras_mean
            K, R, T = create_cameras_mean(list(self.cameras.values()), camera_args)
            K[:, 0, 2] = camera_args.W / 2
            K[:, 1, 2] = camera_args.H / 2
        elif camera_args.method == 'static':
            assert len(self.subs) == 1, "Only support monocular videos"
            camera = self.cameras[self.subs[0]]
            K = camera['K'][None]
            R = camera['R'][None]
            T = camera['T'][None]
        elif camera_args.method == 'line':
            for key, camera in self.cameras.items():
                R = camera['R']
                T = camera['T']
                center_old = - R.T @ T
                print(key, center_old.T[0])
            camera = self.cameras[str(camera_args.ref_sub)]
            K = camera['K'][None]
            R = camera['R'][None]
            T = camera['T'][None]
            t = np.linspace(0., 1., camera_args.allstep).reshape(-1, 1)
            t = t - 0.33
            t[t<0.] = 0.
            t = t/t.max()
            start = np.array(camera_args.center_start).reshape(1, 3)
            end = np.array(camera_args.center_end).reshape(1, 3)
            center = end * t + start * (1-t)
            K = K.repeat(camera_args.allstep, 0)
            R = R.repeat(camera_args.allstep, 0)
            T = - np.einsum('fab,fb->fa', R, center)
            T = T.reshape(-1, 3, 1)

        K[:, :2] *= scale

        if demo_args is None:
            infos = self._demo_play_frames(self.ranges, K, R, T)
            return infos
        # create scripts
        if demo_args.mode == 'scripts':
            infos = self._demo_script(self.ranges, K, R, T, demo_args.stages)
        elif demo_args.mode == 'keyframe+rotate':
            infos = self._demo_keyframe(self.ranges, K, R, T, demo_args.frame, demo_args.nFrames)
        else:
            raise NotImplementedError
        return infos

    def get_allnames(self, root, subs, ranges, image_args):
        return []

    def read_image(self, imgname, image_args, info):
        return self.blank

class BaseNovelPose(BaseBase):
    def __init__(self, root, object_keys, ignore_keys, object_args, sample_args, camera_args):
        super().__init__(split='demo')
        self.root = root
        self.parse_object_args(object_keys, object_args, ignore_keys)
        self.params = self.load_all_smpl(root)
        self.cameras = self.create_cameras(camera_args)
        self.infos = self.combine_frame_and_camera(self.params, self.cameras)
        self.H = camera_args.H
        self.W = camera_args.W
        self.sample_args = sample_args
        self.check = False

    def load_all_smpl(self, root):
        from glob import glob
        from tqdm import tqdm
        smplnames = sorted(glob(join(root, 'smpl', '*.json')))
        params = []
        for nf, smplname in enumerate(tqdm(smplnames)):
            frame = int(os.path.basename(smplname).split('.')[0])
            nv, sub = 0, 'novelpose0'
            # nf, frame = 0, 0
            imgname = 'smplname'
            params.append({
                'imgname': imgname,
                'sub': sub,
                'frame': frame,
                'nf': nf,
                'nv': nv,
                'index': nf,
            })
        return params
    
    def combine_frame_and_camera(self, params, cameras):
        infos = []
        for i in range(len(params)):
            info = params[i].copy()
            info['camera'] = cameras[0]
            infos.append(info)
        return infos

    def read_blank_image(self, *args):
        img = np.zeros([self.H, self.W, 3], dtype=np.float32)
        return img

    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        info = self.infos[index]
        objects = self.get_objects(self.root, info, self.object_keys, self.object_args)

        info = self.infos[index]
        imgname = info['imgname']
        img = self.read_blank_image(imgname, info)
        back_mask = np.ones_like(img[:, :, 0])
        # sample the ray from image
        ray_o, ray_d, rgb, coord = self.sample_ray(img, back_mask, info, objects)
        ret = self.sample_near_far(rgb, ray_o, ray_d, coord, objects)
        info['nf'] = 0
        meta = {
            'H': img.shape[0], 'W': img.shape[1],
            'index': index,
            'nframe': info['nf'], 'nview': info['nv'],
            'sub': info['sub'],
            'keys': list(objects.keys()),
        }
        if self.check:
            meta['img'] = img
        meta['object_keys'] = meta['keys']
        ret['meta'] = meta
        return ret

class BaseCanonical(BaseNovelPose):
    def __init__(self, nFrames, **kwargs):
        super().__init__(**kwargs)
        self.nFrames = nFrames

    def load_all_smpl(self, root):
        smpl = read_smpl(root)[0]
        for key in ['poses', 'Rh', 'Th']:
            smpl[key] = np.zeros_like(smpl[key])
        return smpl
    
    def combine_frame_and_camera(self, params, cameras):
        infos = []
        for i in range(len(cameras)):
            info = params.copy()
            info['camera'] = cameras[i]
            info.update({
                'imgname': f'view_{i}_frame_0',
                'sub': 'novel_' + str(i),
                'frame': 0,
                'nf': 0,
                'nv': i,
                'index': i,
            })
            infos.append(info)
        return infos
    
    def get_objects(self, root, info, object_keys, object_args):
        objects = {'human_0': AABBSampler(split='test', bounds=[[-1, -1.3, -0.3], [1, 0.7, 0.3]])}
        objects['human_0'].feature['shapes'] = self.params['shapes']
        return objects

if __name__ == '__main__':
    from ...config import Config, load_object
    from copy import deepcopy
    # config = Config.load('config/neuralbody/dataset/multiview_custom.yml')
    # config = Config.load('config/neuralbody/dataset/neuralbody_lightstage.yml')
    config = Config.load('config/neuralbody/dataset/neuralbody_soccer.yml')
    data_share = config.pop('data_share_args')
    data_share.root = os.environ['data']
    for split in ['train', 'val', 'demo']:
        data = deepcopy(data_share)
        data.merge_from_other_cfg(config['data_{}_args'.format(split)])
        config['data_{}_args'.format(split)] = data
    print(config)
    dataset = load_object(config.data_train_module, config.data_train_args)
    dataset.debug = True
    for data in dataset:
        print(data.keys())