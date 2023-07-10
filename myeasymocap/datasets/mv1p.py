from easymocap.mytools.camera_utils import read_cameras
from easymocap.mytools.debug_utils import log, myerror, mywarn
from easymocap.mytools.file_utils import read_json
from .basedata import ImageDataBase, read_mv_images, find_best_people, find_all_people
import os
from os.path import join
import numpy as np
import cv2
from collections import defaultdict

panoptic15_in_body15 = [1,0,8,5,6,7,12,13,14,2,3,4,9,10,11]

def convert_body15_panoptic15(keypoints):
    k3d_panoptic15 = keypoints[..., panoptic15_in_body15,: ]
    return k3d_panoptic15

def convert_panoptic15_body15(keypoints):
    keypoints_b15 = np.zeros_like(keypoints)
    keypoints_b15[..., panoptic15_in_body15, :] = keypoints
    return keypoints_b15

def padding_and_stack(datas):
    shapes = {}
    for data in datas:
        if len(data) == 0:
            continue
        for key, value in data.items():
            if key not in shapes.keys():
                shapes[key] = value.shape
    collect = {key: np.zeros((len(datas), *shapes[key])) for key in shapes.keys()}
    for i, data in enumerate(datas):
        for key, value in data.items():
            collect[key][i] = value
    return collect

def padding_empty(datas):
    shapes = {}
    for data in datas:
        if len(data) == 0:
            continue
        for key, value in data.items():
            if key not in shapes.keys():
                shapes[key] = value.shape[1:]
    collect = {key: [None for data in datas] for key in shapes.keys()}
    for i, data in enumerate(datas):
        for key, shape in shapes.items():
            if key not in data.keys():
                print('[Dataset] padding empty view {} of {}'.format(i, key))
                collect[key][i] = np.zeros((0, *shape), dtype=np.float32)
            else:
                collect[key][i] = data[key]
    return collect

def parse_frames(pafs_frame, H, W):
    # 解析单帧的
    res = {
        'joints': [],
        'pafs': {}
    }
    joints = pafs_frame[1:1+3*25]
    for i in range(25):
        value = np.fromstring(joints[3*i+2], sep=' ').reshape(3, -1).T
        value[:, 0] = value[:, 0] * W
        value[:, 1] = value[:, 1] * H
        res['joints'].append(value.astype(np.float32))
    # parse pafs
    pafs = pafs_frame[1+3*25+1:]
    for npart in range(26):
        label = pafs[3*npart+0].split(' ')[2:]
        label = (int(label[0]), int(label[1]))
        shape = pafs[3*npart+1].split(' ')[2:]
        w, h = int(shape[0]), int(shape[1])
        value = np.fromstring(pafs[3*npart+2], sep=' ').reshape(w, h).astype(np.float32)
        res['pafs'][label] = value
    return res

def read_4dassociation(pafs, H, W):
    outputs = []
    # 解析paf文件
    with open(pafs, 'r') as f:
        pafs = f.readlines()
    indices = []
    for i, line in enumerate(pafs):
        if line.startswith('# newframes:'):
            indices.append([i])
        elif line.startswith('# end frames:'):
            indices[-1].append(i)
    print('[Read OpenPose] Totally {} frames'.format(len(indices)))
    for (start, end) in indices:
        pafs_frame = pafs[start+1:end]
        pafs_frame = list(map(lambda x:x.strip(), pafs_frame))
        frames = parse_frames(pafs_frame, H, W)
        outputs.append(frames)
    return outputs

class MVDataset(ImageDataBase):
    def __init__(self, root, subs, subs_vis, ranges, read_image=False, reader={}, filter={}) -> None:
        super().__init__(root, subs, ranges, read_image)
        self.subs_vis = subs_vis
        self.length = 0
        for key, value in reader.items():
            if key == 'images':
                self.try_to_extract_images(root, value)
                data, meta = read_mv_images(root, value['root'], value['ext'], subs)
                self.length = len(data)
            elif key == 'image_shape':
                imgnames = self.infos['images'][0]
                shapes = []
                for imgname in imgnames:
                    img = cv2.imread(imgname)
                    height, width, _ = img.shape
                    log('[{}] sub {} shape {}'.format(self.__class__.__name__, imgname, img.shape))
                    shapes.append([height, width])
                data = [shapes]
                meta = {}
            elif key == 'annots':
                data, meta = read_mv_images(root, value['root'], value['ext'], subs)
                if self.length > 0:
                    if self.length != len(data):
                        myerror('annots length {} not equal to images length {}.'.format(len(data), self.length))
                        data = data[:self.length]
                else:
                    self.length = len(data)
            elif key == 'openpose':
                # 读取open pose
                if len(subs) == 0:
                    pafs = sorted(os.listdir(join(root, value['root'])))
                else:
                    pafs = [f'{sub}.txt' for sub in subs]
                results = []
                for nv, paf in enumerate(pafs):
                    pafname = join(root, value['root'], paf)
                    infos = read_4dassociation(pafname, H=self.infos['image_shape'][0][nv][0], W=self.infos['image_shape'][0][nv][1])
                    results.append(infos)
                data = [[d[i] for d in results] for i in range(self.length)]
                meta = {}
            elif key == 'cameras':
                if 'with_sub' in value.keys():
                    raise NotImplementedError
                else:
                    cameras = read_cameras(os.path.join(root, value['root']))
                    if 'remove_k3' in value.keys():
                        for cam, camera in cameras.items():
                            camera['dist'][:, 4] = 0.
                    data = [cameras]
                    meta = {}
            elif key in ['pelvis']:
                continue
            elif key == 'keypoints3d':
                k3droot = value['root']
                filenames = sorted(os.listdir(k3droot))[:self.length]
                res_key = value.get('key', 'pred')
                data = []
                for filename in filenames:
                    results = read_json(join(k3droot, filename))
                    if 'pids' not in results.keys():
                        # 擅自补全
                        results['pids'] = list(range(len(results[res_key])))
                    data.append({
                        'pids': results['pids'],
                        'keypoints3d': np.array(results[res_key], dtype=np.float32)
                    })
                    if data[-1]['keypoints3d'].shape[-1] == 3:
                        mywarn('The input keypoints dont have confidence')
                        data[-1]['keypoints3d'] = np.concatenate([data[-1]['keypoints3d'], np.ones_like(data[-1]['keypoints3d'][..., :1])], axis=-1)
                    if 'conversion' in value.keys():
                        if value['conversion'] == 'panoptic15_to_body15':
                            data[-1]['keypoints3d'] = convert_panoptic15_body15(data[-1]['keypoints3d'])
            else:
                raise ValueError(f'Unknown reader: {key}')
            self.infos[key] = data
            self.meta.update(meta)
        self.reader = reader
        self.filter = filter
        if len(self.subs) == 0:
            self.subs = self.meta['subs']
        if len(self.subs_vis) == 1:
            if self.subs_vis[0] == '_all_':
                self.subs_vis = self.subs
            elif self.subs_vis[0] == '_sample_4_':
                self.subs_vis = [self.subs[0], self.subs[len(self.subs)//3], self.subs[(len(self.subs)*2//3)], self.subs[-1]]
        self.check_frames_length()
    
    @staticmethod
    def read_annots(annotnames):
        val = []
        for annname in annotnames:
            annots = read_json(annname)['annots']
            # select the best people
            annots = find_best_people(annots)
            val.append(annots)
        val = padding_and_stack(val)
        return val
    
    def filter_openpose(self, candidates, pafs):
        for nv, candview in enumerate(candidates):
            H=self.infos['image_shape'][0][nv][0]
            W=self.infos['image_shape'][0][nv][1]
            for cand in candview:
                if 'border' in self.filter.keys():
                    border = self.filter['border'] * max(H, W)
                    flag = (cand[:, 0] > border) & (cand[:, 0] < W - border) & (cand[:, 1] > border) & (cand[:, 1] < H - border)
                    cand[~flag] = 0
        return candidates, pafs

    def __getitem__(self, index):
        frame = self.frames[index]
        ret = {}
        for key, value in self.infos.items():
            if len(value) == 1:
                ret[key] = value[0]
            elif frame >= len(value):
                myerror(f'[{self.__class__.__name__}] {key}: index {frame} out of range {len(value)}')
            else:
                ret[key] = value[frame]
        ret_list = defaultdict(list)
        for key, val in ret.items():
            if key == 'annots':
                ret_list[key] = self.read_annots(val)
            elif key == 'cameras':
                for sub in self.subs:
                    select = {k: val[sub][k] for k in ['K', 'R', 'T', 'dist', 'P']}
                    ret_list[key].append(select)
                ret_list[key] = padding_and_stack(ret_list[key])
            elif key == 'images':
                if self.flag_read_image:
                    for i, sub in enumerate(self.subs):
                        imgname = val[i]
                        if sub in self.subs_vis or self.subs_vis == 'all':
                            img = self.read_image(imgname)
                        else:
                            img = imgname
                        ret_list[key].append(img)
                        ret_list['imgnames'].append(imgname)
                else:
                    ret_list[key] = val
                    ret_list['imgnames'] = val
            elif key == 'openpose':
                ret_list[key] = [v['joints'] for v in val]
                # 同时返回PAF
                ret_list[key+'_paf'] = [v['pafs'] for v in val]
                # check一下PAF:
                for nv in range(len(ret_list[key])):
                    ret_list[key+'_paf'][nv][(8, 1)] = ret_list[key+'_paf'][nv].pop((1, 8)).T
                ret_list[key], ret_list[key+'_paf'] = self.filter_openpose(ret_list[key], ret_list[key+'_paf'])
            elif key == 'keypoints3d':
                ret_list['keypoints3d'] = val['keypoints3d']
                if 'pids' in val.keys():
                    ret_list['pids'] = val['pids']
                else:
                    ret_list['pids'] = list(range(len(val['keypoints3d'])))
            elif key in ['image_shape']:
                pass
            else:
                print('[Dataset] Unknown key: {}'.format(key))
        ret_list.update(ret_list.pop('annots', {}))
        for key, val in self.reader.items():
            if key == 'pelvis' and 'annots' in self.reader.keys(): # load pelvis from annots.keypoints
                ret_list[key] = [d[:, val.root_id] for d in ret_list['keypoints']]
            elif key == 'pelvis' and 'openpose' in self.reader.keys():
                ret_list[key] = [d[val.root_id] for d in ret_list['openpose']]
        ret_list['meta'] = {
            'subs': self.subs,
            'index': index,
            'frame': frame,
            'image_shape': ret['image_shape'],
            'imgnames': ret_list['imgnames'],
        }
        return ret_list

    def check(self, index):
        raise NotImplementedError
    
    def __str__(self) -> str:
        pre = super().__str__()
        pre += '''    subs_vis: {}'''.format(self.subs_vis)
        return pre

class MVMP(MVDataset):
    def read_annots(self, annotnames):
        val = []
        for annname in annotnames:
            annots = read_json(annname)['annots']
            # 在这里进行filter，去掉不需要的2D
            annots_valid = []
            for annot in annots:
                flag = True
                if 'bbox_size' in self.filter.keys():
                    bbox_size = self.filter['bbox_size']
                    bbox = annot['bbox']
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area < bbox_size:
                        flag = False
                if flag:
                    annots_valid.append(annot)
            annots = annots_valid
            # select the best people
            annots = find_all_people(annots)
            val.append(annots)
        val = padding_empty(val)
        return val
    
    def check(self, index):
        data = self.__getitem__(index)
        from easymocap.mytools.vis_base import plot_bbox, merge, plot_keypoints_auto
        # check the subs vis
        vis = []
        for nv, sub in enumerate(self.subs):
            if sub not in self.subs_vis:continue
            img = data['images'][nv].copy()
            bbox = data['bbox'][nv]
            kpts = data['keypoints'][nv]
            for i in range(bbox.shape[0]):
                plot_bbox(img, bbox[i], pid=i)
                plot_keypoints_auto(img, kpts[i], pid=i, use_limb_color=False)
            vis.append(img)
        vis = merge(vis)
        cv2.imwrite('debug/{}_{:06d}.jpg'.format(self.__class__.__name__, index), vis)

if __name__ == '__main__':
    config = '''
args:
    root: /nas/ZJUMoCap/Part0/313
    subs: []
    subs_vis: ['01', '07', '13', '19']
    ranges: [0, 100, 1]
    read_image: False
    reader:
        images:
            root: images
            ext: .jpg
        annots:
            root: annots
            ext: .json
        cameras: # 兼容所有帧的相机参数不同的情况
            root: ''
'''
    import yaml
    config = yaml.load(config, Loader=yaml.FullLoader)
    dataset = MVDataset(**config['args'])
    for i in range(len(dataset)):
        data = dataset[i]