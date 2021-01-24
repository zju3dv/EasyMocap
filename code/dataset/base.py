'''
  @ Date: 2021-01-13 16:53:55
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-24 22:27:01
  @ FilePath: /EasyMocapRelease/code/dataset/base.py
'''
import os
import json
from os.path import join
from torch.utils.data.dataset import Dataset
import cv2
import os, sys
import numpy as np
code_path = join(os.path.dirname(__file__), '..')
sys.path.append(code_path)

from mytools.camera_utils import read_camera, undistort, write_camera, get_fundamental_matrix
from mytools.vis_base import merge, plot_bbox, plot_keypoints

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)


def read_annot(annotname, mode='body25'):
    data = read_json(annotname)
    if not isinstance(data, list):
        data = data['annots']
    for i in range(len(data)):
        if 'id' not in data[i].keys():
            data[i]['id'] = data[i].pop('personID')
        if 'keypoints2d' in data[i].keys() and 'keypoints' not in data[i].keys():
            data[i]['keypoints'] = data[i].pop('keypoints2d')
        for key in ['bbox', 'keypoints', 'handl2d', 'handr2d', 'face2d']:
            if key not in data[i].keys():continue
            data[i][key] = np.array(data[i][key])
            if key == 'face2d':
                # TODO: Make parameters, 17 is the offset for the eye brows,
                # etc. 51 is the total number of FLAME compatible landmarks
                data[i][key] = data[i][key][17:17+51, :]
        if mode == 'body25':
            data[i]['keypoints'] = data[i]['keypoints']
        elif mode == 'body15':
            data[i]['keypoints'] = data[i]['keypoints'][:15, :]
        elif mode == 'total':
            data[i]['keypoints'] = np.vstack([data[i][key] for key in ['keypoints', 'handl2d', 'handr2d', 'face2d']])
        elif mode == 'bodyhand':
            data[i]['keypoints'] = np.vstack([data[i][key] for key in ['keypoints', 'handl2d', 'handr2d']])
        elif mode == 'bodyhandface':
            data[i]['keypoints'] = np.vstack([data[i][key] for key in ['keypoints', 'handl2d', 'handr2d', 'face2d']])
    data.sort(key=lambda x:x['id'])
    return data

def get_bbox_from_pose(pose_2d, img, rate = 0.1):
    # this function returns bounding box from the 2D pose
    # here use pose_2d[:, -1] instead of pose_2d[:, 2]
    # because when vis reprojection, the result will be (x, y, depth, conf)
    validIdx = pose_2d[:, -1] > 0
    if validIdx.sum() == 0:
        return [0, 0, 100, 100, 0]
    y_min = int(min(pose_2d[validIdx, 1]))
    y_max = int(max(pose_2d[validIdx, 1]))
    x_min = int(min(pose_2d[validIdx, 0]))
    x_max = int(max(pose_2d[validIdx, 0]))
    dx = (x_max - x_min)*rate
    dy = (y_max - y_min)*rate
    # 后面加上类别这些
    bbox = [x_min-dx, y_min-dy, x_max+dx, y_max+dy, 1]
    correct_bbox(img, bbox)
    return bbox

def correct_bbox(img, bbox):
    # this function corrects the bbox, which is out of image
    w = img.shape[0]
    h = img.shape[1]
    if bbox[2] <= 0 or bbox[0] >= h or bbox[1] >= w or bbox[3] <= 0:
        bbox[4] = 0
    return bbox
class FileWriter:
    def __init__(self, output_path, config=None, basenames=[], cfg=None) -> None:
        self.out = output_path
        keys = ['keypoints3d', 'match', 'smpl', 'skel', 'repro', 'keypoints']
        output_dict = {key:join(self.out, key) for key in keys}
        # for key, p in output_dict.items():
            # os.makedirs(p, exist_ok=True)
        self.output_dict = output_dict
        
        self.basenames = basenames
        if cfg is not None:
            print(cfg, file=open(join(output_path, 'exp.yml'), 'w'))
        self.save_origin = False
        self.config = config
    
    def write_keypoints3d(self, results, nf):
        os.makedirs(self.output_dict['keypoints3d'], exist_ok=True)
        savename = join(self.output_dict['keypoints3d'], '{:06d}.json'.format(nf))
        save_json(savename, results)
    
    def vis_detections(self, images, lDetections, nf, key='keypoints', to_img=True, vis_id=True):
        os.makedirs(self.output_dict[key], exist_ok=True)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            for det in lDetections[nv]:
                if key == 'match':
                    pid = det['id_match']
                else:
                    pid = det['id']
                if key not in det.keys():
                    keypoints = det['keypoints']
                else:
                    keypoints = det[key]
                if 'bbox' not in det.keys():
                    bbox = get_bbox_from_pose(keypoints, img)
                else:
                    bbox = det['bbox']
                plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
                plot_keypoints(img, keypoints, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        image_vis = merge(images_vis, resize=not self.save_origin)
        if to_img:
            savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            cv2.imwrite(savename, image_vis)
        return image_vis
    
    def write_smpl(self, results, nf):
        os.makedirs(self.output_dict['smpl'], exist_ok=True)
        format_out = {'float_kind':lambda x: "%.3f" % x}
        filename = join(self.output_dict['smpl'], '{:06d}.json'.format(nf))
        with open(filename, 'w') as f:
            f.write('[\n')
            for idata, data in enumerate(results):
                f.write('    {\n')
                output = {}
                output['id'] = data['id']
                for key in ['Rh', 'Th', 'poses', 'expression', 'shapes']:
                    if key not in data.keys():continue
                    output[key] = np.array2string(data[key], max_line_width=1000, separator=', ', formatter=format_out)
                for key in output.keys():
                    f.write('        \"{}\": {}'.format(key, output[key]))
                    if key != 'shapes':
                        f.write(',\n')
                    else:
                        f.write('\n')
                        
                f.write('    }')
                if idata != len(results) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write(']\n')

    def vis_smpl(self, render_data_, nf, images, cameras, mode='smpl', add_back=False):
        out = join(self.out, mode)
        os.makedirs(out, exist_ok=True)
        from visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None)
        if isinstance(render_data_, list): # different view have different data
            for nv, render_data in enumerate(render_data_):
                render_results = render.render(render_data, cameras, images)
                image_vis = merge(render_results, resize=not self.save_origin)
                savename = join(out, '{:06d}_{:02d}.jpg'.format(nf, nv))
                cv2.imwrite(savename, image_vis)
        else:
            render_results = render.render(render_data_, cameras, images, add_back=add_back)
            image_vis = merge(render_results, resize=not self.save_origin)
            savename = join(out, '{:06d}.jpg'.format(nf))
            cv2.imwrite(savename, image_vis)

def readReasultsTxt(outname, isA4d=True):
    res_ = []
    with open(outname, "r") as file:
        lines = file.readlines()
        if len(lines) < 2:
            return res_
        nPerson, nJoints = int(lines[0]), int(lines[1])
        # 只包含每个人的结果
        lines = lines[1:]
        # 每个人的都写了关键点数量
        line_per_person = 1 + 1 + nJoints
        for i in range(nPerson):
            trackId = int(lines[i*line_per_person+1])
            content = ''.join(lines[i*line_per_person+2:i*line_per_person+2+nJoints])
            pose3d = np.fromstring(content, dtype=float, sep=' ').reshape((nJoints, 4))
            if isA4d:
                # association4d 的关节顺序和正常的定义不一样
                pose3d = pose3d[[4, 1, 5, 9, 13, 6, 10, 14, 0, 2, 7, 11, 3, 8, 12], :]
            res_.append({'id':trackId, 'keypoints3d':np.array(pose3d)})
    return res_

def readResultsJson(outname):
    with open(outname) as f:
        data = json.load(f)
    res_ = []
    for d in data:
        pose3d = np.array(d['keypoints3d'])
        if pose3d.shape[0] > 25:
            # 对于有手的情况，把手的根节点赋值成body25上的点
            pose3d[25, :] = pose3d[7, :]
            pose3d[46, :] = pose3d[4, :]
        res_.append({
            'id': d['id'] if 'id' in d.keys() else d['personID'],
            'keypoints3d': pose3d
        })
    return res_

class VideoBase(Dataset):
    """Dataset for single sequence data
    """
    def __init__(self, image_root, annot_root, out=None, config={}, mode='body15', no_img=False) -> None:
        self.image_root = image_root
        self.annot_root = annot_root
        self.mode = mode
        self.no_img = no_img
        self.config = config
        assert out is not None
        self.out = out
        self.writer = FileWriter(self.out, config=config)
        imgnames = sorted(os.listdir(self.image_root))
        self.imagelist = imgnames
        self.annotlist = sorted(os.listdir(self.annot_root))
        self.nFrames = len(self.imagelist)
        self.undis = False
        self.read_camera()
    
    def read_camera(self):
        # 读入相机参数
        annname = join(self.annot_root, self.annotlist[0])
        data = read_json(annname)
        if 'K' not in data.keys():
            height, width = data['height'], data['width']
            focal = 1.2*max(height, width)
            K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
        else:
            K = np.array(data['K']).reshape(3, 3)
        self.camera = {'K':K ,'R': np.eye(3), 'T': np.zeros((3, 1))}

    def __getitem__(self, index: int):
        imgname = join(self.image_root, self.imagelist[index])
        annname = join(self.annot_root, self.annotlist[index])
        assert os.path.exists(imgname), imgname
        assert os.path.exists(annname), annname
        assert os.path.basename(imgname).split('.')[0] == os.path.basename(annname).split('.')[0], (imgname, annname)
        if not self.no_img:
            img = cv2.imread(imgname)
        else:
            img = None
        annot = read_annot(annname, self.mode)
        return img, annot
    
    def __len__(self) -> int:
        return self.nFrames
    
    def write_smpl(self, peopleDict, nf):
        results = []
        for pid, people in peopleDict.items():
            result = {'id': pid}
            result.update(people.body_params)
            results.append(result)
        self.writer.write_smpl(results, nf)
    
    def vis_detections(self, image, detections, nf, to_img=True):
        return self.writer.vis_detections([image], [detections], nf, 
            key='keypoints', to_img=to_img, vis_id=True)
    
    def vis_repro(self, peopleDict, image, annots, nf):
        # 可视化重投影的关键点与输入的关键点
        detections = []
        for pid, data in peopleDict.items():
            keypoints3d = (data.keypoints3d @ self.camera['R'].T + self.camera['T'].T) @ self.camera['K'].T
            keypoints3d[:, :2] /= keypoints3d[:, 2:]
            keypoints3d = np.hstack([keypoints3d, data.keypoints3d[:, -1:]])
            det = {
                'id': pid,
                'repro': keypoints3d
            }
            detections.append(det)
        return self.writer.vis_detections([image], [detections], nf, key='repro',
            to_img=True, vis_id=False)

    def vis_smpl(self, peopleDict, faces, image, nf, sub_vis=[], 
        mode='smpl', extra_data=[], add_back=True,
        axis=np.array([1., 0., 0.]), degree=0., fix_center=None):
        # 为了统一接口，旋转视角的在此处实现，只在单视角的数据中使用
        # 通过修改相机参数实现
        # 相机参数的修正可以通过计算点的中心来获得
        # render the smpl to each view
        render_data = {}
        for pid, data in peopleDict.items():
            render_data[pid] = {
                'vertices': data.vertices, 'faces': faces, 
                'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        for iid, extra in enumerate(extra_data):
            render_data[10000+iid] = {
                'vertices': extra['vertices'],
                'faces': extra['faces'],
                'colors': extra['colors'],
                'name': extra['name']
            }
        camera = {}
        for key in self.camera.keys():
            camera[key] = self.camera[key][None, :, :]
        # render another view point
        if np.abs(degree) > 1e-3:
            vertices_all = np.vstack([data.vertices for data in peopleDict.values()])
            if fix_center is None:
                center = np.mean(vertices_all, axis=0, keepdims=True)
                new_center = center.copy()
                new_center[:, 0:2] = 0
            else:
                center = fix_center.copy()
                new_center = fix_center.copy()
                new_center[:, 2] *= 1.5
            direc = np.array(axis)
            rot, _ = cv2.Rodrigues(direc*degree/90*np.pi/2)
            # If we rorate the data, it is like:
            # V = Rnew @ (V0 - center) + new_center
            #   = Rnew @ V0 - Rnew @ center + new_center
            # combine with the camera
            # VV = Rc(Rnew @ V0 - Rnew @ center + new_center) + Tc
            #    = Rc@Rnew @ V0 + Rc @ (new_center - Rnew@center) + Tc
            blank = np.zeros_like(image, dtype=np.uint8) + 255
            images = [image, blank]
            Rnew = camera['R'][0] @ rot
            Tnew = camera['R'][0] @ (new_center.T - rot @ center.T) + camera['T'][0]
            camera['K'] = np.vstack([camera['K'], camera['K']])
            camera['R'] = np.vstack([camera['R'], Rnew[None, :, :]])
            camera['T'] = np.vstack([camera['T'], Tnew[None, :, :]])
        else:
            images = [image]
        self.writer.vis_smpl(render_data, nf, images, camera, mode, add_back=add_back)

class MVBase(Dataset):
    """ Dataset for multiview data
    """
    def __init__(self, root, cams=[], out=None, config={}, 
        image_root='images', annot_root='annots', 
        mode='body25',
        undis=True, no_img=False) -> None:
        self.root = root
        self.image_root = join(root, image_root)
        self.annot_root = join(root, annot_root)
        self.mode = mode
        self.undis = undis
        self.no_img = no_img
        self.config = config
        # results path
        # the results store keypoints3d
        self.skel_path = None
        if out is None:
            out = join(root, 'output')
        self.out = out
        self.writer = FileWriter(self.out, config=config)
        
        if len(cams) == 0:
            cams = sorted([i for i in os.listdir(self.image_root) if os.path.isdir(join(self.image_root, i))])
            if cams[0].isdigit(): # 对于使用数字命名的文件夹
                cams.sort(key=lambda x:int(x))
        self.cams = cams
        self.imagelist = {}
        self.annotlist = {}
        for cam in cams: #TODO: 增加start,end
            imgnames = sorted(os.listdir(join(self.image_root, cam)))
            self.imagelist[cam] = imgnames
            self.annotlist[cam] = sorted(os.listdir(join(self.annot_root, cam)))
        nFrames = min([len(val) for key, val in self.imagelist.items()])
        self.nFrames = nFrames
        self.nViews = len(cams)
        self.read_camera()
    
    def read_camera(self):
        path = self.root
        # 读入相机参数
        intri_name = join(path, 'intri.yml')
        extri_name = join(path, 'extri.yml')
        if os.path.exists(intri_name) and os.path.exists(extri_name):
            self.cameras = read_camera(intri_name, extri_name, self.cams)
            self.cameras.pop('basenames')
            self.cameras_for_affinity = [[cam['invK'], cam['R'], cam['T']] for cam in [self.cameras[name] for name in self.cams]]
            self.Pall = [self.cameras[cam]['P'] for cam in self.cams]
            self.Fall = get_fundamental_matrix(self.cameras, self.cams)
        else:
            print('!!!there is no camera parameters, maybe bug', intri_name, extri_name)
            self.cameras = None

    def undistort(self, images):
        if self.cameras is not None and len(images) > 0:
            images_ = []
            for nv in range(self.nViews):
                mtx = self.cameras[self.cams[nv]]['K']
                dist = self.cameras[self.cams[nv]]['dist']
                frame = cv2.undistort(images[nv], mtx, dist, None)
                images_.append(frame)
        else:
            images_ = images
        return images_
    
    def undis_det(self, lDetections):
        for nv in range(len(lDetections)):
            camera = self.cameras[self.cams[nv]]
            for det in lDetections[nv]:
                det['bbox'] = undistort(camera, bbox=det['bbox'])
                keypoints = det['keypoints']
                det['keypoints'] = undistort(camera, keypoints=keypoints[None, :, :])[1][0]
        return lDetections

    def __getitem__(self, index: int):
        images, annots = [], []
        for cam in self.cams:
            imgname = join(self.image_root, cam, self.imagelist[cam][index])
            annname = join(self.annot_root, cam, self.annotlist[cam][index])
            assert os.path.exists(imgname), imgname
            assert os.path.exists(annname), annname
            assert self.imagelist[cam][index].split('.')[0] == self.annotlist[cam][index].split('.')[0]
            if not self.no_img:
                img = cv2.imread(imgname)
                images.append(img)
            # TODO:这里直接取了0
            annot = read_annot(annname, self.mode)
            annots.append(annot)
        if self.undis:
            images = self.undistort(images)
            annots = self.undis_det(annots)
        return images, annots
    
    def __len__(self) -> int:
        return self.nFrames
    
    def vis_detections(self, images, lDetections, nf, to_img=True, sub_vis=[]):
        if len(sub_vis) != 0:
            valid_idx = [self.cams.index(i) for i in sub_vis]
            images = [images[i] for i in valid_idx]
            lDetections = [lDetections[i] for i in valid_idx]
        return self.writer.vis_detections(images, lDetections, nf, 
            key='keypoints', to_img=to_img, vis_id=True)
    
    def vis_match(self, images, lDetections, nf, to_img=True, sub_vis=[]):
        if len(sub_vis) != 0:
            valid_idx = [self.cams.index(i) for i in sub_vis]
            images = [images[i] for i in valid_idx]
            lDetections = [lDetections[i] for i in valid_idx]
        return self.writer.vis_detections(images, lDetections, nf, 
            key='match', to_img=to_img, vis_id=True)
    
    def write_keypoints3d(self, peopleDict, nf):
        results = []
        for pid, people in peopleDict.items():
            result = {'id': pid, 'keypoints3d': people.keypoints3d.tolist()}
            results.append(result)
        self.writer.write_keypoints3d(results, nf)

    def write_smpl(self, peopleDict, nf):
        results = []
        for pid, people in peopleDict.items():
            result = {'id': pid}
            result.update(people.body_params)
            results.append(result)
        self.writer.write_smpl(results, nf)

    def read_skel(self, nf, mode='none'):
        if mode == 'a4d':
            outname = join(self.skel_path, '{}.txt'.format(nf))
            assert os.path.exists(outname), outname
            skels = readReasultsTxt(outname)
        elif mode == 'none':
            outname = join(self.skel_path, '{:06d}.json'.format(nf))
            assert os.path.exists(outname), outname
            skels = readResultsJson(outname)
        else:
            import ipdb; ipdb.set_trace()
        return skels
    
    def read_smpl(self, nf):
        outname = join(self.skel_path, '{:06d}.json'.format(nf))
        assert os.path.exists(outname), outname
        datas = read_json(outname)
        outputs = []
        for data in datas:
            for key in ['Rh', 'Th', 'poses', 'shapes']:
                data[key] = np.array(data[key])
            outputs.append(data)
        return outputs