'''
  @ Date: 2021-01-13 16:53:55
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-14 19:55:58
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

from mytools.camera_utils import read_camera, undistort, write_camera
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


def read_annot(annotname, add_hand_face=False):
    data = read_json(annotname)['annots']
    for i in range(len(data)):
        data[i]['id'] = data[i].pop('personID')
        for key in ['bbox', 'keypoints', 'handl2d', 'handr2d', 'face2d']:
            if key not in data[i].keys():continue
            data[i][key] = np.array(data[i][key])
    return data

def get_bbox_from_pose(pose_2d, img, rate = 0.1):
    # this function returns bounding box from the 2D pose
    validIdx = pose_2d[:, 2] > 0
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
        keys = ['keypoints3d', 'smpl', 'repro', 'keypoints']
        output_dict = {key:join(self.out, key) for key in keys}
        for key, p in output_dict.items():
            os.makedirs(p, exist_ok=True)
        self.output_dict = output_dict
        
        self.basenames = basenames
        if cfg is not None:
            print(cfg, file=open(join(output_path, 'exp.yml'), 'w'))
        self.save_origin = False
        self.config = config
    
    def write_keypoints3d(self, results, nf):
        savename = join(self.output_dict['keypoints3d'], '{:06d}.json'.format(nf))
        save_json(savename, results)
    
    def vis_detections(self, images, lDetections, nf, key='keypoints', to_img=True, vis_id=True):
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            for det in lDetections[nv]:
                keypoints = det[key]
                bbox = det.pop('bbox', get_bbox_from_pose(keypoints, img))
                # bbox = det['bbox']
                plot_bbox(img, bbox, pid=det['id'], vis_id=vis_id)
                plot_keypoints(img, keypoints, pid=det['id'], config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        image_vis = merge(images_vis, resize=not self.save_origin)
        if to_img:
            savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            cv2.imwrite(savename, image_vis)
        return image_vis
    
    def write_smpl(self, results, nf):
        format_out = {'float_kind':lambda x: "%.3f" % x}
        filename = join(self.output_dict['smpl'], '{:06d}.json'.format(nf))
        with open(filename, 'w') as f:
            f.write('[\n')
            for data in results:
                f.write('    {\n')
                output = {}
                output['id'] = data['id']
                output['Rh']   = np.array2string(data['Rh'], max_line_width=1000, separator=', ', formatter=format_out)
                output['Th']   = np.array2string(data['Th'], max_line_width=1000, separator=', ', formatter=format_out)
                output['poses'] = np.array2string(data['poses'], max_line_width=1000, separator=', ', formatter=format_out)
                output['shapes']  = np.array2string(data['shapes'], max_line_width=1000, separator=', ', formatter=format_out)
                for key in ['id', 'Rh', 'Th', 'poses', 'shapes']:
                    f.write('        \"{}\": {},\n'.format(key, output[key]))
                f.write('    },\n')
            f.write(']\n')

    def vis_smpl(self, render_data, nf, images, cameras):
        from visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None)
        render_results = render.render(render_data, cameras, images)
        image_vis = merge(render_results, resize=not self.save_origin)
        savename = join(self.output_dict['smpl'], '{:06d}.jpg'.format(nf))
        cv2.imwrite(savename, image_vis)

class MVBase(Dataset):
    """ Dataset for multiview data
    """
    def __init__(self, root, cams=[], out=None, config={}, 
        image_root='images', annot_root='annots', 
        add_hand_face=True,
        undis=True, no_img=False) -> None:
        self.root = root
        self.image_root = join(root, image_root)
        self.annot_root = join(root, annot_root)
        self.add_hand_face = add_hand_face
        self.undis = undis
        self.no_img = no_img
        self.config = config
        
        if out is None:
            out = join(root, 'output')
        self.out = out
        self.writer = FileWriter(self.out, config=config)
        
        if len(cams) == 0:
            cams = sorted([i for i in os.listdir(self.image_root) if os.path.isdir(join(self.image_root, i))])
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
            annot = read_annot(annname, self.add_hand_face)
            annots.append(annot)
        if self.undis:
            images = self.undistort(images)
            annots = self.undis_det(annots)
        return images, annots
    
    def __len__(self) -> int:
        return self.nFrames