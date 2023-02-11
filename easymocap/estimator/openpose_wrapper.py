'''
  @ Date: 2021-08-21 14:16:38
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-05-23 23:10:43
  @ FilePath: /EasyMocapPublic/easymocap/estimator/openpose_wrapper.py
'''
import os
import shutil
from tqdm import tqdm
from .wrapper_base import bbox_from_keypoints, create_annot_file, check_result
from ..mytools import read_json
from ..annotator.file_utils import save_annot
from os.path import join
import numpy as np
import cv2
from glob import glob
from multiprocessing import Process

def run_openpose(image_root, annot_root, config):
    image_root = os.path.realpath(image_root)
    annot_root = os.path.realpath(annot_root)

    os.makedirs(annot_root, exist_ok=True)
    pwd = os.getcwd()
    if os.name != 'nt':
        cmd = './build/examples/openpose/openpose.bin --image_dir {} --write_json {} --display 0'.format(
            image_root, annot_root)
    else:
        cmd = 'bin\\OpenPoseDemo.exe --image_dir {} --write_json {} --display 0'.format(
            os.path.abspath(image_root), os.path.abspath(annot_root))
    if config['res'] != 1:
        cmd = cmd + ' --net_resolution -1x{}'.format(int(16*((368*config['res'])//16)))
    if config['hand']:
        cmd = cmd + ' --hand'
    if config['face']:
        cmd = cmd + ' --face'
    if config['vis']:
        cmd = cmd + ' --write_images {}'.format(annot_root)
    else:
        cmd = cmd + ' --render_pose 0'
    os.chdir(config['root'])
    print(cmd)
    os.system(cmd)
    os.chdir(pwd)

def convert_from_openpose(src, dst, image_root, ext):
    # convert the 2d pose from openpose
    inputlist = sorted(os.listdir(src))
    for inp in tqdm(inputlist, desc='{:10s}'.format(os.path.basename(dst))):
        annots = load_openpose(join(src, inp))
        base = inp.replace('_keypoints.json', '')
        annotname = join(dst, base+'.json')
        imgname = join(image_root, inp.replace('_keypoints.json', ext))
        annot = create_annot_file(annotname, imgname)
        annot['annots'] = annots
        save_annot(annotname, annot)

global_tasks = []

def extract_2d(image_root, annot_root, tmp_root, config):
    if check_result(image_root, annot_root):
        return global_tasks
    if not check_result(image_root, tmp_root):
        run_openpose(image_root, tmp_root, config)
    # TODO: add current task to global_tasks
    thread = Process(target=convert_from_openpose, 
        args=(tmp_root, annot_root, image_root, config['ext'])) # 应该不存在任何数据竞争
    thread.start()
    global_tasks.append(thread)
    return global_tasks

def load_openpose(opname):
    mapname = {
        'face_keypoints_2d':'face2d', 
        'hand_left_keypoints_2d':'handl2d', 
        'hand_right_keypoints_2d':'handr2d'}
    assert os.path.exists(opname), opname
    data = read_json(opname)
    out = []
    pid = 0
    for i, d in enumerate(data['people']):
        keypoints = d['pose_keypoints_2d']
        keypoints = np.array(keypoints).reshape(-1, 3)
        annot = {
            'bbox': bbox_from_keypoints(keypoints),
            'personID': pid + i,
            'keypoints': keypoints.tolist(),
            'isKeyframe': False
        }
        bbox = annot['bbox']
        if bbox[-1] < 0.01:
            continue
        annot['area'] = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])
        for key in mapname.keys():
            if len(d[key]) == 0:
                continue
            kpts = np.array(d[key]).reshape(-1, 3)
            annot[mapname[key]] = kpts.tolist()
            annot['bbox_'+mapname[key]] = bbox_from_keypoints(kpts)
        out.append(annot)
    out.sort(key=lambda x:-x['area'])
    for i in range(len(out)):
        out[i]['personID'] = pid + i
    return out

def get_crop(image, bbox, rot, scale=1.2):
    l, t, r, b, _ = bbox
    cx = (l+r)/2
    cy = (t+b)/2
    wx = (r-l)*scale/2
    wy = (b-t)*scale/2
    l = cx - wx
    r = cx + wx
    t = cy - wy
    b = cy + wy
    l = max(0, int(l+0.5))
    t = max(0, int(t+0.5))
    r = min(image.shape[1], int(r+0.5))
    b = min(image.shape[0], int(b+0.5))
    crop = image[t:b, l:r].copy()
    crop = np.ascontiguousarray(crop)
    # rotate the image
    if rot == 180:
        crop = cv2.flip(crop, -1)
    return crop, (l, t)

def transoform_foot(crop_shape, start, rot, keypoints, kpts_old=None):
    l, t = start
    if rot == 180:
        keypoints[..., 0] = crop_shape[1] - keypoints[..., 0] - 1
        keypoints[..., 1] = crop_shape[0] - keypoints[..., 1] - 1
    keypoints[..., 0] += l
    keypoints[..., 1] += t
    if kpts_old is None:
        kpts_op = keypoints[0]
        return kpts_op
    # 选择最好的
    kpts_np = np.array(kpts_old)
    dist = np.linalg.norm(kpts_np[None, :15, :2] - keypoints[:, :15, :2], axis=-1)
    conf = np.minimum(kpts_np[None, :15, 2], keypoints[:, :15, 2])
    dist = (dist * conf).sum(axis=-1)/conf.sum(axis=-1)*conf.shape[1]/(conf>0).sum(axis=-1)
    best = dist.argmin()
    kpts_op = keypoints[best]
    # TODO:判断一下关键点
    # 这里以HRNet的估计为准
    # WARN: disable feet
    # 判断OpenPose的脚与HRNet的脚是否重合
    if (kpts_np[[11, 14], -1] > 0.3).all() and (kpts_op[[11, 14], -1] > 0.3).all():
        dist_ll = np.linalg.norm(kpts_np[11, :2] - kpts_op[11, :2])
        dist_rr = np.linalg.norm(kpts_np[14, :2] - kpts_op[14, :2])
        dist_lr = np.linalg.norm(kpts_np[11, :2] - kpts_op[14, :2])
        dist_rl = np.linalg.norm(kpts_np[14, :2] - kpts_op[11, :2])
        if dist_lr < dist_ll and dist_rl < dist_rr:
            kpts_op[[19, 20, 21, 22, 23, 24]] = kpts_op[[22, 23, 24, 19, 20, 21]]
    # if (kpts_np[[11, 14], -1] > 0.3).all() and (kpts_op[[19, 22], -1] > 0.3).all():
    #     if np.linalg.norm(kpts_op[19, :2] - kpts_np[11, :2]) \
    #         < np.linalg.norm(kpts_op[19, :2] - kpts_np[14, :2])\
    #         and np.linalg.norm(kpts_op[22, :2] - kpts_np[11, :2]) \
    #             > np.linalg.norm(kpts_op[22, :2] - kpts_np[14, :2]):
    #             kpts_op[[19, 20, 21, 22, 23, 24]] = kpts_op[[22, 23, 24, 19, 20, 21]]
    #             print('[info] swap left and right')
    # 直接选择第一个
    kpts_np[19:] = kpts_op[19:]
    return kpts_np

def filter_feet(kpts):
    # 判断左脚
    if (kpts[[13, 14, 19], -1]>0).all():
        l_feet = ((kpts[[19,20,21],-1]>0)*np.linalg.norm(kpts[[19, 20, 21], :2] - kpts[14, :2], axis=-1)).max()
        l_leg = np.linalg.norm(kpts[13, :2] - kpts[14, :2])
        if  l_leg < 1.5 * l_feet:
            kpts[[19, 20, 21]] = 0.
            print('[LOG] remove left ankle {} < {}'.format(l_leg, l_feet))
    # 判断右脚
    if (kpts[[10, 11], -1]>0).all():
        l_feet = ((kpts[[22, 23, 24],-1]>0)*np.linalg.norm(kpts[[22, 23, 24], :2] - kpts[11, :2], axis=-1)).max()
        l_leg = np.linalg.norm(kpts[10, :2] - kpts[11, :2])
        if l_leg < 1.5 * l_feet:
            kpts[[22, 23, 24]] = 0.
            print('[LOG] remove right ankle {} < {}'.format(l_leg, l_feet))
    return kpts

class FeetEstimatorByCrop:
    def __init__(self, openpose, tmpdir=None, fullbody=False, hand=False, face=False) -> None:
        self.openpose = openpose
        if tmpdir is None:
            tmpdir = os.path.abspath(join('./', 'tmp'))
        else:
            tmpdir = os.path.abspath(tmpdir)
        self.tmpdir = tmpdir
        self.fullbody = fullbody
        self.hand = hand
        self.face = face
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(join(tmpdir, 'images'), exist_ok=True)
        self.config = {
            'root': self.openpose,
            'res': 1,
            'hand': hand, # detect hand when in fullbody mode
            'face': face,
            'vis': False,
        }

    def detect_foot(self, image_root, annot_root, ext):
        # TODO:换成取heatmap的最大值
        THRES = 0.3
        imgnames = sorted(glob(join(image_root, '*'+ext)))
        if len(imgnames) == 0:
            # 尝试换成png格式
            ext = '.png'
            imgnames = sorted(glob(join(image_root, '*'+ext)))
        infos = {}
        crop_counts = 0
        for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
            base = os.path.basename(imgname).replace(ext, '')
            annotname = join(annot_root, base+'.json')
            annots = read_json(annotname)
            image = cv2.imread(imgname)
            if 'detect_feet' in annots.keys() and annots['detect_feet'] and False:
                continue
            infos[base] = {}
            for i, annot in enumerate(annots['annots']):
                bbox = annot['bbox']
                # 判断bbox大小
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if width < 100 and height < 100:
                    continue
                rot = 0
                if not self.fullbody:
                    kpts = np.array(annot['keypoints'])
                    # 判断有没有脚的关键点
                    if kpts[11][-1] < THRES and kpts[14][-1] < THRES:
                        continue
                    # 判断(1, 8)的朝向
                    if kpts[1][-1] < THRES or kpts[8][-1] < THRES:
                        continue
                    dir_1_8 = np.array([kpts[1][0]-kpts[8][0], kpts[1][1]-kpts[8][1]])
                    dir_1_8 = dir_1_8 / np.linalg.norm(dir_1_8)
                    if dir_1_8[1] > 0.8:
                        rot = 180
                crop, start = get_crop(image, bbox, rot)
                cropname = join(self.tmpdir, 'images', f'{base}_{i}.jpg')
                infos[base][i] = {
                    'crop_shape': crop.shape,
                    'start': start,
                    'rot': rot,
                    'name': f'{base}_{i}.json'
                }
                cv2.imwrite(cropname, crop)
                crop_counts += 1
        tmp_image_root = join(self.tmpdir, 'images')
        tmp_annot_root = join(self.tmpdir, 'annots')
        tmp_tmp_root = join(self.tmpdir, 'openpose')
        print(len(os.listdir(tmp_image_root)), crop_counts)
        run_openpose(tmp_image_root, tmp_tmp_root, self.config)
        convert_from_openpose(tmp_tmp_root, tmp_annot_root, tmp_image_root, '.jpg') # 应该不存在任何数据竞争
        for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
            base = os.path.basename(imgname).replace(ext, '')
            annotname = join(annot_root, base+'.json')
            annots_ori = read_json(annotname)
            if base not in infos.keys():
                continue
            for i, annots in enumerate(annots_ori['annots']):
                if i not in infos[base].keys():
                    continue
                info = infos[base][i]
                cropname = join(tmp_annot_root, info['name'])
                if not os.path.exists(cropname):
                    print('[WARN] {} not exists!'.format(cropname))
                    continue
                annots_sub = read_json(cropname)['annots']
                if len(annots_sub) < 1:
                    continue
                keypoints = np.stack([np.array(d['keypoints']) for d in annots_sub])
                if self.hand or self.face:
                    for key in ['handl2d', 'handr2d', 'face2d']:
                        if key in annots_sub[0].keys():
                            khand = np.array(annots_sub[0][key])[None]
                            annots[key] = transoform_foot(info['crop_shape'], info['start'], info['rot'], khand, None)
                            annots['bbox_'+key] = bbox_from_keypoints(annots[key])
                if self.fullbody:
                    kpts_np = transoform_foot(info['crop_shape'], info['start'], info['rot'], keypoints, None)
                else:
                    kpts = annots['keypoints']
                    kpts_np = transoform_foot(info['crop_shape'], info['start'], info['rot'], keypoints, kpts)
                    if False: # WARN: disable filter feet
                        kpts_np = filter_feet(kpts_np)
                annots['keypoints'] = kpts_np
            annots_ori['detect_feet'] = True
            save_annot(annotname, annots_ori)

class FeetEstimator:
    def __init__(self, openpose='/media/qing/Project/openpose') -> None:
        import sys
        sys.path.append('{}/build_py/python'.format(openpose))
        from openpose import pyopenpose as op
        opWrapper = op.WrapperPython()
        params = dict()
        params["model_folder"] = "{}/models".format(openpose)
        opWrapper.configure(params)
        opWrapper.start()
        self.wrapper = opWrapper
        self.rect = op.Rectangle
        self.datum = op.Datum
        try:
            self.vec = op.VectorDatum
        except:
            self.vec = lambda x:x

    def detect_foot(self, image_root, annot_root, ext):
        # TODO:换成取heatmap的最大值
        THRES = 0.3
        imgnames = sorted(glob(join(image_root, '*'+ext)))
        for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
            base = os.path.basename(imgname).replace(ext, '')
            annotname = join(annot_root, base+'.json')
            annots = read_json(annotname)
            image = cv2.imread(imgname)
            for annot in annots['annots']:
                bbox = annot['bbox']
                kpts = np.array(annot['keypoints'])
                # 判断(1, 8)的朝向
                if kpts[1][-1] < THRES or kpts[8][-1] < THRES:
                    continue
                dir_1_8 = np.array([kpts[1][0]-kpts[8][0], kpts[1][1]-kpts[8][1]])
                dir_1_8 = dir_1_8 / np.linalg.norm(dir_1_8)
                if dir_1_8[1] > 0.8:
                    rot = 180
                else:
                    rot = 0
                kpts = self._detect_with_bbox(image, kpts, bbox, rot=rot)
                kpts = filter_feet()
                annot['keypoints'] = kpts
            save_annot(annotname, annots)

    def _detect_with_bbox(self, image, kpts, bbox, rot=0):
        crop, start = get_crop(image, bbox, rot)
        datum = self.datum()
        datum.cvInputData = crop
        self.wrapper.emplaceAndPop(self.vec([datum]))
        # keypoints: (N, 25, 3)
        keypoints = datum.poseKeypoints
        if len(keypoints.shape) < 3:
            print(keypoints)
            print('Not detect person!')
            return kpts
        kpts_np = transoform_foot(crop.shape, start, rot, keypoints, kpts)
        return kpts_np