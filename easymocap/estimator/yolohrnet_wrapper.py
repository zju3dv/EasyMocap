from ..annotator.file_utils import read_json
from .wrapper_base import check_result, create_annot_file, save_annot
from glob import glob
from os.path import join
from tqdm import tqdm
import os
import cv2
import numpy as np

def detect_frame(detector, img, pid=0, only_bbox=False):
    lDetections = detector.detect([img], only_bbox=only_bbox)[0]
    annots = []
    for i in range(len(lDetections)):
        annot = {
            'bbox': [float(d) for d in lDetections[i]['bbox']],
            'personID': pid + i,
            'isKeyframe': False
        }
        if not only_bbox:
            annot['keypoints'] = lDetections[i]['keypoints'].tolist()
        annots.append(annot)
    return annots

def extract_bbox(image_root, annot_root, ext, **config):
    force = config.pop('force')
    if check_result(image_root, annot_root) and not force:
        return 0
    import torch
    from .YOLOv4 import YOLOv4
    device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
    detector = YOLOv4(device=device, **config)
    imgnames = sorted(glob(join(image_root, '*'+ext)))
    if len(imgnames) == 0:
        ext = '.png'
        imgnames = sorted(glob(join(image_root, '*'+ext)))
    # run_yolo(image_root, )
    for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
        base = os.path.basename(imgname).replace(ext, '')
        annotname = join(annot_root, base+'.json')
        annot = create_annot_file(annotname, imgname)
        image = cv2.imread(imgname)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.predict_single(image_rgb)
        annots = []
        pid = 0
        for i in range(len(detections)):
            annot_ = {
                'bbox': [float(d) for d in detections[i]],
                'isKeyframe': False
            }
            annot_['area'] = max(annot_['bbox'][2] - annot_['bbox'][0], annot_['bbox'][3] - annot_['bbox'][1])**2
            annots.append(annot_)
        annots.sort(key=lambda x:-x['area'])
        # re-assign the person ID
        for i in range(len(annots)):
            annots[i]['personID'] = i + pid
        annot['annots'] = annots
        save_annot(annotname, annot)

def extract_hrnet(image_root, annot_root, ext, **config):
    config.pop('force')
    import torch
    imgnames = sorted(glob(join(image_root, '*'+ext)))
    import torch
    device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
    from .HRNet import SimpleHRNet
    estimator = SimpleHRNet(device=device, **config)

    for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
        base = os.path.basename(imgname).replace(ext, '')
        annotname = join(annot_root, base+'.json')
        annots = read_json(annotname)
        detections = np.array([data['bbox'] for data in annots['annots']])
        image = cv2.imread(imgname)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points2d = estimator.predict(image_rgb, detections)
        for i in range(detections.shape[0]):
            annot_ = annots['annots'][i]
            annot_['keypoints'] = points2d[i]
        save_annot(annotname, annots)

def extract_yolo_hrnet(image_root, annot_root, ext, config_yolo, config_hrnet):
    config_yolo.pop('ext', None)
    imgnames = sorted(glob(join(image_root, '*{}'.format(ext))))
    import torch
    device = torch.device('cuda')
    from .YOLOv4 import YOLOv4
    device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
    detector = YOLOv4(device=device, **config_yolo)
    from .HRNet import SimpleHRNet
    estimator = SimpleHRNet(device=device, **config_hrnet)

    for nf, imgname in enumerate(tqdm(imgnames, desc=os.path.basename(image_root))):
        base = os.path.basename(imgname).replace(ext, '')
        annotname = join(annot_root, base+'.json')
        annot = create_annot_file(annotname, imgname)
        img0 = cv2.imread(imgname)
        annot = create_annot_file(annotname, imgname)
        image = cv2.imread(imgname)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.predict_single(image_rgb)
        # forward_hrnet
        points2d = estimator.predict(image_rgb, detections)
        annots = []
        pid = 0
        for i in range(len(detections)):
            annot_ = {
                'bbox': [float(d) for d in detections[i]],
                'keypoints': points2d[i],
                'isKeyframe': False
            }
            annot_['area'] = max(annot_['bbox'][2] - annot_['bbox'][0], annot_['bbox'][3] - annot_['bbox'][1])**2
            annots.append(annot_)
        annots.sort(key=lambda x:-x['area'])
        # re-assign the person ID
        for i in range(len(annots)):
            annots[i]['personID'] = i + pid
        annot['annots'] = annots
        save_annot(annotname, annot)