import torch
import numpy as np
import os
import cv2
from os.path import join
import pickle

def check_modelpath(paths):
    if isinstance(paths, str):
        assert os.path.exists(paths), paths
        return paths
    elif isinstance(paths, list):
        for path in paths:
            if os.path.exists(path):
                print(f'Found model in {path}')
                break
        else:
            print(f'No model found in {paths}!')
            raise FileExistsError
        return path
    else:
        raise NotImplementedError

class BaseYOLOv5:
    def __init__(self, ckpt=None, model='yolov5m', name='object2d', multiview=True) -> None:
        if ckpt is not None:
            ckpt = check_modelpath(ckpt)
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', ckpt)
        else:
            print('[{}] Not given ckpt, use default yolov5'.format(self.__class__.__name__))
            self.model = torch.hub.load('ultralytics/yolov5', model)
        self.multiview = multiview
        self.name = name
    
    def dump(self, cachename, output):
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        with open(cachename, 'wb') as f:
            pickle.dump(output, f)
        return output
    
    def load(self, cachename):
        with open(cachename, 'rb') as f:
            output = pickle.load(f)
        return output

    def check_cache(self, imgname):
        basename = os.path.basename(imgname)
        imgext = '.' + basename.split('.')[-1]
        nv = imgname.split(os.sep)[-2]
        cachename = join(self.output, self.name, nv, basename.replace(imgext, '.npy'))
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        if os.path.exists(cachename):
            output = self.load(cachename)
            return True, output, cachename
        else:
            return False, None, cachename
    
    def check_image(self, img_or_name):
        if isinstance(img_or_name, str):
            images = cv2.imread(img_or_name)
        else:
            images = img_or_name
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        return images
    
    @torch.no_grad()
    def detect(self, image, imgname):
        flag, cache, cachename = self.check_cache(imgname)
        if flag:
            return cache
        image = self.check_image(imgname)
        results = self.model(image) #RGB images[:,:,::-1]
        arrays = np.array(results.pandas().xyxy[0])
        res = {
            'results': arrays,
            'image_shape': image.shape,
        }
        self.dump(cachename, res)
        return res
    
    @staticmethod
    def select_class(results, name):
        select = []
        for i, res in enumerate(results['results']):
            classname = res[6]
            if classname != name:
                continue
            box = res[:5]
            select.append(box)
        select = np.stack(select)
        return select, results

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        # Naive: select the best
        idx = np.argsort(select[:, -1])[::-1]
        return select[idx[0:1]]

    def __call__(self, images, imgnames): # 这里好像默认是多视角了，需要继承一下单视角的
        squeeze = False
        if not isinstance(images, list):
            images = [images]
            imgnames = [imgnames]
            squeeze = True
        detects = {'bbox': [[] for _ in range(len(images))]}
        for nv in range(len(images)):
            res = self.detect(images[nv], imgnames[nv])            
            select, res = self.select_class(res, self.name)
            if len(select) == 0:
                select = np.zeros((0,5), dtype=np.float32)
            else:
                select = np.stack(select).astype(np.float32)
            # TODO: add track here
            select = self.select_bbox(select, res, imgnames[nv])
            detects['bbox'][nv] = select
        if squeeze:
            detects['bbox'] = detects['bbox'][0]
        return detects

class YoloWithTrack(BaseYOLOv5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.track_cache = {}

    @staticmethod
    def calculate_iou(bbox_pre, bbox_now):
        area_now = (bbox_now[:, 2] - bbox_now[:, 0])*(bbox_now[:, 3]-bbox_now[:, 1])
        area_pre = (bbox_pre[:, 2] - bbox_pre[:, 0])*(bbox_pre[:, 3]-bbox_pre[:, 1])
        # compute IOU
        # max of left
        xx1 = np.maximum(bbox_now[:, 0], bbox_pre[:, 0])
        yy1 = np.maximum(bbox_now[:, 1], bbox_pre[:, 1])
        # min of right
        xx2 = np.minimum(bbox_now[:, 0+2], bbox_pre[:, 0+2])
        yy2 = np.minimum(bbox_now[:, 1+2], bbox_pre[:, 1+2])
        # w h
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        over = (w*h)/(area_pre+area_now-w*h)
        return over

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        sub = os.path.basename(os.path.dirname(imgname))
        frame = int(os.path.basename(imgname).split('.')[0])
        if sub not in self.track_cache:
            # select the best
            select = super().select_bbox(select, results, imgname)
            self.track_cache[sub] = {
                'frame': [frame],
                'bbox': [select]
            }
            return select
        bbox_pre = self.track_cache[sub]['bbox'][-1]
        iou = self.calculate_iou(bbox_pre, select)
        idx = iou.argmax()
        select = select[idx:idx+1]
        self.track_cache[sub]['frame'].append(frame)
        self.track_cache[sub]['bbox'].append(select)
        return select

class MultiPerson(BaseYOLOv5):
    def __init__(self, min_length, max_length, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        print('[{}] Only keep the bbox in [{}, {}]'.format(self.__class__.__name__, min_length, max_length))

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        # 判断一下面积
        area = np.sqrt((select[:, 2] - select[:, 0])*(select[:, 3]-select[:, 1]))
        valid = (area > self.min_length) & (area < self.max_length)
        height, width, _ = results['image_shape']
        # set the limit of left and right
        valid = valid & (select[:, 2] > self.min_length * 1.5) & (select[:, 0] < width - self.min_length * 1.5)
        return select[valid]

class DetectToPelvis:
    def __init__(self, key) -> None:
        self.key = key
        self.multiview = True
    
    def __call__(self, **kwargs):
        key = self.key
        val = kwargs[key]
        ret = {'pelvis': []}
        for nv in range(len(val)):
            bbox = val[nv]
            center = np.stack([(bbox[:, 0] + bbox[:, 2])/2, (bbox[:, 1] + bbox[:, 3])/2, bbox[:, -1]], axis=-1)
            ret['pelvis'].append(center)
        return ret

class Yolo_model:
    def __init__(self, mode, yolo_ckpt, multiview, repo_or_dir = 'ultralytics/yolov5', source='github') -> None:
        yolo_ckpt = check_modelpath(yolo_ckpt)
        self.model = torch.hub.load(repo_or_dir, 'custom', yolo_ckpt, source=source)
        self.min_detect_thres = 0.3
        self.mode = mode # 'fullimg' # 'bboxcrop'
        self.output = 'output'
        self.name = 'yolo'
        self.multiview = multiview
    @torch.no_grad()
    def det_step(self, img_or_name, imgname, bbox=[]):

        basename = os.path.basename(imgname)
        if self.multiview:
            nv = imgname.split('/')[-2]
            cachename = join(self.output, self.name, nv, basename.replace('.jpg', '.pkl'))
        else:
            cachename = join(self.output, self.name, basename.replace('.jpg', '.pkl'))
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        if os.path.exists(cachename):
            with open(cachename, 'rb') as f:
                output = pickle.load(f)
            return output

        if isinstance(img_or_name,str):
            images = cv2.imread(img_or_name)
        else:
            images = img_or_name

        if self.mode == 'bboxcrop':
            bbox[0] = max(0,bbox[0])
            bbox[1] = max(0,bbox[1])
            crop = images[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),::-1]
        else:
            crop = images[:,:,::-1]
        # print("[yolo img shape] ",crop.shape)
        results = self.model(crop) #RGB images[:,:,::-1]
        # breakpoint()
        arrays = np.array(results.pandas().xyxy[0])
        bboxes = {
            'bbox':[],
            'bbox_handl':[],
            'bbox_handr':[],
            'pelvis':[],
            'pelvis_l':[],
            'pelvis_r':[]
        }

        for i, res in enumerate(arrays):
            classid = res[5]
            box = res[:5]
            if self.mode == 'bboxcrop':
                box[0]+=bbox[0]
                box[2]+=bbox[0]
                box[1]+=bbox[1]
                box[3]+=bbox[1]
            if False:
                vis = images.copy()
                cpimg = crop.copy()
                from easymocap.mytools.vis_base import plot_bbox
                plot_bbox(vis,box,0)
                plot_bbox(cpimg,res[:5],0)
                cv2.imshow('vis',vis)
                # cv2.waitKey(0)
                cv2.imshow('crop',cpimg)
                cv2.waitKey(0)
                breakpoint()
            if box[4] < self.min_detect_thres:
                continue
            if classid==0:
                bboxes['bbox'].append(box)
            elif classid==1:
                bboxes['bbox_handl'].append(box)
                bboxes['pelvis_l'].append([(box[0]+box[2])/2,(box[1]+box[3])/2,box[-1]])
            elif classid==2:
                bboxes['bbox_handr'].append(box)
                bboxes['pelvis_r'].append([(box[0]+box[2])/2,(box[1]+box[3])/2,box[-1]])
        if(len(bboxes['bbox_handl'])==0):
            # bboxes['bbox_handl'].append(np.zeros((0, 5)))
            # bboxes['pelvis_l'].append(np.zeros((0, 3)))
            bboxes['bbox_handl'].append(np.zeros((5)))
            bboxes['pelvis_l'].append(np.zeros((3)))
            
        if(len(bboxes['bbox_handr'])==0):
            # bboxes['bbox_handr'].append(np.zeros((0, 5)))
            # bboxes['pelvis_r'].append(np.zeros((0, 3)))
            bboxes['bbox_handr'].append(np.zeros((5)))
            bboxes['pelvis_r'].append(np.zeros((3)))
        if(len(bboxes['bbox'])==0):
            bboxes['bbox'].append(np.zeros((5)))
        bboxes['bbox'] = np.array(bboxes['bbox'])
        if isinstance(imgname,str):
            with open(cachename, 'wb') as f:
                pickle.dump(bboxes, f)
        return bboxes
    def __call__(self, images, imgname, bbox=[]):
        return self.det_step(images, imgname, bbox)


class Yolo_model_hand_mvmp(Yolo_model):
    @torch.no_grad()
    def __call__(self, bbox, images, imgnames):
        ret = {
            'pelvis_l':[],
            'pelvis_r':[],
            # 'pelvis':[],
            'bbox_handl':[],
            'bbox_handr':[],
        }
        for nv in range(len(images)):
            img = images[nv]
            imgname = imgnames[nv]
            if self.mode == 'bboxcrop':
                bboxes = {
                    'bbox':[],
                    'bbox_handl':[],
                    'bbox_handr':[],
                    'pelvis_l':[],
                    'pelvis_r':[]
                }
                for pid in range(len(bbox[nv])):
                    bboxes_ = self.det_step(img, imgname, bbox[nv][pid])
                    for key in bboxes.keys():
                        bboxes[key].append(bboxes_[key])
            else:
                bboxes = self.det_step(img, imgname)
            for k in ret.keys():
                ret[k].append(np.array(bboxes[k]))

        return ret