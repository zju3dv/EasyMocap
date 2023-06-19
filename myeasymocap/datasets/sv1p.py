from .basedata import ImageDataBase, read_mv_images, find_best_people
from easymocap.mytools.debug_utils import log, myerror, mywarn
from easymocap.mytools.camera_utils import read_cameras
from easymocap.mytools.file_utils import read_json
import os
import numpy as np
import cv2

class SVDataset(ImageDataBase):
    '''
        这个数据只用来返回单段的视频数据，不用来返回多段的视频数据
    '''
    def __init__(self, root, subs, ranges, read_image=False, reader={}) -> None:
        super().__init__(root, subs, ranges, read_image)
        assert len(subs) == 1, 'SVDataset only support one subject'
        for key, value in reader.items():
            if key == 'images':
                self.try_to_extract_images(root, value)
                data, meta = read_mv_images(root, value['root'], value['ext'], subs)
                data = [d[0] for d in data]
                self.length = len(data)
            elif key == 'image_shape':
                imgname = self.infos['images'][0]
                shapes = []
                assert os.path.exists(imgname), "image {} not exists".format(imgname)
                img = cv2.imread(imgname)
                assert img is not None, "image {} read failed".format(imgname)
                height, width, _ = img.shape
                log('[{}] sub {} shape {}'.format(self.__class__.__name__, imgname, img.shape))
                shapes.append([height, width])
                data = shapes
            elif key == 'annots':
                data, meta = read_mv_images(root, value['root'], value['ext'], subs)
                data = [d[0] for d in data]
                if self.length > 0:
                    assert self.length == len(data), \
                        myerror('annots length {} not equal to images length {}.'.format(len(data), self.length))
                else:
                    self.length = len(data)
            elif key == 'cameras':
                myerror('暂时没有实现相机参数')
                raise NotImplementedError
            else:
                raise ValueError(f'Unknown reader: {key}')
            self.infos[key] = data
            self.meta.update(meta)
        # check cameras:
        if 'cameras' not in self.infos:
            mywarn('[{}] No camera info, use default camera'.format(self.__class__.__name__))
            imgname0 = self.infos['images'][0]
            img = self.read_image(imgname0)
            height, width = img.shape[:2]
            log('[{}] Read shape {} from image {}'.format(self.__class__.__name__, img.shape, imgname0))
            focal = 1.2*min(height, width) # as colmap
            log('[{}] Set a fix focal length {}'.format(self.__class__.__name__, focal))
            K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
            camera = {'K':K ,'R': np.eye(3), 'T': np.zeros((3, 1)), 'dist': np.zeros((1, 5))}
            for key, val in camera.items():
                camera[key] = val.astype(np.float32)
            self.infos['cameras'] = [camera]
        self.check_frames_length()
        self.find_best_people = find_best_people
    
    def __getitem__(self, index):
        frame = self.frames[index]
        ret = {}
        for key, value in self.infos.items():
            if len(value) == 1:
                ret[key] = value[0]
            elif index >= len(value):
                myerror(f'[{self.__class__.__name__}] {key}: index {frame} out of range {len(value)}')
            else:
                ret[key] = value[frame]
        ret_new = {}
        for key, val in ret.items():
            if key == 'annots':
                annots = read_json(val)['annots']
                # select the best people
                annots = self.find_best_people(annots)
                ret_new.update(annots)
            elif key == 'cameras':
                ret_new[key] = val
            elif key == 'images':
                ret_new['imgnames'] = val
                if self.flag_read_image:
                    img = self.read_image(val)
                    ret_new[key] = img
                else:
                    ret_new[key] = val
            elif key == 'image_shape':
                ret_new['image_shape'] = val
        ret_new['meta'] = {
            'subs': self.subs,
            'index': index,
            'frame': self.frames[index],
            'image_shape': ret_new['image_shape'],
            'imgnames': ret_new['imgnames'],
        }
        return ret_new

class SVHandL(SVDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.find_best_people = self._find_best_hand
    
    def _find_best_hand(self, annots):
        assert len(annots) == 1, 'SVHandL only support one person'
        annot = annots[0]
        ret = {
            'bbox': np.array(annot['bbox_handl2d'], dtype=np.float32),
            'keypoints': np.array(annot['handl2d'], dtype=np.float32),
        }
        return ret

if __name__ == '__main__':
    cfg = '''
module: myeasymocap.datasets.1v1p.MonoDataset
args:
  root: /nas/home/shuaiqing/EasyMocapDoc/demo/1v1p
  subs: ['0+000553+000965']
  ranges: [0, 99999, 1]
  read_image: True
  reader:
    images:
      root: images
      ext: .jpg
    annots:
      root: annots
      ext: .json
'''
    import yaml
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)
    dataset = SVDataset(**cfg['args'])
    print(dataset)
    for i in range(len(dataset)):
        data = dataset[i]