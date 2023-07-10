import os
from os.path import join
import numpy as np
import cv2
from easymocap.mytools.debug_utils import log, myerror, mywarn

class ImageDataBase:
    def __init__(self, root, subs, ranges, read_image) -> None:
        assert root != 'TO_BE_FILLED', 'You must set the root of dataset'
        assert os.path.exists(root), f'root {root} not exists'
        self.root = root
        self.subs = subs
        self.ranges = ranges
        self.flag_read_image = read_image
        self.infos = {}
        self.meta = {}
    
    def check_frames_length(self):
        if len(self.ranges) == 0:
            self.ranges = [0, self.length, 1]
        if self.ranges[1] > self.length:
            self.ranges[1] = self.length
        self.frames = list(range(*self.ranges))
        self.length = len(self.frames)

    def try_to_extract_images(self, root, value):
        if not os.path.exists(os.path.join(root, value['root'])) and os.path.exists(os.path.join(root, 'videos')):
            print('[{}] Cannot find the images but find the videos, try to extract it'.format(self.__class__.__name__))
            for videoname in sorted(os.listdir(os.path.join(root, 'videos'))):
                videoext = '.' + videoname.split('.')[-1]
                outdir = join(root, value['root'], videoname.replace(videoext, ''))
                os.makedirs(outdir, exist_ok=True)
                cmd = 'ffmpeg -i {videoname} -q:v 1 -start_number 0 {outdir}/%06d.jpg'.format(
                    videoname=join(root, 'videos', videoname),
                    outdir=outdir
                )
                os.system(cmd)

    def __str__(self) -> str:
        return f''' [Dataset] {self.__class__.__name__}
    root  : {self.root}
    subs  : {self.subs}
    ranges: {self.ranges}
'''

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return self.length
    
    def read_image(self, imgname, cameras=None):
        assert os.path.exists(imgname), "image {} not exists".format(imgname)
        sub = os.path.basename(os.path.dirname(imgname))
        img = cv2.imread(imgname)
        if cameras is None:
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

def read_mv_images(root, root_images, ext, subs):
    assert os.path.exists(os.path.join(root, root_images)), f'root {root}/{root_images} not exists'
    if len(subs) == 0:
        subs = sorted(os.listdir(os.path.join(root, root_images)))
        if subs[0].isdigit():
            subs = sorted(subs, key=lambda x: int(x))
    imagelists = []
    log(f'Found {len(subs)} subjects in {root}/{root_images}')
    for sub in subs:
        images = sorted(os.listdir(os.path.join(root, root_images, sub)))
        images = [os.path.join(root, root_images, sub, image) for image in images if image.endswith(ext)]
        log(f'  -> Found {len(images)} {root_images} in {sub}.')
        imagelists.append(images)
    min_length = min([len(image) for image in imagelists])
    log(f'  -> Min length: {min_length}')
    imagenames = [[image[i] for image in imagelists] for i in range(min_length)]
    return imagenames, {'subs': subs}

def FloatArray(x):
    return np.array(x, dtype=np.float32)

def find_best_people(annots):
    if len(annots) == 0:
        return {}
    # TODO: find the best
    annot = annots[0]
    bbox = FloatArray(annot['bbox'])
    if 'keypoints' not in annot.keys():
        return {}
    keypoints = FloatArray(annot['keypoints'])
    return {'bbox': bbox, 'keypoints': keypoints}

def find_all_people(annots):
    if len(annots) == 0:
        return {}
    bbox = FloatArray([annot['bbox'] for annot in annots])
    keypoints = FloatArray([annot['keypoints'] for annot in annots])
    return {'bbox': bbox, 'keypoints': keypoints}