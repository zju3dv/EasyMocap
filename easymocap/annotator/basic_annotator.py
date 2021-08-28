import shutil
import cv2
from tqdm import tqdm
from .basic_keyboard import register_keys
from .basic_visualize import plot_text, resize_to_screen, merge
from .basic_callback import point_callback, CV_KEY, get_key
from .file_utils import load_annot_to_tmp, read_json, save_annot

class ComposedCallback:
    def __init__(self, callbacks=[point_callback], processes=[]) -> None:
        self.callbacks = callbacks
        self.processes = processes
    
    def call(self, event, x, y, flags, param):
        scale = param['scale']
        x, y = int(round(x/scale)), int(round(y/scale))
        for callback in self.callbacks:
            callback(event, x, y, flags, param)
        for key in ['click', 'start', 'end']:
            if param[key] is not None:
                break
        else:
            return 0
        for process in self.processes:
            process(**param)

def get_valid_yn():
    while True:
        key = input('Saving this annotations? [y/n]')
        if key in ['y', 'n']:
            break
        print('Please specify [y/n]')
    return key

restore_key = {
    'body25': ('bbox', 'keypoints'),
    'handl': ('bbox_handl2d', 'handl2d'),
    'handr': ('bbox_handr2d', 'handr2d'),
}
class AnnotBase:
    def __init__(self, dataset, key_funcs={}, callbacks=[], vis_funcs=[],
        name = 'main', body='body25',
        start=0, end=100000, step=10, no_window=False) -> None:
        self.name = name
        self.dataset = dataset
        self.nFrames = len(dataset)
        self.step = step
        self.register_keys = register_keys.copy()
        self.register_keys.update(key_funcs)
        self.no_img = False
        if resize_to_screen not in vis_funcs:
            vis_funcs += [resize_to_screen]
        self.vis_funcs = vis_funcs
        self.start = start
        self.end = end

        self.isOpen = True
        self._frame = self.start
        self.visited_frames = set([self._frame])
        bbox_name, kpts_name = restore_key[body]
        self.param = {
            'frame': 0, 'nFrames': self.nFrames,
            'kpts_name': kpts_name, 'bbox_name': bbox_name,
            'select': {bbox_name: -1, 'corner': -1}, 
            'click': None, 
            'name': name,
            'capture_screen':False}
        self.set_frame(self.start)
        self.no_window = no_window
        if not no_window:
            cv2.namedWindow(self.name)
            callback = ComposedCallback(processes=callbacks)
            cv2.setMouseCallback(self.name, callback.call, self.param)
    
    @property
    def working(self):
        param = self.param
        flag = False
        if param['click'] is not None or param['start'] is not None:
            flag = True
        for key in self.param['select']:
            if self.param['select'][key] != -1:
                flag = True
        return flag

    @staticmethod
    def clear_working(param):
        param['click'] = None
        param['start'] = None
        param['end'] = None
        for key in param['select']:
            param['select'][key] = -1

    def save_and_quit(self, key=None):
        self.frame = self.frame
        self.isOpen = False
        cv2.destroyWindow(self.name)
        # get the input
        if key is None:
            key = get_valid_yn()
        if key == 'n':
            return 0
        for frame in tqdm(self.visited_frames, desc='writing'):
            self.dataset.isTmp = True
            _, annname = self.dataset[frame]
            self.dataset.isTmp = False
            _, annname_ = self.dataset[frame]
            if annname is not None:
                shutil.copy(annname, annname_)
        
    @property
    def frame(self):
        return self._frame

    def previous(self):
        if self.frame == 0:
            print('Reach to the first frame')
            return None
        imgname, annname = self.dataset[self.frame-1]
        annots = load_annot_to_tmp(annname)
        return annots

    @staticmethod
    def set_param(param, imgname, annname, nf, no_img=False):
        annots = load_annot_to_tmp(annname)
        # 清空键盘
        for key in ['click', 'start', 'end']:
            param[key] = None
        # 清空选中
        for key in param['select']:
            param['select'][key] = -1
        param['imgname'] = imgname
        param['annname'] = annname
        param['frame'] = nf
        param['annots'] = annots
        if not no_img:
            img0 = cv2.imread(imgname)
            param['img0'] = img0
            # param['pid'] = len(annot['annots'])
            param['scale'] = min(CV_KEY.WINDOW_HEIGHT/img0.shape[0], CV_KEY.WINDOW_WIDTH/img0.shape[1])
            # param['scale'] = 1

    def set_frame(self, nf):
        param = self.param
        if 'annots' in param.keys():
            save_annot(param['annname'], param['annots'])
        self.clear_working(param)
        imgname, annname = self.dataset[nf]
        self.set_param(param, imgname, annname, nf, no_img=self.no_img)
        
    @frame.setter
    def frame(self, value):
        self.visited_frames.add(value)
        self._frame = value
        # save current frames
        save_annot(self.param['annname'], self.param['annots'])
        self.set_frame(value)
        
    def run(self, key=None, noshow=False):
        if key is None:
            key = chr(get_key())
        if key in self.register_keys.keys():
            self.register_keys[key](self, param=self.param)
        if not self.isOpen:
            return 0
        if noshow:
            return 0
        img = self.param['img0'].copy()
        for func in self.vis_funcs:
            img = func(img, **self.param)
        if not self.no_window:
            cv2.imshow(self.name, img)

class AnnotMV:
    def __init__(self, datasets, key_funcs={}, key_funcs_view={}, callbacks=[], vis_funcs=[], vis_funcs_all=[], 
        name='main', step=100, body='body25', start=0, end=100000) -> None:
        self.subs = list(datasets.keys())
        self.annotdict = {}
        self.nFrames = end
        for sub, dataset in datasets.items():
            annot = AnnotBase(dataset, key_funcs={}, callbacks=callbacks, vis_funcs=vis_funcs,
                name=sub, step=step, body=body, start=start, end=end)
            self.annotdict[sub] = annot
            self.nFrames = min(self.nFrames, annot.nFrames)
        self.isOpen = True
        # self.register_keys_view = {key:register_keys[key] for key in 'q'}
        self.register_keys_view = {}
        if 'w' not in key_funcs:
            for key in 'wasd':
                self.register_keys_view[key] = register_keys[key]
        self.register_keys_view.update(key_funcs_view)
        self.register_keys = {
            'Q': register_keys['q'],
            'h': register_keys['H'],
            'A': register_keys['A']
        }
        self.register_keys.update(key_funcs)
        self.vis_funcs_all = vis_funcs_all
        self.name = name
        self.param = {}
    
    @property
    def frame(self):
        sub = list(self.annotdict.keys())[0]
        return self.annotdict[sub].frame

    @property
    def working(self):
        return False
    
    def save_and_quit(self):
        key = get_valid_yn()
        for sub, annot in self.annotdict.items():
            annot.save_and_quit(key)
        self.isOpen = False

    def run(self, key=None, noshow=False):
        if key is None:
            key = chr(get_key())
        for sub, annot in self.annotdict.items():
            if key in self.register_keys_view.keys():
                self.register_keys_view[key](annot, param=annot.param)
            else:
                annot.run(key='')
        if key in self.register_keys.keys():
            self.register_keys[key](self, param=self.param)
        if len(self.vis_funcs_all) > 0 or True:
            imgs = []
            for sub in self.subs:
                img = self.annotdict[sub].param['img0'].copy()
                for func in self.vis_funcs_all:
                    img = func(img, sub, param=self.annotdict[sub].param)
                imgs.append(img)
            for func in [merge, resize_to_screen]:
                imgs = func(imgs, scale=0.1)
            cv2.imshow(self.name, imgs)

import numpy as np
def callback_select_image(click, select, ranges, **kwargs):
    if click is None:
        return 0
    ranges = np.array(ranges)
    click = np.array(click).reshape(1, -1)
    res = (click[:, 0]>ranges[:, 0])&(click[:, 0]<ranges[:, 2])&(click[:, 1]>ranges[:, 1])&(click[:, 1]<ranges[:, 3])
    if res.any():
        select['camera'] = int(np.where(res)[0])

class AnnotMVMain:
    def __init__(self, datasets, key_funcs={}, key_funcs_view={}, callbacks=[], vis_funcs=[], vis_funcs_all=[], 
        name='main', step=100, body='body25', start=0, end=100000) -> None:
        self.subs = list(datasets.keys())
        self.annotdict = {}
        self.nFrames = end
        for sub, dataset in datasets.items():
            annot = AnnotBase(dataset, key_funcs={}, callbacks=callbacks, vis_funcs=vis_funcs,
                name=sub, step=step, body=body, start=start, end=end, no_window=True)
            self.annotdict[sub] = annot
            self.nFrames = min(self.nFrames, annot.nFrames)
        self.isOpen = True
        self.register_keys_view = {}
        self.register_keys = {
            'Q': register_keys['q'],
            'h': register_keys['H'],
            'A': register_keys['A']
        }
        self.register_keys.update(key_funcs)
        self.vis_funcs_all = vis_funcs_all
        self.name = name
        imgs = self.load_images()
        imgs, ranges = merge(imgs, ret_range=True)
        self.param = {
            'scale': 0.45, 'ranges': ranges,
            'click': None, 'start': None, 'end': None,
            'select': {'camera': -1}}
        callbacks = [callback_select_image]
        cv2.namedWindow(self.name)
        callback = ComposedCallback(processes=callbacks)
        cv2.setMouseCallback(self.name, callback.call, self.param)
    
    @property
    def frame(self):
        sub = list(self.annotdict.keys())[0]
        return self.annotdict[sub].frame

    @property
    def working(self):
        return False
    
    def save_and_quit(self, key=None):
        if key is None:
            key = get_valid_yn()
        for sub, annot in self.annotdict.items():
            annot.save_and_quit(key)
        self.isOpen = False

    def load_images(self):
        imgs = []
        for sub in self.subs:
            img = self.annotdict[sub].param['img0'].copy()
            imgs.append(img)
        return imgs

    def run(self, key=None, noshow=False):
        if key is None:
            key = chr(get_key())
        active_v = self.param['select']['camera']
        if active_v == -1:
            # run the key for all cameras
            if key in self.register_keys.keys():
                self.register_keys[key](self, param=self.param)
            else:
                for sub in self.subs:
                    self.annotdict[sub].run(key)
        else:
            # run the key for the selected cameras
            self.annotdict[self.subs[active_v]].run(key=key)
        if len(self.vis_funcs_all) > 0:
            imgs = []
            for nv, sub in enumerate(self.subs):
                img = self.annotdict[sub].param['img0'].copy()
                for func in self.vis_funcs_all:
                    # img = func(img, sub, param=self.annotdict[sub].param)
                    img = func(img, **self.annotdict[sub].param)
                if self.param['select']['camera'] == nv:
                    cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), img.shape[1]//100)
                # img = plot_text(img, self.annotdict[sub].param['annots'], self.annotdict[sub].param['imgname'])
                imgs.append(img)
            for func in [merge, resize_to_screen]:
                imgs = func(imgs, scale=0.45)
            cv2.imshow(self.name, imgs)

def load_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the sub folder lists when in video mode')
    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--annot', type=str, default='annots')
    parser.add_argument('--body', type=str, default='handl')
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    # new arguments
    parser.add_argument('--start', type=int, default=0, help='frame start')
    parser.add_argument('--end', type=int, default=100000, help='frame end')
    return parser

def parse_parser(parser):
    import os
    from os.path import join
    args = parser.parse_args()
    if args.from_file is not None and args.from_file.endswith('.txt'):
        assert os.path.exists(args.from_file), args.from_file
        with open(args.from_file) as f:
            datas = f.readlines()
            subs = [d for d in datas if not d.startswith('#')]
            subs = [d.rstrip().replace('https://www.youtube.com/watch?v=', '') for d in subs]
        newsubs = sorted(os.listdir(join(args.path, 'images')))
        clips = []
        for newsub in newsubs:
            if newsub in subs:
                continue
            if newsub.split('+')[0] in subs:
                clips.append(newsub)
        for sub in subs:
            if os.path.exists(join(args.path, 'images', sub)):
                clips.append(sub)
        args.sub = sorted(clips)
    elif args.from_file is not None and args.from_file.endswith('.json'):
        data = read_json(args.from_file)
        args.sub = sorted([v['vid'] for v in data])
    elif len(args.sub) == 0:
        args.sub = sorted(os.listdir(join(args.path, 'images')))
        if args.sub[0].isdigit():
            args.sub = sorted(args.sub, key=lambda x:int(x))
    helps = """
    Demo code for annotation:
    - Input : {}
    -      => {}
    -      => {}
""".format(args.path, ', '.join(args.sub), args.annot)
    print(helps)
    return args