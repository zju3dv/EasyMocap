import shutil
import cv2
import os
from tqdm import tqdm
from .basic_keyboard import print_help, register_keys
from .basic_visualize import plot_text, resize_to_screen, merge
from .basic_callback import point_callback, CV_KEY, get_key
from .bbox_callback import callback_select_image
from .file_utils import load_annot_to_tmp, read_json, save_annot
import copy
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
            ret = process(**param)
            if ret: # 操作成功，结束
                param['click'] = None
                param['start'] = None
                param['end'] = None

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

class BaseWindow:
    # 2021.10.11: 考虑新的层次的抽象
    # 这个实现的基类只包含了打开一个窗口->进行标注->关闭窗口 功能
    def __init__(self, name, param, register_keys, vis_funcs, callbacks) -> None:
        self.name = name
        register_keys['h'] = print_help
        register_keys['q'] = self.quit
        register_keys['x'] = self.clear
        register_keys['Q'] = self.quit_without_save
        self.register_keys = register_keys
        self.vis_funcs  = vis_funcs
        self.isOpen = True
        param['click'] = None
        param['start'] = None
        param['end'] = None
        self.param0 = copy.deepcopy(param)
        self.param = param
        cv2.namedWindow(self.name)
        callback = ComposedCallback(processes=callbacks)
        cv2.setMouseCallback(self.name, callback.call, self.param)

    def quit_without_save(self, annotator, param):
        self.quit(annotator, param, save=False)
    
    def clear(self, annotator, param):
        select = param['select']
        for key in select.keys():
            select[key] = -1
        for key in ['click', 'start', 'end']:
            self.param[key] = None

    def quit(self, annotator, param, save=True):
        for key in ['click', 'start', 'end']:
            if self.param[key] is not None:
                self.param[key] = None
                break
        else:
            if not save:
                for key in self.param.keys():
                    self.param[key] = self.param0[key]
            else:
                self.save_and_quit()
            self.isOpen = False
            cv2.destroyWindow(self.name)

    def save_and_quit(self, key=None):
        pass

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
        cv2.imshow(self.name, img)

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
            'body': body,
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
                shutil.copyfile(annname, annname_)
        
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
            assert os.path.exists(imgname), imgname
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
        return img

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
            imgs = merge(imgs, square=True)
            imgs = resize_to_screen(imgs, scale=CV_KEY.WINDOW_HEIGHT/imgs.shape[0])
            cv2.imshow(self.name, imgs)


class AnnotMVMerge(BaseWindow):
    # 这个类的设计理念是
    # 只负责整体的合并与可视化，不要考虑具体的操作
    restore_key = {
        'body25': ('bbox', 'keypoints'),
        'handl': ('bbox_handl2d', 'handl2d'),
        'handr': ('bbox_handr2d', 'handr2d'),
        'face': ('bbox_face2d', 'face2d'),
    }
    def __init__(self, datasets, register_keys, vis_funcs, vis_funcs_view, callbacks, body) -> None:
        self.subs = list(datasets.keys())
        self.isOpen = True
        for key in 'wasd':
            register_keys[key] = self.get_move(key)
        register_keys['q'] = self.quit
        register_keys['h'] = print_help
        register_keys['x'] = self.clear
        self.register_keys = register_keys
        self.vis_funcs = vis_funcs
        self.vis_funcs_view = vis_funcs_view
        self.callbacks = callbacks
        self.datasets = datasets
        self.name = 'main'
        frames = {sub:0 for sub in self.subs}
        self.body = body
        bbox_name, kpts_name = self.restore_key[body]

        self.params_view = {sub:{
            'body': body, 'bbox_name': bbox_name, 'kpts_name': kpts_name,
            'select': {bbox_name:-1}} for sub in self.subs}
        imgs, annots = self.load_images_annots(self.datasets, frames)
        img0, ranges = merge(imgs, ret_range=True)
        scale = 10000./img0.shape[0]
        self.nFrames = len(self.datasets[self.subs[0]])
        self.start = 0
        self.end = self.nFrames
        self.step = 50
        self.visited_frames = {sub: set([self.start]) for sub in self.subs}

        self.param = {
            'scale': scale, 'ranges': ranges,
            'click': None, 'start': None, 'end': None,
            'frames': frames,
            'body': body, 'bbox_name': bbox_name, 'kpts_name': kpts_name,
            'select': {'camera': -1, bbox_name:-1, 'corner': -1}}
        self.param['imgs'] = imgs
        self.param['annots'] = annots

        self.no_window = False
        cv2.namedWindow(self.name)
        callback = ComposedCallback(processes=callbacks)
        cv2.setMouseCallback(self.name, callback.call, self.param)
    
    def save_and_quit(self, key=None):
        self.isOpen = False
        self.update_param()
        cv2.destroyWindow(self.name)
        # get the input
        if key is None:
            key = get_valid_yn()
        if key == 'n':
            return 0
        for nv, sub in enumerate(self.subs):
            dataset = self.datasets[sub]
            for frame in tqdm(self.visited_frames[sub], desc='writing'):
                dataset.isTmp = True
                _, annname = dataset[frame]
                dataset.isTmp = False
                _, annname_ = dataset[frame]
                if annname is not None:
                    print(annname, annname_)
                    shutil.copyfile(annname, annname_)

    @property
    def frame(self):
        return list(self.param['frames'].values())[0]

    def update_param(self):
        # 先保存
        for nv, sub in enumerate(self.subs):
            self.visited_frames[sub].add(self.param['frames'][sub])
            save_annot(self.params_view[sub]['annname'], self.param['annots'][nv])
        imgs, annots = self.load_images_annots(self.datasets, self.param['frames'])
        self.param['imgs'] = imgs
        self.param['annots'] = annots

    def move(self, delta):
        for sub in self.subs:
            self.param['frames'][sub] += delta
        self.update_param()

    @staticmethod
    def get_move(wasd):
        get_frame = {
            'a': lambda x, f: f - 1,
            'd': lambda x, f: f + 1,
            'w': lambda x, f: f - x.step,
            's': lambda x, f: f + x.step
        }[wasd]
        text = {
            'a': 'Move to last frame',
            'd': 'Move to next frame',
            'w': 'Move to last step frame',
            's': 'Move to next step frame'
        }
        clip_frame = lambda x, f: max(x.start, min(x.nFrames-1, min(x.end-1, f)))
        def move(annotator, **kwargs):
            newframe = get_frame(annotator, annotator.frame)
            newframe = clip_frame(annotator, newframe)
            annotator.move(newframe - annotator.frame)
        move.__doc__ = text[wasd]
        return move

    def load_images_annots(self, datasets, frames):
        imgs, annots = [], []
        for sub, dataset in datasets.items():
            imgname, annname = dataset[frames[sub]]
            img = cv2.imread(imgname)
            annot = load_annot_to_tmp(annname)
            imgs.append(img)
            annots.append(annot)
            self.params_view[sub]['imgname'] = imgname
            self.params_view[sub]['annname'] = annname
        return imgs, annots
    
    def run(self, key=None, noshow=False):
        # 更新选中
        if key is None:
            key = chr(get_key())
        actv = self.param['select']['camera']
        for sub in self.subs:
            for sel in self.params_view[sub]['select']:
                self.params_view[sub]['select'][sel] = -1
        if actv != -1:
            self.params_view[self.subs[actv]]['select'].update(self.param['select'])
        if key in self.register_keys.keys():
            func = self.register_keys[key]
            if isinstance(func, list):
                [f(self, param=self.param) for f in func]
            else:
                func(self, param=self.param)
        if not self.isOpen:
            return 0
        if noshow:
            return 0
        imgs = self.param['imgs']
        imgs = [img.copy() for img in imgs]
        for nv, img in enumerate(imgs):
            for func in self.vis_funcs_view:
                img = func(img, self.param['annots'][nv], name=self.subs[nv], **self.params_view[self.subs[nv]])
            if nv == actv:
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), img.shape[1]//100)
            imgs[nv] = img
        img = merge(imgs)
        for func in self.vis_funcs:
            img = func(img, **self.param)
        cv2.imshow(self.name, img)

class AnnotMVMain:
    def __init__(self, datasets, key_funcs={}, key_funcs_view={}, callbacks=[], vis_funcs=[], vis_funcs_all=[], 
        name='main', step=100, body='body25', start=0, end=100000,
        scale=0.5) -> None:
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
        for key, val in self.register_keys.items():
            print(key, val.__doc__)
        self.vis_funcs_all = vis_funcs_all
        self.name = name
        imgs = self.load_images()
        imgs, ranges = merge(imgs, ret_range=True, square=True)
        self.scale = scale
        self.param = {
            'scale': scale, 'ranges': ranges,
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
            elif ord(key) != 255:
                for sub in self.subs:
                    self.annotdict[sub].run(key)
        elif key == 'x':
            self.param['select']['camera'] = -1
            self.param['click'] = None
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
            imgs = merge(imgs, square=True)
            for func in [resize_to_screen]:
                # scale here
                imgs = func(imgs, scale=self.scale)
            cv2.imshow(self.name, imgs)

def load_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the sub folder lists when in video mode')
    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--annot', type=str, default='annots')
    parser.add_argument('--body', type=str, default='body25')
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
    
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
        newsubs = sorted(os.listdir(join(args.path, args.image)))
        clips = []
        for newsub in newsubs:
            if newsub in subs:
                continue
            if newsub.split('+')[0] in subs:
                clips.append(newsub)
        for sub in subs:
            if os.path.exists(join(args.path, args.image, sub)):
                clips.append(sub)
        args.sub = sorted(clips)
    elif args.from_file is not None and args.from_file.endswith('.json'):
        data = read_json(args.from_file)
        args.sub = sorted([v['vid'] for v in data])
    elif len(args.sub) == 0:
        if not os.path.exists(join(args.path, args.image)):
            print('{} not exists, Please run extract_image first'.format(join(args.path, args.image)))
            raise FileNotFoundError
        subs = sorted(os.listdir(join(args.path, args.image)))
        subs = [s for s in subs if os.path.isdir(join(args.path, args.image, s)) and not s.startswith('._')]
        if len(subs) > 0 and subs[0].isdigit():
            subs = sorted(subs, key=lambda x:int(x))
        args.sub = subs
    helps = """
    Demo code for annotation:
    - Input : {}
    -      => "{}"
    -      => {}
""".format(args.path, '", "'.join(args.sub), args.annot)
    print(helps)
    return args