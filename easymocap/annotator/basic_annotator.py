import shutil
import cv2
from .basic_keyboard import register_keys
from .basic_visualize import resize_to_screen
from .basic_callback import point_callback, CV_KEY, get_key
from .file_utils import load_annot_to_tmp, save_annot

class ComposedCallback:
    def __init__(self, callbacks=[point_callback], processes=[]) -> None:
        self.callbacks = callbacks
        self.processes = processes
    
    def call(self, event, x, y, flags, param):
        scale = param['scale']
        x, y = int(x/scale), int(y/scale)
        for callback in self.callbacks:
            callback(event, x, y, flags, param)
        for key in ['click', 'start', 'end']:
            if param[key] is not None:
                break
        else:
            return 0
        for process in self.processes:
            process(**param)

class AnnotBase:
    def __init__(self, dataset, key_funcs={}, callbacks=[], vis_funcs=[],
        name = 'main',
        step=1) -> None:
        self.name = name
        self.dataset = dataset
        self.nFrames = len(dataset)
        self.step = step
        self.register_keys = register_keys.copy()
        self.register_keys.update(key_funcs)

        self.vis_funcs = vis_funcs + [resize_to_screen]
        self.isOpen = True
        self._frame = 0
        self.visited_frames = set([self._frame])
        self.param = {'select': {'bbox': -1, 'corner': -1}, 
            'start': None, 'end': None, 'click': None, 
            'capture_screen':False}
        self.set_frame(0)
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

    def clear_working(self):
        self.param['click'] = None
        self.param['start'] = None
        self.param['end'] = None
        for key in self.param['select']:
            self.param['select'][key] = -1

    def save_and_quit(self):
        self.frame = self.frame
        self.isOpen = False
        cv2.destroyWindow(self.name)
        # get the input
        while True:
            key = input('Saving this annotations? [y/n]')
            if key in ['y', 'n']:
                break
            print('Please specify [y/n]')
        if key == 'n':
            return 0
        if key == 'n':
            return 0
        for frame in self.visited_frames:
            self.dataset.isTmp = True
            _, annname = self.dataset[frame]
            self.dataset.isTmp = False
            _, annname_ = self.dataset[frame]
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

    def set_frame(self, nf):
        self.clear_working()
        imgname, annname = self.dataset[nf]
        img0 = cv2.imread(imgname)
        annots = load_annot_to_tmp(annname)
        # 清空键盘
        for key in ['click', 'start', 'end']:
            self.param[key] = None
        # 清空选中
        for key in self.param['select']:
            self.param['select'][key] = -1
        self.param['imgname'] = imgname
        self.param['annname'] = annname
        self.param['frame'] = nf
        self.param['annots'] = annots
        self.param['img0'] = img0
        # self.param['pid'] = len(annot['annots'])
        self.param['scale'] = min(CV_KEY.WINDOW_HEIGHT/img0.shape[0], CV_KEY.WINDOW_WIDTH/img0.shape[1])

    @frame.setter
    def frame(self, value):
        self.visited_frames.add(value)
        self._frame = value
        # save current frames
        save_annot(self.param['annname'], self.param['annots'])
        self.set_frame(value)
        
    def run(self, key=None):
        if key is None:
            key = chr(get_key())
        if key in self.register_keys.keys():
            self.register_keys[key](self, param=self.param)
        if not self.isOpen:
            return 0
        img = self.param['img0'].copy()
        for func in self.vis_funcs:
            img = func(img, **self.param)
        cv2.imshow(self.name, img)