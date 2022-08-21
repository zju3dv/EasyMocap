'''
  @ Date: 2021-06-09 09:57:23
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-14 21:37:34
  @ FilePath: /EasyMocapPublic/apps/annotation/annot_clip.py
'''
# 功能：
# 1. 快速预览图像
# 2. 设置起点
# 3. 设置终点
# 不兼容的接口：没有标注文件
from easymocap.mytools.debug_utils import myerror, mywarn, run_cmd
from easymocap.mytools.vis_base import plot_line
from easymocap.annotator.basic_annotator import AnnotBase, parse_parser
from easymocap.annotator import ImageFolder
from easymocap.annotator import plot_text
from easymocap.annotator.basic_visualize import capture_screen, resize_to_screen
from easymocap.mytools import read_json, save_json
from easymocap.annotator.basic_keyboard import get_any_move
from os.path import join
import os
import numpy as np
import cv2

class Clips:
    def __init__(self, path) -> None:
        self.temp = join(path, 'clips.json')
        if os.path.exists(self.temp):
            self.annots = read_json(self.temp)
        else:
            self.annots = {}
        self.start_ = None
        self.end_ = None
        self.clips = []
        self.sub_ = None
    
    @property
    def sub(self):
        return self.sub_
    
    @sub.setter
    def sub(self, value):
        self.sub_ = value
        if value in self.annots.keys():
            self.clips = self.annots[value]
        else:
            self.annots[value] = []
            self.clips = self.annots[value]
        self.print(0)

    def start(self, annotator, **kwargs):
        self.start_ = annotator.frame
        print('>>> Start clip from frame {:6d}'.format(annotator.frame))

    def end(self, annotator, **kwargs):
        self.end_ = annotator.frame
        print('>>> End clip from frame {:6d}'.format(annotator.frame))
    
    def add(self, annotator, **kwargs):
        if self.start_ is None:
            print('[clip] Please check the start!')
            return 0
        if self.end_ is None:
            print('[clip] Please check the end!')
            return 0
        print('[{}, {})'.format(self.start_, self.end_))
        self.clips.append([self.start_, self.end_])
        self.start_ = None
        self.end_ = None
    
    def delete(self, annotator, **kwargs):
        frame = annotator.frame
        ind = -1
        for i, (start, end) in enumerate(self.clips):
            if frame > start and frame < end:
                ind = i
                break
        else:
            print('[clip] current not in any clip')
            return 0
        self.clips.pop(ind)
    
    def print(self, annotator, **kwargs):
        print('{}: '.format(self.sub))
        for (start, end) in self.clips:
            print(' - [{}, {})'.format(start, end))

    def save(self):
        save_json(self.temp, self.annots)
    
    def vis_clips(self, img, frame, nFrames, **kwargs):
        COL_CLIP = (0, 0, 255)
        COL_NEW = (0, 0, 255)
        width = img.shape[1]
        pos = lambda x: int(width*(x+1)/nFrames)
        lw = 12
        # 可视化标注的clips
        for (start, end) in self.clips:
            plot_line(img, (pos(start), lw/2), (pos(end), lw/2), lw, COL_CLIP)
        # 可视化当前的标注
        if self.start_ is not None:
            top = pos(self.start_)
            pts = np.array([[top, lw], [top-lw, lw*4], [top, lw*4]])
            cv2.fillPoly(img, [pts], COL_NEW)
        if self.end_ is not None:
            top = pos(self.end_)
            pts = np.array([[top, lw], [top, lw*4], [top+lw, lw*4]])
            cv2.fillPoly(img, [pts], COL_NEW)
        return img

def annot_example(path, sub, skip=False):
    # define datasets
    # define visualize
    if not os.path.exists(join(path, 'images', sub)):
        mywarn('[annot] No such sub: {}'.format(sub))
        return 0
    clip = Clips(path)
    vis_funcs = [resize_to_screen, plot_text, clip.vis_clips, capture_screen]
    clip.sub = sub
    if skip and len(clip.clips) > 0:
        return 0
    key_funcs = {
        'j': clip.start,
        'k': clip.end,
        'l': clip.add,
        'x': clip.delete,
        'v': clip.print,
        'w': get_any_move(-10),
        's': get_any_move(10),
        'f': get_any_move(100),
        'g': get_any_move(-100)
    }

    dataset = ImageFolder(path, sub=sub, no_annot=True)
    print('[Info] Totally {} frames'.format(len(dataset)))
    # construct annotations
    annotator = AnnotBase(
        dataset=dataset, 
        key_funcs=key_funcs,
        vis_funcs=vis_funcs)
    while annotator.isOpen:
        annotator.run()
    clip.save()

def copy_clips(path, out):
    from tqdm import tqdm
    import shutil
    from easymocap.mytools.debug_utils import log, mywarn, mkdir
    temp = join(path, 'clips.json')
    assert os.path.exists(temp), temp
    annots = read_json(temp)
    for key, clips in tqdm(annots.items()):
        for start, end in clips:
            outname = '{}+{:06d}+{:06d}'.format(key, start, end)
            outdir = join(out, 'images', outname)
            if os.path.exists(outdir) and len(os.listdir(outdir)) == end - start:
                mywarn('[copy] Skip {}'.format(outname))
                continue
            # check the input image
            srcname0 = join(path, 'images', key, '{:06d}.jpg'.format(start))
            srcname1 = join(path, 'images', key, '{:06d}.jpg'.format(end))
            if not os.path.exists(srcname0) or not os.path.exists(srcname1):
                myerror('[copy] No such file: {}, {}'.format(srcname0, srcname1))
            log('[copy] {}'.format(outname))
            mkdir(outdir)
            # copy the images
            for nnf, nf in enumerate(tqdm(range(start, end), desc='copy {}'.format(outname))):
                srcname = join(path, 'images', key, '{:06d}.jpg'.format(nf))
                dstname = join(outdir, '{:06d}.jpg'.format(nnf))
                shutil.copyfile(srcname, dstname)

def copy_mv_clips(path, out):
    temp = join(path, 'clips.json')
    assert os.path.exists(temp), temp
    annots = read_json(temp)
    clips = list(annots.values())[0]
    for start, end in clips:
        if out is None:
            outdir = path + '+{:06d}+{:06d}'.format(start, end)
        else:
            outdir = out + '+{:06d}+{:06d}'.format(start, end)
        print(outdir)
        cmd = f'python3 scripts/preprocess/copy_dataset.py {path} {outdir} --start {start} --end {end}'
        if len(args.sub) > 0:
            cmd += ' --subs {}'.format(' '.join(args.sub))
        if args.strip is not None:
            cmd += ' --strip {}'.format(args.strip)
        run_cmd(cmd)

if __name__ == "__main__":
    from easymocap.annotator import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--strip', type=str, default=None)
    parser.add_argument('--copy', action='store_true')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--mv', action='store_true')
    parser.add_argument('--sub_ignore', type=str, nargs='+', default=[])
    args = parse_parser(parser)

    args.sub = [i for i in args.sub if i not in args.sub_ignore]

    if args.copy:
        print(args.path, args.out)
        if args.mv:
            copy_mv_clips(args.path, args.out)
        else:
            if args.out is None:
                myerror('[copy] No output path')
                exit(0)
            copy_clips(args.path, args.out)
    else:
        if args.mv:
            annot_example(args.path, sub=args.sub[0], skip=args.skip)
        else:
            for sub in args.sub:
                annot_example(args.path, sub=sub, skip=args.skip)