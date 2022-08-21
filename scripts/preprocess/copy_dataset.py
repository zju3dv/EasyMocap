'''
  @ Date: 2021-06-14 15:39:26
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-02 21:50:40
  @ FilePath: /EasyMocapPublic/scripts/preprocess/copy_dataset.py
'''
import os
from os.path import join
import shutil
from tqdm import tqdm
from glob import glob
import cv2

from easymocap.mytools.debug_utils import myerror, mywarn

mkdir = lambda x:os.makedirs(x, exist_ok=True)

import json

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def copy_dataset(inp, out, start, end, step, keys, args):
    copy_keys = {
        'images': args.ext,
        'annots': '.json',
        'mask-schp': '.png',
    }
    copy_share_keys = {
        'output-keypoints3d/keypoints3d': '.json'
    }
    mkdir(out)
    if os.path.exists(join(inp, 'intri.yml')):
        shutil.copyfile(join(inp, 'intri.yml'), join(out, 'intri.yml'))
        shutil.copyfile(join(inp, 'extri.yml'), join(out, 'extri.yml'))
    if os.path.exists(join(inp, 'match_name.json')):
        names = read_json(join(inp, 'match_name.json'))
        names = names[start:end:step]
        save_json(join(out, 'match_name.json'), names)
    if os.path.exists(join(inp, 'sync_time.txt')):
        import numpy as np
        times = np.loadtxt(join(inp, 'sync_time.txt'))
        times = times.reshape(times.shape[0], -1)
        times = times[:, start:end:step]
        np.savetxt(join(out, 'sync_time.txt'), times, fmt='%10d')
    os.system('touch ' + join(out, '{}-{}-{}'.format(start, end, step)))
    for copy, ext in copy_share_keys.items():
        if not os.path.exists(join(inp, copy)):
            continue
        if len(args.frames) == 0:
            ranges = [i for i in range(start, end, step)]
        else:
            ranges = args.frames
        outdir = join(out, copy)
        if os.path.exists(outdir) and len(os.listdir(outdir)) == len(ranges):
            pass
        os.makedirs(outdir, exist_ok=True)
        for nnf, nf in enumerate(tqdm(ranges, desc='{}'.format(copy))):
            oldname = join(inp, copy, '{:06d}{}'.format(nf, ext))
            if not os.path.exists(oldname):
                mywarn('{} not exists'.format(oldname))
                continue
            newname = join(outdir, '{:06d}{}'.format(nnf, ext))
            shutil.copyfile(oldname, newname)

    for copy in keys:
        ext = copy_keys.get(copy, '.json')
        if not os.path.exists(join(inp, copy)):
            continue
        if len(args.subs) == 0:
            subs = sorted(os.listdir(join(inp, copy)))
            subs = [s for s in subs if os.path.isdir(join(inp, copy, s))]
        else:
            subs = args.subs
        for sub in subs:
            if not os.path.exists(join(inp, copy)):
                continue
            outdir = join(out, copy, sub.replace(args.strip, ''))
            os.makedirs(outdir, exist_ok=True)
            if args.end == -1:
                oldnames = sorted(glob(join(inp, copy, sub, '*{}'.format(ext))))
                end = len(oldnames)
                print('{} has {} frames'.format(sub, end))
            if args.sample == -1:
                if len(args.frames) == 0:
                    ranges = [i for i in range(start, end, step)]
                else:
                    ranges = args.frames
            else:
                ranges = [(i/args.sample)*(end-start-2*args.strip_frame)+start+args.strip_frame for i in range(args.sample)]
                ranges = [int(i+0.5) for i in ranges]
            if os.path.exists(outdir) and len(os.listdir(outdir)) == len(ranges):
                mywarn('[copy] Skip {}'.format(outdir))
                continue
            for nnf, nf in enumerate(tqdm(ranges, desc='{}:{}'.format(sub, copy))):
                oldname = join(inp, copy, sub, '{:06d}{}'.format(nf, ext))
                if not os.path.exists(oldname):
                    oldnames = sorted(glob(join(inp, copy, sub, '{:06d}_*{}'.format(nf, ext))))
                    if len(oldnames) == 0:
                        myerror('{} not exists'.format(oldname))
                        import ipdb;ipdb.set_trace()
                    else:
                        for oldname in oldnames:
                            newname = join(outdir, os.path.basename(oldname).replace('{:06d}'.format(nf), '{:06d}'.format(nnf)))
                            shutil.copyfile(oldname, newname)                            
                else:
                    newname = join(outdir, '{:06d}{}'.format(nnf, ext))
                    if copy == 'images' and args.scale != 1:
                        img = cv2.imread(oldname)
                        img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
                        cv2.imwrite(newname, img)
                    else:
                        shutil.copyfile(oldname, newname)
        # make videos
        if copy == 'images' and args.make_video:
            os.makedirs(join(out, 'videos'), exist_ok=True)
            for sub in subs:
                shell = '{} -y -i {}/images/{}/%06d{} -vcodec libx264 {}/videos/{}.mp4 -loglevel quiet'.format(
                    args.ffmpeg, out, sub, ext, out, sub
                )
                print(shell)
                os.system(shell)

def export(root, out, keys):
    mkdir(out)
    for key in keys:
        src = join(root, key)
        dst = join(out, key)
        if key == 'videos':
            if os.path.exists(src):
                shutil.copytree(src, dst)
            else:
                mkdir(dst)
                subs = sorted(os.listdir(join(root, 'images')))
                for sub in subs:
                    cmd = '{ffmpeg} -r {fps} -i {inp}/%06d.jpg -vcodec libx264 {out}'.format(
                        ffmpeg=args.ffmpeg, fps=50, inp=join(root, 'images', sub),
                        out=join(dst, sub+'.mp4')
                    )
                    os.system(cmd)
        if not os.path.exists(src):
            print(src)
            continue
        shutil.copytree(src, dst)
    for name in ['intri.yml', 'extri.yml']:
        if os.path.exists(join(root, name)):
            shutil.copyfile(join(root, name), join(out, name))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--strip', type=str, default='')
    parser.add_argument('--keys', type=str, nargs='+', default=['images', 'annots', 'chessboard'])
    parser.add_argument('--subs', type=str, nargs='+', default=[])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--strip_frame', type=int, default=0,
        help='remove the start frames and end frames')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--sample', type=int, default=-1,
        help='use this flag to sample a fixed number of frames')
    parser.add_argument('--frames', type=int, default=[], nargs='+')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--make_video', action='store_true')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if args.export:
        export(args.path, args.out, args.keys)
    else:
        copy_dataset(args.path, args.out, start=args.start, end=args.end, step=args.step, keys=args.keys, args=args)
