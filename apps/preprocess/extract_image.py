'''
  @ Date: 2021-08-19 22:06:13
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-10-23 16:02:43
  @ FilePath: /EasyMocap/apps/preprocess/extract_image.py
'''
# extract image from videos
import os
from os.path import join
from glob import glob

extensions = ['.mp4', '.webm', '.flv', '.MP4', '.MOV', '.mov', '.avi']

def run(cmd):
    print(cmd)
    os.system(cmd)

def extract_images(path, ffmpeg, image):
    videos = sorted(sum([
        glob(join(path, 'videos', '*'+ext)) for ext in extensions
        ], [])
    )
    for videoname in videos:
        sub = '.'.join(os.path.basename(videoname).split('.')[:-1])
        sub = sub.replace(args.strip, '')
        outpath = join(path, image, sub)
        os.makedirs(outpath, exist_ok=True)
        other_cmd = ''
        if args.num != -1:
            other_cmd += '-vframes {}'.format(args.num)
        if args.transpose != -1:
            other_cmd += '-vf transpose={}'.format(args.transpose)
        cmd = '{} -i {} {} -q:v 1 -start_number 0 {}/%06d.jpg'.format(
            ffmpeg, videoname, other_cmd, outpath)
        run(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--strip', type=str, default='')
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--transpose', type=int, default=-1)
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    extract_images(args.path, args.ffmpeg, args.image)