'''
  @ Date: 2021-11-27 16:50:33
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-19 21:37:49
  @ FilePath: /EasyMocapPublic/easymocap/visualize/ffmpeg_wrapper.py
'''
import shutil
import os
from os.path import join
from glob import glob
from tqdm import tqdm
class VideoMaker:
    def __init__(self, restart=True, fps_in=50, fps_out=50, remove_images=False,
        reorder=False,
        ext='.jpg', debug=False) -> None:
        self.restart = ' -y' if restart else ''
        self.fps_in = ' -r {}'.format(fps_in)
        self.remove_images = remove_images
        cmd = ' -pix_fmt yuv420p -vcodec libx264'
        cmd += ' -r {}'.format(fps_out)
        if ext == '.png':
            cmd += ' -profile:v main'
        self.cmd = cmd
        self.ext = ext
        self.shell = 'ffmpeg{restart}{fps_in} -i {path}/%06d{ext} -vf scale="2*ceil(iw/2):2*ceil(ih/2)"{cmd} {path}.mp4'
        if not debug:
            self.shell += ' -loglevel quiet'
        self.reorder = reorder
    
    def make_video(self, path):
        imgnames = sorted(glob(join(path, '*'+self.ext)))
        if len(imgnames) == 0:
            print('[ffmpeg] No images in folder {}'.format(path))
            return 0
        firstname = imgnames[0]
        index = os.path.basename(firstname).replace(self.ext, '')
        if index.isdigit():
            index = int(index)
            if index != 0:
                self.reorder = True
        if self.reorder:
            tmpdir = '/tmp/ffmpeg-tmp'
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
            os.makedirs(tmpdir, exist_ok=True)
            for nf, imgname in tqdm(enumerate(imgnames), desc='copy to /tmp'):
                tmpname = join(tmpdir, '{:06d}{}'.format(nf, self.ext))
                shutil.copyfile(imgname, tmpname)
            path_ori = path
            path = tmpdir
        cmd = self.shell.format(
            restart=self.restart,
            fps_in=self.fps_in,
            cmd=self.cmd,
            path=path,
            ext=self.ext
        )
        print(cmd)
        os.system(cmd)
        if self.reorder:
            shutil.copy(path+'.mp4', path_ori+'.mp4')
        if self.remove_images:
            shutil.rmtree(path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reorder', action='store_true')
    args = parser.parse_args()
    video_maker = VideoMaker(
        restart=True, fps_in=args.fps, fps_out=args.fps, remove_images=args.remove, ext=args.ext,
        reorder=args.reorder,
        debug=args.debug)
    video_maker.make_video(args.path)