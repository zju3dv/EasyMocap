import os
import shutil
from os.path import join
from glob import glob
from easymocap.mytools.debug_utils import log, mywarn, myerror, run_cmd

class MakeVideo:
    def __init__(self, fps, keep_image, output='tmp') -> None:
        self.output = output
        self.fps = fps
        self.debug = False
        self.keep_image = keep_image
    
    def __call__(self):
        restart = ' -y '
        fps_in = fps_out = self.fps
        fps_in = ' -r {}'.format(fps_in)
        path = self.output
        ext = '.jpg'
        cmd = ' -pix_fmt yuv420p -vcodec libx264'
        cmd += ' -r {}'.format(fps_out)
        if ext == '.png':
            cmd += ' -profile:v main'
        pathlist = sorted(os.listdir(path))
        pathlist = [join(path, p) for p in pathlist if os.path.isdir(join(path, p))]
        for path in pathlist:
            imgnames = glob(join(path, '*{}'.format(ext)))
            if len(imgnames) == 0:
                continue
            shell = f'ffmpeg{restart}{fps_in} -i "{path}/%06d{ext}" -vf scale="2*ceil(iw/2):2*ceil(ih/2)"{cmd} "{path}.mp4"'
            if not self.debug:
                shell += ' -loglevel quiet'
            print(shell)
            os.system(shell)
            # 确认一下文件已经生成了
            if not os.path.exists(path+'.mp4'):
                mywarn('Video {} is not generated'.format(path+'.mp4'))
                shell = shell.replace(' -loglevel quiet', '')
                run_cmd(shell)
            else:
                if not self.keep_image:
                    shutil.rmtree(path)