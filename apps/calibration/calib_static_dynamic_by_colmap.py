'''
  @ Date: 2022-05-09 12:32:50
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 16:31:29
  @ FilePath: /EasyMocapPublic/apps/calibration/calib_static_dynamic_by_colmap.py
'''
import os
from os.path import join
import shutil
from easymocap.mytools.debug_utils import log, myerror, mywarn, run_cmd, mkdir
from easymocap.mytools.colmap_wrapper import colmap_feature_extract, colmap_feature_match

from tqdm import tqdm

def copy_images(data, out, nf=0):
    subs = sorted(os.listdir(data))
    image_names = []
    for sub in subs:
        srcname = join(data, sub, '{:06d}.jpg'.format(nf))
        dstname = join(out, '{}.jpg'.format(sub))
        os.makedirs(os.path.dirname(dstname), exist_ok=True)
        shutil.copyfile(srcname, dstname)
        image_names.append(dstname)
    return image_names

def copy_to_newdir(path, out, num):
    statics = copy_images(join(path, 'images'), join(out, 'images', 'static'), nf=0)
    scannames = sorted(os.listdir(join(path, 'scan')))
    if num != -1:
        log('[copy] sample {} from {} images'.format(num, len(scannames)))
        scannames = scannames[::len(scannames)//num]
    scans = []
    for name in tqdm(scannames):
        srcname = join(path, 'scan', name)
        dstname = join(out, 'images', 'scan', name)
        os.makedirs(os.path.dirname(dstname), exist_ok=True)
        shutil.copyfile(srcname, dstname)
        scans.append(dstname)
    return statics, scans

def sparse_recon(path, statics, scans, colmap):
    colmap_feature_extract(colmap, path, share_camera=False, add_mask=False, gpu=args.gpu,
        share_camera_per_folder=True)
    colmap_feature_match(colmap, path, gpu=args.gpu)
    mkdir(join(path, 'sparse'))
    cmd = f'{colmap} mapper --database_path {path}/database.db --image_path {path}/images --output_path {path}/sparse \
--Mapper.ba_refine_principal_point 1 \
--Mapper.ba_global_max_num_iterations 1000 \
'
    run_cmd(cmd)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(usage=
    '''This script is used to calibrate a scene with a moving videos and multiple images captured by static cameras
''')
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--colmap', type=str, default=None)
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--step', type=int, default=800)
    parser.add_argument('--add_mask', action='store_true')
    parser.add_argument('--no_camera', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    data = args.path
    out_root = args.out

    for ext in ['mp4', 'mov', 'avi']:
        scanname = join(args.path, 'scan.{}'.format(ext))
        if os.path.exists(scanname):
            break
    else:
        myerror('[error] {} not found'.format('scan.mp4, scan.mov, scan.avi'))
        exit(1)
    scandir = join(args.path, 'scan')
    if not os.path.exists(scandir):
        os.makedirs(scandir)
        cmd = f'ffmpeg -i {scanname} -q:v 1 -start_number 0 -r {args.step} {scandir}/%06d.jpg -loglevel quiet'
        run_cmd(cmd)
    # copy to output dir
    statics, scans = copy_to_newdir(args.path, out_root, num=args.num)
    sparse_recon(out_root, statics, scans, args.colmap)