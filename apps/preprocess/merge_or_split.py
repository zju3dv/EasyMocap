'''
  @ Date: 2022-07-25 21:56:58
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-25 21:59:29
  @ FilePath: /EasyMocapPublic/apps/preprocess/merge_or_split.py
'''
import shutil
from easymocap.mytools.debug_utils import log, mkdir, mywarn, run_cmd
import os
from os.path import join
from tqdm import tqdm

def merge_directories(root, out):
    mkdir(join(out, 'images', 'merge'))
    sequence = sorted(os.listdir(root))
    log('>>> Totally {} sub-folders'.format(len(sequence)))
    records = []
    for seq in tqdm(sequence, 'check sequence'):
        subs = sorted(os.listdir(join(root, seq, 'images')))
        for sub in subs:
            imgnames = sorted(os.listdir(join(root, seq, 'images', sub)))
            for imgname in imgnames:
                records.append((seq, sub, imgname))
    log('>>> Totally {} records'.format(len(records)))
    for (seq, sub, imgname) in tqdm(records):
        srcname = join(root, seq, 'images', sub, imgname)
        dstname = join(out, 'images', 'merge', '{}+{}+{}'.format(seq, sub, imgname))
        if not os.path.exists(dstname):
            cmd = 'ln -s {} {}'.format(srcname, dstname)
            # run_cmd(cmd)
            shutil.copyfile(srcname, dstname)
    with open(join(out, 'log.txt'), 'w') as f:
        for (seq, sub, imgname) in tqdm(records, 'writing'):
            f.write('{},{},{}\r\n'.format(seq, sub, imgname))

def split_directories(root, out):
    with open(join(out, 'log.txt'), 'r') as f:
        records = f.readlines()
    for record in tqdm(records):
        seq, sub, imgname = record.strip().split(',')
        imgname = imgname.replace('.jpg', '.json')
        srcname = join(out, 'annots', 'merge', '{}+{}+{}'.format(seq, sub, imgname))
        dstname = join(root, seq, 'annots', sub, imgname)
        if not os.path.exists(os.path.dirname(dstname)):
            os.makedirs(os.path.dirname(dstname))
        shutil.copyfile(srcname, dstname)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()

    if args.merge:
        merge_directories(args.root, args.out)
    if args.split:
        split_directories(args.root, args.out)
    