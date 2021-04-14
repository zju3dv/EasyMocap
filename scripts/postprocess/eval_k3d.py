'''
  @ Date: 2021-03-05 19:29:49
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-31 22:46:05
  @ FilePath: /EasyMocap/scripts/postprocess/eval_k3d.py
'''
# Evaluate any 3d keypoints
from glob import glob
from tqdm import tqdm
from os.path import join
import os
import numpy as np
from easymocap.dataset import CONFIG
from easymocap.mytools.reader import read_keypoints3d
from easymocap.mytools import read_camera
from eval_utils import keypoints_error
from pprint import pprint

class Conversion:
    def __init__(self, type_i, type_o, type_e=None):
        names_i = CONFIG[type_i]['joint_names']
        names_o = CONFIG[type_o]['joint_names']
        if type_e is None:
            self.commons = [i for i in names_o if i in names_i]
        else:
            names_e = CONFIG[type_e]['joint_names']
            self.commons = [i for i in names_e if i in names_i and i in names_o]
        self.idx_i = [names_i.index(i) for i in self.commons]
        self.idx_o = [names_o.index(i) for i in self.commons]

    def inp(self, inp):
        return inp[..., self.idx_i, :]
    
    def out(self, out):
        return out[..., self.idx_o, :]
    
    def __call__(self, inp, out):
        return inp[..., self.idx_i, :], out[..., self.idx_o, :]

def run_eval_keypoints(inp, out, type_i, type_o, step_gt, mode='single', args=None):
    # 遍历输出文件夹
    conversion = Conversion(type_i, type_o)
    inplists = sorted(glob(join(inp, '*.json')))[::step_gt]
    outlists = sorted(glob(join(out, '*.json')))[args.start:args.end]
    assert len(inplists) == len(outlists), '{} != {}'.format(len(inplists), len(outlists))
    results = []
    for nf, inpname in enumerate(tqdm(inplists)):
        outname = outlists[nf]
        gts = read_keypoints3d(inpname)
        ests = read_keypoints3d(outname)
        # 将GT转换到当前坐标系
        for gt in gts:
            gt['keypoints3d'] = conversion.inp(gt['keypoints3d'])
            if gt['keypoints3d'].shape[1] == 3:
                gt['keypoints3d'] = np.hstack([gt['keypoints3d'], np.ones((gt['keypoints3d'].shape[0], 1))])
        for est in ests:
            est['keypoints3d'] = conversion.out(est['keypoints3d'])
            if est['keypoints3d'].shape[1] == 3:
                est['keypoints3d'] = np.hstack([est['keypoints3d'], np.ones((est['keypoints3d'].shape[0], 1))])
        # 这一步将交换est的顺序
        if mode == 'single':
            # 单人的：直接匹配上
            pass
        elif mode == 'matched': # ID已经匹配过了
            pass 
        else: # 进行匹配
            # 把估计的id都清空
            for est in ests:
                est['id'] = -1
            # 计算距离先
            kpts_gt = np.stack([v['keypoints3d'] for v in gts])
            kpts_dt = np.stack([v['keypoints3d'] for v in ests])
            distances = np.linalg.norm(kpts_gt[:, None, :, :3] - kpts_dt[None, :, :, :3], axis=-1)
            conf = (kpts_gt[:, None, :, -1] > 0) * (kpts_dt[None, :, :, -1] > 0)
            dist = (distances * conf).sum(axis=-1)/conf.sum(axis=-1)
            # 贪婪的匹配
            ests_new = []
            for igt, gt in enumerate(gts):
                bestid = np.argmin(dist[igt])
                ests_new.append(ests[bestid])
            ests = ests_new
        # 计算误差
        for i, data in enumerate(gts):
            kpts_gt = data['keypoints3d']
            kpts_est = ests[i]['keypoints3d']
            # 计算各种误差，存成字典
            result = keypoints_error(kpts_gt, kpts_est, conversion.commons, joint_level=args.joint, use_align=args.align)
            result['nf'] = nf
            result['id'] = data['id']
            results.append(result)
    write_to_csv(join(out, '..', 'report.csv'), results)
    return 0
    keys = list(results[list(results.keys())[0]][0].keys())
    reports = {}
    for pid, result in results.items():
        vals = {key: sum([res[key] for res in result])/len(result) for key in keys}
        reports[pid] = vals
    from tabulate import tabulate
    headers = [''] + keys
    table = []
    for pid, report in reports.items():
        res = ['{}'.format(pid)] + ['{:.2f}'.format(report[key]) for key in keys]
        table.append(res)
    savename = 'tmp.txt'
    print(tabulate(table, headers, tablefmt='fancy_grid'))
    print(tabulate(table, headers, tablefmt='fancy_grid'), file=open(savename, 'w'))

def write_to_csv(filename, results):
    from tabulate import tabulate
    keys = list(results[0].keys())
    headers, table = [], []
    for key in keys:
        if isinstance(results[0][key], float):
            headers.append(key)
            table.append('{:.3f}'.format(sum([res[key] for res in results])/len(results)))
    print('>> Totally {} samples:'.format(len(results)))
    print(tabulate([table], headers, tablefmt='fancy_grid'))
    with open(filename, 'w') as f:
        # 写入头
        header = list(results[0].keys())
        f.write(','.join(header) + '\n')
        for res in results:
            f.write(','.join(['{}'.format(res[key]) for key in header]) + '\n')

def run_eval_keypoints_mono(inp, out, type_i, type_o, type_e, step_gt, cam_path, mode='single'):
    conversion = Conversion(type_i, type_o, type_e)
    inplists = sorted(glob(join(inp, '*.json')))[::step_gt]
    # TODO:only evaluate a subset of views
    if len(args.sub) == 0:
        views = sorted(os.listdir(out))
    else:
        views = args.sub
    # read camera
    cameras = read_camera(join(cam_path, 'intri.yml'), join(cam_path, 'extri.yml'), views)
    cameras = {key:cameras[key] for key in views}
    if args.cam_res is not None:
        cameras_res = read_camera(join(args.cam_res, 'intri.yml'), join(args.cam_res, 'extri.yml'), views)
        cameras_res = {key:cameras_res[key] for key in views}

    results = []
    for view in views:
        outlists = sorted(glob(join(out, view, '*.json')))
        RT = cameras[view]['RT']
        for outname in outlists:
            basename = os.path.basename(outname)
            gtname = join(inp, basename)
            gts = read_keypoints3d(gtname)
            ests = read_keypoints3d(outname)
            # 将GT转换到当前坐标系
            for gt in gts:
                keypoints3d = conversion.inp(gt['keypoints3d'])
                conf = keypoints3d[:, -1:].copy()
                keypoints3d[:, -1] = 1
                keypoints3d = (RT @ keypoints3d.T).T
                gt['keypoints3d'] = np.hstack([keypoints3d, conf])
            for est in ests:
                est['keypoints3d'] = conversion.out(est['keypoints3d'])
                if est['keypoints3d'].shape[1] == 3:
                    # 增加置信度为1
                    est['keypoints3d'] = np.hstack([est['keypoints3d'], np.ones((est['keypoints3d'].shape[0], 1))])
            # 计算误差
            for i, data in enumerate(gts):
                kpts_gt = data['keypoints3d']
                kpts_est = ests[i]['keypoints3d']
                # 计算各种误差，存成字典
                result = keypoints_error(kpts_gt, kpts_est, conversion.commons, joint_level=args.joint, use_align=True)
                result['pid'] = data['id']
                result['view'] = view
                results.append(result)

    write_to_csv(join(out, '..', 'report.csv'), results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--type_i', type=str, default='body25',
        help='Type of ground-truth keypoints')
    parser.add_argument('--type_o', type=str, default='body25',
        help='Type of output keypoints')
    parser.add_argument('--type_e', type=str, default=None,
        help='Type of evaluation keypoints')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'matched', 'greedy'],
        help='the mode of match 3d person')
    # parser.add_argument('--dataset', type=str, default='h36m')
    parser.add_argument('--start', type=int, default=0,
        help='frame start')
    parser.add_argument('--end', type=int, default=100000,
        help='frame end')    
    parser.add_argument('--step', type=int, default=1,
        help='frame step')
    parser.add_argument('--step_gt', type=int, default=1)
    parser.add_argument('--joint', action='store_true',
        help='report each joint')
    parser.add_argument('--align', action='store_true',
        help='report each joint')
    # Multiple views dataset
    parser.add_argument('--mono', action='store_true',
        help='use this option if the estimated joints use monocular images. \
            The results are stored in different folders.')
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the sub folder lists when in video mode')
    parser.add_argument('--cam', type=str, default=None)
    parser.add_argument('--cam_res', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.mono:
        run_eval_keypoints_mono(args.path, args.out, args.type_i, args.type_o, args.type_e, cam_path=args.cam, step_gt=args.step_gt, mode=args.mode)
    else:
        run_eval_keypoints(args.path, args.out, args.type_i, args.type_o, args.step_gt, mode=args.mode, args=args)
