import re
import numpy as np
import os, sys
import cv2
import shutil
from os.path import join
from tqdm import trange, tqdm
from multiprocessing import Pool
import json

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def parseImg(imgname):
    """ 解析图像名称
    
    Arguments:
        imgname {str} -- 
    
    Returns:
        dic -- 包含文件图像信息的字典
    """
    s = re.search(
        '(?P<id>\d+)_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_(?P<hour>\d{2})-(?P<min>\d{2})-(?P<sec>\d{2})\.(?P<ms>\d{3})',
        imgname)
    assert s is not None, imgname
    dic = s.groupdict()
    for key in dic.keys():
        dic[key] = int(dic[key])
    dic['time'] = dic['ms'] + dic['sec'] * 1000 + dic['min'] * 60000 + dic['hour'] * 60000 * 60
    return dic


def getCamNum(x):
    return x.split('_B')[1]


def getImgId(x):
    return x.split('_')[4]


def findBeginEnd(images_info):
    begin_time = 0
    end_time = np.inf
    for key in images_info:
        first_frame = images_info[key]['first_frame']
        last_frame = images_info[key]['last_frame']
        curr_f_time = images_info[key][first_frame]['time']
        curr_e_time = images_info[key][last_frame]['time']
        if curr_f_time > begin_time:
            begin_time = curr_f_time
        if curr_e_time < end_time:
            end_time = curr_e_time
    return begin_time, end_time


def findRef(images_info):
    ref_cam = 0
    min_frame = np.inf

    for key in images_info:
        first_frame = images_info[key]['first_frame']
        last_frame = images_info[key]['last_frame']
        f_id = images_info[key][first_frame]['id']
        e_id = images_info[key][last_frame]['id']
        if (e_id - f_id) < min_frame:
            min_frame = e_id - f_id
            ref_cam = key
    return ref_cam


def findNearest(cam_info, time):
    # 找time最接近的帧的名称
    select_frame = ''
    # WARN: 确保cam_info是有序的
    img_pre = None
    for idx, img in enumerate(cam_info.keys()):
        if isinstance(cam_info[img], dict):
            if cam_info[img]['time'] < time:
                img_pre = img
                continue
            else:
                select_frame = img
                break
    # 判断一下处于边界上的两帧，哪一帧的时间更接近
    if img_pre is not None:
        if abs(time - cam_info[img_pre]['time']) < abs(time - cam_info[img]['time']):
            select_frame = img_pre
    return select_frame


def get_filelists(path, save_path):
    cameralists = sorted(os.listdir(path), key=lambda x: getCamNum(x))
    images_info = {}
    for camname in cameralists:
        images_info[camname] = {}
        imglists = listdir([path, camname])
        imglists.sort(key=lambda x: getImgId(x))
        for imgname in tqdm(imglists, desc=camname):
            images_info[camname][imgname] = parseImg(imgname)
        images_info[camname]['first_frame'] = imglists[0]
        images_info[camname]['last_frame'] = imglists[-1]
        # print(images_info[camname])
    # 寻找最晚开始最早结束的时间
    begin_time, end_time = findBeginEnd(images_info)
    print('begin time: {}, end time: {}'.format(begin_time, end_time))
    # 寻找帧率最低的视频，以这个视频为参考
    if args.ref is None:
        ref_cam = findRef(images_info)
    else:
        ref_cam = args.ref

    print('The reference camera is {}'.format(ref_cam))
    # 以帧率最低的相机为参考，对每一帧寻找其他相机时间最接近的帧
    output_info = {key: [] for key in cameralists}
    for imgname in tqdm(images_info[ref_cam].keys(), 'sync'):
        if isinstance(images_info[ref_cam][imgname], dict):
            cur_time = images_info[ref_cam][imgname]['time']
            if cur_time < begin_time:
                continue
            if cur_time > end_time:
                break
            for cam in cameralists:
                if cam == ref_cam:
                    select = imgname
                else:
                    select = findNearest(images_info[cam], cur_time)
                output_info[cam].append(select)
    # 将图片保存
    mkdir(save_path)
    # 保存匹配信息
    # TODO:增加匹配时间差的指标
    import json
    with open(join(save_path, 'match_info.json'), 'w') as f:
        json.dump(output_info, f, indent=4)
    for cam in cameralists:
        mkdir(join(save_path, cam))
        for i, imgname in enumerate(tqdm(output_info[cam], desc=cam)):
            src = join(path, cam, imgname)
            dst = join(save_path, cam, '%06d.jpg' % i)
            img = cv2.imread(src)
            if img.shape[0] == 2048:
                img = cv2.resize(img, (1024, 1024), cv2.INTER_NEAREST)
            cv2.imwrite(dst, img)

def getFileDict(path):
    cams = sorted(os.listdir(path))
    cams = [cam for cam in cams if os.path.isdir(join(path, cam))]
    cams = list(filter(
        lambda x:\
            x.startswith('Camera')\
            and x not in filter_list
            , cams)) # B6相机同步有问题 不要使用了
    results = {}
    for cam in cams:
        # 注意：lightstage的图像直接sort就是按照顺序了的
        files = sorted(os.listdir(join(path, cam)))
        files = [f for f in files if f.endswith('.jpg')]
        results[cam] = files
    return cams, results

def sync_by_name(imagelists, times_all, cams):
    # 选择开始帧
    start = max([t[0] for t in times_all.values()])
    # 弹出开始帧以前的数据
    for cam in cams:
        times = times_all[cam].tolist()
        while times[0] < start:
            times.pop(0)
            imagelists[cam].pop(0)
        times_all[cam] = np.array(times)
    # 选择参考视角的时候,应该选择与其他视角的距离最近的作为参考
    best_distances = []
    for cam in cams:
        # 分别对每个进行设置, 使用第一帧的时间,留有余地
        ref_time = times_all[cam][1]
        distances = []
        for c in cams:
            dist = np.abs(times_all[c] - ref_time).min()
            distances.append(dist)
        print('{:10s}: {:.2f}'.format(cam, sum(distances)/len(cams)))
        best_distances.append(sum(distances)/len(cams))
    best_distances = np.array(best_distances)
    ref_view = best_distances.argmin()
    if args.ref is None:
        ref_cam = cams[best_distances.argmin()]
    else:
        ref_cam = args.ref
        ref_view = cams.index(ref_cam)

    times_all = [times_all[cam] for cam in cams]
    print('Select reference view: ', cams[ref_view])
    if False:
        distance = np.eye((dimGroups[-1]))
        for nv0 in range(len(times_all)-1):
            for nv1 in range(nv0+1, len(times_all)):
                dist = np.abs(times_all[nv0][:, None] - times_all[nv1][None, :])
                dist = (MAX_DIST - dist)/MAX_DIST
                dist[dist<0] = 0
                distance[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]] = dist
                distance[dimGroups[nv1]:dimGroups[nv1+1], dimGroups[nv0]:dimGroups[nv0+1]] = dist.T
        matched, ref_view = match_dtw(distance, dimGroups, debug=args.debug)
    elif True:
        # 使用最近邻选择
        matched = []
        for nv in range(len(times_all)):
            dist = np.abs(times_all[ref_view][:, None] - times_all[nv][None, :])
            rows = np.arange(dist.shape[0])
            argmin0 = dist.argmin(axis=1)
            # 直接选择最近的吧
            # 去掉开头
            for i in range(argmin0.shape[0]):
                if argmin0[i] == argmin0[i+1]:
                    argmin0[i] = -1
                else:
                    break
            # 去掉结尾
            for i in range(1, argmin0.shape[0]):
                if argmin0[-i] == argmin0[-i-1]:
                    argmin0[-i] = -1
                else:
                    break
            matched.append(argmin0)
        matched = np.stack(matched)
    elif False:
        # 1. 首先判断一下所有视角的最接近的点
        nViews = len(times_all)
        TIME_STEP = 20
        REF_OFFSET = 20 # 给参考视角增加一个帧的偏移，保证所有相机都正常开启了，同时增加一个帧的结束，保证所有相机都结束了
        views_ref = [ref_view]
        matched = {
            ref_view:np.arange(REF_OFFSET, times_all[ref_view].shape[0]-REF_OFFSET)
        }
        while True:
            times_mean = np.stack([times_all[ref][matched[ref]] for ref in matched.keys()])
            times_mean = np.mean(times_mean, axis=0)
            infos = []
            for nv in range(nViews):
                if nv in matched.keys():
                    continue
                if False:
                    dist_all = []
                    for ref, indices in matched.items():
                        dist = np.abs(times_all[ref][indices, None] - times_all[nv][None, :])
                        dist[dist>TIME_STEP] = 10*TIME_STEP
                        dist_all.append(dist)
                    dist = np.stack(dist_all).sum(axis=0)
                    dist = dist / len(matched.keys())
                else:
                    dist = np.abs(times_mean[:, None] - times_all[nv][None, :])
                argmin0 = dist.argmin(axis=1)
                rows = np.arange(dist.shape[0])
                dist_sum = dist.min(axis=1).mean()
                infos.append({
                    'v': nv,
                    'dist_sum': dist_sum,
                    'argmin': argmin0
                })
                print(nv, dist_sum)
            if len(infos) == 0:
                break
            infos.sort(key=lambda x:x['dist_sum'])
            print('Select view: ', infos[0]['v'], infos[0]['dist_sum'])
            matched[infos[0]['v']] = infos[0]['argmin']
        matched = np.stack([matched[nv] for nv in range(nViews)])
    else:
        # 选择一个开头，计算最佳的偏移
        # 开始帧：所有的开始帧中的最晚的一帧
        # 假定恒定帧率，只需要选择一个开头就好了
        nViews = len(times_all)
        start_t = max([t[0] for t in times_all])
        # 留出10帧来操作
        start_f = [np.where(t>start_t)[0][0] + 10 for t in times_all]
        start_t = times_all[ref_view][start_f[ref_view]]
        valid_f = [[np.where(t<start_t)[0][-1],np.where(t>=start_t)[0][0]] for t in times_all]
        from copy import deepcopy
        valid_f_copy = deepcopy(valid_f)
        import matplotlib as mpl
        mpl.use('TkAgg')
        import matplotlib.pyplot as plt

        while True:
            min_v, min_t = -1, 1e10
            min_info, max_info = [], []
            max_v, max_t = -1, -1
            for nv in range(nViews):
                if len(valid_f[nv]) == 1:
                    continue
                # 存在多个的
                min_info.append({
                    'v': nv,
                    't': times_all[nv][valid_f[nv][0]]
                })
                max_info.append({
                    'v': nv,
                    't': times_all[nv][valid_f[nv][-1]]
                })
            # 判断最小和最大的弹出谁
            min_info.sort(key=lambda x:x['t'])
            max_info.sort(key=lambda x:-x['t'])
            if len(min_info) > 1 and len(max_info) > 1:
                # delta_min = min_info[1]['t'] - min_info[0]['t']
                # delta_max = max_info[0]['t'] - max_info[1]['t']
                delta_max = max_info[0]['t'] - start_t
                delta_min = start_t - min_info[0]['t']
                if delta_max > delta_min:
                    valid_f[max_info[0]['v']].pop(-1)
                else:
                    valid_f[min_info[0]['v']].pop(0)
            else:
                nv = min_info[0]['v']
                t_min = times_all[nv][valid_f[0]]
                t_max = times_all[nv][valid_f[1]]
                delta_min = start_t - t_min
                delta_max = t_max - start_t
                if delta_max > delta_min:
                    valid_f[nv].pop(-1)
                else:
                    valid_f[nv].pop(0)
                break
        plt.plot([0, nViews], [start_t, start_t])
        for nv in range(len(valid_f)):
            if len(valid_f[nv]) > 1:
                start, end = valid_f[nv]
                start, end = times_all[nv][start], times_all[nv][end]
                plt.plot([nv, nv], [start, end])
            else:
                start, end = valid_f_copy[nv][0], valid_f_copy[nv][-1]
                start, end = times_all[nv][start], times_all[nv][end]
                plt.plot([nv, nv], [start, end])
                plt.scatter(nv, times_all[nv][valid_f[nv]])
        plt.show()
        matched = np.arange(times_all[ref_view].shape[0]).reshape(1, -1).repeat(nViews, 0)
        matched = np.arange(2).reshape(1, -1).repeat(nViews, 0)
        start = np.array(valid_f).reshape(-1, 1)
        matched += start
        shape = np.array([t.shape[0] for t in times_all]).reshape(-1, 1) - 10
        matched[matched<0] = -1
        # matched[matched>shape] = -1
    matched = matched[:, (matched!=-1).all(axis=0)]
    matched_time = np.zeros_like(matched)
    for nv in range(matched.shape[0]):
        matched_time[nv] = times_all[nv][matched[nv]]
    max_time = matched_time.max(axis=0)
    min_time = matched_time.min(axis=0)
    diff = max_time - min_time
    step = matched_time[:, 1:] - matched_time[:, :-1]
    headers = ['camera', 'start', 'end', 'delta_mean', 'delta_min', 'delta_max', 'diff_max', 'diff_min', 'diff_mean']
    infos = []
    dist_to_ref_all = 0
    for nv, cam in enumerate(cams):
        dist_to_ref = (matched_time[nv] - matched_time[ref_view]).tolist()
        dist_to_ref_all += np.abs(dist_to_ref).mean()
        dist_to_ref.sort(key=lambda x: abs(x))
        infos.append([cam, matched_time[nv, 0], matched_time[nv, -1], step[nv].mean(), step[nv].min(), step[nv].max(), dist_to_ref[-1], dist_to_ref[0], np.abs(np.array(dist_to_ref)).mean()])
    print(tabulate(infos, headers=headers))
    # import matplotlib.pyplot as plt
    # plt.plot(times_all[7][:100])
    # plt.plot(times_all[ref_view][:100])
    # plt.show()
    # import ipdb;ipdb.set_trace()
    print("Max sync difference = {}ms, Mean max sync difference = {:.1f}ms".format(diff.max(), diff.mean()))
    print("Mean sync diff : {}".format(dist_to_ref_all/len(cams)))
    if not args.nocheck: import ipdb;ipdb.set_trace()
    return matched, matched_time

def copy_func(src, dst):
    if args.keep2048:
        shutil.copyfile(src, dst)
    else:
        img = cv2.imread(src)
        img = cv2.resize(img, (1024, 1024))
        if colors_params is not None:
            sub = os.path.basename(os.path.dirname(dst))
            M = colors_params[sub]
            img = (np.clip((img.astype(np.float32)/255.) @ M, 0., 1.) * 255).astype(np.uint8)
        cv2.imwrite(dst, img)

def copy_func_batch(src: list, dst: list):
    assert(len(src) == len(dst))
    for i in tqdm(range(len(src))):
        copy_func(src[i], dst[i])

THREAD_CNT = 8

def copy_with_match(path, out, matched, imagelists, cams, multiple_thread = False):
    print('---')
    print('Copy {} to {}'.format(path, out))
    print('---')
    pad_2 = lambda x:'{:02d}'.format(int(x))
    remove_cam = lambda x:x.replace('Camera_B', '').replace('Camera_', '').replace('Camera (', '').replace(')', '')
    cvt_viewname = lambda x:pad_2(remove_cam(x))

    reports = [[] for _ in range(matched.shape[1])]
    for nv in tqdm(range(matched.shape[0])):
        outdir = join(out, 'images', cvt_viewname(cams[nv]))
        if os.path.exists(outdir):
            if matched.shape[1] == len(os.listdir(outdir)):
                print('exists enough images')
                continue
            else:
                print('exists not enough images')
        else:
            os.makedirs(outdir, exist_ok=True)
        imgname_old_s = [[] for _ in range(THREAD_CNT)]
        imgname_new_s = [[] for _ in range(THREAD_CNT)]
        for nfnew in range(matched.shape[1]):
            nf = matched[nv, nfnew]
            imgname_old = join(path, cams[nv], imagelists[cams[nv]][nf])
            imgname_old_s[nfnew % THREAD_CNT].append(imgname_old)
            imgname_new_s[nfnew % THREAD_CNT].append(join(outdir, '{:06d}.jpg'.format(nfnew)))
            reports[nfnew].append(imgname_old)
        if multiple_thread:
            import threading
            threads = []
            for i in range(THREAD_CNT):
                thread = threading.Thread(target=copy_func_batch, args=(imgname_old_s[i], imgname_new_s[i])) # 应该不存在任何数据竞争
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
        else:
            for nfnew in tqdm(range(matched.shape[1]), desc='{}'.format(cams[nv])):
                nf = matched[nv, nfnew]
                imgname_old = join(path, cams[nv], imagelists[cams[nv]][nf])
                imgname_new = join(outdir, '{:06d}.jpg'.format(nfnew))
                copy_func(imgname_old, imgname_new)
    save_json(join(out, 'match_name.json'), reports)

from tabulate import tabulate
def parse_time(imagelists, cams):
    times_all = {}
    headers = ['camera', 'frames', 'mean', 'min', 'max', 'number>mean', 'start', 'end']
    MAX_STEP = 20
    infos = []
    start_time = -1
    for cam in cams:
        times = []
        for imgname in imagelists[cam]:
            time = parseImg(imgname)['time']
            times.append(time)
        times = np.array(times)
        times_all[cam] = times
        if start_time < 0:
            start_time = times[0]
        else:
            start_time = min(start_time, times[0])
    print('Start time: {}'.format(start_time))
    for cam in cams:
        times = times_all[cam]
        times -= start_time
        delta = times[1:] - times[:-1]
        infos.append([cam, times.shape[0], 
            delta.mean(),
            '{}/{}'.format(delta.min(), delta.argmin()), 
            '{}/{}'.format(delta.max(), delta.argmax()), 
            (delta>delta.mean()).sum(), 
            times[0]%60000,
            times[-1]%60000])
    print(tabulate(infos, headers=headers))
    return times_all

def soft_sync(path, out, multiple_thread = False):
    os.makedirs(out, exist_ok=True)
    # 获取图像名称
    cams, imagelists = getFileDict(path)
    if args.static:
        # 静止场景，直接保存第一帧图像
        matched = np.zeros((len(cams), 1), dtype=np.int)
    elif args.nosync:
        assert len(cams) == 1
        times_all = parse_time(imagelists, cams)
        matched = np.arange(0, len(imagelists[cams[0]])).reshape(1, -1)
        # matched = np.arange((1, len(imagelists[cams[0]])), dtype=np.int)
    else:
        # 获取图像时间
        times_all = parse_time(imagelists, cams)
        matched, matched_time = sync_by_name(imagelists, times_all, cams)
        matched = matched[:, ::args.step]
        times_all = {key:val.tolist() for key, val in times_all.items()}
        save_json(join(out, 'timestamp.json'), times_all)
        np.savetxt(join(out, 'sync_time.txt'), matched_time-matched_time.min(), fmt='%10d')
        # 保存图像
    copy_with_match(path, out, matched, imagelists, cams, multiple_thread)

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage='''
    origin=/path/to/CoreView_xxx
    data=/path/to/output/xxx
    - convert data: python3 scripts/dataset/pre_lightstage.py ${origin} ${data} --mp --ref_min
    - keep origin resolution: --keep2048
    - only copy one frame: --static
    - set the color adjustment: --color /path/to/color
    - skip the check: --nocheck
''')
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--filter', type=str, nargs='+', default=[])
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--ref', type=str, default=None)
    parser.add_argument('--color', type=str, default=None)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--keep2048', action='store_true')
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--ref_min', action='store_true')
    parser.add_argument('--nosync', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--mp", action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--nocheck', action='store_true')
    args = parser.parse_args()

    # Reading color adjustment
    if args.color is not None:
        colors_params = {}
        for sub in sorted(os.listdir(args.color)):
            colors = read_json(join(args.color, sub, '000000.json'))
            colors_params[sub] = np.array(colors['params'], dtype=np.float32)
        print('Reading color adjustment from {}'.format(args.color))
    else:
        colors_params = None
    cmd = 'python3 ' + ' '.join(sys.argv)
    os.makedirs(args.out, exist_ok=True)
    print(cmd, file=open(join(args.out, 'cmd.log'), 'w'))
    filter_list = args.filter

    if args.check:
        timestamp = np.loadtxt(join(args.out, 'sync_time.txt'), dtype=np.int)
        timestamp = timestamp[:, :10]
        t = np.arange(timestamp.shape[1])
        import matplotlib as mpl
        mpl.use('TkAgg')
        import matplotlib.pyplot as plt
        for nv in range(timestamp.shape[0]):
            plt.plot(t, timestamp[nv])
        for nf in range(timestamp.shape[1]-1):
            plt.plot([nf, nf+1], [timestamp[:, nf].mean(), timestamp[:, nf].mean()], c='k')
            plt.plot([nf, nf+1], [timestamp[:, nf].min(), timestamp[:, nf].min()], c='r')
            plt.plot([nf, nf+1], [timestamp[:, nf].max(), timestamp[:, nf].max()], c='g')
        plt.show()
        import ipdb; ipdb.set_trace()
    else:
        soft_sync(args.path, args.out, multiple_thread = args.mp)
