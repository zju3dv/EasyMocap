'''
  @ Date: 2020-12-01 22:14:11
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-30 21:33:40
  @ FilePath: /EasyMocap/scripts/postprocess/eval_shelf.py
'''
import os
import sys
from os.path import join
import re
import json
import time
import scipy.io as scio
import numpy as np
from tqdm import tqdm

def save_json(output, json_path):
    os.system('mkdir -p {}'.format(os.path.dirname(json_path)))    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=4)

def is_right(model_start_point, model_end_point, gt_strat_point, gt_end_point, alpha=0.5):
    bone_lenth = np.linalg.norm ( gt_end_point - gt_strat_point )
    start_difference = np.linalg.norm ( gt_strat_point - model_start_point )
    end_difference = np.linalg.norm ( gt_end_point - model_end_point )
    return ((start_difference + end_difference) / 2) <= alpha * bone_lenth

def openpose2shelf3D(pose3d, score):
    """
    transform coco order(our method output) 3d pose to shelf dataset order with interpolation
    :param pose3d: np.array with shape nJx3
    :return: 3D pose in shelf order with shape 14x3
    """
    shelf_pose = np.zeros ( (14, 3) )
    shelf_score = np.zeros ( (14, 1) )

    # coco2shelf = np.array ( [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9] )
    openpose2shelf = np.array([11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7])
    shelf_pose[0: 12] += pose3d[openpose2shelf]
    shelf_score[0: 12] += score[openpose2shelf]
    if True:
        shelf_pose[12] = pose3d[1]  # Use middle of shoulder to init
        shelf_pose[13] = pose3d[0]  # use nose to init
        shelf_pose[13] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array ( [0.75, 0.75, 1.5] )
        shelf_pose[12] = shelf_pose[12] + (pose3d[0] - shelf_pose[12]) * np.array ( [1. / 2., 1. / 2., 1. / 2.] )
        shelf_score[12] = score[0]*score[1]
        shelf_score[13] = score[0]*score[1]
    else:
        shelf_pose[12] = pose3d[1]
        shelf_pose[13] = pose3d[0]
    return shelf_pose, shelf_score

def convert_openpose_shelf(keypoints3d):
    shelf15 = np.zeros((15, 4))
    openpose2shelf = np.array([11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 0, 8])
    shelf15 = keypoints3d[openpose2shelf].copy()
    # interp head
    faceDir = np.cross(shelf15[12, :3] - shelf15[14, :3], shelf15[8, :3] - shelf15[9, :3])
    faceDir = faceDir/np.linalg.norm(faceDir)
    zDir = np.array([0., 0., 1.])
    shoulderCenter = (keypoints3d[2, :3] + keypoints3d[5, :3])/2.
    # headCenter = (keypoints3d[15, :3] + keypoints3d[16, :3])/2.
    headCenter = (keypoints3d[17, :3] + keypoints3d[18, :3])/2.

    shelf15[12, :3] = shoulderCenter + (headCenter - shoulderCenter) * 0.5
    shelf15[13, :3] = shelf15[12, :3] + faceDir * 0.125 + zDir * 0.145
    return shelf15

def convert_openpose_shelf1(keypoints3d):
    shelf15 = np.zeros((15, 4))
    openpose2shelf = np.array([11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 0, 8])
    shelf15 = keypoints3d[openpose2shelf].copy()
    # interp head
    faceDir = np.cross(keypoints3d[1, :3] - keypoints3d[8, :3], keypoints3d[2, :3] - shelf15[5, :3])
    faceDir = faceDir/np.linalg.norm(faceDir)
    
    upDir = keypoints3d[1, :3] - keypoints3d[8, :3]
    upDir = upDir/np.linalg.norm(upDir)
    
    shoulderCenter = keypoints3d[1, :3]
    ear = (keypoints3d[17, :3] + keypoints3d[18, :3])/2 - keypoints3d[1, :3]
    eye = (keypoints3d[15, :3] + keypoints3d[16, :3])/2 - keypoints3d[1, :3]
    nose = keypoints3d[0, :3] - keypoints3d[1, :3]
    head = (ear + eye + nose)/3.
    noseLen = np.linalg.norm(head)
    noseDir = head / noseLen
    headDir = (noseDir * 2 + upDir)
    headDir = headDir / np.linalg.norm(headDir)

    neck = shoulderCenter + noseLen*headDir * 0.5

    shelf15[12, :3] = neck
    shelf15[13, :3] = neck + headDir * noseLen * 0.8
    return shelf15

def convert_shelf_shelfgt(keypoints):
    gt_hip = (keypoints[2] + keypoints[3]) / 2
    gt = np.vstack((keypoints, gt_hip))
    return gt

def vectorize_distance(a, b):
    """
    Calculate euclid distance on each row of a and b
    :param a: Nx... np.array
    :param b: Mx... np.array
    :return: MxN np.array representing correspond distance
    """
    N = a.shape[0]
    a = a.reshape ( N, -1 )
    M = b.shape[0]
    b = b.reshape ( M, -1 )
    a2 = np.tile ( np.sum ( a ** 2, axis=1 ).reshape ( -1, 1 ), (1, M) )
    b2 = np.tile ( np.sum ( b ** 2, axis=1 ), (N, 1) )
    dist = a2 + b2 - 2 * (a @ b.T)
    return np.sqrt ( dist )

def distance(a, b, score):
    # a: (N, J, 3)
    # b: (M, J, 3)
    # score: (M, J, 1)
    # return: (M, N)
    a = a[None, :, :, :]
    b = b[:, None, :, :]
    score = score[:, None, :, 0]
    diff = np.sum((a - b)**2, axis=3)*score
    dist = diff.sum(axis=2)/score.sum(axis=2)
    return np.sqrt(dist)

def _readResult(filename, isA4d):
    import json
    with open(filename, "r") as file:
        datas = json.load(file)
    res_ = []
    for data in datas:
        trackId = data['id']
        keypoints3d = np.array(data['keypoints3d'])
        if (keypoints3d[:, 3]>0).sum() > 1:
            res_.append({'id':trackId, 'keypoints3d': keypoints3d})
    if isA4d:
        # association4d 的关节顺序和正常的定义不一样
        for r in res_:
            r['keypoints3d'] = r['keypoints3d'][[4, 1, 5, 9, 13, 6, 10, 14, 0, 2, 7, 11, 3, 8, 12], :]
    return res_
    
def readResult(filePath, range_=None, isA4d=None):
    res = {}
    if range_ is None:
        from glob import glob
        filelists = glob(join(filePath, '*.txt'))
        range_ = [i for i in range(len(filelists))]        
    if isA4d is None:
        isA4d = args.a4d
    for imgId in tqdm(range_):
        res[imgId] = _readResult(join(filePath, '{:06d}.json'.format(imgId)), isA4d)
    return res

class ShelfGT:
    def __init__(self, actor3D) -> None:
        self.actor3D = actor3D
        self.actor3D = self.actor3D[:3]

    def __getitem__(self, index):
        results = []
        for pid in range(len(self.actor3D)):
            gt_pose = self.actor3D[pid][index-2][0]
            if gt_pose.shape == (1, 0) or gt_pose.shape == (0, 0):
                continue
            keypoints3d = convert_shelf_shelfgt(gt_pose)
            results.append({'id': pid, 'keypoints3d': keypoints3d})
        return results

def write_to_csv(filename, results, id_wise=True):
    keys = [key for key in results[0].keys() if isinstance(results[0][key], float)]
    if id_wise:
        ids = list(set([res['id'] for res in results]))
    header = [''] + ['{:s}'.format(key.replace(' ', '')) for key in keys]
    contents = []
    if id_wise:
        for pid in ids:
            content = ['{}'.format(pid)]
            for key in keys:
                vals = [res[key] for res in results if res['id'] == pid]
                content.append('{:.3f}'.format(sum(vals)/len(vals)))
            contents.append(content)
        # 计算平均值
        content = ['Mean']
        for i, key in enumerate(keys):
            content.append('{:.3f}'.format(sum([float(con[i+1]) for con in contents])/len(ids)))
        contents.append(content)
    else:
        content = ['Mean']
        for key in keys:
            content.append('{:.3f}'.format(sum([res[key] for res in results])/len(results)))
        contents.append(content)
    import tabulate
    print(tabulate.tabulate(contents, header, tablefmt='fancy_grid'))
    print(tabulate.tabulate(contents, header, tablefmt='fancy_grid'), file=open(filename.replace('.csv', '.txt'), 'w'))

    with open(filename, 'w') as f:
        # 写入头
        header = list(results[0].keys())
        f.write(','.join(header) + '\n')
        for res in results:
            f.write(','.join(['{}'.format(res[key]) for key in header]) + '\n')

def evaluate(actor3D, range_, out):
    shelfgt = ShelfGT(actor3D)
    check_result = np.zeros ( (len ( actor3D[0] ), len ( actor3D ), 10), dtype=np.int32 )
    result = readResult(out, range_)
    bones = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [12, 14]]
    start = [ 9, 8, 10, 7, 3, 2, 4, 1, 12, 12,]
    end =   [10, 7, 11, 6, 4, 1, 5, 0, 13, 14]
    names = ["Left Upper Arm", "Right Upper Arm", "Left Lower Arm", "Right Lower Arm", "Left Upper Leg", "Right Upper Leg", "Left Lower Leg", "Right Lower Leg", "Head", "Torso" ]

    results = []
    for img_id in tqdm(range_):
        # 转化成model_poses
        ests = []
        for res in result[img_id]:
            ests.append({'id': res['id'], 'keypoints3d': convert_openpose_shelf1(res['keypoints3d'])})
        gts = shelfgt[img_id]
        if len(gts) < 1:
            continue
        # 匹配最近的
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
            da = np.linalg.norm(kpts_gt[start, :3] - kpts_est[start, :3], axis=1)
            db = np.linalg.norm(kpts_gt[end, :3] - kpts_est[end, :3], axis=1)
            l = np.linalg.norm(kpts_gt[start, :3] - kpts_gt[end, :3], axis=1)
            isright = 1.0*((da + db) < l)
            if args.joint:
                res = {name: isright[i] for i, name in enumerate(names)}
            else:
                res = {}
            res['Mean'] = isright.mean()
            res['nf'] = img_id
            res['id'] = data['id']
            results.append(res)
    write_to_csv(join(out, '..', 'report.csv'), results)
    return 0

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument('--out', type=str, default='output/')
    parser.add_argument('--gt_path', type=str, default='config/evaluation/actorsGT_shelf.mat')
    parser.add_argument('--setting', type=str, default='shelf')
    parser.add_argument('--a4d', action='store_true')
    parser.add_argument('--joint', action='store_true')

    args = parser.parse_args ()
    if args.setting == 'shelf':
        test_range = range ( 302, 602)
        # test_range = range (2000, 3200)
    elif args.setting == 'campus':
        test_range = [i for i in range ( 350, 471 )] + [i for i in range ( 650, 751 )]
    else:
        raise NotImplementedError

    actorsGT = scio.loadmat (args.gt_path)
    test_actor3D = actorsGT['actor3D'][0]
    if False:
        valid = np.zeros((3200, 4))
        for nf in range(3200):
            for pid in range(4):
                if test_actor3D[pid][nf].item().shape[0] == 14:
                    valid[nf, pid] = 1
        import matplotlib.pyplot as plt
        plt.plot(valid.sum(axis=1))
        plt.show()
        import ipdb; ipdb.set_trace()
    evaluate(test_actor3D, test_range, args.out)
