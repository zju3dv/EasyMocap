'''
  @ Date: 2021-06-10 15:39:55
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-10 16:03:13
  @ FilePath: /EasyMocap/easymocap/annotator/keypoints_keyboard.py
'''
import numpy as np

def set_unvisible(self, param, **kwargs):
    "set the selected joints unvisible"
    bbox_name, kpts_name = param['bbox_name'], param['kpts_name']
    select = param['select']
    if select[bbox_name] == -1:
        return 0
    if select['joints'] == -1:
        return 0
    param['annots']['annots'][select[bbox_name]][kpts_name][select['joints']][-1] = 0.

def set_unvisible_according_previous(self, param, **kwargs):
    "set the selected joints unvisible if previous unvisible"
    previous = self.previous()
    select = param['select']
    bbox_name, kpts_name = param['bbox_name'], param['kpts_name']
    if select[bbox_name] == -1:
        return 0
    pid = param['annots']['annots'][select[bbox_name]]['personID']
    kpts_now = param['annots']['annots'][select[bbox_name]][kpts_name]
    for annots in previous['annots']:
        if annots['personID'] == pid:
            kpts_old = annots[kpts_name]
            for nj in range(len(kpts_old)):
                kpts_now[nj][2] = min(kpts_old[nj][2], kpts_now[nj][2])

def set_face_unvisible(self, param, **kwargs):
    "set the face unvisible"
    select = param['select']
    bbox_name, kpts_name = param['bbox_name'], param['kpts_name']
    if select[bbox_name] == -1:
        return 0
    for i in [15, 16, 17, 18]:
        param['annots']['annots'][select[bbox_name]][kpts_name][i][-1] = 0.

def mirror_keypoints2d(self, param, **kwargs):
    "mirror the keypoints2d"
    select = param['select']
    bbox_name, kpts_name = param['bbox_name'], param['kpts_name']
    if select[bbox_name] == -1:
        return 0
    kpts = param['annots']['annots'][select[bbox_name]][kpts_name]
    for pairs in [[(2, 5), (3, 6), (4, 7)], [(15, 16), (17, 18)], [(9, 12), (10, 13), (11, 14), (21, 24), (19, 22), (20, 23)]]:
        for i, j in pairs:
            kpts[i], kpts[j] = kpts[j], kpts[i]

def mirror_keypoints2d_leg(self, param, **kwargs):
    "mirror the keypoints2d of legs and feet"
    select = param['select']
    bbox_name, kpts_name = param['bbox_name'], param['kpts_name']
    if select[bbox_name] == -1:
        return 0
    kpts = param['annots']['annots'][select[bbox_name]][kpts_name]
    for pairs in [[(9, 12), (10, 13), (11, 14), (21, 24), (19, 22), (20, 23)]]:
        for i, j in pairs:
            kpts[i], kpts[j] = kpts[j], kpts[i]

def check_track(self, param):
    "check the tracking keypoints"
    if self.frame == 0:
        return 0
    bbox_name, kpts_name = param['bbox_name'], param['kpts_name']
    annots_pre = self.previous()['annots']
    annots = param['annots']['annots']
    if len(annots) == 0 or len(annots_pre) == 0 or len(annots) != len(annots_pre):
        param['stop'] = True
        return 0
    for data in annots:
        for data_pre in annots_pre:
            if data_pre['personID'] != data['personID']:
                continue
            l, t, r, b, c = data_pre[bbox_name][:5]
            bbox_size = max(r-l, b-t)
            keypoints_now = np.array(data[kpts_name])
            keypoints_pre = np.array(data_pre[kpts_name])
            conf = np.sqrt(keypoints_now[:, -1] * keypoints_pre[:, -1])
            diff = np.linalg.norm(keypoints_now[:, :2] - keypoints_pre[:, :2], axis=-1)
            dist = np.sum(diff * conf, axis=-1)/np.sum(conf, axis=-1)/bbox_size
            print('{}: {:.2f}'.format(data['personID'], dist))
            if dist > 0.05:
                param['stop'] = True