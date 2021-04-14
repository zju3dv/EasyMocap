import numpy as np
import cv2

MIN_PIXEL = 50
def callback_select_bbox_corner(start, end, annots, select, **kwargs):
    if start is None or end is None:
        select['corner'] = -1
        return 0
    if start[0] == end[0] and start[1] == end[1]:
        return 0
    # 判断选择了哪个角点
    annots = annots['annots']
    start = np.array(start)[None, :]
    if select['bbox'] == -1 and select['corner'] == -1:
        for i in range(len(annots)):
            l, t, r, b = annots[i]['bbox'][:4]
            corners = np.array([(l, t), (l, b), (r, t), (r, b)])
            dist = np.linalg.norm(corners - start, axis=1)
            mindist = dist.min()
            if mindist < MIN_PIXEL:
                mincor = dist.argmin()
                select['bbox'] = i
                select['corner'] = mincor
                break
        else:
            select['corner'] = -1
    elif select['bbox'] != -1 and select['corner'] == -1:
        i = select['bbox']
        l, t, r, b = annots[i]['bbox'][:4]
        corners = np.array([(l, t), (l, b), (r, t), (r, b)])
        dist = np.linalg.norm(corners - start, axis=1)
        mindist = dist.min()
        if mindist < MIN_PIXEL:
            mincor = dist.argmin()
            select['corner'] = mincor
    elif select['bbox'] != -1 and select['corner'] != -1:
        # Move the corner
        x, y = end
        (i, j) = [(0, 1), (0, 3), (2, 1), (2, 3)][select['corner']]
        data = annots[select['bbox']]
        data['bbox'][i] = x
        data['bbox'][j] = y
    elif select['bbox'] == -1 and select['corner'] != -1:
        select['corner'] = -1

def callback_select_bbox_center(click, annots, select, **kwargs):
    if click is None:
        return 0
    annots = annots['annots']
    bboxes = np.array([d['bbox'] for d in annots])
    center = (bboxes[:, [2, 3]] + bboxes[:, [0, 1]])/2
    click = np.array(click)[None, :]
    dist = np.linalg.norm(click - center, axis=1)
    mindist, minid = dist.min(), dist.argmin()
    if mindist < MIN_PIXEL:
        select['bbox'] = minid

def auto_pose_track(self, param, **kwargs):
    "auto tracking with poses"
    MAX_SPEED = 100
    if self.frame == 0:
        return 0
    previous = self.previous()
    annots = param['annots']['annots']
    keypoints_pre = np.array([d['keypoints'] for d in previous['annots']])
    keypoints_now = np.array([d['keypoints'] for d in annots])
    conf = np.sqrt(keypoints_now[:, None, :, -1] * keypoints_pre[None, :, :, -1])
    diff = np.linalg.norm(keypoints_now[:, None, :, :] - keypoints_pre[None, :, :, :], axis=-1)
    dist = np.sum(diff * conf, axis=-1)/np.sum(conf, axis=-1)
    nows, pres = np.where(dist < MAX_SPEED)
    edges = []
    for n, p in zip(nows, pres):
        edges.append((n, p, dist[n, p]))
    edges.sort(key=lambda x:x[2])
    used_n, used_p = [], []
    for n, p, _ in edges:
        if n in used_n or p in used_p:
            continue
        annots[n]['personID'] = previous['annots'][p]['personID']
        used_n.append(n)
        used_p.append(p)
    # TODO:stop when missing
    