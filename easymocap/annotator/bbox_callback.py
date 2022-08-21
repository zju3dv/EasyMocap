import numpy as np
from ..dataset.config import CONFIG

MIN_PIXEL = 50
def findNearestPoint(points, click):
    # points: (N, 2)
    # click : [x, y]
    click = np.array(click)
    if len(points.shape) == 2:
        click = click[None, :]
    elif len(points.shape) == 3:
        click = click[None, None, :]
    dist = np.linalg.norm(points - click, axis=-1)
    if dist.min() < MIN_PIXEL:
        idx = np.unravel_index(dist.argmin(), dist.shape)
        return True, idx
    else:
        return False, (-1, -1)

def callback_select_bbox_corner(start, end, annots, select, bbox_name, **kwargs):
    if start is None or end is None:
        select['corner'] = -1
        return 0
    if start[0] == end[0] and start[1] == end[1]:
        return 0
    # 判断选择了哪个角点
    annots = annots['annots']
    if len(annots) == 0:
        return 0
    # not select a bbox
    if select[bbox_name] == -1 and select['corner'] == -1:
        corners = []
        for i in range(len(annots)):
            l, t, r, b = annots[i][bbox_name][:4]
            corner = np.array([(l, t), (l, b), (r, t), (r, b), ((l+r)/2, (t+b)/2)])
            corners.append(corner)
        corners = np.stack(corners)
        flag, minid = findNearestPoint(corners, start)
        if flag:
            select[bbox_name] = minid[0]
            select['corner'] = minid[1]
        else:
            select['corner'] = -1
    # have selected a bbox, not select a corner
    elif select[bbox_name] != -1 and select['corner'] == -1:
        i = select[bbox_name]
        l, t, r, b = annots[i][bbox_name][:4]
        corners = np.array([(l, t), (l, b), (r, t), (r, b), ((l+r)/2, (t+b)/2)])
        flag, minid = findNearestPoint(corners, start)
        if flag:
            select['corner'] = minid[0]
    # have selected a bbox, and select a corner
    elif select[bbox_name] != -1 and select['corner'] != -1:
        x, y = end
        # Move the corner
        if select['corner'] < 4:
            (i, j) = [(0, 1), (0, 3), (2, 1), (2, 3)][select['corner']]
            data = annots[select[bbox_name]]
            data[bbox_name][i] = x
            data[bbox_name][j] = y
        # Move the center
        else:
            bbox = annots[select[bbox_name]][bbox_name]
            w = (bbox[2] - bbox[0])/2
            h = (bbox[3] - bbox[1])/2
            bbox[0] = x - w
            bbox[1] = y - h
            bbox[2] = x + w
            bbox[3] = y + h

    elif select[bbox_name] == -1 and select['corner'] != -1:
        select['corner'] = -1

def callback_select_bbox_center(click, annots, select, bbox_name, min_pixel=-1, **kwargs):
    if click is None:
        return 0
    if min_pixel == -1:
        min_pixel = MIN_PIXEL
    annots = annots['annots']
    if len(annots) == 0:
        return 0
    bboxes = np.array([d[bbox_name] for d in annots])
    center = (bboxes[:, [2, 3]] + bboxes[:, [0, 1]])/2
    click = np.array(click)[None, :]
    dist = np.linalg.norm(click - center, axis=1)
    mindist, minid = dist.min(), dist.argmin()
    if mindist < min_pixel:
        select[bbox_name] = minid

def get_auto_track(mode='kpts'):
    MAX_SPEED = 100
    if mode == 'bbox':
        MAX_SPEED = 0.2
    def auto_track(self, param, **kwargs):
        if self.frame == 0:
            return 0
        previous = self.previous()
        annots = param['annots']['annots']
        bbox_name = param['bbox_name']
        kpts_name = param['kpts_name']
        if len(annots) == 0:
            return 0
        if len(previous['annots']) == 0:
            return 0
        if mode == 'kpts':
            keypoints_pre = np.array([d[kpts_name] for d in previous['annots']])
            keypoints_now = np.array([d[kpts_name] for d in annots])
            conf = np.sqrt(keypoints_now[:, None, :, -1] * keypoints_pre[None, :, :, -1])
            diff = np.linalg.norm(keypoints_now[:, None, :, :2] - keypoints_pre[None, :, :, :2], axis=-1)
            dist = np.sum(diff * conf, axis=-1)/np.sum(conf, axis=-1)
        elif mode == bbox_name:
            # 计算IoU
            bbox_pre = np.array([d[bbox_name] for d in previous['annots']])
            bbox_now = np.array([d[bbox_name] for d in annots])
            bbox_pre = bbox_pre[None]
            bbox_now = bbox_now[:, None]
            areas_pre = (bbox_pre[..., 2] - bbox_pre[..., 0]) * (bbox_pre[..., 3] - bbox_pre[..., 1])
            areas_now = (bbox_now[..., 2] - bbox_now[..., 0]) * (bbox_now[..., 3] - bbox_now[..., 1])
            # 左边界的大值
            xx1 = np.maximum(bbox_pre[..., 0], bbox_now[..., 0])
            yy1 = np.maximum(bbox_pre[..., 1], bbox_now[..., 1])
            # 右边界的小值
            xx2 = np.minimum(bbox_pre[..., 2], bbox_now[..., 2])
            yy2 = np.minimum(bbox_pre[..., 3], bbox_now[..., 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            over = inter / (areas_pre + areas_now - inter)
            dist = 1 - over
            # diff = np.linalg.norm(bbox_now[:, None, :4] - bbox_pre[None, :, :4], axis=-1)
            # bbox_size = np.max(bbox_pre[:, [2, 3]] - bbox_pre[:, [0, 1]], axis=1)[None, :]
            # diff = diff / bbox_size
            # dist = diff
        else:
            raise NotImplementedError
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
        pre_ids = [d['personID'] for d in previous['annots']]
        if len(used_p) != len(pre_ids):
            param['stop'] = True
            print('>>> Stop because missing key: {}'.format(
                [i for i in pre_ids if i not in used_p]))
            print(dist)
        max_id = max(pre_ids) + 1
        for i in range(len(annots)):
            if i in used_n:
                continue
            annots[i]['personID'] = max_id
            max_id += 1
    auto_track.__doc__ = 'auto track the {}'.format(mode)
    return auto_track

def copy_previous_missing(self, param, **kwargs):
    "copy the missing person of previous frame"
    if self.frame == 0:
        return 0
    previous = self.previous()
    annots = param['annots']['annots']
    pre_ids = [d.get('personID', d.get('id')) for d in previous['annots']]
    now_ids = [d.get('personID', d.get('id')) for d in annots]
    for i in range(len(pre_ids)):
        if pre_ids[i] not in now_ids:
            annots.append(previous['annots'][i])

def copy_previous_bbox(self, param, **kwargs):
    "copy the annots of previous frame"
    if self.frame == 0:
        return 0
    previous = self.previous()
    annots = param['annots']['annots'] = previous['annots']

def create_bbox(self, param, **kwargs):
    "add new boundbox"
    start, end = param['start'], param['end']
    if start is None or end is None:
        return 0
    annots = param['annots']['annots']
    nowids = [d['personID'] for d in annots]
    bbox_name, kpts_name = param['bbox_name'], param['kpts_name']
    if len(nowids) == 0:
        maxID = 0
    else:
        maxID = max(nowids) + 1
    data = {
        'personID': maxID,
        bbox_name: [start[0], start[1], end[0], end[1], 1],
        kpts_name: [[0., 0., 0.] for _ in range(25)]
    }
    annots.append(data)
    param['start'], param['end'] = None, None

def create_bbox_mv(self, param, **kwargs):
    "add new boundbox"
    start, end = param['start'], param['end']
    if start is None or end is None:
        return 0
    nv = param['select']['camera']
    if nv == -1:
        return 0
    ranges = param['ranges']
    start = (start[0]-ranges[nv][0], start[1]-ranges[nv][1])
    end = (end[0]-ranges[nv][0], end[1]-ranges[nv][1])
    annots = param['annots'][nv]['annots']

    nowids = [d['personID'] for d in annots]
    body = param['body']
    bbox_name, kpts_name = param['bbox_name'], param['kpts_name']
    if len(nowids) == 0:
        maxID = 0
    else:
        maxID = max(nowids) + 1
    data = {
        'personID': maxID,
        bbox_name: [start[0], start[1], end[0], end[1], 1],
        kpts_name: [[0., 0., 0.] for _ in range(CONFIG[body]['nJoints'])]
    }
    annots.append(data)
    param['start'], param['end'] = None, None

def delete_bbox(self, param, **kwargs):
    "delete the person"
    bbox_name = param['bbox_name']
    active = param['select'][bbox_name]
    if active == -1:
        return 0
    else:
        param['annots']['annots'].pop(active)
        param['select'][bbox_name] = -1
    return 0

def delete_all_bbox(self, param, **kwargs):
    "delete the person"
    bbox_name = param['bbox_name']
    param['annots']['annots'] = []
    param['select'][bbox_name] = -1
    return 0

def callback_select_image(click, select, ranges, **kwargs):
    if click is None:
        return 0
    ranges = np.array(ranges)
    click = np.array(click).reshape(1, -1)
    res = (click[:, 0]>ranges[:, 0])&(click[:, 0]<ranges[:, 2])&(click[:, 1]>ranges[:, 1])&(click[:, 1]<ranges[:, 3])
    if res.any():
        select['camera'] = int(np.where(res)[0])

def callback_select_image_bbox(click, start, end, select, ranges, annots, bbox_name='bbox', **kwargs):
    if click is None:
        return 0
    ranges = np.array(ranges)
    click = np.array(click).reshape(1, -1)
    res = (click[:, 0]>ranges[:, 0])&(click[:, 0]<ranges[:, 2])&(click[:, 1]>ranges[:, 1])&(click[:, 1]<ranges[:, 3])
    if res.any():
        select['camera'] = int(np.where(res)[0])
    # 判断是否在人体bbox里面
    nv = select['camera']
    if nv == -1:
        return 0
    click_view = click[0] - ranges[nv][:2]
    callback_select_bbox_center(click_view, annots[nv], select, bbox_name, min_pixel=MIN_PIXEL*2)

def callback_move_bbox(start, end, click, select, annots, ranges, bbox_name='bbox', **kwargs):
    if start is None or end is None:
        return 0
    nv, nb = select['camera'], select[bbox_name]
    if nv == -1 or nb == -1:
        return 0
    start = (start[0]-ranges[nv][0], start[1]-ranges[nv][1])
    end = (end[0]-ranges[nv][0], end[1]-ranges[nv][1])
    annots = annots[nv]['annots']
    # 判断start是否在bbox的角点附近
    i = select[bbox_name]
    if select['corner'] == -1:
        l, t, r, b = annots[i][bbox_name][:4]
        corners = np.array([(l, t), (l, b), (r, t), (r, b), ((l+r)/2, (t+b)/2)])
        flag, minid = findNearestPoint(corners, start)
        if flag:
            select['corner'] = minid[0]
        else:
            flag, minid = findNearestPoint(corners, end)
            if flag:
                select['corner'] = minid[0]
            else:
                select['corner'] = -1
    if select['corner'] == -1:
        return 0
    x, y = end
     # Move the corner
    if select['corner'] < 4:
        (i, j) = [(0, 1), (0, 3), (2, 1), (2, 3)][select['corner']]
        data = annots[select[bbox_name]]
        data[bbox_name][i] = x
        data[bbox_name][j] = y
    # Move the center
    else:
        bbox = annots[select[bbox_name]][bbox_name]
        w = (bbox[2] - bbox[0])/2
        h = (bbox[3] - bbox[1])/2
        bbox[0] = x - w
        bbox[1] = y - h
        bbox[2] = x + w
        bbox[3] = y + h