import numpy as np
from easymocap.dataset.mirror import flipPoint2D

CONF_VANISHING_ANNOT = 2.
def clear_vanish_points(self, param):
    "remove all vanishing points"
    annots = param['annots']
    annots['vanish_line'] = [[], [], []]
    annots['vanish_point'] = [[], [], []]

def clear_body_points(self, param):
    "remove vanish lines of body"
    annots = param['annots']
    for i in range(3):
        vanish_lines = []
        for data in annots['vanish_line'][i]:
            if data[0][-1] > 1 and data[1][-1] > 1:
                vanish_lines.append(data)
        annots['vanish_line'][i] = vanish_lines
        if len(vanish_lines) > 1:
            annots['vanish_point'][i] = update_vanish_points(vanish_lines)


def calc_vanishpoint(keypoints2d, thres=0.3):
    '''
        keypoints2d: (2, N, 3)
    '''
    valid_idx = []
    for nj in range(keypoints2d.shape[1]):
        if keypoints2d[0, nj, 2] > thres and keypoints2d[1, nj, 2] > thres:
            valid_idx.append(nj)
    assert len(valid_idx) > 0, 'ATTN: cannot calculate the mirror pose'

    keypoints2d = keypoints2d[:, valid_idx]
    # weight: (N, 1)
    weight = keypoints2d[:, :, 2:].mean(axis=0)
    conf = weight.mean()
    A = np.hstack([
        keypoints2d[1, :, 1:2] - keypoints2d[0, :, 1:2],
        -(keypoints2d[1, :, 0:1] - keypoints2d[0, :, 0:1])
    ])
    b = -keypoints2d[0, :, 0:1]*(keypoints2d[1, :, 1:2] - keypoints2d[0, :, 1:2]) \
        + keypoints2d[0, :, 1:2] * (keypoints2d[1, :, 0:1] - keypoints2d[0, :, 0:1])
    b = -b
    A = A * weight
    b = b * weight
    avgInsec = np.linalg.inv(A.T @ A) @ (A.T @ b)
    result = np.zeros(3)
    result[0] = avgInsec[0, 0]
    result[1] = avgInsec[1, 0]
    result[2] = conf
    return result
    
def update_vanish_points(lines):
    vline0 = np.array(lines).transpose(1, 0, 2)
    # vline0 = np.dstack((vline0, np.ones((vline0.shape[0], vline0.shape[1], 1))))
    dim1points = vline0.copy()
    points = calc_vanishpoint(dim1points)
    return points.tolist()
    
def get_record_vanish_lines(index):
    def record_vanish_lines(self, param, **kwargs):
        "record vanish lines, X: mirror edge, Y: into mirror, Z: Up"
        annots = param['annots']
        if 'vanish_line' not in annots.keys():
            annots['vanish_line'] = [[], [], []]
        if 'vanish_point' not in annots.keys():
            annots['vanish_point'] = [[], [], []]
        start, end = param['start'], param['end']
        if start is not None and end is not None:
            annots['vanish_line'][index].append([[start[0], start[1], CONF_VANISHING_ANNOT], [end[0], end[1], CONF_VANISHING_ANNOT]])
            # 更新vanish point
            param['start'] = None
            param['end'] = None
        if len(annots['vanish_line'][index]) > 1:
            for val in annots['vanish_line'][index]:
                if len(val[0]) == 2:
                    val[0].append(CONF_VANISHING_ANNOT)
                    val[1].append(CONF_VANISHING_ANNOT)
            annots['vanish_point'][index] = update_vanish_points(annots['vanish_line'][index])
    func = record_vanish_lines
    text = ['parallel to mirror edges', 'vertical to mirror', 'vertical to ground']
    func.__doc__ = 'vanish line of ' + text[index]
    return record_vanish_lines

def vanish_point_from_body(self, param, **kwargs):
    "calculating the vanish point from human keypoints"
    annots = param['annots']
    bodies = annots['annots']
    if len(bodies) < 2:
        return 0
    assert len(bodies) == 2, 'Please make sure that there are only two bboxes!'
    kpts0 = np.array(bodies[0]['keypoints'])
    kpts1 = flipPoint2D(np.array(bodies[1]['keypoints']))
    vanish_line = annots['vanish_line'][1] # the y-dim
    MIN_CONF = 0.5
    for i in range(15):
        conf = min(kpts0[i, -1], kpts1[i, -1])
        if kpts0[i, -1] > MIN_CONF and kpts1[i, -1] > MIN_CONF:
            vanish_line.append([[kpts0[i, 0], kpts0[i, 1], conf], [kpts1[i, 0], kpts1[i, 1], conf]])
    if len(vanish_line) > 1:
        annots['vanish_point'][1] = update_vanish_points(vanish_line)

def copy_edges(self, param, **kwargs):
    "copy the static edges from previous frame"
    if self.frame == 0:
        return 0
    previous = self.previous()
    annots = param['annots']

    # copy the vanish points
    vanish_lines_pre = previous['vanish_line']
    vanish_lines = param['annots']['vanish_line']
    for i in range(3):
        vanish_lines[i] = []
        for data in vanish_lines_pre[i]:
            if data[0][-1] > 1 and data[1][-1] > 1:
                vanish_lines[i].append(data)
        if len(vanish_lines[i]) > 1:
            annots['vanish_point'][i] = update_vanish_points(vanish_lines[i])

def get_calc_intrinsic(mode='xy'):
    def calc_intrinsic(self, param, **kwargs):
        "calculating intrinsic matrix according to vanish points"
        annots = param['annots']
        if mode == 'xy':
            point0 = annots['vanish_point'][0]
            point1 = annots['vanish_point'][1]
        elif mode == 'yz':
            point0 = annots['vanish_point'][1]
            point1 = annots['vanish_point'][2]
        else:
            import ipdb; ipdb.set_trace()
        if len(point0) < 1 or len(point1) < 1:
            return 0
        vanish_point = np.stack([np.array(point0), np.array(point1)])
        K = np.eye(3)
        H = annots['height']
        W = annots['width']
        K = np.eye(3)
        K[0, 2] = W/2
        K[1, 2] = H/2
        print(vanish_point)
        vanish_point[:, 0] -= W/2
        vanish_point[:, 1] -= H/2
        print(vanish_point)
        focal = np.sqrt(-(vanish_point[0][0]*vanish_point[1][0] + vanish_point[0][1]*vanish_point[1][1]))
        
        K[0, 0] = focal
        K[1, 1] = focal
        annots['K'] = K.tolist()
        print('>>> estimated K: ')
        print(K)
    calc_intrinsic.__doc__ = 'calculate K with {}'.format(mode)
    return calc_intrinsic