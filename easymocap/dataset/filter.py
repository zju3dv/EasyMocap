import numpy as np

class BaseCrit:
    def __init__(self, log, **kwargs) -> None:
        self.log = log

    def __call__(self, keypoints, bbox, **kwargs) -> bool:
        return True
    
    def __str__(self) -> str:
        return "default filter"

class CritMinJoints(BaseCrit):
    def __init__(self, min_joints, log, **kwargs):
        super().__init__(log)
        self.min_joints = min_joints

    def __call__(self, keypoints, **kwargs):
        return (keypoints[:, 2] > 0.).sum() > self.min_joints
    
    def __str__(self) -> str:
        return "remove the detections less than {} joints".format(self.min_joints)

class CritWithTorso(BaseCrit):
    def __init__(self, torso_idx, min_conf, log, **kwargs) -> None:
        super().__init__(log)
        self.idx = torso_idx
        self.min_conf = min_conf
    
    def __call__(self, keypoints, bbox, **kwargs) -> bool:
        return (keypoints[self.idx, 2] > self.min_conf).all()
    
    def __str__(self) -> str:
        return "remove the human without torso {}".format(self.idx)

class CritNoBorder(BaseCrit):
    def __init__(self, rate, height, width, log) -> None:
        super().__init__(log)
        self.height = height
        self.width = width
        self.border = rate * max(self.height, self.width)
        self.leftidx =  [3, 4, 10, 11]
        self.rightidx = [6, 7, 13, 14]

    def __call__(self, keypoints, bbox, **kwargs) -> bool:
        l, t, r, b, c = bbox[:5]
        if t < self.border: # 跳过上面部分被截掉的
            pass
        if l < self.border or r > self.width - self.border:
            if self.log:print('[Crit2d]: {}'.format(' '.join(['%8.3f'%(i) for i in bbox])))
            if self.log:print('[Error] Left or right')
            dist = np.linalg.norm(keypoints[self.leftidx, :2] - keypoints[self.rightidx, :2], axis=1)
            bbox_size = b - t
            dist = dist/bbox_size
            if dist.min() < 1e-2:
                return False
            else:
                return True
        if b > self.height:
            if self.log:print('[Error] bottom')
        return True
    
    def __str__(self) -> str:
        return "remove the human in the border"

class ComposedFilter:
    def __init__(self, filters, min_conf) -> None:
        self.filters = filters
        self.min_conf = min_conf

    def __call__(self, keypoints, **kwargs) -> bool:
        conf = keypoints[:, 2]
        conf[conf<self.min_conf] = 0
        valid = conf>self.min_conf
        center = keypoints[valid, :2].mean(axis=0, keepdims=True)
        keypoints[conf<self.min_conf, :2] = center
        for filt in self.filters:
            if not filt(keypoints=keypoints, **kwargs):
                return False
        return True
    
    def nms(self, annots):
        # This function do nothing
        if len(annots) < 2:
            return annots
        keypoints = np.stack([annot['keypoints'] for annot in annots])
        bbox = np.stack([annot['bbox'] for annot in annots])
        bbox_size = np.max(np.abs(bbox[:, [1, 3]] - bbox[:, [0, 2]]), axis=1)
        bbox_size = np.maximum(bbox_size[:, None], bbox_size[None, :])
        dist = np.linalg.norm(keypoints[:, None, :, :2] - keypoints[None, :, :, :2], axis=-1)
        conf = (keypoints[:, None, :, 2] > 0) * (keypoints[None, :, :, 2] > 0)
        dist = (dist * conf).sum(axis=2)/conf.sum(axis=2)/bbox_size
        return annots

    def __str__(self) -> str:
        indent = ' ' * 4
        res = indent + 'Composed Filters: \n'
        for filt in self.filters:
            res_ = indent + indent + '{:15s}'.format(filt.__class__.__name__) + ': ' + str(filt) + '\n'
            res += res_
        return res

def make_filter(param):
    filters = []
    for key, val in param.filter.items():
        filters.append(globals()[key](log=param.log, width=param.width, height=param.height, **val))
    comp = ComposedFilter(filters, param.min_conf)
    print(comp)
    return comp

