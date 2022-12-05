'''
  @ Date: 2021-03-15 12:23:12
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-11-08 21:43:37
  @ FilePath: /EasyMocapPublic/easymocap/mytools/file_utils.py
'''
import os
import json
import numpy as np
from os.path import join

mkdir = lambda x:os.makedirs(x, exist_ok=True)
# mkout = lambda x:mkdir(os.path.dirname(x)) if x is not None
def mkout(x):
    if x is not None:
        mkdir(os.path.dirname(x))
def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        try:
            data = json.load(f)
        except:
            print('Reading error {}'.format(path))
            data = []
    return data

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def save_numpy_dict(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    res = {}
    for key, val in data.items():
        res[key] = val.tolist()
    with open(file, 'w') as f:
        json.dump(res, f, indent=4)

def read_numpy_dict(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    for key, val in data.items():
        data[key] = np.array(val, dtype=np.float32)
    return data

def append_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    if os.path.exists(file):
        res = read_json(file)
        assert isinstance(res, list)
        res.append(data)
        data = res
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

save_annot = save_json

def getFileList(root, ext='.jpg'):
    files = []
    dirs = os.listdir(root)
    while len(dirs) > 0:
        path = dirs.pop()
        fullname = join(root, path)
        if os.path.isfile(fullname) and fullname.endswith(ext):
            files.append(path)
        elif os.path.isdir(fullname):
            for s in os.listdir(fullname):
                newDir = join(path, s)
                dirs.append(newDir)
    files = sorted(files)
    return files

def read_annot(annotname, mode='body25'):
    data = read_json(annotname)
    if not isinstance(data, list):
        data = data['annots']
    for i in range(len(data)):
        if 'id' not in data[i].keys():
            data[i]['id'] = data[i].pop('personID')
        if 'keypoints2d' in data[i].keys() and 'keypoints' not in data[i].keys():
            data[i]['keypoints'] = data[i].pop('keypoints2d')
        for key in ['bbox', 'keypoints', 
            'bbox_handl2d', 'handl2d', 
            'bbox_handr2d', 'handr2d', 
            'bbox_face2d', 'face2d']:
            if key not in data[i].keys():continue
            data[i][key] = np.array(data[i][key])
            if key == 'face2d':
                # TODO: Make parameters, 17 is the offset for the eye brows,
                # etc. 51 is the total number of FLAME compatible landmarks
                data[i][key] = data[i][key][17:17+51, :]
        if 'bbox' in data[i].keys():
            data[i]['bbox'] = data[i]['bbox'][:5]
            if data[i]['bbox'][-1] < 0.001:
                print('{}/{} bbox conf = 0, may be error'.format(annotname, i))
                data[i]['bbox'][-1] = 0
        # combine the basic results
        if mode == 'body25':
            data[i]['keypoints'] = data[i].get('keypoints', np.zeros((25, 3)))
        elif mode == 'body15':
            data[i]['keypoints'] = data[i]['keypoints'][:15, :]
        elif mode in ['handl', 'handr']:
            data[i]['keypoints'] = np.array(data[i][mode+'2d']).astype(np.float32)
            key = 'bbox_'+mode+'2d'
            if key not in data[i].keys():
                data[i]['bbox'] = np.array(get_bbox_from_pose(data[i]['keypoints'])).astype(np.float32)
            else:
                data[i]['bbox'] = data[i]['bbox_'+mode+'2d'][:5]
        elif mode == 'total':
            data[i]['keypoints'] = np.vstack([data[i][key] for key in ['keypoints', 'handl2d', 'handr2d', 'face2d']])
        elif mode == 'bodyhand':
            data[i]['keypoints'] = np.vstack([data[i][key] for key in ['keypoints', 'handl2d', 'handr2d']])
        elif mode == 'bodyhandface':
            data[i]['keypoints'] = np.vstack([data[i][key] for key in ['keypoints', 'handl2d', 'handr2d', 'face2d']])
        conf = data[i]['keypoints'][..., -1]
        conf[conf<0] = 0
    data.sort(key=lambda x:x['id'])
    return data

def array2raw(array, separator=' ', fmt='%.3f'):
    assert len(array.shape) == 2, 'Only support MxN matrix, {}'.format(array.shape)
    res = []
    for data in array:
        res.append(separator.join([fmt%(d) for d in data]))
    
    
def myarray2string(array, separator=', ', fmt='%7.7f', indent=8):
    assert len(array.shape) == 2, 'Only support MxN matrix, {}'.format(array.shape)
    blank = ' ' * indent
    res = ['[']
    for i in range(array.shape[0]):
        res.append(blank + '  ' + '[{}]'.format(separator.join([fmt%(d) for d in array[i]])))
        if i != array.shape[0] -1:
            res[-1] += ', '
    res.append(blank + ']')
    return '\r\n'.join(res)

def write_common_results(dumpname=None, results=[], keys=[], fmt='%2.3f'):
    format_out = {'float_kind':lambda x: fmt % x}
    out_text = []
    out_text.append('[\n')
    for idata, data in enumerate(results):
        out_text.append('    {\n')
        output = {}
        output['id'] = data['id']
        for k in ['type']:
            if k in data.keys():output[k] = '\"{}\"'.format(data[k])
        keys_current = [k for k in keys if k in data.keys()]
        for key in keys_current:
            # BUG: This function will failed if the rows of the data[key] is too large
            # output[key] = np.array2string(data[key], max_line_width=1000, separator=', ', formatter=format_out)
            output[key] = myarray2string(data[key], separator=', ', fmt=fmt)
        for key in output.keys():
            out_text.append('        \"{}\": {}'.format(key, output[key]))
            if key != keys_current[-1]:
                out_text.append(',\n')
            else:
                out_text.append('\n')
        out_text.append('    }')
        if idata != len(results) - 1:
            out_text.append(',\n')
        else:
            out_text.append('\n')
    out_text.append(']\n')
    if dumpname is not None:
        mkout(dumpname)
        with open(dumpname, 'w') as f:
            f.writelines(out_text)
    else:
        return ''.join(out_text)

def write_keypoints3d(dumpname, results, keys = ['keypoints3d']):
    # TODO:rewrite it
    write_common_results(dumpname, results, keys, fmt='%6.7f')

def write_vertices(dumpname, results):
    keys = ['vertices']
    write_common_results(dumpname, results, keys, fmt='%6.5f')

def write_smpl(dumpname, results):
    keys = ['Rh', 'Th', 'poses', 'handl', 'handr', 'expression', 'shapes']
    write_common_results(dumpname, results, keys)

def batch_bbox_from_pose(keypoints2d, height, width, rate=0.1):
    # TODO:write this in batch
    bboxes = np.zeros((keypoints2d.shape[0], 5), dtype=np.float32)
    border = 20
    for bn in range(keypoints2d.shape[0]):
        valid = keypoints2d[bn, :, -1] > 0
        if valid.sum() == 0:
            continue
        p2d = keypoints2d[bn, valid, :2]
        x_min, y_min = p2d.min(axis=0)
        x_max, y_max = p2d.max(axis=0)
        x_mean, y_mean = p2d.mean(axis=0)
        if x_mean < -border or y_mean < -border or x_mean > width + border or y_mean > height + border:
            continue
        dx = (x_max - x_min)*rate
        dy = (y_max - y_min)*rate
        bboxes[bn] = [x_min-dx, y_min-dy, x_max+dx, y_max+dy, 1]
    return bboxes

def get_bbox_from_pose(pose_2d, img=None, rate = 0.1):
    # this function returns bounding box from the 2D pose
    # here use pose_2d[:, -1] instead of pose_2d[:, 2]
    # because when vis reprojection, the result will be (x, y, depth, conf)
    validIdx = pose_2d[:, -1] > 0
    if validIdx.sum() == 0:
        return [0, 0, 100, 100, 0]
    y_min = int(min(pose_2d[validIdx, 1]))
    y_max = int(max(pose_2d[validIdx, 1]))
    x_min = int(min(pose_2d[validIdx, 0]))
    x_max = int(max(pose_2d[validIdx, 0]))
    dx = (x_max - x_min)*rate
    dy = (y_max - y_min)*rate
    # 后面加上类别这些
    bbox = [x_min-dx, y_min-dy, x_max+dx, y_max+dy, 1]
    if img is not None:
        correct_bbox(img, bbox)
    return bbox

def correct_bbox(img, bbox):
    # this function corrects the bbox, which is out of image
    w = img.shape[0]
    h = img.shape[1]
    if bbox[2] <= 0 or bbox[0] >= h or bbox[1] >= w or bbox[3] <= 0:
        bbox[4] = 0
    return bbox

def merge_params(param_list, share_shape=True):
    output = {}
    for key in ['poses', 'shapes', 'Rh', 'Th', 'expression']:
        if key in param_list[0].keys():
            output[key] = np.vstack([v[key] for v in param_list])
    if share_shape:
        output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
    return output

def select_nf(params_all, nf):
    output = {}
    for key in ['poses', 'Rh', 'Th']:
        output[key] = params_all[key][nf:nf+1, :]
    if 'expression' in params_all.keys():
        output['expression'] = params_all['expression'][nf:nf+1, :]
    if params_all['shapes'].shape[0] == 1:
        output['shapes'] = params_all['shapes']
    else:
        output['shapes'] = params_all['shapes'][nf:nf+1, :]
    return output