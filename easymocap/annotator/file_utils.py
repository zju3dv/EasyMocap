'''
  @ Date: 2021-06-09 10:16:46
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-21 16:55:26
  @ FilePath: /EasyMocapPublic/easymocap/annotator/file_utils.py
'''
import os
import json
import numpy as np
from os.path import join
import shutil
from ..mytools.file_utils import myarray2string

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if file is None:
        return 0
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

tobool = lambda x: 'true' if x else 'false'

def annot2string(data):
    out_text = []
    out_text.append('{\n')
    keysbase = ['filename', 'height', 'width', 'annots', 'isKeyframe']
    keys_other = [key for key in data.keys() if key not in keysbase]
    for key in keysbase[:-1] + keys_other + ['isKeyframe']:
        value = data[key]
        indent = 4
        if key != 'annots':
            if key == 'isKeyframe':
                res = '"{}": {}'.format(key, tobool(value))
            elif key == 'filename':
                res = '"{}": "{}",'.format(key, value.replace('\\', "\\\\"))
            elif isinstance(value, str):
                res = '"{}": "{}",'.format(key, value)
            elif isinstance(value, bool):
                res = '"{}": {},'.format(key, tobool(value))
            elif isinstance(value, int):
                res = '"{}": {},'.format(key, value)
            elif isinstance(value, np.ndarray):
                #TODO: pretty array
                res = '"{}": {},'.format(key, myarray2string(value, indent=0))
            else:
                res = '"{}": {},'.format(key, value)
            out_text.append(indent * ' ' + res+'\n')
        else:
            out_text.append(indent * ' ' + '"annots": [\n')
            for n, annot in enumerate(value):
                head = (indent + 4) * " " + "{\n"
                ind = (indent + 8) * " "
                pid = ind + '"personID": {},\n'.format(annot['personID'])
                out_text.append(head)
                out_text.append(pid)
                for ckey in ['class']:
                    if ckey not in annot.keys():
                        continue
                    info_class = ind + '"class": "{}",\n'.format(annot['class'])
                    out_text.append(info_class)
                for bkey in ['bbox', 'bbox_handl2d', 'bbox_handr2d', 'bbox_face2d']:
                    if bkey not in annot.keys():
                        continue
                    bbox = ind + '"{}": [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}],\n'.format(bkey, *annot[bkey][:5])
                    out_text.append(bbox)
                for bkey in ['keypoints', 'handl2d', 'handr2d', 'face2d', 'keypoints3d']:
                    if bkey not in annot.keys():
                        continue
                    val = np.array(annot[bkey])
                    conf = val[:, -1]
                    conf[conf<0] = 0
                    ret = myarray2string(val, fmt='%7.2f', indent=12)
                    kpts = ind + '"{}": '.format(bkey) + ret + ',\n'
                    out_text.append(kpts)
                if 'params' in annot.keys():
                    out_text.append(ind + '"params": {\n')
                    keys = list(annot['params'].keys())
                    for vkey, val in annot['params'].items():
                        val = np.array(val)
                        ret = myarray2string(val, fmt='%7.2f', indent=4*4)
                        kpts = ind + 4*' ' + '"{}": '.format(vkey) + ret
                        if vkey == keys[-1]:
                            kpts += '\n'
                        else:
                            kpts += ',\n'
                        out_text.append(kpts)
                    out_text.append(ind + '},\n')
                for rkey in ['isKeyframe']:
                    val = annot.get(rkey, False)
                    bkey = ind + '"{}": {}\n'.format(rkey, tobool(val))
                tail = (indent + 4) * " " + "}"
                if n == len(value) - 1:
                    tail += '\n'
                else:
                    tail += ',\n'
                out_text.extend([bkey, tail])
            out_text.append(indent * ' ' + '],\n')
    out_text.append('}\n')
    out_text = ''.join(out_text)
    return out_text

def save_annot(file, data):
    if file is None:
        return 0
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    if 'filename' not in data.keys():
        if data.get('isList', False):
            data = data['annots']
        save_json(file, data)
        return 0
    out_text = annot2string(data)
    print(out_text, file=open(file, 'w'))

def getFileList(root, ext='.jpg', max=-1, ret_full=False):
    files = []
    dirs = sorted(os.listdir(root))
    while len(dirs) > 0:
        path = dirs.pop()
        if path.startswith('.'):continue
        fullname = join(root, path)
        if os.path.isfile(fullname) and fullname.endswith(ext):
            if ret_full:
                files.append(fullname)
            else:
                files.append(path)
        elif os.path.isdir(fullname):
            names = sorted(os.listdir(fullname))
            if max != -1 and os.path.isfile(join(fullname, names[0])):
                names = names[:max]
            for s in names:
                newDir = join(path, s)
                dirs.append(newDir)
    files = sorted(files)
    return files

def load_annot_to_tmp(annotname):
    if annotname is None:
        return {}
    if not os.path.exists(annotname):
        dirname = os.path.dirname(annotname)
        os.makedirs(dirname, exist_ok=True)
        shutil.copyfile(annotname.replace('_tmp', ''), annotname)
    annot = read_json(annotname)
    if isinstance(annot, list):
        annot = {'annots': annot, 'isKeyframe': False, 'isList': True}
    return annot