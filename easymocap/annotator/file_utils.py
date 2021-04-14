import os
import json
import numpy as np
from os.path import join
import shutil

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
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

def load_annot_to_tmp(annotname):
    if not os.path.exists(annotname):
        dirname = os.path.dirname(annotname)
        os.makedirs(dirname, exist_ok=True)
        shutil.copy(annotname.replace('_tmp', ''), annotname)
    annot = read_json(annotname)
    return annot