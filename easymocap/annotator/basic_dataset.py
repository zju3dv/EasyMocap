'''
  @ Date: 2021-04-15 16:57:53
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-07-14 22:15:26
  @ FilePath: /EasyMocap/easymocap/annotator/basic_dataset.py
'''
from os.path import join
import os
import shutil
from .file_utils import getFileList

class ImageFolder:
    def __init__(self, path, sub=None, image='images', annot='annots', no_annot=False, ext='.jpg', remove_tmp=True) -> None:
        self.root = path
        self.image = image
        self.annot = annot
        self.image_root = join(path, self.image)
        self.annot_root = join(path, self.annot)
        if not os.path.exists(self.annot_root):
            no_annot = True
        self.annot_root_tmp = join(path, self.annot + '_tmp')
        if os.path.exists(self.annot_root_tmp) and remove_tmp:
            shutil.rmtree(self.annot_root_tmp)
        if sub is None:
            self.imgnames = getFileList(self.image_root, ext=ext)
            if not no_annot:
                self.annnames = getFileList(self.annot_root, ext='.json')
        else:
            self.imgnames = getFileList(join(self.image_root, sub), ext=ext)
            self.imgnames = [join(sub, name) for name in self.imgnames]
            if not no_annot:
                self.annnames = getFileList(join(self.annot_root, sub), ext='.json')
                self.annnames = [join(sub, name) for name in self.annnames]
                length = min(len(self.imgnames), len(self.annnames))
                self.imgnames = self.imgnames[:length]
                self.annnames = self.annnames[:length]
                # assert len(self.imgnames) == len(self.annnames)
        self.isTmp = True
        self.no_annot = no_annot
    
    def __getitem__(self, index):
        imgname = join(self.image_root, self.imgnames[index])
        if self.no_annot:
            annname = None
        else:
            if self.isTmp:
                annname = join(self.annot_root_tmp, self.annnames[index])
            else:
                annname = join(self.annot_root, self.annnames[index])
        return imgname, annname
    
    def __len__(self):
        return len(self.imgnames)
    
    def __str__(self) -> str:
        return '{}: {} images'.format(self.root, len(self))

class MVBase:
    def __init__(self, path, subs, annot='annots') -> None:
        self.root = path
        self.subs = subs
        self.image = 'images'
        self.annot = annot
        self.image_root = join(path, self.image)
        self.annot_root = join(path, self.annot)
        self.annot_root_tmp = join(path, self.annot + '_tmp')
        assert len(subs) > 0, subs
        self.imgnames, self.annnames = {}, {}
        for sub in subs:
            imgnames = getFileList(join(self.image_root, sub), ext='.jpg')
            annnames = getFileList(join(self.annot_root, sub), ext='.json')
            imgnames = [join(sub, name) for name in imgnames]
            annnames = [join(sub, name) for name in annnames]
            self.imgnames[sub] = imgnames
            self.annnames[sub] = annnames
        self.isTmp = True
        
    
    def __getitem__(self, index):
        imgnames, annnames = {}, {}
        for sub in self.subs:
            imgnames[sub] = join(self.image_root, self.imgnames[sub][index])
            if self.isTmp:
                annname = join(self.annot_root_tmp, self.annnames[sub][index])
            else:
                annname = join(self.annot_root, self.annnames[sub][index])
            annnames[sub] = annname
        return imgnames, annnames
    
    def __len__(self):
        return len(self.imgnames[self.subs[0]])