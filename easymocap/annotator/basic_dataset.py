from os.path import join
from .file_utils import getFileList

class ImageFolder:
    def __init__(self, path, sub=None, annot='annots') -> None:
        self.root = path
        self.image = 'images'
        self.annot = annot
        self.image_root = join(path, self.image)
        self.annot_root = join(path, self.annot)
        self.annot_root_tmp = join(path, self.annot + '_tmp')
        if sub is None:
            self.imgnames = getFileList(self.image_root, ext='.jpg')
            self.annnames = getFileList(self.annot_root, ext='.json')
        else:
            self.imgnames = getFileList(join(self.image_root, sub), ext='.jpg')
            self.annnames = getFileList(join(self.annot_root, sub), ext='.json')
            self.imgnames = [join(sub, name) for name in self.imgnames]
            self.annnames = [join(sub, name) for name in self.annnames]
        self.isTmp = True
        assert len(self.imgnames) == len(self.annnames)
    
    def __getitem__(self, index):
        imgname = join(self.image_root, self.imgnames[index])
        if self.isTmp:
            annname = join(self.annot_root_tmp, self.annnames[index])
        else:
            annname = join(self.annot_root, self.annnames[index])
        return imgname, annname
    
    def __len__(self):
        return len(self.imgnames)