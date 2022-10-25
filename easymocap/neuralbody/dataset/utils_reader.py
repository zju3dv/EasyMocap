import  cv2
import numpy as np
from ...mytools.file_utils import read_json

def img_to_numpy(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    return img

def numpy_to_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img*255).astype(np.uint8)
    return img

def read_json_with_cache(filename, file_cache):
    if filename not in file_cache.keys():
        data = read_json(filename)
        file_cache[filename] = data
    return file_cache[filename]


semantic_dict = {
    'background': 0,
    'hat': 1,
    'hair': 1,
    'glove': 1,
    'sunglasses': 1,
    'upper_cloth': 2,
    'dress': 1, #x
    'coat': 1,
    'sock': 1,
    'pant': 3,
    'jumpsuit': 1,
    'scarf': 1,
    'skirt': 1,
    'face': 1,
    'left_leg': 1,
    'right_leg': 1,
    'left_arm': 1,
    'right_arm': 1,
    'left_shoe': 1,
    'right_shoe': 1,
}

semantic_labels = list(semantic_dict.keys())
semantic_dim = len(semantic_labels)

def get_schp_palette(num_cls=256):
    # Copied from SCHP
    """ Returns the color map for visualizing the segmentation mask.
    Inputs:
        num_cls: Number of classes.
    Returns:
        The color map.
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    palette = np.array(palette, dtype=np.uint8)
    palette = palette.reshape(-1, 3)  # n_cls, 3
    return palette

palette = get_schp_palette(semantic_dim)

def parse_semantic(semantic):
    msk_cihp = (semantic * 255).astype(np.int)  # H, W, 3 
    sem_msk = np.zeros(msk_cihp.shape[:2], dtype=np.int64)
    for i, rgb in enumerate(palette):
        if i == 0:continue
        belong = np.abs(msk_cihp - rgb).sum(axis=-1) < 4
        sem_msk[belong] = semantic_dict[semantic_labels[i]]
    return sem_msk