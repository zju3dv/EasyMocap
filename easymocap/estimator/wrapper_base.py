import os
import cv2
import numpy as np
from ..annotator.file_utils import save_annot

def check_result(image_root, annot_root):
    if os.path.exists(annot_root):
        # check the number of images and keypoints
        nimg = len(os.listdir(image_root))
        nann = len(os.listdir(annot_root))
        print('Check {} == {}'.format(nimg, nann))
        if nimg == nann:
            return True
    return False

def create_annot_file(annotname, imgname):
    assert os.path.exists(imgname), imgname
    img = cv2.imread(imgname)
    height, width = img.shape[0], img.shape[1]
    imgnamesep = imgname.split(os.sep)
    filename = os.sep.join(imgnamesep[imgnamesep.index('images'):])
    annot = {
        'filename':filename,
        'height':height,
        'width':width,
        'annots': [],
        'isKeyframe': False
    }
    save_annot(annotname, annot)
    return annot

def bbox_from_keypoints(keypoints, rescale=1.2, detection_thresh=0.05, MIN_PIXEL=5):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:,-1] > detection_thresh
    if valid.sum() < 3:
        return [0, 0, 100, 100, 0]
    valid_keypoints = keypoints[valid][:,:-1]
    center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))/2
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    if bbox_size[0] < MIN_PIXEL or bbox_size[1] < MIN_PIXEL:
        return [0, 0, 100, 100, 0]
    bbox_size = bbox_size * rescale
    bbox = [
        center[0] - bbox_size[0]/2, 
        center[1] - bbox_size[1]/2,
        center[0] + bbox_size[0]/2, 
        center[1] + bbox_size[1]/2,
        keypoints[valid, 2].mean()
    ]
    bbox = np.array(bbox).tolist()
    return bbox