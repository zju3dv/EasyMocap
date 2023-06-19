# 2023.06.15
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=OMjuVQiDYJKF&uniqifier=1
# pip install -q mediapipe==0.10.0
import os
import numpy as np
import cv2
# !wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except:
    print('Please install the mediapipe by\npip install -q mediapipe==0.10.0')
    raise ModuleNotFoundError

VisionRunningMode = mp.tasks.vision.RunningMode

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
    return bbox

class MediaPipe:
    NUM_HAND = 21
    def create_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.ckpt)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=2,
                                            running_mode=VisionRunningMode.VIDEO)
        detector = vision.HandLandmarker.create_from_options(options)
        return detector

    def __init__(self, ckpt) -> None:
        if not os.path.exists(ckpt):
            cmd = 'wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            print('Cannot find {}, try to download it'.format(ckpt))
            print(cmd)
            os.system(cmd)
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            cmd = 'mv hand_landmarker.task {}'.format(os.path.dirname(ckpt))
            os.system(cmd)
        self.ckpt = ckpt
        self.detector = {}
        self.timestamp = 0
    
    @staticmethod
    def to_array(pose, W, H):
        N = len(pose)
        if N == 0:
            return np.zeros((1, 21, 3))
        res = np.zeros((N, 21, 3))
        for nper in range(N):
            for i in range(len(pose[nper])):
                res[nper, i, 0] = pose[nper][i].x * W
                res[nper, i, 1] = pose[nper][i].y * H
                res[nper, i, 2] = pose[nper][i].visibility
        res[..., 0] = W - res[..., 0] - 1
        return res

    def get_hand(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((1, self.NUM_HAND, 3))
            return bodies
        poses = self.to_array(pose, W, H)
        poses[..., 2] = 1.
        return poses
    
    def __call__(self, imgnames, images):
        squeeze = False
        if not isinstance(imgnames, list):
            imgnames = [imgnames]
            images = [images]
            squeeze = True
        # STEP 3: Load the input image.
        nViews = len(images)
        keypoints = []
        bboxes = []
        for nv in range(nViews):
            if isinstance(images[nv], str):
                images[nv] = cv2.imread(images[nv])
            sub = os.path.basename(os.path.dirname(imgnames[nv]))
            if sub not in self.detector.keys():
                self.detector[sub] = self.create_detector()
            image_ = cv2.cvtColor(images[nv], cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image_.shape
            image_ = cv2.flip(image_, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_)
            detection_result = self.detector[sub].detect_for_video(mp_image, self.timestamp)
            handl2d = self.get_hand(detection_result.hand_landmarks, image_width, image_height)
            keypoints.append(handl2d[:1])
            bboxes.append(bbox_from_keypoints(handl2d[0]))

        keypoints = np.vstack(keypoints)
        bboxes = np.stack(bboxes)
        if squeeze:
            keypoints = keypoints[0]
            bboxes = bboxes[0]
        self.timestamp += 33 # 假设30fps
        return {
            'keypoints': keypoints,
            'bbox': bboxes,
        }