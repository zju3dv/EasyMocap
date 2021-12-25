import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
from ..mytools import Timer

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

class Detector:
    NUM_BODY = 33
    NUM_HAND = 21
    NUM_FACE = 468
    def __init__(self, nViews, to_openpose, model_type, show=False, **cfg) -> None:
        self.nViews = nViews
        self.to_openpose = to_openpose
        self.model_type = model_type
        self.show = show
        if self.to_openpose:
            self.NUM_BODY = 25
            self.openpose25_in_33 = [0, 0, 12, 14, 16, 11, 13, 15, 0, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7, 31, 31, 29, 32, 32, 30]
        if model_type == 'holistic':
            model_name = mp_holistic.Holistic
        elif model_type == 'pose':
            model_name = mp.solutions.pose.Pose
        elif model_type == 'face':
            model_name = mp.solutions.face_mesh.FaceMesh
            cfg.pop('model_complexity')
            cfg['max_num_faces'] = 1
        elif model_type in ['hand', 'handl', 'handr']:
            model_name = mp.solutions.hands.Hands
        else:
            raise NotImplementedError
        self.models = [
            model_name(**cfg) for nv in range(nViews)
        ]
    
    @staticmethod
    def to_array(pose, W, H, start=0):
        N = len(pose.landmark) - start
        res = np.zeros((N, 3))
        for i in range(start, len(pose.landmark)):
            res[i-start, 0] = pose.landmark[i].x * W
            res[i-start, 1] = pose.landmark[i].y * H
            res[i-start, 2] = pose.landmark[i].visibility
        return res

    def get_body(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((self.NUM_BODY, 3))
            return bodies, [0, 0, 100, 100, 0]
        poses = self.to_array(pose, W, H)
        if self.to_openpose:
            poses = poses[self.openpose25_in_33]
            poses[8, :2] = poses[[9, 12], :2].mean(axis=0)
            poses[8, 2] = poses[[9, 12], 2].min(axis=0)
            poses[1, :2] = poses[[2, 5], :2].mean(axis=0)
            poses[1, 2] = poses[[2, 5], 2].min(axis=0)
        return poses, bbox_from_keypoints(poses)
    
    def get_hand(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((self.NUM_HAND, 3))
            return bodies, [0, 0, 100, 100, 0.]
        poses = self.to_array(pose, W, H)
        poses[:, 2] = 1.
        return poses, bbox_from_keypoints(poses)

    def get_face(self, pose, W, H):
        if pose is None:
            bodies = np.zeros((self.NUM_FACE, 3))
            return bodies, [0, 0, 100, 100, 0]
        poses = self.to_array(pose, W, H)
        poses[:, 2] = 1.
        return poses, bbox_from_keypoints(poses)

    def vis(self, image, annots, nv=0):
        from easymocap.mytools.vis_base import plot_keypoints
        from easymocap.dataset.config import CONFIG
        annots = annots['annots'][0]
        if 'keypoints' in annots.keys():
            kpts = annots['keypoints']
            if self.to_openpose:
                config = CONFIG['body25']
            else:
                config = CONFIG['mpbody']
            plot_keypoints(image, kpts, 0, config)
        if 'face2d' in annots.keys():
            kpts = annots['face2d']
            plot_keypoints(image, kpts, 0, CONFIG['mpface'], use_limb_color=False)
            if len(kpts) > 468:
                plot_keypoints(image, kpts[468:], 0, {'kintree': [[4, 1], [1, 2], [2, 3], [3, 4], [9, 6], [6, 7], [7, 8], [8, 9]]}, use_limb_color=False)
        if 'handl2d' in annots.keys():
            kpts = annots['handl2d']
            plot_keypoints(image, kpts, 1, CONFIG['hand'], use_limb_color=True)
        if 'handr2d' in annots.keys():
            kpts = annots['handr2d']
            plot_keypoints(image, kpts, 1, CONFIG['hand'], use_limb_color=True)
        cv2.imshow('vis{}'.format(nv), image)
        cv2.waitKey(5)

    def process_body(self, data, results, image_width, image_height):
        if self.model_type in ['pose', 'holistic']:
            keypoints, bbox = self.get_body(results.pose_landmarks, image_width, image_height)
            data['keypoints'] = keypoints
            data['bbox'] = bbox
    
    def process_hand(self, data, results, image_width, image_height):
        lm = {'Left': None, 'Right': None}
        if self.model_type in ['hand', 'handl', 'handr']:
            if results.multi_hand_landmarks:
                for i in range(len(results.multi_hand_landmarks)):
                    label = results.multi_handedness[i].classification[0].label
                    if lm[label] is not None:
                        continue
                    lm[label] = results.multi_hand_landmarks[i]
            if self.model_type == 'handl':
                lm['Right'] = None
            elif self.model_type == 'handr':
                lm['Left'] = None
        elif self.model_type == 'holistic':
            lm = {'Left': results.left_hand_landmarks, 'Right': results.right_hand_landmarks}
        if self.model_type in ['holistic', 'hand', 'handl', 'handr']:
            handl, bbox_handl = self.get_hand(lm['Left'], image_width, image_height)
            handr, bbox_handr = self.get_hand(lm['Right'], image_width, image_height)

            # flip
            if self.model_type != 'holistic':
                handl[:, 0] = image_width - handl[:, 0] - 1
                handr[:, 0] = image_width - handr[:, 0] - 1
                bbox_handl[0] = image_width - bbox_handl[0] - 1
                bbox_handl[2] = image_width - bbox_handl[2] - 1
                bbox_handr[0] = image_width - bbox_handr[0] - 1
                bbox_handr[2] = image_width - bbox_handr[2] - 1
                bbox_handl[0], bbox_handl[2] = bbox_handl[2], bbox_handl[0]
                bbox_handr[0], bbox_handr[2] = bbox_handr[2], bbox_handl[0]
            if self.model_type in ['hand', 'handl', 'holistic']:
                data['handl2d'] = handl.tolist()
                data['bbox_handl2d'] = bbox_handl
            if self.model_type in ['hand', 'handr', 'holistic']:
                data['handr2d'] = handr.tolist()
                data['bbox_handr2d'] = bbox_handr
    
    def process_face(self, data, results, image_width, image_height, image=None):
        if self.model_type == 'holistic':
            face2d, bbox_face2d = self.get_face(results.face_landmarks, image_width, image_height)
            data['face2d'] = face2d
            data['bbox_face2d'] = bbox_face2d
        elif self.model_type == 'face':
            if results.multi_face_landmarks:
                # only select the first
                face_landmarks = results.multi_face_landmarks[0]
            else:
                face_landmarks = None
            face2d, bbox_face2d = self.get_face(face_landmarks, image_width, image_height)
            data['face2d'] = face2d
            data['bbox_face2d'] = bbox_face2d

    def __call__(self, images):
        annots_all = []
        for nv, image_ in enumerate(images):
            image_height, image_width, _ = image_.shape
            image = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
            if self.model_type in ['hand', 'handl', 'handr']:
                image = cv2.flip(image, 1)
            image.flags.writeable = False
            with Timer('- detect', True):
                results = self.models[nv].process(image)
            data = {
                'personID': 0,
            }
            self.process_body(data, results, image_width, image_height)
            self.process_hand(data, results, image_width, image_height)
            with Timer('- face', True):
                self.process_face(data, results, image_width, image_height, image=image)
            annots = {
                'filename': '{}/run.jpg'.format(nv),
                'height': image_height,
                'width': image_width,
                'annots': [
                    data
                ],
                'isKeyframe': False
            }
            if self.show:
                self.vis(image_, annots, nv)
            annots_all.append(annots)
            # results.face_landmarks
        return annots_all

def extract_2d(image_root, annot_root, config, mode='holistic'):
    from .wrapper_base import check_result, save_annot
    force = config.pop('force')
    if check_result(image_root, annot_root) and not force:
        return 0
    from glob import glob
    from os.path import join
    ext = config.pop('ext')
    import os
    from tqdm import tqdm
    if mode == 'holistic' or mode == 'pose':
        to_openpose = True
    else:
        to_openpose = False
    detector = Detector(nViews=1, to_openpose=to_openpose, model_type=mode, show=False, **config)
    imgnames = sorted(glob(join(image_root, '*'+ext)))
    for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
        base = os.path.basename(imgname).replace(ext, '')
        annotname = join(annot_root, base+'.json')
        image = cv2.imread(imgname)
        annots = detector([image])[0]
        annots['filename'] = os.sep.join(imgname.split(os.sep)[-2:])
        save_annot(annotname, annots)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    path = args.path
    mp_hands = mp.solutions.hands
    from glob import glob
    from os.path import join
    imgnames = sorted(glob(join(path, '*.jpg')))
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        for imgname in imgnames:
            image = cv2.imread(imgname)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break