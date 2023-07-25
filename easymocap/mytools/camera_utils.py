import cv2
import numpy as np
import os
from os.path import join
class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out+'\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.6f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'int':
            self._write('{}: {}'.format(key, value))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'int':
            output = int(self.fs.getNode(key).real())
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def read_intri(intri_name):
    assert os.path.exists(intri_name), intri_name
    intri = FileStorage(intri_name)
    camnames = intri.read('names', dt='list')
    cameras = {}
    for key in camnames:
        cam = {}
        cam['K'] = intri.read('K_{}'.format(key))
        cam['invK'] = np.linalg.inv(cam['K'])
        cam['dist'] = intri.read('dist_{}'.format(key))
        cameras[key] = cam
    return cameras

def write_intri(intri_name, cameras):
    if not os.path.exists(os.path.dirname(intri_name)):
        os.makedirs(os.path.dirname(intri_name))
    intri = FileStorage(intri_name, True)
    results = {}
    camnames = list(cameras.keys())
    intri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        K, dist = val['K'], val['dist']
        assert K.shape == (3, 3), K.shape
        assert dist.shape == (1, 5) or dist.shape == (5, 1) or dist.shape == (1, 4) or dist.shape == (4, 1), dist.shape
        intri.write('K_{}'.format(key), K)
        intri.write('dist_{}'.format(key), dist.flatten()[None])

def write_extri(extri_name, cameras):
    if not os.path.exists(os.path.dirname(extri_name)):
        os.makedirs(os.path.dirname(extri_name))
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = list(cameras.keys())
    extri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])
    return 0

def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format( cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        H = intri.read('H_{}'.format(cam), dt='int')
        W = intri.read('W_{}'.format(cam), dt='int')
        if H is None or W is None:
            print('[camera] no H or W for {}'.format(cam))
            H, W = -1, -1
        cams[cam]['H'] = H
        cams[cam]['W'] = W
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        assert Rvec is not None, cam
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['Rvec'] = Rvec
        cams[cam]['T'] = Tvec
        cams[cam]['center'] = - Rvec.T @ Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
        if cams[cam]['dist'] is None:
            cams[cam]['dist'] = intri.read('D_{}'.format(cam))
            if cams[cam]['dist'] is None:
                print('[camera] no dist for {}'.format(cam))
    cams['basenames'] = cam_names
    return cams

def read_cameras(path, intri='intri.yml', extri='extri.yml', subs=[]):
    cameras = read_camera(join(path, intri), join(path, extri))
    cameras.pop('basenames')
    if len(subs) > 0:
        cameras = {key:cameras[key].astype(np.float32) for key in subs}
    return cameras

def write_camera(camera, path):
    from os.path import join
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['dist'])
        if 'H' in val.keys() and 'W' in val.keys():
            intri.write('H_{}'.format(key), val['H'], dt='int')
            intri.write('W_{}'.format(key), val['W'], dt='int')
        assert val['R'].shape == (3, 3), f"{val['R'].shape} must == (3, 3)"
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])

def camera_from_img(img):
    height, width = img.shape[0], img.shape[1]
    # focal = 1.2*max(height, width) # as colmap
    focal = 1.2*min(height, width) # as colmap
    K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
    camera = {'K':K ,'R': np.eye(3), 'T': np.zeros((3, 1)), 'dist': np.zeros((1, 5))}
    camera['invK'] = np.linalg.inv(camera['K'])
    camera['P'] = camera['K'] @ np.hstack((camera['R'], camera['T']))
    return camera

class Undistort:
    distortMap = {}
    @classmethod
    def image(cls, frame, K, dist, sub=None, interp=cv2.INTER_NEAREST):
        if sub is None:
            return cv2.undistort(frame, K, dist, None)
        else:
            if sub not in cls.distortMap.keys():
                h,  w = frame.shape[:2]
                mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K, (w,h), 5)
                cls.distortMap[sub] = (mapx, mapy)
            mapx, mapy = cls.distortMap[sub]
            img = cv2.remap(frame, mapx, mapy, interp)
            return img

    @staticmethod
    def points(keypoints, K, dist):
        # keypoints: (N, 3)
        assert len(keypoints.shape) == 2, keypoints.shape
        kpts = keypoints[:, None, :2]
        kpts = np.ascontiguousarray(kpts)
        kpts = cv2.undistortPoints(kpts, K, dist, P=K)
        keypoints = np.hstack([kpts[:, 0], keypoints[:, 2:]])
        return keypoints
    
    @staticmethod
    def bbox(bbox, K, dist):
        keypoints = np.array([[bbox[0], bbox[1], 1], [bbox[2], bbox[3], 1]])
        kpts = Undistort.points(keypoints, K, dist)
        bbox = np.array([kpts[0, 0], kpts[0, 1], kpts[1, 0], kpts[1, 1], bbox[4]])
        return bbox

class Distort:
    @staticmethod
    def points(keypoints, K, dist):
        pass

    @staticmethod
    def bbox(bbox, K, dist):
        keypoints = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
        k3d = cv2.convertPointsToHomogeneous(keypoints)
        k3d = (np.linalg.inv(K) @ k3d[:, 0].T).T[:, None]
        k2d, _ = cv2.projectPoints(k3d, np.zeros((3,)), np.zeros((3,)), K, dist)
        k2d = k2d[:, 0]
        bbox = np.array([k2d[0,0], k2d[0,1], k2d[1, 0], k2d[1, 1], bbox[-1]])
        return bbox

def unproj(kpts, invK):
    homo = np.hstack([kpts[:, :2], np.ones_like(kpts[:, :1])])
    homo = homo @ invK.T
    return np.hstack([homo[:, :2], kpts[:, 2:]])
class UndistortFisheye:
    @staticmethod
    def image(frame, K, dist):
        Knew = K.copy()
        frame = cv2.fisheye.undistortImage(frame, K, dist, Knew=Knew)
        return frame, Knew

    @staticmethod
    def points(keypoints, K, dist, Knew):
        # keypoints: (N, 3)
        assert len(keypoints.shape) == 2, keypoints.shape
        kpts = keypoints[:, None, :2]
        kpts = np.ascontiguousarray(kpts)
        kpts = cv2.fisheye.undistortPoints(kpts, K, dist, P=Knew)
        keypoints = np.hstack([kpts[:, 0], keypoints[:, 2:]])
        return keypoints
    
    @staticmethod
    def bbox(bbox, K, dist, Knew):
        keypoints = np.array([[bbox[0], bbox[1], 1], [bbox[2], bbox[3], 1]])
        kpts = UndistortFisheye.points(keypoints, K, dist, Knew)
        bbox = np.array([kpts[0, 0], kpts[0, 1], kpts[1, 0], kpts[1, 1], bbox[4]])
        return bbox


def get_Pall(cameras, camnames):
    Pall = np.stack([cameras[cam]['K'] @ np.hstack((cameras[cam]['R'], cameras[cam]['T'])) for cam in camnames])
    return Pall

def get_fundamental_matrix(cameras, basenames):
    skew_op = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: np.linalg.inv(K_0).T @ (
            R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
    fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: fundamental_op (K_0, RT_0[:, :3], RT_0[:, 3], K_1,
                                                                          RT_1[:, :3], RT_1[:, 3] )
    F = np.zeros((len(basenames), len(basenames), 3, 3))  # N x N x 3 x 3 matrix
    F = {(icam, jcam): np.zeros((3, 3)) for jcam in basenames for icam in basenames}
    for icam in basenames:
        for jcam in basenames:
            F[(icam, jcam)] += fundamental_RT_op(cameras[icam]['K'], cameras[icam]['RT'], cameras[jcam]['K'], cameras[jcam]['RT'])
            if F[(icam, jcam)].sum() == 0:
                F[(icam, jcam)] += 1e-12  # to avoid nan
    return F

def interp_cameras(cameras, keys, step=20, loop=True, allstep=-1, **kwargs):
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    if allstep != -1:
        tall = np.linspace(0., 1., allstep+1)[:-1].reshape(-1, 1, 1)
    elif allstep == -1 and loop:
        tall = np.linspace(0., 1., 1+step*len(keys))[:-1].reshape(-1, 1, 1)
    elif allstep == -1 and not loop:
        tall = np.linspace(0., 1., 1+step*(len(keys)-1))[:-1].reshape(-1, 1, 1)
    cameras_new = {}
    for ik in range(len(keys)):
        if ik == len(keys) -1 and not loop:
            break
        if loop:
            start, end = (ik * tall.shape[0])//len(keys),     int((ik+1)*tall.shape[0])//len(keys)
            print(ik, start, end, tall.shape)
        else:
            start, end = (ik * tall.shape[0])//(len(keys)-1), int((ik+1)*tall.shape[0])//(len(keys)-1)
        t = tall[start:end].copy()
        t = (t-t.min())/(t.max()-t.min())
        left, right = keys[ik], keys[0 if ik == len(keys)-1 else ik + 1]
        camera_left = cameras[left]
        camera_right = cameras[right]
        # 插值相机中心: center = - R.T @ T
        center_l = - camera_left['R'].T @ camera_left['T']
        center_r = - camera_right['R'].T @ camera_right['T']
        center_l, center_r = center_l[None], center_r[None]
        if False:
            centers = center_l * (1-t) + center_r * t
        else:
            # 球面插值
            norm_l, norm_r = np.linalg.norm(center_l), np.linalg.norm(center_r)
            center_l, center_r = center_l/norm_l, center_r/norm_r
            costheta = (center_l*center_r).sum()
            sintheta = np.sqrt(1. - costheta**2)
            theta = np.arctan2(sintheta, costheta)
            centers = (np.sin(theta*(1-t)) * center_l + np.sin(theta * t) * center_r)/sintheta
            norm = norm_l * (1-t) + norm_r * t
            centers = centers * norm
        key_rots = R.from_matrix(np.stack([camera_left['R'], camera_right['R']]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(t.squeeze()).as_matrix()
        # 计算相机T RX + T = 0 => T = - R @ X
        T = - np.einsum('bmn,bno->bmo', interp_rots, centers)
        K = camera_left['K'] * (1-t) + camera_right['K'] * t
        for i in range(T.shape[0]):
            cameras_new['{}-{}-{}'.format(left, right, i)] = \
                {
                    'K': K[i],
                    'dist': np.zeros((1, 5)),
                    'R': interp_rots[i],
                    'T': T[i]
                }
    return cameras_new