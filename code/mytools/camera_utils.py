import cv2
import numpy as np
from tqdm import tqdm
import os

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.version = '.'.join(version.split('.')[:2])
        if isWrite:
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        else:
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        cv2.FileStorage.release(self.fs)

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            cv2.FileStorage.write(self.fs, key, value)
        elif dt == 'list':
            # import ipdb;ipdb.set_trace()
            if self.version == '4.4': # 4.4
                # self.fs.write(key, '[')
                # for elem in value:
                #     self.fs.write('none', elem)
                # self.fs.write('none', ']')
                # import ipdb; ipdb.set_trace()
                self.fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for elem in value:
                    self.fs.write('', elem)
                self.fs.endWriteStruct()
            else: # 3.4
                self.fs.write(key, '[')
                for elem in value:
                    self.fs.write('none', elem)
                self.fs.write('none', ']')

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
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def _FindChessboardCorners(img, patternSize = (9, 6)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retval, corners = cv2.findChessboardCorners(img, patternSize, 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not retval:
        return False, False
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    return True, corners

def FindChessboardCorners(image_names, patternSize=(9, 6), gridSize=0.1, debug=False, remove=False, resize_rate = 1):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    object_points = np.zeros((patternSize[1]*patternSize[0], 3), np.float32)
    object_points[:,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1,2)
    object_points = object_points * gridSize
    # Arrays to store object points and image points from all the images.
    infos = []
    for fname in tqdm(image_names, desc='detecting chessboard'):
        assert os.path.exists(fname), fname
        tmpname = fname+'.corners.txt'
        img = cv2.imread(fname)
        img = cv2.resize(img, None, fx=resize_rate, fy=resize_rate)
        if debug:
            if img.shape[0] > 1000:
                show = cv2.resize(img, None, fx=0.5, fy=0.5)
            else:
                show = img
            cv2.imshow('img', show)
            cv2.waitKey(10)
            

        if os.path.exists(tmpname) and not debug:
            ret = True
            tmp = np.loadtxt(tmpname)
            if len(tmp.shape) < 2:
                ret = False
            else:
                corners = tmp.reshape((-1, 1, 2))
                corners = np.ascontiguousarray(corners.astype(np.float32))
        else:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = _FindChessboardCorners(gray, patternSize)
            if not ret:
                ret, corners = _FindChessboardCorners(gray, patternSize, False)
                print('Not found in adaptive mode, but retry = {}'.format(ret))
            # If found, add object points, image points (after refining them)
            if ret:
                np.savetxt(tmpname, corners[:, 0])
            else:
                np.savetxt(tmpname, np.empty((0, 2), dtype=np.float32))
        if ret:
            infos.append({
                'point_object': object_points,
                'point_image': corners,
                'image': img,
                'image_name': fname
            })
            if debug:
                show = img.copy()
                show = cv2.drawChessboardCorners(show, patternSize, corners, ret)
                if show.shape[0] > 1000:
                    show = cv2.resize(show, None, fx=0.5, fy=0.5)
                cv2.imshow('img', show)
                cv2.waitKey(10)
        elif remove:
            os.remove(fname)
            os.remove(tmpname)
    return infos


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_intri(intri_name):
    assert os.path.exists(intri_name), intri_name
    intri = FileStorage(intri_name)
    camnames = intri.read('names', dt='list')
    from collections import OrderedDict
    cameras = OrderedDict()
    for key in camnames:
        cam = {}
        cam['K'] = intri.read('K_{}'.format(key))
        cam['invK'] = np.linalg.inv(cam['K'])
        cam['dist'] = intri.read('dist_{}'.format(key))
        cameras[key] = cam
    return cameras

def write_extri(camera, path, base='extri.yml'):
    extri_name = os.path.join(path, base)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        key = key_.split('.')[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])
    return 0

def read_camera(intri_name, extri_name, cam_names=[0,1,2,3]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format( cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['T'] = Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams

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
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])
        
def undistort(camera, frame=None, keypoints=None, output=None, bbox=None):
    # bbox: 1, 7
    mtx = camera['K']
    dist = camera['dist']
    if frame is not None:
        frame = cv2.undistort(frame, mtx, dist, None)
    if output is not None:
        output = cv2.undistort(output, mtx, dist, None)
    if keypoints is not None:
        for nP in range(keypoints.shape[0]):
            kpts = keypoints[nP][:, None, :2]
            kpts = np.ascontiguousarray(kpts)
            kpts = cv2.undistortPoints(kpts, mtx, dist, P=mtx)
            keypoints[nP, :, :2] = kpts[:, 0]
    if bbox is not None:
        kpts = np.zeros((2, 1, 2))
        kpts[0, 0, 0] = bbox[0]
        kpts[0, 0, 1] = bbox[1]
        kpts[1, 0, 0] = bbox[2]
        kpts[1, 0, 1] = bbox[3]
        kpts = cv2.undistortPoints(kpts, mtx, dist, P=mtx)
        bbox[0] = kpts[0, 0, 0]
        bbox[1] = kpts[0, 0, 1]
        bbox[2] = kpts[1, 0, 0]
        bbox[3] = kpts[1, 0, 1]
        return bbox
    return frame, keypoints, output

def get_bbox(points_set, H, W, thres=0.1, scale=1.2):
    bboxes = np.zeros((points_set.shape[0], 6))
    for iv in range(points_set.shape[0]):
        pose = points_set[iv, :, :]
        use_idx = pose[:,2] > thres
        if np.sum(use_idx) < 1:
            continue
        ll, rr = np.min(pose[use_idx, 0]), np.max(pose[use_idx, 0])
        bb, tt = np.min(pose[use_idx, 1]), np.max(pose[use_idx, 1])
        center = (int((ll + rr) / 2), int((bb + tt) / 2))
        length = [int(scale*(rr-ll)/2), int(scale*(tt-bb)/2)]
        l = max(0, center[0] - length[0])
        r = min(W, center[0] + length[0]) # img.shape[1]
        b = max(0, center[1] - length[1])
        t = min(H, center[1] + length[1]) # img.shape[0]
        conf = pose[:, 2].mean()
        cls_conf = pose[use_idx, 2].mean()
        bboxes[iv, 0] = l
        bboxes[iv, 1] = r
        bboxes[iv, 2] = b
        bboxes[iv, 3] = t
        bboxes[iv, 4] = conf
        bboxes[iv, 5] = cls_conf
    return bboxes

def filterKeypoints(keypoints, thres = 0.1, min_width=40, \
    min_height=40, min_area= 50000, min_count=6):
    add_list = []
    # TODO:并行化
    for ik in range(keypoints.shape[0]):
        pose = keypoints[ik]
        vis_count = np.sum(pose[:15, 2] > thres)  #TODO:
        if vis_count < min_count:
            continue
        ll, rr = np.min(pose[pose[:,2]>thres,0]), np.max(pose[pose[:,2]>thres,0])
        bb, tt = np.min(pose[pose[:,2]>thres,1]), np.max(pose[pose[:,2]>thres,1])
        center = (int((ll+rr)/2), int((bb+tt)/2))
        length = [int(1.2*(rr-ll)/2), int(1.2*(tt-bb)/2)]
        l = center[0] - length[0]
        r = center[0] + length[0]
        b = center[1] - length[1]
        t = center[1] + length[1]
        if (r - l) < min_width:
            continue
        if (t - b) < min_height:
            continue
        if (r - l)*(t - b) < min_area:
            continue
        add_list.append(ik)
    keypoints = keypoints[add_list, :, :]
    return keypoints, add_list


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
