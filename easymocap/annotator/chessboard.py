'''
  @ Date: 2021-04-13 16:14:36
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-25 20:56:26
  @ FilePath: /EasyMocapRelease/easymocap/annotator/chessboard.py
'''
import numpy as np
import cv2
from func_timeout import func_set_timeout

def getChessboard3d(pattern, gridSize, axis='xy'):
    object_points = np.zeros((pattern[1]*pattern[0], 3), np.float32)
    # 注意：这里为了让标定板z轴朝上，设定了短边是x，长边是y
    object_points[:,:2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1,2)
    object_points[:, [0, 1]] = object_points[:, [1, 0]]
    object_points = object_points * gridSize
    if axis == 'zx':
        object_points = object_points[:, [1, 2, 0]]
    return object_points

colors_chessboard_bar = [
    [0, 0, 255],
    [0, 128, 255],
    [0, 200, 200],
    [0, 255, 0],
    [200, 200, 0],
    [255, 0, 0],
    [255, 0, 250]
]

def get_lines_chessboard(pattern=(9, 6)):
    w, h = pattern[0], pattern[1]
    lines = []
    lines_cols = []
    for i in range(w*h-1):
        lines.append([i, i+1])
        lines_cols.append(colors_chessboard_bar[(i//w)%len(colors_chessboard_bar)])
    return lines, lines_cols

def _findChessboardCorners(img, pattern, debug):
    "basic function"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retval, corners = cv2.findChessboardCorners(img, pattern, 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
    if not retval:
        return False, None
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    corners = corners.squeeze()
    return True, corners

def _findChessboardCornersAdapt(img, pattern, debug):
    "Adapt mode"
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, 21, 2)
    return _findChessboardCorners(img, pattern, debug)

@func_set_timeout(5)
def findChessboardCorners(img, annots, pattern, debug=False):
    conf = sum([v[2] for v in annots['keypoints2d']])
    if annots['visited'] and conf > 0:
        return True
    elif annots['visited']:
        return None
    annots['visited'] = True
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    for func in [_findChessboardCorners, _findChessboardCornersAdapt]:
        ret, corners = func(gray, pattern, debug)
        if ret:break
    else:
        return None
    # found the corners
    show = img.copy()
    show = cv2.drawChessboardCorners(show, pattern, corners, ret)
    assert corners.shape[0] == len(annots['keypoints2d'])
    corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
    annots['keypoints2d'] = corners.tolist()
    return show

def create_chessboard(path, keypoints3d, out='annots'):
    from tqdm import tqdm
    from os.path import join
    from .file_utils import getFileList, save_json, read_json
    import os
    keypoints2d = np.zeros((keypoints3d.shape[0], 3))
    imgnames = getFileList(join(path, 'images'), ext='.jpg', max=1)
    imgnames = [join('images', i) for i in imgnames]
    template = {
        'keypoints3d': keypoints3d.tolist(),
        'keypoints2d': keypoints2d.tolist(),
        'visited': False
    }
    for imgname in tqdm(imgnames, desc='create template chessboard'):
        annname = imgname.replace('images', out).replace('.jpg', '.json')
        annname = join(path, annname)
        if not os.path.exists(annname):
            save_json(annname, template)
        elif True:
            annots = read_json(annname)
            annots['keypoints3d'] = template['keypoints3d']
            save_json(annname, annots)


def detect_charuco(image, aruco_type, long, short, squareLength, aruco_len):
    ARUCO_DICT = {
        "4X4_50": cv2.aruco.DICT_4X4_50,
        "4X4_100": cv2.aruco.DICT_4X4_100,
        "5X5_100": cv2.aruco.DICT_5X5_100,
        "5X5_250": cv2.aruco.DICT_5X5_250,
    }
    # 创建ChArUco标定板
    dictionary = cv2.aruco.getPredefinedDictionary(dict=ARUCO_DICT[aruco_type])
    board = cv2.aruco.CharucoBoard_create(
        squaresY=long,
        squaresX=short,
        squareLength=squareLength,
        markerLength=aruco_len,
        dictionary=dictionary,
    )
    corners = board.chessboardCorners
    # ATTN: exchange the XY
    corners3d = corners[:, [1, 0, 2]]
    keypoints2d = np.zeros_like(corners3d)
    # 查找标志块的左上角点
    corners, ids, _ = cv2.aruco.detectMarkers(
        image=image, dictionary=dictionary, parameters=None
    )
    # 棋盘格黑白块内角点
    if ids is not None:
        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=image, board=board
        )
        if retval:
            ids = charucoIds[:, 0]
            pts = charucoCorners[:, 0]
            keypoints2d[ids, :2] = pts
            keypoints2d[ids, 2] = 1.
    else:
        retval = False
    return retval, keypoints2d, corners3d

class CharucoBoard:
    def __init__(self, long, short, squareLength, aruco_len, aruco_type) -> None:    
        '''
            short,long 分别表示短边、长边的格子数.
            squareLength,aruco_len 分别表示棋盘格的边长与aruco的边长.
            aruco_type 表示Aruco的类型 4X4表示aruco中的白色格子是4x4的 _50表示aruco字典中有多少种aruco.
        '''
        # 定义现有的Aruco类型
        self.ARUCO_DICT = {
            "4X4_50": cv2.aruco.DICT_4X4_50,
            "4X4_100": cv2.aruco.DICT_4X4_100,
            "5X5_100": cv2.aruco.DICT_5X5_100,
            "5X5_250": cv2.aruco.DICT_5X5_250,
        }
        # 创建ChArUco标定板
        dictionary = cv2.aruco.getPredefinedDictionary(dict=self.ARUCO_DICT[aruco_type])
        board = cv2.aruco.CharucoBoard_create(
            squaresY=long,
            squaresX=short,
            squareLength=squareLength,
            markerLength=aruco_len,
            dictionary=dictionary,
        )
        corners = board.chessboardCorners
        # ATTN: exchange the XY
        corners = corners[:, [1, 0, 2]]
        self.template = {
            'keypoints3d': corners,
            'keypoints2d': np.zeros_like(corners),
            'pattern': (long-1, short-1),
            'grid_size': squareLength,
            'visted': False
        }
        print(corners.shape)
        self.dictionary = dictionary
        self.board = board
    
    def detect(self, img_color, annots):
        # 查找标志块的左上角点
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=img_color, dictionary=self.dictionary, parameters=None
        )
        # 棋盘格黑白块内角点
        if ids is not None:
            retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=img_color, board=self.board
            )
        else:
            retval = False
        if retval:
            # 绘制棋盘格黑白块内角点
            cv2.aruco.drawDetectedCornersCharuco(
                img_color, charucoCorners, charucoIds, [0, 0, 255]
            )
            if False:
                cv2.aruco.drawDetectedMarkers(
                    image=img_color, corners=corners, ids=ids, borderColor=None
                )

            ids = charucoIds[:, 0]
            pts = charucoCorners[:, 0]
            annots['keypoints2d'][ids, :2] = pts
            annots['keypoints2d'][ids, 2] = 1.
            # if args.show:
            #     img_color = cv2.resize(img_color, None, fx=0.5, fy=0.5)
            #     cv2.imshow('vis', img_color)
            #     cv2.waitKey(0)
            # visname = imgname.replace(images, output)
            # os.makedirs(os.path.dirname(visname), exist_ok=True)
            # cv2.imwrite(visname, img_color)
        else:
            # mywarn('Cannot find in {}'.format(imgname))
            pass
        
    def __call__(self, imgname, images='images', output='output'):
        import os
        from .file_utils import read_json, save_json
        import copy
        img_color = cv2.imread(imgname)
        annotname = imgname.replace('images', 'chessboard').replace('.jpg', '.json')
        if os.path.exists(annotname):
            annots = read_json(annotname)
            if annots['visited']:
                return
        else:
            annots = copy.deepcopy(self.template)
        annots['visited'] = True
        self.detect(img_color, annots)
        annots['keypoints2d'] = annots['keypoints2d'].tolist()
        annots['keypoints3d'] = annots['keypoints3d'].tolist()
        save_json(annotname, annots)