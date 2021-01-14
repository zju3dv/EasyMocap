'''
 * @ Date: 2020-09-26 16:52:55
 * @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-13 14:04:46
  @ FilePath: /EasyMocap/code/dataset/config.py
'''
import numpy as np

CONFIG = {}

CONFIG['body25'] = {'kintree':
   [[ 1,  0],
    [ 2,  1],
    [ 3,  2],
    [ 4,  3],
    [ 5,  1],
    [ 6,  5],
    [ 7,  6],
    [ 8,  1],
    [ 9,  8],
    [10,  9],
    [11, 10],
    [12,  8],
    [13, 12],
    [14, 13],
    [15,  0],
    [16,  0],
    [17, 15],
    [18, 16],
    [19, 14],
    [20, 19],
    [21, 14],
    [22, 11],
    [23, 22],
    [24, 11]]}

CONFIG['body15'] = {'kintree':
   [[ 1,  0],
    [ 2,  1],
    [ 3,  2],
    [ 4,  3],
    [ 5,  1],
    [ 6,  5],
    [ 7,  6],
    [ 8,  1],
    [ 9,  8],
    [10,  9],
    [11, 10],
    [12,  8],
    [13, 12],
    [14, 13],]}
    
CONFIG['hand'] = {'kintree':
      [[ 1,  0],
       [ 2,  1],
       [ 3,  2],
       [ 4,  3],
       [ 5,  0],
       [ 6,  5],
       [ 7,  6],
       [ 8,  7],
       [ 9,  0],
       [10,  9],
       [11, 10],
       [12, 11],
       [13,  0],
       [14, 13],
       [15, 14],
       [16, 15],
       [17,  0],
       [18, 17],
       [19, 18],
       [20, 19]]
}

CONFIG['bodyhand'] = {'kintree':
   [[ 1,  0],
    [ 2,  1],
    [ 3,  2],
    [ 4,  3],
    [ 5,  1],
    [ 6,  5],
    [ 7,  6],
    [ 8,  1],
    [ 9,  8],
    [10,  9],
    [11, 10],
    [12,  8],
    [13, 12],
    [14, 13],
    [15,  0],
    [16,  0],
    [17, 15],
    [18, 16],
    [19, 14],
    [20, 19],
    [21, 14],
    [22, 11],
    [23, 22],
    [24, 11],
    [26, 25],  # handl
    [27, 26],
    [28, 27],
    [29, 28],
    [30, 25],
    [31, 30],
    [32, 31],
    [33, 32],
    [34, 25],
    [35, 34],
    [36, 35],
    [37, 36],
    [38, 25],
    [39, 38],
    [40, 39],
    [41, 40],
    [42, 25],
    [43, 42],
    [44, 43],
    [45, 44],
    [47, 46], # handr
    [48, 47],
    [49, 48],
    [50, 49],
    [51, 46],
    [52, 51],
    [53, 52],
    [54, 53],
    [55, 46],
    [56, 55],
    [57, 56],
    [58, 57],
    [59, 46],
    [60, 59],
    [61, 60],
    [62, 61],
    [63, 46],
    [64, 63],
    [65, 64],
    [66, 65]
    ]
}

CONFIG['face'] = {'kintree':[ [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16], #outline (ignored)
                [17,18],[18,19],[19,20],[20,21], #right eyebrow
                [22,23],[23,24],[24,25],[25,26], #left eyebrow
                [27,28],[28,29],[29,30],   #nose upper part
                [31,32],[32,33],[33,34],[34,35], #nose lower part
                [36,37],[37,38],[38,39],[39,40],[40,41],[41,36], #right eye
                [42,43],[43,44],[44,45],[45,46],[46,47],[47,42], #left eye
                [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,48], #Lip outline
                [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60] #Lip inner line 
                ]}
NJOINTS_BODY = 25
NJOINTS_HAND = 21
NJOINTS_FACE = 70
NLIMBS_BODY = len(CONFIG['body25']['kintree'])
NLIMBS_HAND = len(CONFIG['hand']['kintree'])
NLIMBS_FACE = len(CONFIG['face']['kintree'])

def getKintree(name='total'):
    if name == 'total':
        # order: body25, face, rhand, lhand
        kintree = CONFIG['body25']['kintree'] + CONFIG['hand']['kintree'] + CONFIG['hand']['kintree'] + CONFIG['face']['kintree']
        kintree = np.array(kintree)
        kintree[NLIMBS_BODY:NLIMBS_BODY + NLIMBS_HAND] += NJOINTS_BODY
        kintree[NLIMBS_BODY + NLIMBS_HAND:NLIMBS_BODY + 2*NLIMBS_HAND] += NJOINTS_BODY + NJOINTS_HAND
        kintree[NLIMBS_BODY + 2*NLIMBS_HAND:] += NJOINTS_BODY + 2*NJOINTS_HAND
    elif name == 'smplh':
        # order: body25, lhand, rhand
        kintree = CONFIG['body25']['kintree'] + CONFIG['hand']['kintree'] + CONFIG['hand']['kintree']
        kintree = np.array(kintree)
        kintree[NLIMBS_BODY:NLIMBS_BODY + NLIMBS_HAND] += NJOINTS_BODY
        kintree[NLIMBS_BODY + NLIMBS_HAND:NLIMBS_BODY + 2*NLIMBS_HAND] += NJOINTS_BODY + NJOINTS_HAND
    return kintree
CONFIG['total'] = {}
CONFIG['total']['kintree'] = getKintree('total')

COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]

def coco17tobody25(points2d):
    dim = 3
    if len(points2d.shape) == 2:
        points2d = points2d[None, :, :]
        dim = 2
    kpts = np.zeros((points2d.shape[0], 25, 3))
    kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
    kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
    if dim == 2:
        kpts = kpts[0]
    return kpts

