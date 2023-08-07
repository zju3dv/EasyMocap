'''
 * @ Date: 2020-09-26 16:52:55
 * @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-30 15:45:07
  @ FilePath: /EasyMocapPublic/easymocap/dataset/config.py
'''
import numpy as np

CONFIG = {
    'points': {
        'nJoints': 1,
        'kintree': []
    }
}

CONFIG['smpl'] = {'nJoints': 24, 'kintree': 
    [
        [ 0, 1 ],
        [ 0, 2 ],
        [ 0, 3 ],
        [ 1, 4 ],
        [ 2, 5 ],
        [ 3, 6 ],
        [ 4, 7 ],
        [ 5, 8 ],
        [ 6, 9 ],
        [ 7, 10],
        [ 8, 11],
        [ 9, 12],
        [ 9, 13],
        [ 9, 14],
        [12, 15],
        [13, 16],
        [14, 17],
        [16, 18],
        [17, 19],
        [18, 20],
        [19, 21],
        [20, 22],
        [21, 23],
    ], 
    'joint_names': [
        'MidHip',            # 0
        'LUpLeg',       # 1
        'RUpLeg',      # 2
        'spine',           # 3
        'LLeg',         # 4
        'RLeg',        # 5
        'spine1',          # 6
        'LFoot',        # 7
        'RFoot',       # 8
        'spine2',          # 9
        'LToeBase',     # 10
        'RToeBase',    # 11
        'neck',            # 12
        'LShoulder',    # 13
        'RShoulder',   # 14
        'head',            # 15
        'LArm',         # 16
        'RArm',        # 17
        'LForeArm',     # 18
        'RForeArm',    # 19
        'LHand',        # 20
        'RHand',       # 21
        'LHandIndex1',  # 22
        'RHandIndex1', # 23
    ]
}

CONFIG['smplh'] = {'nJoints': 52, 'kintree': 
    [
       [         1,          0],
       [         2,          0],
       [         3,          0],
       [         4,          1],
       [         5,          2],
       [         6,          3],
       [         7,          4],
       [         8,          5],
       [         9,          6],
       [        10,          7],
       [        11,          8],
       [        12,          9],
       [        13,          9],
       [        14,          9],
       [        15,         12],
       [        16,         13],
       [        17,         14],
       [        18,         16],
       [        19,         17],
       [        20,         18],
       [        21,         19],
       [        22,         20],
       [        23,         22],
       [        24,         23],
       [        25,         20],
       [        26,         25],
       [        27,         26],
       [        28,         20],
       [        29,         28],
       [        30,         29],
       [        31,         20],
       [        32,         31],
       [        33,         32],
       [        34,         20],
       [        35,         34],
       [        36,         35],
       [        37,         21],
       [        38,         37],
       [        39,         38],
       [        40,         21],
       [        41,         40],
       [        42,         41],
       [        43,         21],
       [        44,         43],
       [        45,         44],
       [        46,         21],
       [        47,         46],
       [        48,         47],
       [        49,         21],
       [        50,         49],
       [        51,         50]
    ], 
    'joint_names': [
        'MidHip',            # 0
        'LUpLeg',       # 1
        'RUpLeg',      # 2
        'spine',           # 3
        'LLeg',         # 4
        'RLeg',        # 5
        'spine1',          # 6
        'LFoot',        # 7
        'RFoot',       # 8
        'spine2',          # 9
        'LToeBase',     # 10
        'RToeBase',    # 11
        'neck',            # 12
        'LShoulder',    # 13
        'RShoulder',   # 14
        'head',            # 15
        'LArm',         # 16
        'RArm',        # 17
        'LForeArm',     # 18
        'RForeArm',    # 19
        'LHand',        # 20
        'RHand',       # 21
        'LHandIndex1',  # 22
        'RHandIndex1', # 23
    ]
}

CONFIG['coco'] = {
    'nJoints': 17,
    'kintree': [
        [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [5, 12], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]
    ],
}

CONFIG['coco_17'] = CONFIG['coco']

CONFIG['body25'] = {'nJoints': 25, 'kintree':
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
    [24, 11]], 
    'joint_names': [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel"]}
CONFIG['body25']['kintree_order'] = [
    [1, 8], # 躯干放在最前面
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [8, 9],
    [8, 12],
    [9, 10],
    [10, 11],
    [12, 13],
    [13, 14],
    [1, 0],
    [0, 15],
    [0, 16],
    [15, 17],
    [16, 18],
    [11, 22],
    [11, 24],
    [22, 23],
    [14, 19],
    [19, 20],
    [14, 21]
]
CONFIG['body25']['colors'] = ['k', 'r', 'r', 'r', 'b', 'b', 'b', 'k', 'r', 'r', 'r', 'b', 'b', 'b', 'r', 'b', 'r', 'b', 'b', 'b', 'b', 'r', 'r', 'r']
CONFIG['body25']['skeleton'] = \
{
    ( 0,  1): {'mean': 0.228, 'std': 0.046}, # Nose     ->Neck     
    ( 1,  2): {'mean': 0.144, 'std': 0.029}, # Neck     ->RShoulder
    ( 2,  3): {'mean': 0.283, 'std': 0.057}, # RShoulder->RElbow   
    ( 3,  4): {'mean': 0.258, 'std': 0.052}, # RElbow   ->RWrist   
    ( 1,  5): {'mean': 0.145, 'std': 0.029}, # Neck     ->LShoulder
    ( 5,  6): {'mean': 0.281, 'std': 0.056}, # LShoulder->LElbow   
    ( 6,  7): {'mean': 0.258, 'std': 0.052}, # LElbow   ->LWrist   
    ( 1,  8): {'mean': 0.483, 'std': 0.097}, # Neck     ->MidHip   
    ( 8,  9): {'mean': 0.106, 'std': 0.021}, # MidHip   ->RHip     
    ( 9, 10): {'mean': 0.438, 'std': 0.088}, # RHip     ->RKnee    
    (10, 11): {'mean': 0.406, 'std': 0.081}, # RKnee    ->RAnkle   
    ( 8, 12): {'mean': 0.106, 'std': 0.021}, # MidHip   ->LHip     
    (12, 13): {'mean': 0.438, 'std': 0.088}, # LHip     ->LKnee    
    (13, 14): {'mean': 0.408, 'std': 0.082}, # LKnee    ->LAnkle   
    ( 0, 15): {'mean': 0.043, 'std': 0.009}, # Nose     ->REye     
    ( 0, 16): {'mean': 0.043, 'std': 0.009}, # Nose     ->LEye     
    (15, 17): {'mean': 0.105, 'std': 0.021}, # REye     ->REar     
    (16, 18): {'mean': 0.104, 'std': 0.021}, # LEye     ->LEar     
    (14, 19): {'mean': 0.180, 'std': 0.036}, # LAnkle   ->LBigToe  
    (19, 20): {'mean': 0.038, 'std': 0.008}, # LBigToe  ->LSmallToe
    (14, 21): {'mean': 0.044, 'std': 0.009}, # LAnkle   ->LHeel    
    (11, 22): {'mean': 0.182, 'std': 0.036}, # RAnkle   ->RBigToe  
    (22, 23): {'mean': 0.038, 'std': 0.008}, # RBigToe  ->RSmallToe
    (11, 24): {'mean': 0.044, 'std': 0.009}, # RAnkle   ->RHeel    
}

CONFIG['body25vis'] = {
    'nJoints': 25,
    'kintree': [
        [8, 1], # 躯干放在最前面
        [8, 9],
        [8, 12],
        [9, 10],
        [12, 13],
        [10, 11],
        [13, 14],
        [11, 22],
        [14, 19],
        [1, 2],
        [1, 5],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [1, 0]]
}

CONFIG['handvis'] = {
    'nJoints': 21,
    'kintree': [
       [0, 1],
       [0, 5],
       [0, 9],
       [0, 13],
       [0, 17],
       [1, 2],
       [2, 3],
       [3, 4],
        [5, 6],
        [6, 7],
        [7, 8],
        [9, 10],
        [10, 11],
        [11, 12],
        [13, 14],
        [14, 15],
        [15, 16],
        [17, 18],
        [18, 19],
        [19, 20]
    ]
}

CONFIG['body15'] = {'nJoints': 15, 'root': 8,
                    'kintree':
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
    [14, 13]], 'root': 8,}
CONFIG['body15']['joint_names'] = CONFIG['body25']['joint_names'][:15]
CONFIG['body15']['skeleton'] = {key: val for key, val in CONFIG['body25']['skeleton'].items() if key[0] < 15 and key[1] < 15}
CONFIG['body15']['kintree_order'] = CONFIG['body25']['kintree_order'][:14]
CONFIG['body15']['colors'] = CONFIG['body25']['colors'][:15]

CONFIG['body19'] = {'nJoints': 19, 'kintree': [[i, j] for (i, j) in CONFIG['body25']['kintree'] if i < 19 and j < 19]}
CONFIG['body19']['skeleton'] = {key: val for key, val in CONFIG['body25']['skeleton'].items() if key[0] < 19 and key[1] < 19}

CONFIG['panoptic'] = {
    'nJoints': 19,
    'joint_names': ['Neck', 'Nose', 'MidHip', 'LShoulder', 'LElbow', 'LWrist', 'LHip', 'LKnee', 'LAnkle', 'RShoulder','RElbow', 'RWrist', 'RHip','RKnee', 'RAnkle', 'LEye', 'LEar', 'REye', 'REar'],
    'kintree': [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]],
    'colors': ['b' for _ in range(19)]
}

CONFIG['panoptic15'] = {
    'nJoints': 15,
    'root': 2,
    'joint_names': CONFIG['panoptic']['joint_names'][:15],
    'kintree': [[i, j] for (i, j) in CONFIG['panoptic']['kintree'] if i < 15 and j < 15],
    'limb_mean': [0.1129,0.4957,0.1382,0.2547,0.2425,0.1374,0.2549,0.2437,0.1257,0.1256, 0.4641,0.4580,0.4643,0.4589],
    'limb_std': [0.0164,0.0333,0.0078,0.0237,0.0233,0.0085,0.0233,0.0237,0.0076,0.0076, 0.0273,0.0247,0.0272,0.0242],
    'colors': CONFIG['panoptic']['colors'][:15]
}

CONFIG['mpii_16'] = {
    'nJoints': 16,
    'joint_names': ['rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle', 'pelvis', 'thorax', 'upper_neck', 'head_top', 'rwrist', 'relbow', 'rshoulder', 'lshoulder', 'lelbow', 'lwrist'],
    'kintree': [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [10, 11], [11, 12], [12, 7], [13, 14], [14, 15], [13, 7]],
    'colors': ['b' for _ in range(16)]
}

CONFIG['ochuman_19'] = {
    'nJoints': 19,
    'joint_names': ["right_shoulder", "right_elbow", "right_wrist",
                     "left_shoulder", "left_elbow", "left_wrist",
                     "right_hip", "right_knee", "right_ankle",
                     "left_hip", "left_knee", "left_ankle",
                     "head", "neck"] + ['right_ear', 'left_ear', 'nose', 'right_eye', 'left_eye'],
    'kintree': [
        [0, 1], [1, 2], [3, 4], [4, 5],
        [6, 7], [7, 8], [9, 10], [10, 11],
        [13, 0], [13, 3], [0, 3], [6, 9],
        [12, 16], [16, 13], [16, 17], [16, 18], [18, 15], [17, 14],
    ],
    'colors': ['b' for _ in range(19)]
}


CONFIG['chi3d_25'] = {
    'nJoints': 25,
    'joint_names': [],
    'kintree': [[10, 9], [9, 8], [8, 11], [8, 14], [11, 12], [14, 15], [12, 13], [15, 16],
                    [8, 7], [7, 0], [0, 1], [0, 4], [1, 2], [4, 5], [2, 3], [5, 6],
                    [13, 21], [13, 22], [16, 23], [16, 24], [3, 17], [3, 18], [6, 19], [6, 20]],
    'colors': ['b' for _ in range(25)]
}

CONFIG['chi3d_17'] = {
    'nJoints': 17,
    'joint_names': [],
    'kintree': [[10, 9], [9, 8], [8, 11], [8, 14], [11, 12], [14, 15], [12, 13], [15, 16],
                    [8, 7], [7, 0], [0, 1], [0, 4], [1, 2], [4, 5], [2, 3], [5, 6],
                ],
    'colors': ['b' for _ in range(17)]
}


CONFIG['hand'] = {'nJoints': 21, 'kintree':
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
       [20, 19]],
    'colors': [
        '_k', '_k', '_k', '_k', '_r', '_r', '_r', '_r', 
        '_g', '_g', '_g', '_g', '_b', '_b', '_b', '_b', 
        '_y', '_y', '_y', '_y'],
    'colorsrhand': [
        '_pink', '_pink', '_pink', '_pink', '_mint', '_mint', '_mint', '_mint', 
        '_orange', '_orange', '_orange', '_orange', '_mint2', '_mint2', '_mint2', '_mint2', 
        'purple', 'purple', 'purple', 'purple'],
    'joint_names':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

CONFIG['handl'] = CONFIG['hand']
CONFIG['handr'] = CONFIG['hand']
CONFIG['handlr'] = {
    'nJoints': 42,
    'colors': CONFIG['hand']['colors'] + CONFIG['hand']['colorsrhand'],
    'joint_names': CONFIG['hand']['joint_names'] + CONFIG['hand']['joint_names'],
    'kintree': np.vstack((np.array(CONFIG['hand']['kintree']), np.array(CONFIG['hand']['kintree'])+21)).tolist()
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
    [26,  7],  # handl
    [27, 26],
    [28, 27],
    [29, 28],
    [30,  7],
    [31, 30],
    [32, 31],
    [33, 32],
    [34,  7],
    [35, 34],
    [36, 35],
    [37, 36],
    [38,  7],
    [39, 38],
    [40, 39],
    [41, 40],
    [42,  7],
    [43, 42],
    [44, 43],
    [45, 44],
    [47,  4], # handr
    [48, 47],
    [49, 48],
    [50, 49],
    [51,  4],
    [52, 51],
    [53, 52],
    [54, 53],
    [55,  4],
    [56, 55],
    [57, 56],
    [58, 57],
    [59,  4],
    [60, 59],
    [61, 60],
    [62, 61],
    [63,  4],
    [64, 63],
    [65, 64],
    [66, 65]
    ],
    'nJoints': 67,
    'colors': CONFIG['body25']['colors'] + CONFIG['hand']['colors'] + CONFIG['hand']['colors'],
    'skeleton':{
    ( 0,  1): {'mean': 0.251, 'std': 0.050}, 
    ( 1,  2): {'mean': 0.169, 'std': 0.034}, 
    ( 2,  3): {'mean': 0.292, 'std': 0.058}, 
    ( 3,  4): {'mean': 0.275, 'std': 0.055}, 
    ( 1,  5): {'mean': 0.169, 'std': 0.034}, 
    ( 5,  6): {'mean': 0.295, 'std': 0.059}, 
    ( 6,  7): {'mean': 0.278, 'std': 0.056}, 
    ( 1,  8): {'mean': 0.566, 'std': 0.113}, 
    ( 8,  9): {'mean': 0.110, 'std': 0.022}, 
    ( 9, 10): {'mean': 0.398, 'std': 0.080}, 
    (10, 11): {'mean': 0.402, 'std': 0.080}, 
    ( 8, 12): {'mean': 0.111, 'std': 0.022}, 
    (12, 13): {'mean': 0.395, 'std': 0.079}, 
    (13, 14): {'mean': 0.403, 'std': 0.081}, 
    ( 0, 15): {'mean': 0.053, 'std': 0.011}, 
    ( 0, 16): {'mean': 0.056, 'std': 0.011}, 
    (15, 17): {'mean': 0.107, 'std': 0.021}, 
    (16, 18): {'mean': 0.107, 'std': 0.021}, 
    (14, 19): {'mean': 0.180, 'std': 0.036}, 
    (19, 20): {'mean': 0.055, 'std': 0.011}, 
    (14, 21): {'mean': 0.065, 'std': 0.013}, 
    (11, 22): {'mean': 0.169, 'std': 0.034}, 
    (22, 23): {'mean': 0.052, 'std': 0.010}, 
    (11, 24): {'mean': 0.061, 'std': 0.012}, 
    ( 7, 26): {'mean': 0.045, 'std': 0.009}, 
    (26, 27): {'mean': 0.042, 'std': 0.008}, 
    (27, 28): {'mean': 0.035, 'std': 0.007}, 
    (28, 29): {'mean': 0.029, 'std': 0.006}, 
    ( 7, 30): {'mean': 0.102, 'std': 0.020}, 
    (30, 31): {'mean': 0.040, 'std': 0.008}, 
    (31, 32): {'mean': 0.026, 'std': 0.005}, 
    (32, 33): {'mean': 0.023, 'std': 0.005}, 
    ( 7, 34): {'mean': 0.101, 'std': 0.020}, 
    (34, 35): {'mean': 0.043, 'std': 0.009}, 
    (35, 36): {'mean': 0.029, 'std': 0.006}, 
    (36, 37): {'mean': 0.024, 'std': 0.005}, 
    ( 7, 38): {'mean': 0.097, 'std': 0.019}, 
    (38, 39): {'mean': 0.041, 'std': 0.008}, 
    (39, 40): {'mean': 0.027, 'std': 0.005}, 
    (40, 41): {'mean': 0.024, 'std': 0.005}, 
    ( 7, 42): {'mean': 0.095, 'std': 0.019}, 
    (42, 43): {'mean': 0.033, 'std': 0.007}, 
    (43, 44): {'mean': 0.020, 'std': 0.004}, 
    (44, 45): {'mean': 0.018, 'std': 0.004}, 
    ( 4, 47): {'mean': 0.043, 'std': 0.009}, 
    (47, 48): {'mean': 0.041, 'std': 0.008}, 
    (48, 49): {'mean': 0.034, 'std': 0.007}, 
    (49, 50): {'mean': 0.028, 'std': 0.006}, 
    ( 4, 51): {'mean': 0.101, 'std': 0.020}, 
    (51, 52): {'mean': 0.041, 'std': 0.008}, 
    (52, 53): {'mean': 0.026, 'std': 0.005}, 
    (53, 54): {'mean': 0.024, 'std': 0.005}, 
    ( 4, 55): {'mean': 0.100, 'std': 0.020}, 
    (55, 56): {'mean': 0.044, 'std': 0.009}, 
    (56, 57): {'mean': 0.029, 'std': 0.006}, 
    (57, 58): {'mean': 0.023, 'std': 0.005}, 
    ( 4, 59): {'mean': 0.096, 'std': 0.019}, 
    (59, 60): {'mean': 0.040, 'std': 0.008}, 
    (60, 61): {'mean': 0.028, 'std': 0.006}, 
    (61, 62): {'mean': 0.023, 'std': 0.005}, 
    ( 4, 63): {'mean': 0.094, 'std': 0.019}, 
    (63, 64): {'mean': 0.032, 'std': 0.006}, 
    (64, 65): {'mean': 0.020, 'std': 0.004}, 
    (65, 66): {'mean': 0.018, 'std': 0.004}, 
}
}

CONFIG['bodyhandface'] = {'kintree':
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
    [26,  7],  # handl
    [27, 26],
    [28, 27],
    [29, 28],
    [30,  7],
    [31, 30],
    [32, 31],
    [33, 32],
    [34,  7],
    [35, 34],
    [36, 35],
    [37, 36],
    [38,  7],
    [39, 38],
    [40, 39],
    [41, 40],
    [42,  7],
    [43, 42],
    [44, 43],
    [45, 44],
    [47,  4], # handr
    [48, 47],
    [49, 48],
    [50, 49],
    [51,  4],
    [52, 51],
    [53, 52],
    [54, 53],
    [55,  4],
    [56, 55],
    [57, 56],
    [58, 57],
    [59,  4],
    [60, 59],
    [61, 60],
    [62, 61],
    [63,  4],
    [64, 63],
    [65, 64],
    [66, 65],
    [ 67,  68],
    [ 68,  69],
    [ 69,  70],
    [ 70,  71],
    [ 72,  73],
    [ 73,  74],
    [ 74,  75],
    [ 75,  76],
    [ 77,  78],
    [ 78,  79],
    [ 79,  80],
    [ 81,  82],
    [ 82,  83],
    [ 83,  84],
    [ 84,  85],
    [ 86,  87],
    [ 87,  88],
    [ 88,  89],
    [ 89,  90],
    [ 90,  91],
    [ 91,  86],
    [ 92,  93],
    [ 93,  94],
    [ 94,  95],
    [ 95,  96],
    [ 96,  97],
    [ 97,  92],
    [ 98,  99],
    [ 99, 100],
    [100, 101],
    [101, 102],
    [102, 103],
    [103, 104],
    [104, 105],
    [105, 106],
    [106, 107],
    [107, 108],
    [108, 109],
    [109,  98],
    [110, 111],
    [111, 112],
    [112, 113],
    [113, 114],
    [114, 115],
    [115, 116],
    [116, 117],
    [117, 110]
    ],
    'nJoints': 118,
    'skeleton':{
    ( 0,  1): {'mean': 0.251, 'std': 0.050}, 
    ( 1,  2): {'mean': 0.169, 'std': 0.034}, 
    ( 2,  3): {'mean': 0.292, 'std': 0.058}, 
    ( 3,  4): {'mean': 0.275, 'std': 0.055}, 
    ( 1,  5): {'mean': 0.169, 'std': 0.034}, 
    ( 5,  6): {'mean': 0.295, 'std': 0.059}, 
    ( 6,  7): {'mean': 0.278, 'std': 0.056}, 
    ( 1,  8): {'mean': 0.566, 'std': 0.113}, 
    ( 8,  9): {'mean': 0.110, 'std': 0.022}, 
    ( 9, 10): {'mean': 0.398, 'std': 0.080}, 
    (10, 11): {'mean': 0.402, 'std': 0.080}, 
    ( 8, 12): {'mean': 0.111, 'std': 0.022}, 
    (12, 13): {'mean': 0.395, 'std': 0.079}, 
    (13, 14): {'mean': 0.403, 'std': 0.081}, 
    ( 0, 15): {'mean': 0.053, 'std': 0.011}, 
    ( 0, 16): {'mean': 0.056, 'std': 0.011}, 
    (15, 17): {'mean': 0.107, 'std': 0.021}, 
    (16, 18): {'mean': 0.107, 'std': 0.021}, 
    (14, 19): {'mean': 0.180, 'std': 0.036}, 
    (19, 20): {'mean': 0.055, 'std': 0.011}, 
    (14, 21): {'mean': 0.065, 'std': 0.013}, 
    (11, 22): {'mean': 0.169, 'std': 0.034}, 
    (22, 23): {'mean': 0.052, 'std': 0.010}, 
    (11, 24): {'mean': 0.061, 'std': 0.012}, 
    ( 7, 26): {'mean': 0.045, 'std': 0.009}, 
    (26, 27): {'mean': 0.042, 'std': 0.008}, 
    (27, 28): {'mean': 0.035, 'std': 0.007}, 
    (28, 29): {'mean': 0.029, 'std': 0.006}, 
    ( 7, 30): {'mean': 0.102, 'std': 0.020}, 
    (30, 31): {'mean': 0.040, 'std': 0.008}, 
    (31, 32): {'mean': 0.026, 'std': 0.005}, 
    (32, 33): {'mean': 0.023, 'std': 0.005}, 
    ( 7, 34): {'mean': 0.101, 'std': 0.020}, 
    (34, 35): {'mean': 0.043, 'std': 0.009}, 
    (35, 36): {'mean': 0.029, 'std': 0.006}, 
    (36, 37): {'mean': 0.024, 'std': 0.005}, 
    ( 7, 38): {'mean': 0.097, 'std': 0.019}, 
    (38, 39): {'mean': 0.041, 'std': 0.008}, 
    (39, 40): {'mean': 0.027, 'std': 0.005}, 
    (40, 41): {'mean': 0.024, 'std': 0.005}, 
    ( 7, 42): {'mean': 0.095, 'std': 0.019}, 
    (42, 43): {'mean': 0.033, 'std': 0.007}, 
    (43, 44): {'mean': 0.020, 'std': 0.004}, 
    (44, 45): {'mean': 0.018, 'std': 0.004}, 
    ( 4, 47): {'mean': 0.043, 'std': 0.009}, 
    (47, 48): {'mean': 0.041, 'std': 0.008}, 
    (48, 49): {'mean': 0.034, 'std': 0.007}, 
    (49, 50): {'mean': 0.028, 'std': 0.006}, 
    ( 4, 51): {'mean': 0.101, 'std': 0.020}, 
    (51, 52): {'mean': 0.041, 'std': 0.008}, 
    (52, 53): {'mean': 0.026, 'std': 0.005}, 
    (53, 54): {'mean': 0.024, 'std': 0.005}, 
    ( 4, 55): {'mean': 0.100, 'std': 0.020}, 
    (55, 56): {'mean': 0.044, 'std': 0.009}, 
    (56, 57): {'mean': 0.029, 'std': 0.006}, 
    (57, 58): {'mean': 0.023, 'std': 0.005}, 
    ( 4, 59): {'mean': 0.096, 'std': 0.019}, 
    (59, 60): {'mean': 0.040, 'std': 0.008}, 
    (60, 61): {'mean': 0.028, 'std': 0.006}, 
    (61, 62): {'mean': 0.023, 'std': 0.005}, 
    ( 4, 63): {'mean': 0.094, 'std': 0.019}, 
    (63, 64): {'mean': 0.032, 'std': 0.006}, 
    (64, 65): {'mean': 0.020, 'std': 0.004}, 
    (65, 66): {'mean': 0.018, 'std': 0.004}, 
    (67, 68): {'mean': 0.012, 'std': 0.002}, 
    (68, 69): {'mean': 0.013, 'std': 0.003}, 
    (69, 70): {'mean': 0.014, 'std': 0.003}, 
    (70, 71): {'mean': 0.012, 'std': 0.002}, 
    (72, 73): {'mean': 0.014, 'std': 0.003}, 
    (73, 74): {'mean': 0.014, 'std': 0.003}, 
    (74, 75): {'mean': 0.015, 'std': 0.003}, 
    (75, 76): {'mean': 0.013, 'std': 0.003}, 
    (77, 78): {'mean': 0.014, 'std': 0.003}, 
    (78, 79): {'mean': 0.014, 'std': 0.003}, 
    (79, 80): {'mean': 0.015, 'std': 0.003}, 
    (81, 82): {'mean': 0.009, 'std': 0.002}, 
    (82, 83): {'mean': 0.010, 'std': 0.002}, 
    (83, 84): {'mean': 0.010, 'std': 0.002}, 
    (84, 85): {'mean': 0.010, 'std': 0.002}, 
    (86, 87): {'mean': 0.009, 'std': 0.002}, 
    (87, 88): {'mean': 0.009, 'std': 0.002}, 
    (88, 89): {'mean': 0.008, 'std': 0.002}, 
    (89, 90): {'mean': 0.008, 'std': 0.002}, 
    (90, 91): {'mean': 0.009, 'std': 0.002}, 
    (86, 91): {'mean': 0.008, 'std': 0.002}, 
    (92, 93): {'mean': 0.009, 'std': 0.002}, 
    (93, 94): {'mean': 0.009, 'std': 0.002}, 
    (94, 95): {'mean': 0.009, 'std': 0.002}, 
    (95, 96): {'mean': 0.009, 'std': 0.002}, 
    (96, 97): {'mean': 0.009, 'std': 0.002}, 
    (92, 97): {'mean': 0.009, 'std': 0.002}, 
    (98, 99): {'mean': 0.016, 'std': 0.003}, 
    (99, 100): {'mean': 0.013, 'std': 0.003}, 
    (100, 101): {'mean': 0.008, 'std': 0.002}, 
    (101, 102): {'mean': 0.008, 'std': 0.002}, 
    (102, 103): {'mean': 0.012, 'std': 0.002}, 
    (103, 104): {'mean': 0.014, 'std': 0.003}, 
    (104, 105): {'mean': 0.015, 'std': 0.003}, 
    (105, 106): {'mean': 0.012, 'std': 0.002}, 
    (106, 107): {'mean': 0.009, 'std': 0.002}, 
    (107, 108): {'mean': 0.009, 'std': 0.002}, 
    (108, 109): {'mean': 0.013, 'std': 0.003}, 
    (98, 109): {'mean': 0.016, 'std': 0.003}, 
    (110, 111): {'mean': 0.021, 'std': 0.004}, 
    (111, 112): {'mean': 0.009, 'std': 0.002}, 
    (112, 113): {'mean': 0.008, 'std': 0.002}, 
    (113, 114): {'mean': 0.019, 'std': 0.004}, 
    (114, 115): {'mean': 0.018, 'std': 0.004}, 
    (115, 116): {'mean': 0.008, 'std': 0.002}, 
    (116, 117): {'mean': 0.009, 'std': 0.002}, 
    (110, 117): {'mean': 0.020, 'std': 0.004}, 
}
}

face_kintree_without_contour = [[ 0,  1],
       [ 1,  2],
       [ 2,  3],
       [ 3,  4],
       [ 5,  6],
       [ 6,  7],
       [ 7,  8],
       [ 8,  9],
       [10, 11],
       [11, 12],
       [12, 13],
       [14, 15],
       [15, 16],
       [16, 17],
       [17, 18],
       [19, 20],
       [20, 21],
       [21, 22],
       [22, 23],
       [23, 24],
       [24, 19],
       [25, 26],
       [26, 27],
       [27, 28],
       [28, 29],
       [29, 30],
       [30, 25],
       [31, 32],
       [32, 33],
       [33, 34],
       [34, 35],
       [35, 36],
       [36, 37],
       [37, 38],
       [38, 39],
       [39, 40],
       [40, 41],
       [41, 42],
       [42, 31],
       [43, 44],
       [44, 45],
       [45, 46],
       [46, 47],
       [47, 48],
       [48, 49],
       [49, 50],
       [50, 43]]

CONFIG['face'] = {'nJoints': 70,
    'kintree':[ [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16], #outline (ignored)
                [17,18],[18,19],[19,20],[20,21], #right eyebrow
                [22,23],[23,24],[24,25],[25,26], #left eyebrow
                [27,28],[28,29],[29,30],   #nose upper part
                [31,32],[32,33],[33,34],[34,35], #nose lower part
                [36,37],[37,38],[38,39],[39,40],[40,41],[41,36], #right eye
                [42,43],[43,44],[44,45],[45,46],[46,47],[47,42], #left eye
                [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,48], #Lip outline
                [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60] #Lip inner line 
                ], 'colors': ['g' for _ in range(100)]}

CONFIG['h36m'] = {
    'kintree': [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [
    12, 13], [8, 14], [14, 15], [15, 16]], 
    'color': ['r', 'r', 'r', 'g', 'g', 'g', 'k', 'k', 'k', 'k', 'g', 'g', 'g', 'r', 'r', 'r'],
    'joint_names': [
        'hip',  # 0
        'LHip',  # 1
        'LKnee',  # 2
        'LAnkle',  # 3
        'RHip',  # 4
        'RKnee',  # 5
        'RAnkle',  # 6
        'Spine (H36M)',  # 7
        'Neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'LShoulder',  # 11
        'LElbow',  # 12
        'LWrist',  # 13
        'RShoulder',  # 14
        'RElbow',  # 15
        'RWrist',  # 16
    ],
    'nJoints': 17}

CONFIG['h36m_17'] = CONFIG['h36m']

NJOINTS_BODY = 25
NJOINTS_HAND = 21
NJOINTS_FACE = 70
NLIMBS_BODY = len(CONFIG['body25']['kintree'])
NLIMBS_HAND = len(CONFIG['hand']['kintree'])
NLIMBS_FACE = len(CONFIG['face']['kintree'])

def compose(names):
    kintrees = []
    nJoints = 0
    for name in names:
        kintrees.append(np.array(CONFIG[name]['kintree']) + nJoints)
        nJoints += CONFIG[name]['nJoints']
    kintrees = np.vstack(kintrees)
    cfg = {
        'kintree': kintrees.tolist(),
        'nJoints': nJoints
    }
    return cfg

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

CONFIG['total'] = compose(['body25', 'hand', 'hand', 'face'])
COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]

CONFIG['bodyhandface']['joint_names'] = CONFIG['body25']['joint_names']
# default to body25
CONFIG['keypoints2d'] = CONFIG['body25']
CONFIG['handl2d'] = CONFIG['hand']
CONFIG['handr2d'] = CONFIG['hand']
CONFIG['face2d'] = CONFIG['face']

# set mediapipe
CONFIG['mpbody'] = {}
CONFIG['mpbody']['kintree'] = [
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (11, 23),
    (12, 14),
    (12, 24),
    (13, 15),
    (14, 16),
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
    (17, 19),
    (18, 20),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (27, 31),
    (28, 30),
    (28, 32),
    (29, 31),
    (30, 32)
]
CONFIG['mpbody']['nJoints'] = 33
CONFIG['mpbody']['colors'] = ['b', 'r', 'b', 'b', 'b', 'r', 'r', 'r', 'k', 'k', 'b', 'b', 'r', 'r', 'b', 'r', 
    'y', 'r', 'y', 'g', 'b', 'g', 'y', 'g', 'k', 'b', 'r', 'b', 'r', 'b', 'b', 'r', 'r', 'b', 'b']

CONFIG['mpface'] = {}
CONFIG['mpface']['kintree'] = [(270, 409), (176, 149), (37, 0), (84, 17), (318, 324), (293, 334), (386, 385), (7, 163), (33, 246), (17, 314), (374, 380), (251, 389), (390, 373), (267, 269), (295, 285), (389, 356), (173, 133), (33, 7), (377, 152), (158, 157), (405, 321), (54, 103), (263, 466), (324, 308), (67, 109), (409, 291), (157, 173), (454, 323), (388, 387), (78, 191), (148, 176), (311, 310), (39, 37), (249, 390), (144, 145), (402, 318), (80, 81), (310, 415), (153, 154), (384, 398), (397, 365), (234, 127), (103, 67), (282, 295), (338, 297), (378, 400), (127, 162), (321, 375), (375, 291), (317, 402), (81, 82), (154, 155), (91, 181), (334, 296), (297, 332), (269, 270), (150, 136), (109, 10), (356, 454), (58, 132), (312, 311), (152, 148), (415, 308), (161, 160), (296, 336), (65, 55), (61, 146), (78, 95), (380, 381), (398, 362), (361, 288), (246, 161), (162, 21), (0, 267), (82, 13), (132, 93), (314, 405), (10, 338), (178, 87), (387, 386), (381, 382), (70, 63), (61, 185), (14, 317), (105, 66), (300, 293), (382, 362), (88, 178), (185, 40), (46, 53), (284, 251), (400, 377), (136, 172), (323, 361), (13, 312), (21, 54), (172, 58), (373, 374), (163, 144), (276, 283), (53, 52), (365, 379), (379, 378), (146, 91), (263, 249), (283, 282), (87, 14), (145, 153), (155, 133), (93, 234), (66, 107), (95, 88), (159, 158), (52, 65), (332, 284), (40, 39), (191, 80), (63, 105), (181, 84), (466, 388), (149, 150), (288, 397), (160, 159), (385, 384)]
CONFIG['mpface']['nJoints'] = 468

CONFIG['mptotal'] = compose(['mpbody', 'hand', 'hand', 'mpface'])
CONFIG['bodyhandmpface'] = compose(['body25', 'hand', 'hand', 'mpface'])

CONFIG['iris'] = {
    'nJoints': 10,
    'kintree': [[0, 1], [1, 2], [2, 3], [3, 4]]
}

CONFIG['onepoint'] = {
    'nJoints': 1,
    'kintree': []
}

CONFIG['up'] = {
    'nJoints': 79,
    'kintree': []
}

CONFIG['ochuman'] = {
    'nJoints': 19,
    'kintree': [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [14, 17], [15, 18], [17, 16], [18, 16]]
}

CONFIG['mpii'] = {
    'nJoints': 16,
    'kintree': [[0, 1], [1, 2], [3, 4], [4, 5], [2, 6], [3, 6], [6, 7], [7, 8], [8, 9], [10, 11], [11, 12], [7, 12], [7, 13], \
        [13, 14], [14, 15]],
    'joint_names': ['rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 'pelv', 'thrx', 'neck', 'head', 'rwri', 'relb', 'rsho', 'lsho', 'lelb', 'lwri'],
}

CONFIG['h36mltri_17'] = {
    'kintree': [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)],
    'color': ['r', 'r', 'r', 'g', 'g', 'g', 'k', 'k', 'k', 'k', 'g', 'g', 'g', 'r', 'r', 'r'],
    'joint_names': CONFIG['mpii']['joint_names'] + ['Neck/Nose'],
    'nJoints': 17}

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

for skeltype, config in CONFIG.items():
    if 'joint_names' in config.keys():
        torsoid = [config['joint_names'].index(name) if name in config['joint_names'] else None for name in ['LShoulder', 'RShoulder', 'LHip', 'RHip']]
        torsoid = [i for i in torsoid if i is not None]
        config['torso'] = torsoid
    if 'colors' not in config.keys():
        config['colors'] = ['b' for _ in range(len(config['kintree']))]