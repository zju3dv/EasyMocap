# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import numpy as np


def keypoint_hflip(kp, img_width):
    # Flip a keypoint horizontally around the y-axis
    # kp N,2
    if len(kp.shape) == 2:
        kp[:,0] = (img_width - 1.) - kp[:,0]
    elif len(kp.shape) == 3:
        kp[:, :, 0] = (img_width - 1.) - kp[:, :, 0]
    return kp


def convert_kps(joints2d, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), joints2d.shape[-1]))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d


def get_perm_idxs(src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    idxs = [src_names.index(h) for h in dst_names if h in src_names]
    return idxs


def get_mpii3d_test_joint_names():
    return [
        'headtop', # 'head_top',
        'neck',
        'rshoulder',# 'right_shoulder',
        'relbow',# 'right_elbow',
        'rwrist',# 'right_wrist',
        'lshoulder',# 'left_shoulder',
        'lelbow', # 'left_elbow',
        'lwrist', # 'left_wrist',
        'rhip', # 'right_hip',
        'rknee', # 'right_knee',
        'rankle',# 'right_ankle',
        'lhip',# 'left_hip',
        'lknee',# 'left_knee',
        'lankle',# 'left_ankle'
        'hip',# 'pelvis',
        'Spine (H36M)',# 'spine',
        'Head (H36M)',# 'head'
    ]


def get_mpii3d_joint_names():
    return [
        'spine3', # 0,
        'spine4', # 1,
        'spine2', # 2,
        'Spine (H36M)', #'spine', # 3,
        'hip', # 'pelvis', # 4,
        'neck', # 5,
        'Head (H36M)', # 'head', # 6,
        "headtop", # 'head_top', # 7,
        'left_clavicle', # 8,
        "lshoulder", # 'left_shoulder', # 9,
        "lelbow", # 'left_elbow',# 10,
        "lwrist", # 'left_wrist',# 11,
        'left_hand',# 12,
        'right_clavicle',# 13,
        'rshoulder',# 'right_shoulder',# 14,
        'relbow',# 'right_elbow',# 15,
        'rwrist',# 'right_wrist',# 16,
        'right_hand',# 17,
        'lhip', # left_hip',# 18,
        'lknee', # 'left_knee',# 19,
        'lankle', #left ankle # 20
        'left_foot', # 21
        'left_toe', # 22
        "rhip", # 'right_hip',# 23
        "rknee", # 'right_knee',# 24
        "rankle", #'right_ankle', # 25
        'right_foot',# 26
        'right_toe' # 27
    ]


# def get_insta_joint_names():
#     return [
#         'rheel'            ,   # 0
#         'rknee'            ,   # 1
#         'rhip'             ,   # 2
#         'lhip'             ,   # 3
#         'lknee'            ,   # 4
#         'lheel'            ,   # 5
#         'rwrist'           ,   # 6
#         'relbow'           ,   # 7
#         'rshoulder'        ,   # 8
#         'lshoulder'        ,   # 9
#         'lelbow'           ,   # 10
#         'lwrist'           ,   # 11
#         'neck'             ,   # 12
#         'headtop'          ,   # 13
#         'nose'             ,   # 14
#         'leye'             ,   # 15
#         'reye'             ,   # 16
#         'lear'             ,   # 17
#         'rear'             ,   # 18
#         'lbigtoe'          ,   # 19
#         'rbigtoe'          ,   # 20
#         'lsmalltoe'        ,   # 21
#         'rsmalltoe'        ,   # 22
#         'lankle'           ,   # 23
#         'rankle'           ,   # 24
#     ]


def get_insta_joint_names():
    return [
        'OP RHeel',
        'OP RKnee',
        'OP RHip',
        'OP LHip',
        'OP LKnee',
        'OP LHeel',
        'OP RWrist',
        'OP RElbow',
        'OP RShoulder',
        'OP LShoulder',
        'OP LElbow',
        'OP LWrist',
        'OP Neck',
        'headtop',
        'OP Nose',
        'OP LEye',
        'OP REye',
        'OP LEar',
        'OP REar',
        'OP LBigToe',
        'OP RBigToe',
        'OP LSmallToe',
        'OP RSmallToe',
        'OP LAnkle',
        'OP RAnkle',
    ]


def get_mmpose_joint_names():
    # this naming is for the first 23 joints of MMPose
    # does not include hands and face
    return [
        'OP Nose', # 1
        'OP LEye', # 2
        'OP REye', # 3
        'OP LEar', # 4
        'OP REar', # 5
        'OP LShoulder', # 6
        'OP RShoulder', # 7
        'OP LElbow', # 8
        'OP RElbow', # 9
        'OP LWrist', # 10
        'OP RWrist', # 11
        'OP LHip', # 12
        'OP RHip', # 13
        'OP LKnee', # 14
        'OP RKnee', # 15
        'OP LAnkle', # 16
        'OP RAnkle', # 17
        'OP LBigToe', # 18
        'OP LSmallToe', # 19
        'OP LHeel', # 20
        'OP RBigToe', # 21
        'OP RSmallToe', # 22
        'OP RHeel', # 23
    ]


def get_insta_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [4 , 5],
            [6 , 7],
            [7 , 8],
            [8 , 9],
            [9 ,10],
            [2 , 8],
            [3 , 9],
            [10,11],
            [8 ,12],
            [9 ,12],
            [12,13],
            [12,14],
            [14,15],
            [14,16],
            [15,17],
            [16,18],
            [0 ,20],
            [20,22],
            [5 ,19],
            [19,21],
            [5 ,23],
            [0 ,24],
        ])


def get_staf_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [8, 12],
            [12, 13],
            [13, 14],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18],
            [2, 9],
            [5, 12],
            [1, 19],
            [20, 19],
        ]
    )


def get_staf_joint_names():
    return [
        'OP Nose', # 0,
        'OP Neck', # 1,
        'OP RShoulder', # 2,
        'OP RElbow', # 3,
        'OP RWrist', # 4,
        'OP LShoulder', # 5,
        'OP LElbow', # 6,
        'OP LWrist', # 7,
        'OP MidHip', # 8,
        'OP RHip', # 9,
        'OP RKnee', # 10,
        'OP RAnkle', # 11,
        'OP LHip', # 12,
        'OP LKnee', # 13,
        'OP LAnkle', # 14,
        'OP REye', # 15,
        'OP LEye', # 16,
        'OP REar', # 17,
        'OP LEar', # 18,
        'Neck (LSP)', # 19,
        'Top of Head (LSP)', # 20,
    ]


def get_spin_op_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
    ]


def get_openpose_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
    ]


def get_spin_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leye',           # 45 'Left Eye', # 45
        'reye',           # 46 'Right Eye', # 46
        'lear',           # 47 'Left Ear', # 47
        'rear',           # 48 'Right Ear', # 48
    ]

def get_muco3dhp_joint_names():
    return [
        'headtop',
        'thorax',
        'rshoulder',
        'relbow',
        'rwrist',
        'lshoulder',
        'lelbow',
        'lwrist',
        'rhip',
        'rknee',
        'rankle',
        'lhip',
        'lknee',
        'lankle',
        'hip',
        'Spine (H36M)',
        'Head (H36M)',
        'R_Hand',
        'L_Hand',
        'R_Toe',
        'L_Toe'
    ]

def get_h36m_joint_names():
    return [
        'hip',  # 0
        'lhip',  # 1
        'lknee',  # 2
        'lankle',  # 3
        'rhip',  # 4
        'rknee',  # 5
        'rankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'lshoulder',  # 11
        'lelbow',  # 12
        'lwrist',  # 13
        'rshoulder',  # 14
        'relbow',  # 15
        'rwrist',  # 16
    ]


def get_spin_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [1 , 5],
            [5 , 6],
            [6 , 7],
            [1 , 8],
            [8 , 9],
            [9 ,10],
            [10,11],
            [8 ,12],
            [12,13],
            [13,14],
            [0 ,15],
            [0 ,16],
            [15,17],
            [16,18],
            [21,19],
            [19,20],
            [14,21],
            [11,24],
            [24,22],
            [22,23],
            [0 ,38],
        ]
    )


def get_openpose_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [1 , 5],
            [5 , 6],
            [6 , 7],
            [1 , 8],
            [8 , 9],
            [9 ,10],
            [10,11],
            [8 ,12],
            [12,13],
            [13,14],
            [0 ,15],
            [0 ,16],
            [15,17],
            [16,18],
            [21,19],
            [19,20],
            [14,21],
            [11,24],
            [24,22],
            [22,23],
        ]
    )


def get_posetrack_joint_names():
    return [
        "nose",
        "neck",
        "headtop",
        "lear",
        "rear",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhip",
        "rhip",
        "lknee",
        "rknee",
        "lankle",
        "rankle"
    ]


def get_posetrack_original_kp_names():
    return [
        'nose',
        'head_bottom',
        'head_top',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]


def get_pennaction_joint_names():
   return [
       "headtop",   # 0
       "lshoulder", # 1
       "rshoulder", # 2
       "lelbow",    # 3
       "relbow",    # 4
       "lwrist",    # 5
       "rwrist",    # 6
       "lhip" ,     # 7
       "rhip" ,     # 8
       "lknee",     # 9
       "rknee" ,    # 10
       "lankle",    # 11
       "rankle"     # 12
   ]


def get_common_joint_names():
    return [
        "rankle",    # 0  "lankle",    # 0
        "rknee",     # 1  "lknee",     # 1
        "rhip",      # 2  "lhip",      # 2
        "lhip",      # 3  "rhip",      # 3
        "lknee",     # 4  "rknee",     # 4
        "lankle",    # 5  "rankle",    # 5
        "rwrist",    # 6  "lwrist",    # 6
        "relbow",    # 7  "lelbow",    # 7
        "rshoulder", # 8  "lshoulder", # 8
        "lshoulder", # 9  "rshoulder", # 9
        "lelbow",    # 10  "relbow",    # 10
        "lwrist",    # 11  "rwrist",    # 11
        "neck",      # 12  "neck",      # 12
        "headtop",   # 13  "headtop",   # 13
    ]


def get_common_paper_joint_names():
    return [
        "Right Ankle",    # 0  "lankle",    # 0
        "Right Knee",     # 1  "lknee",     # 1
        "Right Hip",      # 2  "lhip",      # 2
        "Left Hip",      # 3  "rhip",      # 3
        "Left Knee",     # 4  "rknee",     # 4
        "Left Ankle",    # 5  "rankle",    # 5
        "Right Wrist",    # 6  "lwrist",    # 6
        "Right Elbow",    # 7  "lelbow",    # 7
        "Right Shoulder", # 8  "lshoulder", # 8
        "Left Shoulder", # 9  "rshoulder", # 9
        "Left Elbow",    # 10  "relbow",    # 10
        "Left Wrist",    # 11  "rwrist",    # 11
        "Neck",      # 12  "neck",      # 12
        "Head",   # 13  "headtop",   # 13
    ]


def get_common_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 2 ],
            [ 8, 9 ],
            [ 9, 3 ],
            [ 2, 3 ],
            [ 8, 12],
            [ 9, 10],
            [12, 9 ],
            [10, 11],
            [12, 13],
        ]
    )


def get_coco_joint_names():
    return [
        "nose",      # 0
        "leye",      # 1
        "reye",      # 2
        "lear",      # 3
        "rear",      # 4
        "lshoulder", # 5
        "rshoulder", # 6
        "lelbow",    # 7
        "relbow",    # 8
        "lwrist",    # 9
        "rwrist",    # 10
        "lhip",      # 11
        "rhip",      # 12
        "lknee",     # 13
        "rknee",     # 14
        "lankle",    # 15
        "rankle",    # 16
    ]


def get_ochuman_joint_names():
    return [
        'rshoulder',
        'relbow',
        'rwrist',
        'lshoulder',
        'lelbow',
        'lwrist',
        'rhip',
        'rknee',
        'rankle',
        'lhip',
        'lknee',
        'lankle',
        'headtop',
        'neck',
        'rear',
        'lear',
        'nose',
        'reye',
        'leye'
    ]


def get_crowdpose_joint_names():
    return [
        'lshoulder',
        'rshoulder',
        'lelbow',
        'relbow',
        'lwrist',
        'rwrist',
        'lhip',
        'rhip',
        'lknee',
        'rknee',
        'lankle',
        'rankle',
        'headtop',
        'neck'
    ]

def get_coco_skeleton():
    # 0  - nose,
    # 1  - leye,
    # 2  - reye,
    # 3  - lear,
    # 4  - rear,
    # 5  - lshoulder,
    # 6  - rshoulder,
    # 7  - lelbow,
    # 8  - relbow,
    # 9  - lwrist,
    # 10 - rwrist,
    # 11 - lhip,
    # 12 - rhip,
    # 13 - lknee,
    # 14 - rknee,
    # 15 - lankle,
    # 16 - rankle,
    return np.array(
        [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [ 5, 11],
            [ 6, 12],
            [ 5, 6 ],
            [ 5, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10],
            [ 1, 2 ],
            [ 0, 1 ],
            [ 0, 2 ],
            [ 1, 3 ],
            [ 2, 4 ],
            [ 3, 5 ],
            [ 4, 6 ]
        ]
    )


def get_mpii_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "hip",       # 6
        "thorax",    # 7
        "neck",      # 8
        "headtop",   # 9
        "rwrist",    # 10
        "relbow",    # 11
        "rshoulder", # 12
        "lshoulder", # 13
        "lelbow",    # 14
        "lwrist",    # 15
    ]


def get_mpii_skeleton():
    # 0  - rankle,
    # 1  - rknee,
    # 2  - rhip,
    # 3  - lhip,
    # 4  - lknee,
    # 5  - lankle,
    # 6  - hip,
    # 7  - thorax,
    # 8  - neck,
    # 9  - headtop,
    # 10 - rwrist,
    # 11 - relbow,
    # 12 - rshoulder,
    # 13 - lshoulder,
    # 14 - lelbow,
    # 15 - lwrist,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 6 ],
            [ 6, 3 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 9 ],
            [ 7, 12],
            [12, 11],
            [11, 10],
            [ 7, 13],
            [13, 14],
            [14, 15]
        ]
    )


def get_aich_joint_names():
    return [
        "rshoulder", # 0
        "relbow",    # 1
        "rwrist",    # 2
        "lshoulder", # 3
        "lelbow",    # 4
        "lwrist",    # 5
        "rhip",      # 6
        "rknee",     # 7
        "rankle",    # 8
        "lhip",      # 9
        "lknee",     # 10
        "lankle",    # 11
        "headtop",   # 12
        "neck",      # 13
    ]


def get_aich_skeleton():
    # 0  - rshoulder,
    # 1  - relbow,
    # 2  - rwrist,
    # 3  - lshoulder,
    # 4  - lelbow,
    # 5  - lwrist,
    # 6  - rhip,
    # 7  - rknee,
    # 8  - rankle,
    # 9  - lhip,
    # 10 - lknee,
    # 11 - lankle,
    # 12 - headtop,
    # 13 - neck,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [13, 0 ],
            [13, 3 ],
            [ 0, 6 ],
            [ 3, 9 ]
        ]
    )


def get_3dpw_joint_names():
    return [
        "nose",      # 0
        "thorax",    # 1
        "rshoulder", # 2
        "relbow",    # 3
        "rwrist",    # 4
        "lshoulder", # 5
        "lelbow",    # 6
        "lwrist",    # 7
        "rhip",      # 8
        "rknee",     # 9
        "rankle",    # 10
        "lhip",      # 11
        "lknee",     # 12
        "lankle",    # 13
    ]


def get_3dpw_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 3 ],
            [ 3, 4 ],
            [ 1, 5 ],
            [ 5, 6 ],
            [ 6, 7 ],
            [ 2, 8 ],
            [ 5, 11],
            [ 8, 11],
            [ 8, 9 ],
            [ 9, 10],
            [11, 12],
            [12, 13]
        ]
    )


def get_smplcoco_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "rwrist",    # 6
        "relbow",    # 7
        "rshoulder", # 8
        "lshoulder", # 9
        "lelbow",    # 10
        "lwrist",    # 11
        "neck",      # 12
        "headtop",   # 13
        "nose",      # 14
        "leye",      # 15
        "reye",      # 16
        "lear",      # 17
        "rear",      # 18
    ]


def get_smplcoco_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 12],
            [12, 9 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [14, 15],
            [15, 17],
            [16, 18],
            [14, 16],
            [ 8, 2 ],
            [ 9, 3 ],
            [ 2, 3 ],
        ]
    )


def get_smpl_joint_names():
    return [
        'hips',            # 0
        'leftUpLeg',       # 1
        'rightUpLeg',      # 2
        'spine',           # 3
        'leftLeg',         # 4
        'rightLeg',        # 5
        'spine1',          # 6
        'leftFoot',        # 7
        'rightFoot',       # 8
        'spine2',          # 9
        'leftToeBase',     # 10
        'rightToeBase',    # 11
        'neck',            # 12
        'leftShoulder',    # 13
        'rightShoulder',   # 14
        'head',            # 15
        'leftArm',         # 16
        'rightArm',        # 17
        'leftForeArm',     # 18
        'rightForeArm',    # 19
        'leftHand',        # 20
        'rightHand',       # 21
        'leftHandIndex1',  # 22
        'rightHandIndex1', # 23
    ]


def get_smpl_paper_joint_names():
    return [
        'Hips',            # 0
        'Left Hip',       # 1
        'Right Hip',      # 2
        'Spine',           # 3
        'Left Knee',         # 4
        'Right Knee',        # 5
        'Spine_1',          # 6
        'Left Ankle',        # 7
        'Right Ankle',       # 8
        'Spine_2',          # 9
        'Left Toe',     # 10
        'Right Toe',    # 11
        'Neck',            # 12
        'Left Shoulder',    # 13
        'Right Shoulder',   # 14
        'Head',            # 15
        'Left Arm',         # 16
        'Right Arm',        # 17
        'Left Elbow',     # 18
        'Right Elbow',    # 19
        'Left Hand',        # 20
        'Right Hand',       # 21
        'Left Thumb',  # 22
        'Right Thumb', # 23
    ]


def get_smpl_neighbor_triplets():
    return [
        [ 0,  1, 2 ],  # 0
        [ 1,  4, 0 ],  # 1
        [ 2,  0, 5 ],  # 2
        [ 3,  0, 6 ],  # 3
        [ 4,  7, 1 ],  # 4
        [ 5,  2, 8 ],  # 5
        [ 6,  3, 9 ],  # 6
        [ 7, 10, 4 ],  # 7
        [ 8,  5, 11],  # 8
        [ 9, 13, 14],  # 9
        [10,  7, 4 ],  # 10
        [11,  8, 5 ],  # 11
        [12,  9, 15],  # 12
        [13, 16, 9 ],  # 13
        [14,  9, 17],  # 14
        [15,  9, 12],  # 15
        [16, 18, 13],  # 16
        [17, 14, 19],  # 17
        [18, 20, 16],  # 18
        [19, 17, 21],  # 19
        [20, 22, 18],  # 20
        [21, 19, 23],  # 21
        [22, 20, 18],  # 22
        [23, 19, 21],  # 23
    ]


def get_smpl_skeleton():
    return np.array(
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
        ]
    )


def map_spin_joints_to_smpl():
    # this function primarily will be used to copy 2D keypoint
    # confidences to pose parameters
    return [
        [(39, 27, 28), 0],  # hip,lhip,rhip->hips
        [(28,), 1],  # lhip->leftUpLeg
        [(27,), 2],  # rhip->rightUpLeg
        [(41, 27, 28, 39), 3],  # Spine->spine
        [(29,), 4],  # lknee->leftLeg
        [(26,), 5],  # rknee->rightLeg
        [(41, 40, 33, 34,), 6],  # spine, thorax ->spine1
        [(30,), 7],  # lankle->leftFoot
        [(25,), 8],  # rankle->rightFoot
        [(40, 33, 34), 9],  # thorax,shoulders->spine2
        [(30,), 10],  # lankle -> leftToe
        [(25,), 11],  # rankle -> rightToe
        [(37, 42, 33, 34), 12],  # neck, shoulders -> neck
        [(34,), 13],  # lshoulder->leftShoulder
        [(33,), 14],  # rshoulder->rightShoulder
        [(33, 34, 38, 43, 44, 45, 46, 47, 48,), 15],  # nose, eyes, ears, headtop, shoulders->head
        [(34,), 16],  # lshoulder->leftArm
        [(33,), 17],  # rshoulder->rightArm
        [(35,), 18],  # lelbow->leftForeArm
        [(32,), 19],  # relbow->rightForeArm
        [(36,), 20],  # lwrist->leftHand
        [(31,), 21],  # rwrist->rightHand
        [(36,), 22],  # lhand -> leftHandIndex
        [(31,), 23],  # rhand -> rightHandIndex
    ]


def map_smpl_to_common():
    return [
        [(11, 8), 0], # rightToe, rightFoot -> rankle
        [(5,), 1], # rightleg -> rknee,
        [(2,), 2], # rhip
        [(1,), 3], # lhip
        [(4,), 4], # leftLeg -> lknee
        [(10, 7), 5], # lefttoe, leftfoot -> lankle
        [(21, 23), 6], # rwrist
        [(18,), 7], # relbow
        [(17, 14), 8],  # rshoulder
        [(16, 13), 9],  # lshoulder
        [(19,), 10],  # lelbow
        [(20, 22), 11],  # lwrist
        [(0, 3, 6, 9, 12), 12],  # neck
        [(15,), 13],  # headtop
    ]


def relation_among_spin_joints():
    # this function primarily will be used to copy 2D keypoint
    # confidences to 3D joints
    return [
        [(), 25],
        [(), 26],
        [(39,), 27],
        [(39,), 28],
        [(), 29],
        [(), 30],
        [(), 31],
        [(), 32],
        [(), 33],
        [(), 34],
        [(), 35],
        [(), 36],
        [(40,42,44,43,38,33,34,), 37],
        [(43,44,45,46,47,48,33,34,), 38],
        [(27,28,), 39],
        [(27,28,37,41,42,), 40],
        [(27,28,39,40,), 41],
        [(37,38,44,45,46,47,48,), 42],
        [(44,45,46,47,48,38,42,37,33,34,), 43],
        [(44,45,46,47,48,38,42,37,33,34), 44],
        [(44,45,46,47,48,38,42,37,33,34), 45],
        [(44,45,46,47,48,38,42,37,33,34), 46],
        [(44,45,46,47,48,38,42,37,33,34), 47],
        [(44,45,46,47,48,38,42,37,33,34), 48],
    ]