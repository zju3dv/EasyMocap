'''
  @ Date: 2022-06-20 15:03:50
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-02 22:38:25
  @ FilePath: /EasyMocapPublic/easymocap/mytools/colmap_wrapper.py
'''

import shutil
import sys
import os
import sqlite3
import numpy as np
from os.path import join
import cv2
from .debug_utils import mkdir, run_cmd, log, mywarn
from .colmap_structure import Camera, Image, CAMERA_MODEL_NAMES
from .colmap_structure import rotmat2qvec
from .colmap_structure import read_points3d_binary

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return image_id1, image_id2

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if blob is None:
        return np.empty((0, 2), dtype=dtype)
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches, extra, config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        if 'qvec' in extra.keys():
            self.execute(
                "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pair_id,) + matches.shape + (array_to_blob(matches), config,
                array_to_blob(extra['F']), array_to_blob(extra['E']), array_to_blob(extra['H']),
                array_to_blob(extra['qvec']), array_to_blob(extra['tvec'])))
        else:
            self.execute(
                "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (pair_id,) + matches.shape + (array_to_blob(matches), config,
                array_to_blob(extra['F']), array_to_blob(extra['E']), array_to_blob(extra['H'])))
    
    def read_images(self):
        image_id_to_name, name_to_image_id = {}, {}
        image_results = self.execute("SELECT * FROM images")
        for result in image_results:
            image_id, name, camera_id, q0, q1, q2, q3, t0, t1, t2 = result
            image_id_to_name[image_id] = name
            name_to_image_id[name] = image_id
        return image_id_to_name, name_to_image_id

    def read_keypoints(self, mapping=None):
        image_id_to_keypoints = {}
        keypoints_results = self.execute("SELECT * FROM keypoints")
        for keypoints_result in keypoints_results:
            image_id, rows, cols, keypoints = keypoints_result
            keypoints = blob_to_array(keypoints, np.float32, (rows, cols))
            if mapping is None:
                image_id_to_keypoints[image_id] = keypoints
            else:
                image_id_to_keypoints[mapping[image_id]] = keypoints
        return image_id_to_keypoints
    
    def read_descriptors(self, mapping=None):
        image_id_to_descriptors = {}
        descriptors_results = self.execute("SELECT * FROM descriptors")
        for descriptors_result in descriptors_results:
            image_id, rows, cols, keypoints = descriptors_result
            keypoints = blob_to_array(keypoints, np.uint8, (rows, cols))
            if mapping is None:
                image_id_to_descriptors[image_id] = keypoints
            else:
                image_id_to_descriptors[mapping[image_id]] = keypoints
        return image_id_to_descriptors

    def read_matches(self, mapping=None):
        matches_results = self.execute("SELECT * FROM matches")
        matches = {}
        for matches_result in matches_results:
            pair_id, rows, cols, match = matches_result
            image_id0, image_id1 = pair_id_to_image_ids(pair_id)
            if rows == 0:
                continue
            match = blob_to_array(match, dtype=np.uint32, shape=(rows, cols))
            if mapping is not None:
                image_id0 = mapping[image_id0]
                image_id1 = mapping[image_id1]
            matches[(image_id0, image_id1)] = match
        return matches
    
    def read_two_view_geometry(self, mapping=None):
        geometry = self.execute("SELECT * FROM two_view_geometries")
        geometries = {}
        for _data in geometry:
            if len(_data) == 10:
                pair_id, rows, cols, data, config, F, E, H, qvec, tvec = _data
                extra = {
                    'F': F,
                    'E': E,
                    'H': H,
                    'qvec': qvec,
                    'tvec': tvec
                }
            elif len(_data) == 8:
                pair_id, rows, cols, data, config, F, E, H = _data
                extra = {
                    'F': F,
                    'E': E,
                    'H': H,
                }
            for key, val in extra.items():
                extra[key] = blob_to_array(val, dtype=np.float64)
            image_id0, image_id1 = pair_id_to_image_ids(pair_id)
            match = blob_to_array(data, dtype=np.uint32, shape=(rows, cols))
            if rows == 0:continue
            if mapping is not None:
                image_id0 = mapping[image_id0]
                image_id1 = mapping[image_id1]
            geometries[(image_id0, image_id1)] = {'matches': match, 'extra': extra, 'config': config}
        return geometries

def create_empty_db(database_path):
    if os.path.exists(database_path):
        mywarn('Removing old database: {}'.format(database_path))
        os.remove(database_path)
    print('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()

def create_cameras(db, cameras, subs, width, height, share_intri=True):
    model = 'OPENCV'
    if share_intri:
        cam_id = 1
        K = cameras[subs[0]]['K']
        D = cameras[subs[0]]['dist'].reshape(1, 5)
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[0, 1], D[0, 2], D[0, 3], D[0, 4], 0, 0, 0
        
        params = [fx, fy, cx, cy, k1, k2, p1, p2]
        # params = [fx, fy, cx, cy, 0, 0, 0, 0]
        camera = Camera(
            id=cam_id,
            model=model,
            width=width,
            height=height,
            params=params
        )
        cameras_colmap = {cam_id: camera}
        cameras_map = {sub:cam_id for sub in subs}
        # 
        db.add_camera(CAMERA_MODEL_NAMES[model].model_id, width, height, params,
                   prior_focal_length=False, camera_id=cam_id)
    else:
        raise NotImplementedError
    return cameras_colmap, cameras_map

def create_images(db, cameras, cameras_map, image_names):
    subs = sorted(list(image_names.keys()))
    images = {}
    for sub, image_name in image_names.items():
        img_id = subs.index(sub) + 1
        R = cameras[sub]['R']
        T = cameras[sub]['T']
        qvec = rotmat2qvec(R)
        tvec = T.T[0]
        image = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=cameras_map[sub],
            name=os.path.basename(image_name),
            xys=[],
            point3D_ids=[]
        )
        images[img_id] = image
        db.add_image(image.name, camera_id=image.camera_id,
            prior_q=image.qvec, prior_t=image.tvec, image_id=img_id)
    return images

def copy_images(data, out, nf=0, copy_func=shutil.copyfile, mask='mask', add_mask=True):
    subs = sorted(os.listdir(join(data, 'images')))
    image_names = {}
    for sub in subs:
        srcname = join(data, 'images', sub, '{:06d}.jpg'.format(nf))
        if not os.path.exists(srcname):
            mywarn('{} not exists, skip'.format(srcname))
            return False
        dstname = join(out, 'images', '{}.jpg'.format(sub))
        image_names[sub] = dstname
        if os.path.exists(dstname):
            continue
        os.makedirs(os.path.dirname(dstname), exist_ok=True)
        copy_func(srcname, dstname)
        mskname = join(data, mask, sub, '{:06d}.png'.format(nf))
        dstname = join(out, 'mask', '{}.jpg.png'.format(sub))
        if os.path.exists(mskname) and add_mask:
            os.makedirs(os.path.dirname(dstname), exist_ok=True)
            copy_func(mskname, dstname)
    return True, image_names

def colmap_feature_extract(colmap, path, share_camera, add_mask, gpu=False,
    share_camera_per_folder=False):
    '''
struct SiftMatchingOptions {
  // Number of threads for feature matching and geometric verification.
  int num_threads = -1;

  // Whether to use the GPU for feature matching.
  bool use_gpu = true;

  // Index of the GPU used for feature matching. For multi-GPU matching,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum distance ratio between first and second best match.
  double max_ratio = 0.8;

  // Maximum distance to best match.
  double max_distance = 0.7;

  // Whether to enable cross checking in matching.
  bool cross_check = true;

  // Maximum number of matches.
  int max_num_matches = 32768;

  // Maximum epipolar error in pixels for geometric verification.
  double max_error = 4.0;

  // Confidence threshold for geometric verification.
  double confidence = 0.999;

  // Minimum/maximum number of RANSAC iterations. Note that this option
  // overrules the min_inlier_ratio option.
  int min_num_trials = 100;
  int max_num_trials = 10000;

  // A priori assumed minimum inlier ratio, which determines the maximum
  // number of iterations.
  double min_inlier_ratio = 0.25;

  // Minimum number of inliers for an image pair to be considered as
  // geometrically verified.
  int min_num_inliers = 15;

  // Whether to attempt to estimate multiple geometric models per image pair.
  bool multiple_models = false;

  // Whether to perform guided matching, if geometric verification succeeds.
  bool guided_matching = false;

  bool Check() const;
};
'''
    cmd = f'{colmap} feature_extractor --database_path {path}/database.db \
--image_path {path}/images \
--SiftExtraction.peak_threshold 0.006 \
--ImageReader.camera_model OPENCV \
'
    if share_camera and not share_camera_per_folder:
        cmd += ' --ImageReader.single_camera 1'
    elif share_camera_per_folder:
        cmd += ' --ImageReader.single_camera_per_folder 1'
    if gpu:
        cmd += ' --SiftExtraction.use_gpu 1'
        cmd += ' --SiftExtraction.gpu_index 0'
    if add_mask:
        cmd += f' --ImageReader.mask_path {path}/mask'
    cmd += f' >> {path}/log.txt'
    run_cmd(cmd)

def colmap_feature_match(colmap, path, gpu=False):
    cmd = f'{colmap} exhaustive_matcher --database_path {path}/database.db \
--SiftMatching.guided_matching 0 \
--SiftMatching.max_ratio 0.8 \
--SiftMatching.max_distance 0.5 \
--SiftMatching.cross_check 1 \
--SiftMatching.max_error 4 \
--SiftMatching.max_num_matches 32768 \
--SiftMatching.confidence 0.9999 \
--SiftMatching.max_num_trials 10000 \
--SiftMatching.min_inlier_ratio 0.25 \
--SiftMatching.min_num_inliers 30'
    if gpu:
        cmd += ' --SiftMatching.use_gpu 1'
        cmd += ' --SiftMatching.gpu_index 0'
    cmd += f' >> {path}/log.txt'
    run_cmd(cmd)

def colmap_ba(colmap, path, with_init=False):
    if with_init:
        cmd = f'{colmap} point_triangulator --database_path {path}/database.db \
--image_path {path}/images \
--input_path {path}/sparse/0 \
--output_path {path}/sparse/0 \
--Mapper.tri_merge_max_reproj_error 3 \
--Mapper.ignore_watermarks 1 \
--Mapper.filter_max_reproj_error 2 \
>> {path}/log.txt'
        run_cmd(cmd)
        cmd = f'{colmap} bundle_adjuster \
--input_path {path}/sparse/0 \
--output_path {path}/sparse/0 \
--BundleAdjustment.max_num_iterations 1000 \
>> {path}/log.txt'
        run_cmd(cmd)
        points3d = read_points3d_binary(join(path, 'sparse', '0', 'points3D.bin'))
        pids = list(points3d.keys())
        mean_error = np.mean([points3d[p].error for p in pids])
        log('Triangulate {} points, mean error: {:.2f} pixel'.format(len(pids), mean_error))
    else:
        mkdir(join(path, 'sparse'))
        cmd = f'{colmap} mapper --database_path {path}/database.db --image_path {path}/images --output_path {path}/sparse \
    --Mapper.ba_refine_principal_point 1 \
    --Mapper.ba_global_max_num_iterations 1000 \
    >> {path}/log.txt'
        run_cmd(cmd)
    

def colmap_dense(colmap, path):
    mkdir(join(path, 'dense'))
    cmd = f'{colmap} image_undistorter --image_path {path}/images --input_path {path}/sparse/0 --output_path {path}/dense --output_type COLMAP --max_image_size 2000'
    run_cmd(cmd)
    cmd = f'{colmap} patch_match_stereo \
--workspace_path {path}/dense \
--workspace_format COLMAP \
--PatchMatchStereo.geom_consistency true \
>> {path}/log.txt'

    run_cmd(cmd)        
    cmd = f'{colmap} stereo_fusion \
--workspace_path {path}/dense \
--workspace_format COLMAP \
--input_type geometric \
--output_path {path}/dense/fused.ply \
>> {path}/log.txt'
    run_cmd(cmd)        
