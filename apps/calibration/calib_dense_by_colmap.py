'''
  @ Date: 2022-06-20 15:00:58
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-06-20 15:47:03
  @ FilePath: /EasyMocapPublic/apps/calibration/calib_dense_by_colmap.py
'''
# This script helps to calibration dense cameras with colmap
from os.path import join
import os
from easymocap.mytools.colmap_wrapper import COLMAPDatabase, colmap_ba, colmap_dense, colmap_feature_match, copy_images, create_empty_db
from easymocap.mytools.colmap_wrapper import colmap_feature_extract
from easymocap.mytools.debug_utils import log

def run_dense(path, colmap, args):
    # out = join(out_root, '{}_{:06d}'.format(seq, nf))
    sparse_dir = join(path, 'sparse', 'model')
    os.makedirs(sparse_dir, exist_ok=True)
    # create blank database
    database_name = join(path, 'database.db')
    create_empty_db(database_name)

    # if not args.no_camera:
    # db = COLMAPDatabase.connect(database_name)
    #     cameras_colmap, cameras_map = create_cameras(db, cameras, list(image_names.keys()))
    #     write_cameras_binary(cameras_colmap, join(sparse_dir, 'cameras.bin'))
    #     images = create_images(db, cameras, cameras_map, image_names)
    #     write_images_binary(images, join(sparse_dir, 'images.bin'))
    #     write_points3d_binary({}, join(sparse_dir, 'points3D.bin'))
    # db.commit()
    # db.close()

    # perform COLMAP extracting and matching
    colmap_feature_extract(colmap, path, args.share_camera, args.add_mask)
    colmap_feature_match(colmap, path)
        # check the matches
    db = COLMAPDatabase.connect(join(path, 'database.db'))
    geometry = db.read_two_view_geometry()
    db.close()
    num_pairs = len(geometry)
    num_matches = []
    for key, val in geometry.items():
        log('cameras: {} has {:5d} matches'.format(key, len(val['matches'])))
        num_matches.append(len(val['matches']))
    # log('[match] {}_{:06d}: {} pairs: {}'.format(seq, nf, num_pairs, sum(num_matches)))
    # if not args.no_camera: continue
    colmap_ba(colmap, path)
    if args.dense:
        colmap_dense(colmap, path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--colmap', type=str, default='colmap')
    parser.add_argument('--f', type=int, default=0)
    parser.add_argument('--share_camera', action='store_true')
    parser.add_argument('--add_mask', action='store_true')
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    copy_images(args.path, args.out, args.f)
    run_dense(args.out, args.colmap, args)
