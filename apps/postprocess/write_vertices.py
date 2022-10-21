'''
  @ Date: 2021-07-23 15:58:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-21 16:33:33
  @ FilePath: /EasyMocapPublic/apps/postprocess/write_vertices.py
'''
from easymocap.config.baseconfig import load_object, Config
from easymocap.mytools import Timer
from easymocap.mytools.file_utils import save_json, write_keypoints3d, write_vertices
from easymocap.mytools.reader import read_smpl
from easymocap.bodymodel.base import Params
# This script helps you to convert SMPL parameters to vertices
from os.path import join
from glob import glob
from tqdm import tqdm
import os

def write_func(tasks):
    for i in tqdm(range(len(tasks))):
        func, name, data = tasks[i]
        func(name, data)

def main(inp, out, body_model):
    filenames = sorted(glob(join(inp, '*.json'))) + sorted(glob(join(inp, '*', '*.json')))
    filenames.sort(key=lambda x:os.path.basename(x))

    write_tasks = []
    threads = []
    timer = False
    for filename in tqdm(filenames):
        with Timer('read', not timer):
            params = read_smpl(filename)
        params_ = Params.merge(params, share_shape=False)
        output = []
        if args.mode == 'vertices' or args.mode == 'mesh':
            with Timer('forward', not timer):
                vertices = body_model(return_verts=True, return_tensor=False, return_smpl_joints=False,
                     **params_)
            for i, data in enumerate(params):
                output.append({
                    'id': data['id'],
                    'vertices': vertices[i]
                })
        elif args.mode == 'keypoints':
            keypoints = body_model(return_verts=False, return_tensor=False, return_smpl_joints=False,
                 **params_)
            for i, data in enumerate(params):
                output.append({
                    'id': data['id'],
                    'type': 'body25',
                    'keypoints3d': keypoints[i]
                })
        elif args.mode == 'smpljoints':
            smpljoints = body_model(return_verts=False, return_tensor=True, return_smpl_joints=True,
                **params_)
            for i, data in enumerate(params):
                output.append({
                    'id': data['id'],
                    'keypoints3d': smpljoints[i]
                })
        basename = filename.replace(inp+'/', '')
        outname = join(out, basename)
        if False:
            import numpy as np
            faces = body_model.faces
            vertices = vertices[0]
            v_face = vertices[faces]
            edge0 = np.linalg.norm(v_face[:, 0] - v_face[:, 1], axis=-1)
            import open3d as o3d
            import ipdb;ipdb.set_trace()
        if args.mode == 'vertices':
            write_tasks.append((write_vertices, outname, output))
            # write_vertices(outname, output)
        elif args.mode == 'mesh':
            # todo
            import trimesh
            for i, data in enumerate(params):
                mesh = trimesh.Trimesh(vertices=vertices[i], faces=body_model.faces)
                outname = join(out, str(data['id'])+'_'+basename.replace('.json', '.obj'))
                mesh.export(outname)
        else:
            if args.debug:
                save_json(outname, {'annots': output})
            else:
                write_keypoints3d(outname, output)
        if len(write_tasks) == 100:
            import threading
            thread = threading.Thread(target=write_func, args=(write_tasks,)) # 应该不存在任何数据竞争
            thread.start()
            threads.append(thread)
            write_tasks = []
    if len(write_tasks) > 0:
        import threading
        thread = threading.Thread(target=write_func, args=(write_tasks,)) # 应该不存在任何数据竞争
        thread.start()
        threads.append(thread)
        write_tasks = []
    for thread in threads:
        thread.join()
    Timer.report()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--mode', type=str, default='vertices',
        choices=['vertices', 'keypoints', 'smpljoints', 'mesh'])
    parser.add_argument('--cfg_model', type=str, 
        default='config/model/smpl_neutral.yml')
    parser.add_argument('--opt_model', type=str, 
        default=[], nargs='+')    
    parser.add_argument('--keypoints', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cfg_model = Config.load(args.cfg_model, args.opt_model)

    with Timer('Loading {}'.format(args.cfg_model)):
        body_model = load_object(cfg_model.module, cfg_model.args)
    os.makedirs(args.out, exist_ok=True)
    main(args.path, args.out, body_model)