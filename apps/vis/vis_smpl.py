'''
  @ Date: 2021-07-19 20:37:16
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-08-28 20:42:44
  @ FilePath: /EasyMocapRelease/apps/vis/vis_smpl.py
'''
from easymocap.config import Config, load_object
import open3d as o3d
from easymocap.visualize.o3dwrapper import Vector3dVector, create_mesh, create_coord
import numpy as np

def update_vis(vis, mesh, body_model, params):
    vertices = body_model(return_verts=True, return_tensor=False, **params)[0]
    mesh.vertices = Vector3dVector(vertices)
    vis.update_geometry(model)
    vis.poll_events()
    vis.update_renderer()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, 
        default='config/model/smpl_neutral.yml')
    parser.add_argument('--key', type=str, 
        default='poses')
    parser.add_argument('--num', type=int, default=50)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    key = args.key
    
    config = Config.load(args.cfg)
    body_model = load_object(config.module, config.args)
    params = body_model.init_params(1)
    vertices = body_model(return_verts=True, return_tensor=False, **params)
    joints = body_model(return_verts=False, return_smpl_joints=True, return_tensor=False, **params)
    
    model = create_mesh(vertices=vertices[0], faces=body_model.faces)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=900, height=900)
    vis.add_geometry(model)
    params = body_model.init_params(1)
    var_ranges = np.linspace(0, np.pi/2, args.num)
    var_ranges = np.concatenate([-var_ranges, -var_ranges[::-1], var_ranges, var_ranges[::-1]])
    for npose in range(54, params[key].shape[1]):
        print('[Vis] {}: {}'.format(key, npose))
        for i in range(var_ranges.shape[0]):
            params[key][0, npose] = var_ranges[i]
            update_vis(vis, model, body_model, params)
    import ipdb; ipdb.set_trace()