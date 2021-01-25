'''
  @ Date: 2021-01-17 21:14:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-25 19:34:46
  @ FilePath: /EasyMocapRelease/code/vis_render.py
'''
# visualize the results by pyrender
import pyrender # first import the pyrender
from collections import namedtuple
from dataset.base import MVBase
from dataset.config import CONFIG
import numpy as np
from tqdm import tqdm
from visualize.geometry import create_ground

Person = namedtuple('Person', ['vertices', 'keypoints3d'])

def inBound(keypoints3d, bound):
    if bound is None:
        return True
    valid = np.where(keypoints3d[:, -1] > 0.01)[0]
    kpts = keypoints3d[valid]
    crit = (kpts[:, 0] > bound[0][0]) & (kpts[:, 0] < bound[1][0]) &\
        (kpts[:, 1] > bound[0][1]) & (kpts[:, 1] < bound[1][1]) &\
        (kpts[:, 2] > bound[0][2]) & (kpts[:, 2] < bound[1][2])
    if crit.sum()/crit.shape[0] < 0.8:
        return False
    else:
        return True 

def visualize(path, sub, out, mode, rend_type, args):
    config = CONFIG[mode]
    no_img = False
    dataset = MVBase(path, cams=sub, config=config,
        undis=args.undis, no_img=no_img, out=out)
    dataset.skel_path = args.skel
    if rend_type in ['skel']:
        from visualize.skelmodel import SkelModel
        body_model = SkelModel(config['nJoints'], config['kintree'])
    elif rend_type in ['mesh']:
        from smplmodel import load_model
        body_model = load_model(args.gender, model_type=args.model)
        smpl_model = body_model
    elif rend_type == 'smplskel':
        from smplmodel import load_model
        smpl_model = load_model(args.gender, model_type=args.model)
        from visualize.skelmodel import SkelModel
        body_model = SkelModel(config['nJoints'], config['kintree'])
    
    dataset.writer.save_origin = args.save_origin
    start, end = args.start, min(args.end, len(dataset))
    bound = None
    if args.scene == 'none':
        ground = create_ground(step=0.5)
    elif args.scene == 'hw':
        ground = create_ground(step=1, xrange=14, yrange=10, two_sides=False)
        bound = [[0, 0, 0], [14, 10, 2.5]]
    else:
        ground = create_ground(step=1, xrange=28, yrange=15, two_sides=False)
    for nf in tqdm(range(start, end), desc='rendering'):
        images, annots = dataset[nf]
        if rend_type == 'skel':
            infos = dataset.read_skel(nf)
        else:
            infos = dataset.read_smpl(nf)
        # body_model: input: keypoints3d/smpl params, output: vertices, (colors)
        # The element of peopleDict must have `id`, `vertices`
        peopleDict = {}
        for info in infos:
            if rend_type == 'skel':
                joints = info['keypoints3d']
            else:
                joints = smpl_model(return_verts=False, return_tensor=False, **info)[0]
            if not inBound(joints, bound):
                continue
            if rend_type == 'smplskel':
                joints = smpl_model(return_verts=False, return_tensor=False, **info)[0]
                joints = np.hstack([joints, np.ones((joints.shape[0], 1))])
                info_new = {'id': info['id'], 'keypoints3d': joints}
                vertices = body_model(return_verts=True, return_tensor=False, **info_new)[0]
            else:
                vertices = body_model(return_verts=True, return_tensor=False, **info)[0]
            peopleDict[info['id']] = Person(vertices=vertices, keypoints3d=None)
        dataset.vis_smpl(peopleDict, faces=body_model.faces, images=images, nf=nf, 
            sub_vis=args.sub_vis, mode=rend_type, extra_data=[ground], add_back=args.add_back)

if __name__ == "__main__":
    from mytools.cmd_loader import load_parser
    parser = load_parser()
    parser.add_argument('--type', type=str, default='mesh', choices=['skel', 'mesh', 'smplskel'])
    parser.add_argument('--scene', type=str, default='none', choices=['none', 'zjub', 'hw'])
    parser.add_argument('--skel', type=str, default=None)
    parser.add_argument('--add_back', action='store_true')
    parser.add_argument('--save_origin', action='store_true')
    args = parser.parse_args()
    visualize(args.path, args.sub, args.out, args.body, args.type, args)