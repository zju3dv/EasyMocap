from operator import imod
import numpy as np
from tqdm import tqdm
from os.path import join
from easymocap.dataset.mv1pmf_mirror import ImageFolderMirror as ImageFolder
from easymocap.mytools import Timer
from easymocap.smplmodel import load_model, merge_params, select_nf
from easymocap.estimator import SPIN, init_with_spin
from easymocap.pipeline.mirror import multi_stage_optimize

def demo_1v1p1f_smpl_mirror(path, body_model, spin_model, args):
    "Optimization for single image"
    # 0. construct the dataset
    dataset = ImageFolder(path, out=args.out, kpts_type=args.body)
    if args.gtK:
        dataset.gtK = True
        dataset.load_gt_cameras()
    start, end = args.start, min(args.end, len(dataset))

    for nf in tqdm(range(start, end, args.step), desc='Optimizing'):
        image, annots = dataset[nf]
        if len(annots) < 2:
            continue
        annots = annots[:2]
        camera = dataset.camera(nf)
        # initialize the SMPL parameters
        body_params_all = []
        bboxes, keypoints2d, pids = [], [], []
        for i, annot in enumerate(annots):
            assert annot['id'] == i, (i, annot['id'])
            result = init_with_spin(body_model, spin_model, image, 
                annot['bbox'], annot['keypoints'], camera)
            body_params_all.append(result['body_params'])
            bboxes.append(annot['bbox'])
            keypoints2d.append(annot['keypoints'])
            pids.append(annot['id'])
        bboxes = np.vstack(bboxes)
        keypoints2d = np.stack(keypoints2d)
        body_params = merge_params(body_params_all)
        # bboxes: (nViews(2), 1, 5); keypoints2d: (nViews(2), 1, nJoints, 3)
        bboxes = bboxes[:, None]
        keypoints2d = keypoints2d[:, None]
        if args.normal:
            normal = dataset.normal(nf)[None, :, :]
        else:
            normal = None
        body_params = multi_stage_optimize(body_model, body_params, bboxes, keypoints2d, Pall=camera['P'], normal=normal, args=args)
        vertices = body_model(return_verts=True, return_tensor=False, **body_params)
        keypoints = body_model(return_verts=False, return_tensor=False, **body_params)
        write_data = [{'id': pids[i], 'keypoints3d': keypoints[i]} for i in range(len(pids))]
        # write out the results
        dataset.write_keypoints3d(write_data, nf)
        for i in range(len(pids)):
            write_data[i].update(select_nf(body_params, i))
        if args.vis_smpl:
            # render the results
            render_data = {pids[i]: {
                'vertices': vertices[i], 
                'faces': body_model.faces, 
                'vid': 0, 'name': 'human_{}'.format(pids[i])} for i in range(len(pids))}
            dataset.vis_smpl(render_data, image, camera, nf)
        dataset.write_smpl(write_data, nf)

def demo_1v1pmf_smpl_mirror(path, body_model, spin_model, args):
    subs = args.sub
    assert len(subs) > 0
    # 遍历所有文件夹
    for sub in subs:
        dataset = ImageFolder(path, subs=[sub], out=args.out, kpts_type=args.body)
        start, end = args.start, min(args.end, len(dataset))
        frames = list(range(start, end, args.step))
        nFrames = len(frames)
        pids = [0, 1]
        body_params_all = {pid:[None for nf in frames] for pid in pids}
        bboxes = {pid:[None for nf in frames] for pid in pids}
        keypoints2d = {pid:[None for nf in frames] for pid in pids}
        for nf in tqdm(frames, desc='loading'):
            image, annots = dataset[nf]
            # 这个时候如果annots不够 不能够跳过了，需要进行补全
            camera = dataset.camera(nf)
            # 初始化每个人的SMPL参数
            for i, annot in enumerate(annots):
                pid = annot['id']
                if pid not in pids:
                    continue
                result = init_with_spin(body_model, spin_model, image, 
                    annot['bbox'], annot['keypoints'], camera)
                body_params_all[pid][nf-start] = result['body_params']
                bboxes[pid][nf-start] = annot['bbox']
                keypoints2d[pid][nf-start] = annot['keypoints']
        # stack [p1f1, p1f2, p1f3, ..., p1fn, p2f1, p2f2, p2f3, ..., p2fn]
        # TODO:for missing bbox
        body_params = merge_params([merge_params(body_params_all[pid]) for pid in pids])
        # bboxes: (nViews, nFrames, 5)
        bboxes = np.stack([np.stack(bboxes[pid]) for pid in pids])
        # keypoints: (nViews, nFrames, nJoints, 3)
        keypoints2d = np.stack([np.stack(keypoints2d[pid]) for pid in pids])
        # optimize
        P = dataset.camera(start)['P']
        if args.normal:
            normal = dataset.normal_all(start=start, end=end)
        else:
            normal = None
        body_params = multi_stage_optimize(body_model, body_params, bboxes, keypoints2d, Pall=P, normal=normal, args=args)
        # write
        vertices = body_model(return_verts=True, return_tensor=False, **body_params)
        keypoints = body_model(return_verts=False, return_tensor=False, **body_params)
        dataset.no_img = not args.vis_smpl
        for nf in tqdm(frames, desc='rendering'):
            idx = nf - start
            write_data = [{'id': pids[i], 'keypoints3d': keypoints[i*nFrames+idx]} for i in range(len(pids))]
            dataset.write_keypoints3d(write_data, nf)
            for i in range(len(pids)):
                write_data[i].update(select_nf(body_params, i*nFrames+idx))
            dataset.write_smpl(write_data, nf)
            # 保存结果
            if args.vis_smpl:
                image, annots = dataset[nf]
                camera = dataset.camera(nf)
                render_data = {pids[i]: {
                    'vertices': vertices[i*nFrames+idx], 
                    'faces': body_model.faces, 
                    'vid': 0, 'name': 'human_{}'.format(pids[i])} for i in range(len(pids))}
                dataset.vis_smpl(render_data, image, camera, nf)

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--skel', type=str, default=None, 
        help='path to keypoints3d')
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--gtK', action='store_true')
    parser.add_argument('--normal', action='store_true',
        help='set to use the normal of the mirror')
    args = parse_parser(parser)
    
    helps = '''
  Demo code for single view and one person with mirror:

    - Input : {}: [{}]
    - Output: {}
    - Body  : {} => {}, {}
    '''.format(args.path, ', '.join(args.sub), args.out,
        args.model, args.gender, args.body)
    print(helps)
    with Timer('Loading {}, {}'.format(args.model, args.gender)):
        body_model = load_model(args.gender, model_type=args.model)
    with Timer('Loading SPIN'):
        spin_model = SPIN(
            SMPL_MEAN_PARAMS='data/models/smpl_mean_params.npz', 
            checkpoint='data/models/spin_checkpoint.pt', 
            device=body_model.device)
    if args.video:
        demo_1v1pmf_smpl_mirror(args.path, body_model, spin_model, args)
    else:
        demo_1v1p1f_smpl_mirror(args.path, body_model, spin_model, args)