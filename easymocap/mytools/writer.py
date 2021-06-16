import os
from os.path import join
import numpy as np
import cv2
# from mytools import save_json, merge
# from ..mytools import merge, plot_bbox, plot_keypoints
# from mytools.file_utils import read_json, save_json, read_annot, read_smpl, write_smpl, get_bbox_from_pose
from .vis_base import plot_bbox, plot_keypoints, merge
from .file_utils import write_keypoints3d, write_smpl, mkout, mkdir

class FileWriter:
    """
        This class provides:
                      |  write  | vis
        - keypoints2d |    x    |  o
        - keypoints3d |    x    |  o
        - smpl        |    x    |  o
    """
    def __init__(self, output_path, config=None, basenames=[], cfg=None) -> None:
        self.out = output_path
        keys = ['keypoints3d', 'match', 'smpl', 'skel', 'repro', 'keypoints']
        output_dict = {key:join(self.out, key) for key in keys}
        self.output_dict = output_dict
        
        self.basenames = basenames
        if cfg is not None:
            print(cfg, file=open(join(output_path, 'exp.yml'), 'w'))
        self.save_origin = False
        self.config = config
    
    def write_keypoints2d(self, ):
        pass

    def vis_keypoints2d_mv(self, images, lDetections, outname=None,
        vis_id=True):
        mkout(outname)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            for det in lDetections[nv]:
                pid = det['id']
                if 'keypoints2d' in det.keys():
                    keypoints = det['keypoints2d']
                else:
                    keypoints = det['keypoints']
                if 'bbox' not in det.keys():
                    bbox = get_bbox_from_pose(keypoints, img)
                else:
                    bbox = det['bbox']
                plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
                plot_keypoints(img, keypoints, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        if len(images_vis) > 1:
            images_vis = merge(images_vis, resize=not self.save_origin)
        else:
            images_vis = images_vis[0]
        if outname is not None:
            # savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            # savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            cv2.imwrite(outname, images_vis)
        return images_vis
    
    def write_keypoints3d(self, results, outname):
        write_keypoints3d(outname, results)
    
    def vis_keypoints3d(self, result, outname):
        # visualize the repro of keypoints3d
        import ipdb; ipdb.set_trace()
    
    def vis_smpl(self, render_data, images, cameras, outname, add_back):
        mkout(outname)
        from ..visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None)
        render_results = render.render(render_data, cameras, images, add_back=add_back)
        image_vis = merge(render_results, resize=not self.save_origin)
        cv2.imwrite(outname, image_vis)
        return image_vis

    def _write_keypoints3d(self, results, nf=-1, base=None):
        os.makedirs(self.output_dict['keypoints3d'], exist_ok=True)
        if base is None:
            base = '{:06d}'.format(nf)
        savename = join(self.output_dict['keypoints3d'], '{}.json'.format(base))
        save_json(savename, results)
    
    def vis_detections(self, images, lDetections, nf, key='keypoints', to_img=True, vis_id=True):
        os.makedirs(self.output_dict[key], exist_ok=True)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            for det in lDetections[nv]:
                if key == 'match' and 'id_match' in det.keys():
                    pid = det['id_match']
                else:
                    pid = det['id']
                if key not in det.keys():
                    keypoints = det['keypoints']
                else:
                    keypoints = det[key]
                if 'bbox' not in det.keys():
                    bbox = get_bbox_from_pose(keypoints, img)
                else:
                    bbox = det['bbox']
                plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
                plot_keypoints(img, keypoints, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        image_vis = merge(images_vis, resize=not self.save_origin)
        if to_img:
            savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            cv2.imwrite(savename, image_vis)
        return image_vis
    
    def write_smpl(self, results, outname):
        write_smpl(outname, results)

    def vis_keypoints3d(self, infos, nf, images, cameras, mode='repro'):
        out = join(self.out, mode)
        os.makedirs(out, exist_ok=True)
        # cameras: (K, R, T)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            K, R, T = cameras['K'][nv], cameras['R'][nv], cameras['T'][nv]
            P = K @ np.hstack([R, T])
            for info in infos:
                pid = info['id']
                keypoints3d = info['keypoints3d']
                # 重投影
                kcam = np.hstack([keypoints3d[:, :3], np.ones((keypoints3d.shape[0], 1))]) @ P.T
                kcam = kcam[:, :2]/kcam[:, 2:]
                k2d = np.hstack((kcam, keypoints3d[:, -1:]))
                bbox = get_bbox_from_pose(k2d, img)
                plot_bbox(img, bbox, pid=pid, vis_id=pid)
                plot_keypoints(img, k2d, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        savename = join(out, '{:06d}.jpg'.format(nf))
        image_vis = merge(images_vis, resize=False)
        cv2.imwrite(savename, image_vis)
        return image_vis

    def _vis_smpl(self, render_data_, nf, images, cameras, mode='smpl', base=None, add_back=False, extra_mesh=[]):
        out = join(self.out, mode)
        os.makedirs(out, exist_ok=True)
        from visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None, extra_mesh=extra_mesh)
        if isinstance(render_data_, list): # different view have different data
            for nv, render_data in enumerate(render_data_):
                render_results = render.render(render_data, cameras, images)
                image_vis = merge(render_results, resize=not self.save_origin)
                savename = join(out, '{:06d}_{:02d}.jpg'.format(nf, nv))
                cv2.imwrite(savename, image_vis)
        else:
            render_results = render.render(render_data_, cameras, images, add_back=add_back)
            image_vis = merge(render_results, resize=not self.save_origin)
            if nf != -1:
                if base is None:
                    base = '{:06d}'.format(nf)
                savename = join(out, '{}.jpg'.format(base))
                cv2.imwrite(savename, image_vis)
            return image_vis