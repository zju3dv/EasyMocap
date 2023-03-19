'''
  @ Date: 2021-11-22 15:48:13
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-05-05 16:43:17
  @ FilePath: /EasyMocapPublic/easymocap/visualize/render_func.py
'''
# 提供各个接口的统一的对外的接口，封装了一些利用基础接口实现的功能
import cv2
class RenderFunc:
    def __init__(self, render) -> None:
        self.render = render
        self.position = {}
    
    def render_all(self, images, results, cameras, extra_mesh):
        camnames = list(images.keys())
        images = [images[cam] for cam in camnames]
        cameras = [cameras[cam] for cam in camnames]
        render_results = self.render.render_image(results, images, cameras, 
            extra_mesh=extra_mesh)
        render_results = {cam: [res[nv] for res in render_results] for nv, cam in enumerate(camnames)}
        return render_results
    
    def render_image(self, images, results, cameras, extra_mesh):
        render_results = self.render_all(images, results, cameras, extra_mesh)
        results = {cam: val[2] for cam, val in render_results.items()}
        return results
    
    def render_mask(self, images, results, cameras, extra_mesh):
        render_results = self.render_all(images, results, cameras, extra_mesh)
        results = {cam: val[0][..., 3] for cam, val in render_results.items()}
        return results
    
    def render_depth(self, images, results, cameras, extra_mesh):
        render_results = self.render_all(images, results, cameras, extra_mesh)
        results = {cam: val[1] for cam, val in render_results.items()}
        return results
    
    def render_color(self, images, results, cameras, extra_mesh):
        render_results = self.render_all(images, results, cameras, extra_mesh)
        results = {cam: val[0][..., :3] for cam, val in render_results.items()}
        return results
    
    def render_corner(self, images, results, cameras, extra_mesh):
        render_results = self.render_all(images, results, cameras, extra_mesh)
        results = {}
        for cam, (color, depth, image) in render_results.items():
            mask = color[:, :, 3]
            H, W = mask.shape[:2]
            scale = 3
            img_inp = cv2.resize(images[cam], (W//scale, H//scale))
            if cam not in self.position.keys():
                top_left = mask[:, :W//2]
                top_right = mask[:, -W//2:]
                if (top_left>0).sum() > (top_right>0).sum():
                    position = 'topleft'
                else:
                    position = 'topright'
                self.position[cam] = position
            position = self.position[cam]
            if position == 'topleft':
                image[:img_inp.shape[0], -img_inp.shape[1]:, :3] = img_inp
            elif position == 'topright':
                image[:img_inp.shape[0], :img_inp.shape[1], :3] = img_inp
            results[cam] = image
        return results

    def factory(self, mode):
        if mode == 'image':
            return self.render_image
        elif mode == 'color':
            return self.render_color
        elif mode == 'depth':
            return self.render_depth
        elif mode == 'corner':
            return self.render_corner
        elif mode == 'mask':
            return self.render_mask
        elif mode == 'instance-mask':
            return self.render_mask
        elif mode.startswith('instance-depth'):
            return self.render_depth        

def get_ext(mode):
    ext = {'image': '.jpg', 'color':'.jpg', 'blend': '.jpg',
            'depth':'.png', 'mask':'.png', 'instance':'.png',
            'instance-mask': '.png', 'instance-depth': '.png',  'instance-depth-twoside': '.png',
            'side': '.jpg'
            }.get(mode, '.jpg')
    return ext

def get_render_func(mode, backend='pyrender'):
    if backend == 'pyrender':
        from .pyrender_wrapper import Renderer
        render = Renderer()
    else:
        raise NotImplementedError
    renderer = RenderFunc(render)
    return renderer.factory(mode)