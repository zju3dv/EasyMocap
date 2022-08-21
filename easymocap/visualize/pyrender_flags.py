'''
  @ Date: 2021-05-13 14:34:27
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-13 14:37:24
  @ FilePath: /EasyMocap/easymocap/visualize/pyrender_flags.py
'''
from pyrender import RenderFlags
render_flags_default =  {
    'flip_wireframe': False,
    'all_wireframe': False,
    'all_solid': True,
    'shadows': False, # TODO:bug exists in shadow mode
    'vertex_normals': False,
    'face_normals': False,
    'cull_faces': True, # set to False
    'point_size': 1.0,
    'rgba':True
}

def get_flags(flags):
    render_flags = render_flags_default.copy()
    render_flags.update(flags)
    
    flags = RenderFlags.NONE
    if render_flags['flip_wireframe']:
        flags |= RenderFlags.FLIP_WIREFRAME
    elif render_flags['all_wireframe']:
        flags |= RenderFlags.ALL_WIREFRAME
    elif render_flags['all_solid']:
        flags |= RenderFlags.ALL_SOLID

    if render_flags['shadows']:
        flags |= RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT
    if render_flags['vertex_normals']:
        flags |= RenderFlags.VERTEX_NORMALS
    if render_flags['face_normals']:
        flags |= RenderFlags.FACE_NORMALS
    if not render_flags['cull_faces']:
        flags |= RenderFlags.SKIP_CULL_FACES
    if render_flags['rgba']:
        flags |= RenderFlags.RGBA
    return flags