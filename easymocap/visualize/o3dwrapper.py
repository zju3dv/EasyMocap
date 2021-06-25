'''
  @ Date: 2021-04-25 15:52:01
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-20 15:22:59
  @ FilePath: /EasyMocap/easymocap/visualize/o3dwrapper.py
'''
import open3d as o3d
import numpy as np
from .geometry import create_ground as create_ground_
from .geometry import create_point as create_point_
from .geometry import create_line as create_line_

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh
load_mesh = o3d.io.read_triangle_mesh
vis = o3d.visualization.draw_geometries

def _create_cylinder():
    # create_cylinder(radius=1.0, height=2.0, resolution=20, split=4, create_uv_map=False)
    pass

def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([1., 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh

def create_point(**kwargs):
    return create_mesh(**create_point_(**kwargs))

def create_line(**kwargs):
    return create_mesh(**create_line_(**kwargs))
    
def create_ground(**kwargs):
    ground = create_ground_(**kwargs)
    return create_mesh(**ground)

def create_coord(camera = [0,0,0], radius=1, scale=1):
    camera_frame = TriangleMesh.create_coordinate_frame(
            size=radius, origin=camera)
    camera_frame.scale(scale)
    return camera_frame

def create_bbox(min_bound=(-3., -3., 0), max_bound=(3., 3., 2), flip=False):
    if flip:
        min_bound_ = min_bound.copy()
        max_bound_ = max_bound.copy()
        min_bound = [min_bound_[0], -max_bound_[1], -max_bound_[2]]
        max_bound = [max_bound_[0], -min_bound_[1], -min_bound_[2]]
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return bbox
