'''
  @ Date: 2021-04-25 15:52:01
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-02 14:27:41
  @ FilePath: /EasyMocapPublic/easymocap/visualize/o3dwrapper.py
'''
import open3d as o3d
import numpy as np
from .geometry import create_ground as create_ground_
from .geometry import create_point as create_point_
from .geometry import create_line as create_line_
from os.path import join

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh
load_mesh = o3d.io.read_triangle_mesh
load_pcd = o3d.io.read_point_cloud
vis = o3d.visualization.draw_geometries
write_mesh = o3d.io.write_triangle_mesh

def _create_cylinder():
    # create_cylinder(radius=1.0, height=2.0, resolution=20, split=4, create_uv_map=False)
    pass

def read_mesh(filename):
    mesh = load_mesh(filename)
    mesh.compute_vertex_normals()
    return mesh

def create_mesh(vertices, faces, colors=None, normal=True, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    if colors is not None and isinstance(colors, np.ndarray):
        mesh.vertex_colors = Vector3dVector(colors)
    elif colors is not None and isinstance(colors, list):
        mesh.paint_uniform_color(colors)
    else:
        mesh.paint_uniform_color([1., 0.8, 0.8])
    if normal:
        mesh.compute_vertex_normals()
    return mesh

def create_pcd(xyz, color=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = Vector3dVector(xyz[:, :3])
    if color is not None:
        pcd.paint_uniform_color(color)
    if colors is not None:
        pcd.colors = Vector3dVector(colors)
    return pcd

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
    if scale != 1:
        camera_frame.scale(scale)
    return camera_frame

def create_bbox(min_bound=(-3., -3., 0), max_bound=(3., 3., 2), flip=False):
    if flip:
        min_bound_ = min_bound.copy()
        max_bound_ = max_bound.copy()
        min_bound = [min_bound_[0], -max_bound_[1], -max_bound_[2]]
        max_bound = [max_bound_[0], -min_bound_[1], -min_bound_[2]]
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bbox.color = [0., 0., 0.]
    return bbox

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def create_rt_bbox(rtbbox):
    corners = get_bound_corners(rtbbox.aabb)
    corners = corners @ rtbbox.R.T + rtbbox.T
    lines = []
    for (i, j) in [(0, 1), (0, 2), (2, 3), (3, 1), 
        (4, 5), (4, 6), (6, 7), (5, 7), 
        (0, 4), (2, 6), (1, 5), (3, 7)]:
        line = create_line(start=corners[i], end=corners[j], r=0.001)
        line.paint_uniform_color([0., 0., 0.])
        lines.append(line)
    return lines

def create_my_bbox(min_bound=(-3., -3., 0), max_bound=(3., 3., 2)):
    # 使用圆柱去创建一个mesh
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return bbox

def create_camera(path=None, cameras=None):
    if cameras is None:
        from ..mytools.camera_utils import read_cameras
        cameras = read_cameras(path)
    from .geometry import create_cameras
    meshes = create_cameras(cameras)
    return create_mesh(**meshes)

def read_and_vis(filename):
    mesh = load_mesh(filename)
    mesh.compute_vertex_normals()
    # if not mesh.has_texture:
    vis([mesh])

if __name__ == "__main__":
    for res in [2, 4, 8, 20]:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=res)
        mesh_sphere.paint_uniform_color([0.6, 0.7, 0.8])
        outname = 'easymocap/visualize/assets/sphere_faces_{}.txt'.format(res)
        np.savetxt(outname, np.asarray(mesh_sphere.triangles), fmt='%6d')
        outname = outname.replace('faces', 'vertices')
        np.savetxt(outname, np.asarray(mesh_sphere.vertices), fmt='%7.3f')
        vis([mesh_sphere])