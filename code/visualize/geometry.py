'''
  @ Date: 2021-01-17 22:44:34
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-25 19:14:20
  @ FilePath: /EasyMocapRelease/code/visualize/geometry.py
'''
import numpy as np
import cv2
import numpy as np

def create_ground(
    center=[0, 0, 0], xdir=[1, 0, 0], ydir=[0, 1, 0], # 位置
    step=1, xrange=10, yrange=10, # 尺寸
    white=[1., 1., 1.], black=[0.,0.,0.], # 颜色
    two_sides=True
    ):
    if isinstance(center, list):
        center = np.array(center)
        xdir = np.array(xdir)
        ydir = np.array(ydir)
    xdir = xdir * step
    ydir = ydir * step
    vertls, trils, colls = [],[],[]
    cnt = 0
    min_x = -xrange if two_sides else 0
    min_y = -yrange if two_sides else 0
    for i in range(min_x, xrange+1):
        for j in range(min_y, yrange+1):
            point0 = center + i*xdir + j*ydir
            point1 = center + (i+1)*xdir + j*ydir
            point2 = center + (i+1)*xdir + (j+1)*ydir
            point3 = center + (i)*xdir + (j+1)*ydir
            if (i%2==0 and j%2==0) or (i%2==1 and j%2==1):
                col = white
            else:
                col = black
            vert = np.stack([point0, point1, point2, point3])
            col = np.stack([col for _ in range(vert.shape[0])])
            tri = np.array([[2, 3, 0], [0, 1, 2]]) + vert.shape[0] * cnt
            cnt += 1
            vertls.append(vert)
            trils.append(tri)
            colls.append(col)
    vertls = np.vstack(vertls)
    trils = np.vstack(trils)
    colls = np.vstack(colls)
    return {'vertices': vertls, 'faces': trils, 'colors': colls, 'name': 'ground'}


def get_rotation_from_two_directions(direc0, direc1):
    direc0 = direc0/np.linalg.norm(direc0)
    direc1 = direc1/np.linalg.norm(direc1)
    rotdir = np.cross(direc0, direc1)
    if np.linalg.norm(rotdir) < 1e-2:
        return np.eye(3)
    rotdir = rotdir/np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc0, direc1))
    rotmat, _ = cv2.Rodrigues(rotdir)
    return rotmat
    
def create_plane(normal, point, width=1, height=1, depth=0.005):
    mesh_box = TriangleMesh.create_box(width=2*width, height=2*height, depth=2*depth)
    mesh_box.paint_uniform_color([0.8, 0.8, 0.8])
    # 根据normal计算旋转
    rotmat = get_rotation_from_two_directions(np.array([0, 0, 1]), normal[0])
    transform0 = np.eye(4)
    transform0[0, 3] = -width
    transform0[1, 3] = -height
    transform0[2, 3] = -depth
    transform = np.eye(4)
    transform[:3, :3] = rotmat
    transform[0, 3] = point[0, 0]
    transform[1, 3] = point[0, 1]
    transform[2, 3] = point[0, 2]
    mesh_box.transform(transform @ transform0)
    return {'vertices': np.asarray(mesh_box.vertices), 'faces': np.asarray(mesh_box.triangles), 'colors': np.asarray(mesh_box.vertex_colors), 'name': 'ground'}
    faces = np.loadtxt('./code/visualize/sphere_faces_20.txt', dtype=np.int)
    vertices = np.loadtxt('./code/visualize/sphere_vertices_20.txt')
    colors = np.ones((vertices.shape[0], 3))
    
    return {'vertices': vertices, 'faces': faces, 'colors': colors, 'name': 'ground'}
