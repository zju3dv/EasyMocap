'''
  @ Date: 2021-01-17 22:44:34
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-08-24 16:28:15
  @ FilePath: /EasyMocap/easymocap/visualize/geometry.py
'''
import numpy as np
import cv2
import numpy as np
from tqdm import tqdm
from os.path import join

def load_sphere():
    cur_dir = os.path.dirname(__file__)
    faces = np.loadtxt(join(cur_dir, 'sphere_faces_20.txt'), dtype=int)
    vertices = np.loadtxt(join(cur_dir, 'sphere_vertices_20.txt'))
    return vertices, faces

def load_cylinder():
    cur_dir = os.path.dirname(__file__)
    faces = np.loadtxt(join(cur_dir, 'cylinder_faces_20.txt'), dtype=int)
    vertices = np.loadtxt(join(cur_dir, 'cylinder_vertices_20.txt'))
    return vertices, faces

def create_point(points, r=0.01):
    """ create sphere

    Args:
        points (array): (N, 3)/(N, 4)
        r (float, optional): radius. Defaults to 0.01.
    """
    points = np.array(points)
    nPoints = points.shape[0]
    vert, face = load_sphere()
    vert = vert * r
    nVerts = vert.shape[0]
    vert = vert[None, :, :].repeat(points.shape[0], 0)
    vert = vert + points[:, None, :3]
    verts = np.vstack(vert)
    face = face[None, :, :].repeat(points.shape[0], 0)
    face = face + nVerts * np.arange(nPoints).reshape(nPoints, 1, 1)
    faces = np.vstack(face)
    return {'vertices': verts, 'faces': faces, 'name': 'points'}

def calRot(axis, direc):
    direc = direc/np.linalg.norm(direc)
    axis = axis/np.linalg.norm(axis)
    rotdir = np.cross(axis, direc)
    rotdir = rotdir/np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc, axis))
    rotmat, _ = cv2.Rodrigues(rotdir)
    return rotmat

def create_line(start, end, r=0.01, col=None):
    length = np.linalg.norm(end[:3] - start[:3])
    vertices, faces = load_cylinder()
    vertices[:, :2] *= r
    vertices[:, 2] *= length/2    
    rotmat = calRot(np.array([0, 0, 1]), end - start)
    vertices = vertices @ rotmat.T + (start + end)/2
    ret = {'vertices': vertices, 'faces': faces, 'name': 'line'}
    if col is not None:
        ret['colors'] = col.reshape(-1, 3).repeat(vertices.shape[0], 0)
    return ret

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
    print('[Vis Info] {}, x: {}, y: {}'.format(center, xdir, ydir))
    xdir = xdir * step
    ydir = ydir * step
    vertls, trils, colls = [],[],[]
    cnt = 0
    min_x = -xrange if two_sides else 0
    min_y = -yrange if two_sides else 0
    for i in range(min_x, xrange):
        for j in range(min_y, yrange):
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

PLANE_VERTICES = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 0., 1.],
    [1., 0., 1.],
    [0., 1., 0.],
    [1., 1., 0.],
    [0., 1., 1.],
    [1., 1., 1.]])
PLANE_FACES = np.array([
    [4, 7, 5],
    [4, 6, 7],
    [0, 2, 4],
    [2, 6, 4],
    [0, 1, 2],
    [1, 3, 2],
    [1, 5, 7],
    [1, 7, 3],
    [2, 3, 7],
    [2, 7, 6],
    [0, 4, 1],
    [1, 4, 5]], dtype=np.int32)

def create_plane(normal, center, dx=1, dy=1, dz=0.005, color=[0.8, 0.8, 0.8]):
    vertices = PLANE_VERTICES.copy()
    vertices[:, 0] = vertices[:, 0]*dx - dx/2
    vertices[:, 1] = vertices[:, 1]*dy - dy/2
    vertices[:, 2] = vertices[:, 2]*dz - dz/2
    # 根据normal计算旋转
    rotmat = get_rotation_from_two_directions(
        np.array([0, 0, 1]), np.array(normal))
    vertices = vertices @ rotmat.T
    vertices += np.array(center).reshape(-1, 3)
    return {'vertices': vertices, 'faces': PLANE_FACES.copy(), 'name': 'plane'}

def merge_meshes(meshes):
    verts = []
    faces = []
    # TODO:add colors
    nVerts = 0
    for mesh in meshes:
        verts.append(mesh['vertices'])
        faces.append(mesh['faces'] + nVerts)
        nVerts += mesh['vertices'].shape[0]
    return {'vertices': np.vstack(verts), 'faces':np.vstack(faces), 'name': 'compose_{}'.format(meshes[0]['name'])}

def create_cameras(cameras):
    vertex = np.array([[0.203982,0.061435,0.00717595],[-0.116019,0.061435,0.00717595],[-0.116019,-0.178565,0.00717595],[0.203982,-0.178565,0.00717595],[0.203982,0.061435,-0.092824],[-0.116019,0.061435,-0.092824],[-0.116019,-0.178565,-0.092824],[0.203982,-0.178565,-0.092824],[0.131154,-0.0361827,0.00717595],[0.131154,-0.0361827,0.092176],[0.122849,-0.015207,0.00717595],[0.122849,-0.015207,0.092176],[0.109589,0.00304419,0.00717595],[0.109589,0.00304419,0.092176],[0.092206,0.0174247,0.00717595],[0.092206,0.0174247,0.092176],[0.071793,0.0270302,0.00717595],[0.071793,0.0270302,0.092176],[0.0496327,0.0312577,0.00717595],[0.0496327,0.0312577,0.092176],[0.0271172,0.0298412,0.00717595],[0.0271172,0.0298412,0.092176],[0.00566135,0.0228697,0.00717595],[0.00566135,0.0228697,0.092176],[-0.0133865,0.0107812,0.00717595],[-0.0133865,0.0107812,0.092176],[-0.02883,-0.0056643,0.00717595],[-0.02883,-0.0056643,0.092176],[-0.0396985,-0.0254336,0.00717595],[-0.0396985,-0.0254336,0.092176],[-0.045309,-0.0472848,0.00717595],[-0.045309,-0.0472848,0.092176],[-0.045309,-0.069845,0.00717595],[-0.045309,-0.069845,0.092176],[-0.0396985,-0.091696,0.00717595],[-0.0396985,-0.091696,0.092176],[-0.02883,-0.111466,0.00717595],[-0.02883,-0.111466,0.092176],[-0.0133865,-0.127911,0.00717595],[-0.0133865,-0.127911,0.092176],[0.00566135,-0.14,0.00717595],[0.00566135,-0.14,0.092176],[0.0271172,-0.146971,0.00717595],[0.0271172,-0.146971,0.092176],[0.0496327,-0.148388,0.00717595],[0.0496327,-0.148388,0.092176],[0.071793,-0.14416,0.00717595],[0.071793,-0.14416,0.092176],[0.092206,-0.134554,0.00717595],[0.092206,-0.134554,0.092176],[0.109589,-0.120174,0.00717595],[0.109589,-0.120174,0.092176],[0.122849,-0.101923,0.00717595],[0.122849,-0.101923,0.092176],[0.131154,-0.080947,0.00717595],[0.131154,-0.080947,0.092176],[0.133982,-0.058565,0.00717595],[0.133982,-0.058565,0.092176],[-0.0074325,0.061435,-0.0372285],[-0.0074325,0.074435,-0.0372285],[-0.0115845,0.061435,-0.0319846],[-0.0115845,0.074435,-0.0319846],[-0.018215,0.061435,-0.0274218],[-0.018215,0.074435,-0.0274218],[-0.0269065,0.061435,-0.0238267],[-0.0269065,0.074435,-0.0238267],[-0.0371125,0.061435,-0.0214253],[-0.0371125,0.074435,-0.0214253],[-0.048193,0.061435,-0.0203685],[-0.048193,0.074435,-0.0203685],[-0.0594505,0.061435,-0.0207226],[-0.0594505,0.074435,-0.0207226],[-0.0701785,0.061435,-0.0224655],[-0.0701785,0.074435,-0.0224655],[-0.0797025,0.061435,-0.0254875],[-0.0797025,0.074435,-0.0254875],[-0.0874245,0.061435,-0.0295989],[-0.0874245,0.074435,-0.0295989],[-0.0928585,0.061435,-0.0345412],[-0.0928585,0.074435,-0.0345412],[-0.0956635,0.061435,-0.040004],[-0.0956635,0.074435,-0.040004],[-0.0956635,0.061435,-0.045644],[-0.0956635,0.074435,-0.045644],[-0.0928585,0.061435,-0.051107],[-0.0928585,0.074435,-0.051107],[-0.0874245,0.061435,-0.056049],[-0.0874245,0.074435,-0.056049],[-0.0797025,0.061435,-0.0601605],[-0.0797025,0.074435,-0.0601605],[-0.0701785,0.061435,-0.0631825],[-0.0701785,0.074435,-0.0631825],[-0.0594505,0.061435,-0.0649255],[-0.0594505,0.074435,-0.0649255],[-0.048193,0.061435,-0.0652795],[-0.048193,0.074435,-0.0652795],[-0.0371125,0.061435,-0.064223],[-0.0371125,0.074435,-0.064223],[-0.0269065,0.061435,-0.0618215],[-0.0269065,0.074435,-0.0618215],[-0.018215,0.061435,-0.0582265],[-0.018215,0.074435,-0.0582265],[-0.0115845,0.061435,-0.0536635],[-0.0115845,0.074435,-0.0536635],[-0.0074325,0.061435,-0.0484195],[-0.0074325,0.074435,-0.0484195],[-0.0060185,0.061435,-0.0428241],[-0.0060185,0.074435,-0.0428241]])*0.5
    tri = [[4,3,2],[1,4,2],[6,1,2],[6,5,1],[8,4,1],[5,8,1],[3,7,2],[7,6,2],[4,7,3],[8,7,4],[6,7,5],[7,8,5],[43,42,44],[42,43,41],[43,46,45],[46,43,44],[58,9,57],[9,58,10],[55,58,57],[56,58,55],[53,54,55],[54,56,55],[12,11,9],[12,9,10],[21,20,22],[20,21,19],[34,33,32],[32,33,31],[35,36,37],[37,36,38],[33,36,35],[36,33,34],[29,30,31],[30,32,31],[40,39,37],[40,37,38],[39,40,41],[40,42,41],[47,48,49],[49,48,50],[48,47,45],[46,48,45],[49,52,51],[52,49,50],[52,53,51],[52,54,53],[14,15,13],[15,14,16],[11,14,13],[12,14,11],[18,17,15],[18,15,16],[17,18,19],[18,20,19],[27,35,37],[17,27,15],[27,53,55],[27,49,51],[11,27,9],[27,47,49],[27,33,35],[23,27,21],[27,39,41],[27,55,57],[9,27,57],[15,27,13],[39,27,37],[47,27,45],[53,27,51],[27,11,13],[43,27,41],[27,29,31],[27,43,45],[27,17,19],[21,27,19],[33,27,31],[27,23,25],[23,24,25],[25,24,26],[24,21,22],[24,23,21],[28,36,34],[42,28,44],[28,58,56],[54,28,56],[52,28,54],[28,34,32],[28,46,44],[18,28,20],[20,28,22],[30,28,32],[40,28,42],[58,28,10],[28,48,46],[28,12,10],[28,14,12],[36,28,38],[28,24,22],[28,40,38],[48,28,50],[28,52,50],[14,28,16],[28,18,16],[24,28,26],[28,27,25],[28,25,26],[28,30,29],[27,28,29],[108,59,60],[59,108,107],[62,59,61],[59,62,60],[103,102,101],[102,103,104],[64,61,63],[64,62,61],[70,67,69],[67,70,68],[70,71,72],[71,70,69],[83,84,82],[83,82,81],[86,85,87],[86,87,88],[86,83,85],[83,86,84],[77,78,75],[75,78,76],[105,106,103],[103,106,104],[108,106,107],[106,105,107],[97,96,95],[96,97,98],[96,93,95],[93,96,94],[93,92,91],[92,93,94],[79,105,103],[59,79,61],[79,93,91],[83,79,85],[85,79,87],[61,79,63],[79,103,101],[65,79,67],[79,99,97],[89,79,91],[79,77,75],[79,59,107],[67,79,69],[79,89,87],[79,73,71],[105,79,107],[79,97,95],[79,71,69],[79,83,81],[99,79,101],[93,79,95],[79,65,63],[73,79,75],[99,100,97],[97,100,98],[102,100,101],[100,99,101],[89,90,87],[87,90,88],[90,89,91],[92,90,91],[66,67,68],[66,65,67],[66,64,63],[65,66,63],[74,75,76],[74,73,75],[71,74,72],[73,74,71],[80,106,108],[74,80,72],[86,80,84],[84,80,82],[64,80,62],[80,108,60],[80,100,102],[62,80,60],[66,80,64],[80,70,72],[80,102,104],[96,80,94],[80,90,92],[70,80,68],[80,86,88],[78,80,76],[106,80,104],[80,96,98],[80,92,94],[100,80,98],[90,80,88],[80,66,68],[80,74,76],[82,80,81],[80,79,81],[80,78,77],[79,80,77]]
    tri = [a[::-1] for a in tri]
    triangles = np.array(tri) - 1
    meshes = []
    for nv, (key, camera) in enumerate(cameras.items()):
        vertices = (camera['R'].T @ (vertex.T - camera['T'])).T
        meshes.append({
            'vertices': vertices, 'faces': triangles, 'name': 'camera_{}'.format(nv), 'vid': nv
        })
    meshes = merge_meshes(meshes)
    return meshes

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

def create_cameras_texture(cameras, imgnames, scale=5e-3):
    import trimesh
    import pyrender
    from PIL import Image
    from os.path import join
    cam_path = join(current_dir, 'objs', 'background.obj')
    meshes = []
    for nv, (key, camera) in enumerate(tqdm(cameras.items(), desc='loading images')):
        cam_trimesh = trimesh.load(cam_path, process=False)
        vert = np.asarray(cam_trimesh.vertices)
        K, R, T = camera['K'], camera['R'], camera['T']
        img = Image.open(imgnames[nv])
        height, width = img.height, img.width
        vert[:, 0] *= width
        vert[:, 1] *= height
        vert[:, 2] *= 0
        vert[:, 0] -= vert[:, 0]*0.5
        vert[:, 1] -= vert[:, 1]*0.5
        vert[:, 1] = - vert[:, 1]
        vert[:, :2] *= scale
        # vert[:, 2] = 1
        cam_trimesh.vertices = (vert - T.T) @ R
        cam_trimesh.visual.material.image = img
        cam_mesh = pyrender.Mesh.from_trimesh(cam_trimesh, smooth=True)
        meshes.append(cam_mesh)
    return meshes

def create_mesh_pyrender(vert, faces, col):
    import trimesh
    import pyrender
    mesh = trimesh.Trimesh(vert, faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=col)
    mesh = pyrender.Mesh.from_trimesh(
        mesh,
        material=material)
    return mesh

if __name__ == "__main__":
    pass