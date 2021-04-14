'''
  @ Date: 2020-11-19 11:39:45
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-20 15:06:28
  @ FilePath: /EasyMocap/code/pyfitting/operation.py
'''
import torch

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def projection(points3d, camera_intri, R=None, T=None, distance=None):
    """ project the 3d points to camera coordinate

    Arguments:
        points3d {Tensor} -- (bn, N, 3)
        camera_intri {Tensor} -- (bn, 3, 3)
        distance {Tensor} -- (bn, 1, 1)
        R: bn, 3, 3
        T: bn, 3, 1
    Returns:
        points2d -- (bn, N, 2)
    """
    if R is not None:
        Rt = torch.transpose(R, 1, 2)
        if T.shape[-1] == 1:
            Tt = torch.transpose(T, 1, 2)
            points3d = torch.matmul(points3d, Rt) + Tt
        else:
            points3d = torch.matmul(points3d, Rt) + T
    
    if distance is None:
        img_points = torch.div(points3d[:, :, :2],
                               points3d[:, :, 2:3])
    else:
        img_points = torch.div(points3d[:, :, :2],
                               distance)
    camera_mat = camera_intri[:, :2, :2]
    center = torch.transpose(camera_intri[:, :2, 2:3], 1, 2)
    img_points = torch.matmul(img_points, camera_mat.transpose(1, 2)) + center
    # img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
        # + center
    return img_points