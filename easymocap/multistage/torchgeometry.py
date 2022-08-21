"""
useful functions to perform conversion between rotation in different format(quaternion, rotation_matrix, euler_angle, axis_angle)
quaternion representation: (w,x,y,z)
code reference: torchgeometry, kornia, https://github.com/MandyMo/pytorch_HMR.
"""

import torch
from torch.nn import functional as F
import numpy as np


# Conversions between different rotation representations, quaternion,rotation matrix,euler and axis angle.

def rot6d_to_rotation_matrix(rot6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        rot6d: torch tensor of shape (batch_size, 6) of 6d rotation representations.
    Returns:
        rotation_matrix: torch tensor of shape (batch_size, 3, 3) of corresponding rotation matrices.
    """
    x = rot6d.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_matrix_to_rot6d(rotation_matrix):
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    Args:
        rotation_matrix: torch tensor of shape (batch_size, 3, 3) of corresponding rotation matrices.
    Returns:
        rot6d: torch tensor of shape (batch_size, 6) of 6d rotation representations.
    """
    v1 = rotation_matrix[:, :, 0:1]
    v2 = rotation_matrix[:, :, 1:2]
    rot6d = torch.cat([v1, v2], dim=-1).reshape(v1.shape[0], 6)
    return rot6d


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion coefficients to rotation matrix.
    Args:
        quaternion: torch tensor of shape (batch_size, 4) in (w, x, y, z) representation.
    Returns:
        rotation matrix corresponding to the quaternion, torch tensor of shape (batch_size, 3, 3)
    """

    norm_quaternion = quaternion
    norm_quaternion = norm_quaternion / \
        norm_quaternion.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quaternion[:, 0], norm_quaternion[:,
                                                        1], norm_quaternion[:, 2], norm_quaternion[:, 3]

    batch_size = quaternion.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotation_matrix = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                                   2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                                   2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(batch_size, 3, 3)
    return rotation_matrix


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    Convert rotation matrix to corresponding quaternion
    Args:
        rotation_matrix: torch tensor of shape (batch_size, 3, 3)
    Returns:
        quaternion: torch tensor of shape(batch_size, 4) in (w, x, y, z) representation.
    """
    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~ mask_d0_d1)
    mask_c2 = (~ mask_d2) * mask_d0_nd1
    mask_c3 = (~ mask_d2) * (~ mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_euler(quaternion, order, epsilon=0):
    """
    Convert quaternion to euler angles.
    Args:
        quaternion: torch tensor of shape (batch_size, 4) in (w, x, y, z) representation.
        order: euler angle representation order, 'zyx' etc.
        epsilon: 
    Returns:
        euler: torch tensor of shape (batch_size, 3) in order.
    """
    assert quaternion.shape[-1] == 4
    original_shape = list(quaternion.shape)
    original_shape[-1] = 3
    q = quaternion.contiguous().view(-1, 4)
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(
            2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(
            2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(
            2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(
            2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(
            2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(
            2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise Exception('unsupported euler order!')

    return torch.stack((x, y, z), dim=1).view(original_shape)


def euler_to_quaternion(euler, order):
    """
    Convert euler angles to quaternion.
    Args:
        euler: torch tensor of shape (batch_size, 3) in order.
        order:
    Returns:
        quaternion: torch tensor of shape (batch_size, 4) in (w, x, y, z) representation.
    """
    assert euler.shape[-1] == 3
    original_shape = list(euler.shape)
    original_shape[-1] = 4
    e = euler.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack((torch.cos(x/2), torch.sin(x/2),
                      torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    ry = torch.stack((torch.cos(y/2), torch.zeros_like(y),
                      torch.sin(y/2), torch.zeros_like(y)), dim=1)
    rz = torch.stack((torch.cos(z/2), torch.zeros_like(z),
                      torch.zeros_like(z), torch.sin(z/2)), dim=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise Exception('unsupported euler order!')
        if result is None:
            result = r
        else:
            result = quaternion_mul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)


def quaternion_to_axis_angle(quaternion):
    """
    Convert quaternion to axis angle.
    based on: https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138
    Args:
        quaternion: torch tensor of shape (batch_size, 4) in (w, x, y, z) representation.
    Returns:
        axis_angle: torch tensor of shape (batch_size, 3)
    """
    epsilon = 1.e-8
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta+epsilon)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def axis_angle_to_quaternion(axis_angle):
    """
    Convert axis angle to quaternion.
    Args:
        axis_angle: torch tensor of shape (batch_size, 3)
    Returns:
        quaternion: torch tensor of shape (batch_size, 4) in (w, x, y, z) representation.
    """
    rotation_matrix = axis_angle_to_rotation_matrix(axis_angle)
    return rotation_matrix_to_quaternion(rotation_matrix)


def axis_angle_to_rotation_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix.
    Args:
        axis_angle: torch tensor of shape (batch_size, 3).
    Returns:
        rotation_matrix: torch tensor of shape (batch_size, 3, 3) of corresponding rotation matrices.
    """

    l1_norm = torch.norm(axis_angle+1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1_norm, dim=-1)
    normalized = torch.div(axis_angle, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quaternion = torch.cat([v_cos, v_sin*normalized], dim=1)
    return quaternion_to_rotation_matrix(quaternion)


def rotation_matrix_to_axis_angle(rotation_matrix):
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_axis_angle(quaternion)


def rotation_matrix_to_euler(rotation_matrix, order):
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_euler(quaternion, order)


def euler_to_rotation_matrix(euler, order):
    quaternion = euler_to_quaternion(euler, order)
    return quaternion_to_rotation_matrix(quaternion)


def axis_angle_to_euler(axis_angle, order):
    quaternion = axis_angle_to_quaternion(axis_angle)
    return quaternion_to_euler(quaternion, order)


def euler_to_axis_angle(euler, order):
    quaternion = euler_to_quaternion(euler, order)
    return quaternion_to_axis_angle(quaternion)

# rotation operations


def quaternion_mul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.contiguous().view(-1, 4, 1),
                      q.contiguous().view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def rotate_vec_by_quaternion(v, q):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def quaternion_fix(quaternion):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    Args:
        quaternion: torch tensor of shape (batch_size, 4)
    Returns:
        quaternion: torch tensor of shape (batch_size, 4)
    """
    quaternion_fixed = quaternion.clone()
    dot_products = torch.sum(quaternion[1:]*quaternion[:-1],dim=-1)
    mask = dot_products < 0
    mask = (torch.cumsum(mask, dim=0) % 2).bool()
    quaternion_fixed[1:][mask] *= -1
    return quaternion_fixed


def quaternion_inverse(quaternion):
    q_conjugate = quaternion.clone()
    q_conjugate[::, 1:] * -1
    q_norm = quaternion[::, 1:].norm(dim=-1) + quaternion[::, 0]**2
    return q_conjugate/q_norm.unsqueeze(-1)


def quaternion_lerp(q1, q2, t):
    q = (1-t)*q1 + t*q2
    q = q/q.norm(dim=-1).unsqueeze(-1)
    return q

def geodesic_dist(q1,q2):
    """
    @q1: torch tensor of shape (frame, joints, 4) quaternion
    @q2: same as q1
    @output: torch tensor of shape (frame, joints)
    """
    q1_conjugate = q1.clone()
    q1_conjugate[:,:,1:] *= -1
    q1_norm = q1[:,:,1:].norm(dim=-1) + q1[:,:,0]**2
    q1_inverse = q1_conjugate/q1_norm.unsqueeze(dim=-1)
    q_between = quaternion_mul(q1_inverse,q2)
    geodesic_dist = quaternion_to_axis_angle(q_between).norm(dim=-1)
    return geodesic_dist

def get_extrinsic(translation, rotation):
    batch_size = translation.shape[0]
    pose = torch.zeros((batch_size, 4, 4))
    pose[:,:3, :3] = rotation
    pose[:,:3, 3] = translation
    pose[:,3, 3] = 1
    extrinsic = torch.inverse(pose)
    return extrinsic[:,:3, 3], extrinsic[:,:3, :3]

def euler_fix_old(euler):
    frame_num = euler.shape[0]
    joint_num = euler.shape[1]
    for l in range(3):
        for j in range(joint_num):
            overall_add = 0.
            for i in range(1,frame_num):
                add1 = overall_add
                add2 = overall_add + 2*np.pi
                add3 = overall_add - 2*np.pi
                previous = euler[i-1,j,l]
                value1 = euler[i,j,l] + add1
                value2 = euler[i,j,l] + add2
                value3 = euler[i,j,l] + add3
                e1 = torch.abs(value1 - previous)
                e2 = torch.abs(value2 - previous)
                e3 = torch.abs(value3 - previous)
                if (e1 <= e2) and (e1 <= e3):
                    euler[i,j,l] = value1
                    overall_add = add1
                if (e2 <= e1) and (e2 <= e3):
                    euler[i, j, l] = value2
                    overall_add = add2
                if (e3 <= e1) and (e3 <= e2):
                    euler[i, j, l] = value3
                    overall_add = add3
    return euler

def euler_fix(euler,rotation_order='zyx'):
    frame_num = euler.shape[0]
    joint_num = euler.shape[1]
    euler_new = euler.clone()
    for j in range(joint_num):
        euler_new[:,j] = euler_filter(euler[:,j],rotation_order)
    return euler_new

'''
euler filter from https://github.com/wesen/blender-euler-filter/blob/master/euler_filter.py.
'''
def euler_distance(e1, e2):
    return abs(e1[0] - e2[0]) + abs(e1[1] - e2[1]) + abs(e1[2] - e2[2])


def euler_axis_index(axis):
    if axis == 'x':
        return 0
    if axis == 'y':
        return 1
    if axis == 'z':
        return 2
    return None

def flip_euler(euler, rotation_mode):
    ret = euler.clone()
    inner_axis = rotation_mode[0]
    outer_axis = rotation_mode[2]
    middle_axis = rotation_mode[1]

    ret[euler_axis_index(inner_axis)] += np.pi
    ret[euler_axis_index(outer_axis)] += np.pi
    ret[euler_axis_index(middle_axis)] *= -1
    ret[euler_axis_index(middle_axis)] += np.pi
    return ret

def naive_flip_diff(a1, a2):
    while abs(a1 - a2) >= np.pi+1e-5:
        if a1 < a2:
            a2 -= 2 * np.pi
        else:
            a2 += 2 * np.pi

    return a2

def euler_filter(euler,rotation_order):
    frame_num = euler.shape[0]
    if frame_num <= 1:
        return euler
    euler_fix  = euler.clone()
    prev = euler[0]
    for i in range(1,frame_num):
        e = euler[i]
        for d in range(3):
            e[d] = naive_flip_diff(prev[d],e[d])
        fe = flip_euler(e,rotation_order)
        for d in range(3):
            fe[d] = naive_flip_diff(prev[d],fe[d])
        
        de = euler_distance(prev,e)
        dfe = euler_distance(prev,fe)
        if dfe < de:
            e = fe
        prev = e
        euler_fix[i] = e
    return euler_fix