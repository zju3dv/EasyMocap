import numpy as np
import cv2
import math
from collections import namedtuple

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    # ATTN: dont't normalize here
    # rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    rays_o = rays_o.astype(np.float32)
    rays_d = rays_d.astype(np.float32)
    return rays_o, rays_d

def project(xyz, K, R, T):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, R.T) + T.T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

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

def get_bounds(xyz, delta=0.05):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    if isinstance(delta, list):
        delta = np.array(delta, dtype=np.float32).reshape(1, 3)
    min_xyz -= delta
    max_xyz += delta
    can_bounds = np.stack([min_xyz, max_xyz], axis=0)
    return can_bounds.astype(np.float32)

# get the body
def sample_rays(bound_sum, mask_back, split, nrays=1024, **kwargs):
    coord_body = np.argwhere(bound_sum*mask_back > 0)
    if split == 'train':
        coord_body = coord_body[np.random.randint(0, len(coord_body), nrays)]
    return coord_body

def generate_weight_coords(bounds, rates, back_mask):
    coords = []
    for key in bounds.keys():
        coord_ = np.argwhere(bounds[key]*back_mask > 0)
        if rates[key] == 1.:
            coords.append(coord_)
        elif rates[key] >= 1.:
            # repeat the interger part
            coord_r = np.vstack([coord_ for _ in range(math.floor(rates[key]))])
            if not isinstance(rates[key], int):
                # repeat the float part
                nsample2 = int(len(coord_)*(rates[key] - math.floor(rates[key])))
                coord_f = coord_[np.random.randint(0, len(coord_), nsample2)]
                coord_ = np.vstack([coord_r, coord_f])
            else:
                coord_ = coord_
        else:
            # sample
            coord_ = coord_[np.random.randint(0, len(coord_), int(len(coord_)*rates[key]))]
        coords.append(coord_)
    coords = np.vstack(coords)
    return coords

def sample_rays_rate(bounds, rates, back_mask, nrays=1024, **kwargs):
    if 'method' in kwargs and kwargs['method'] == 'patch':
        cv2.imwrite('debug/back.jpg', (back_mask*255).astype(np.uint8))
        mask_valid = back_mask
        # 腐蚀一下
        mask_valid[:, 0] = 0
        mask_valid[:, -1] = 0
        mask_valid[0, :] = 0
        mask_valid[-1, :] = 0
        # inp = mask_valid.astype(np.uint8) * 255
        patch_size = kwargs['patch_size']
        kernel = np.ones((2*patch_size//2+1, 2*patch_size//2+1), np.uint8)
        back_mask = cv2.erode(mask_valid, kernel, iterations=1)
        # TODO: 这里每个object的mask并不会被erode掉
        # 导致object的边缘也是会被选中的
    coords = generate_weight_coords(bounds, rates, back_mask)
    if 'method' in kwargs and kwargs['method'] == 'patch':
        patch_size = kwargs['patch_size']
        if False:
            canvas = np.zeros_like(back_mask)
            for (i, j) in coords:
                canvas[i, j] += 1
            canvas /= canvas.max()
            cv2.imwrite('debug.jpg', (canvas*255).astype(np.uint8))
        center = coords[np.random.randint(0, len(coords), kwargs['num_patch'])]
        coords_list = []
        for n_patch in range(center.shape[0]):
            cx, cy = center[n_patch]
            x_min = cx - patch_size//2
            x_max = x_min + patch_size
            y_min = cy - patch_size//2
            y_max = y_min + patch_size
            i, j = np.meshgrid(np.arange(x_min, x_max, dtype=coords.dtype),
                       np.arange(y_min, y_max, dtype=coords.dtype),
                       indexing='xy')
            coord = np.stack([i.reshape(-1), j.reshape(-1)], axis=1)
            coords_list.append(coord)
        coords = np.vstack(coords_list)
    else:
        coords = coords[np.random.randint(0, len(coords), nrays)]
    return coords

class BaseSampler:
    def __init__(self, split) -> None:
        self.split = split
        self._mask = None
        self.feature = {}
        self.feature_input = {}
        self.bounds = np.array([
            [-100., -100., -100.],
            [ 100.,  100., 100.],
        ])

    def mask(self, K, R, T, H, W, **kwargs):
        mask = np.zeros((H, W), dtype=np.uint8) + 1
        return mask

class ComposeSampler(BaseSampler):
    def __init__(self, split, objlist) -> None:
        super().__init__(split)
        self.objlist = objlist
        self.bounds = objlist[0].bounds
    
    def mask(self, K, R, T, H, W, **kwargs):
        mask = None
        for obj in self.objlist:
            mask_ = obj.mask(K, R, T, H, W, **kwargs).astype(np.uint8)
            if mask is None and mask_ is not None:
                mask = mask_
            elif mask_ is not None:
                mask = cv2.bitwise_or(mask, mask_)
            else:
                pass
        return mask
    
    def __call__(self, ray_o, ray_d, coord, depth=None):
        """calculate intersections with 3d bounding box"""
        near = np.zeros((ray_o.shape[0])) + 1e5
        far = np.zeros((ray_o.shape[0])) + 0
        mask = np.zeros((ray_o.shape[0]), dtype=bool)
        for obj in self.objlist:
            near_, far_, mask_at_box_ = obj(ray_o, ray_d, coord, depth)
            near[mask_at_box_] = np.minimum(near[mask_at_box_], near_)
            far[mask_at_box_] = np.maximum(far[mask_at_box_], far_)
            mask[mask_at_box_] = True
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        return near[mask], far[mask], mask

RTBBox = namedtuple('RTBBox', ['aabb', 'R', 'T'])

class NearFarSampler(BaseSampler):
    def __init__(self, split, near, far, depth=None) -> None:
        super().__init__(split)
        self.near = near
        self.far = far
        self.depth = depth
    
    def __call__(self, ray_o, ray_d, coord, depth=None):
        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=False)
        near, far = self.near/norm_d, self.far/norm_d
        mask_at_box = near > 0
        if self.depth is not None:
            depth = self.depth[coord[:, 0], coord[:, 1]]
            self.feature['depth'] = depth
        return near, far, mask_at_box

class AABBSampler(BaseSampler):
    @classmethod
    def from_vertices(cls, split, vertices, delta=0.05, **cfg):
        bounds = get_bounds(vertices, delta)
        return cls(split=split, bounds=bounds, **cfg)

    def __init__(self, split, bounds=None, center=None, scale=None):
        super().__init__(split)
        if bounds is None and center is not None:
            center = center.reshape(1, 3)
            scale = np.array(scale).reshape(1, 3)
            bounds = np.concatenate([center - scale, center + scale], axis=0)
        self.bounds = np.array(bounds).astype(np.float32)
        self.depth_min = 0.05 # 限定最近距离
        # self.method = method
        # self.no_mask = no_mask
        # self.instance = instance
        self._mask = None

    def mask(self, K, R, T, H, W, **kwargs):
        corners_3d = get_bound_corners(self.bounds)
        corners_3d = np.dot(corners_3d, R.T) + T.T
        if (corners_3d[..., -1] < 0.).any(): # some points is behind the camera
            # render the plane by mesh renderer
            ray_o, ray_d = get_rays(H, W, K, R, T)
            _, _, mask = self.get_near_far(ray_o, ray_d, self.bounds, coord=None)
        else:
            xyz = np.dot(corners_3d, K.T)
            corners_2d = xyz[:, :2] / xyz[:, 2:]
            corners_2d = np.round(corners_2d).astype(int)
            mask = np.zeros((H, W), dtype=np.uint8)
            for lines in [[0, 1, 3, 2, 0], [4, 5, 7, 6, 5], [0, 1, 5, 4, 0], [2, 3, 7, 6, 2], [0, 2, 6, 4, 0], [1, 3, 7, 5, 1]]:
                cv2.fillPoly(mask, [corners_2d[lines]], 1)
        self._mask = mask
        return mask
    
    @staticmethod
    def get_near_far(ray_o, ray_d, bounds, coord, depth_min=0.1):
        """ get near and far

        Args:
            ray_o (np): 
            ray_d ([type]): [description]
            bounds ([type]): [description]

        Returns:
            near, far, mask_at_box
            这里的near是实际物理空间中的深度
        """
        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
        viewdir = ray_d/norm_d
        viewdir[(viewdir<1e-5)&(viewdir>-1e-10)] = 1e-5
        viewdir[(viewdir>-1e-5)&(viewdir<1e-10)] = -1e-5
        inv_dir = 1.0/viewdir
        tmin = (bounds[:1] - ray_o[:1])*inv_dir
        tmax = (bounds[1:2] - ray_o[:1])*inv_dir
        # 限定时间是增加的
        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)
        near = np.max(t1, axis=-1)
        far = np.min(t2, axis=-1)
        near = np.maximum(near, depth_min)
        mask_at_box = near < far
        return near, far, mask_at_box
    
    @staticmethod
    def get_near_far_RTBBox(ray_o, ray_d, rtbbox, coord, depth_min=0.1):
        # sample the near far in canonical coordinate
        R, T = rtbbox.R, rtbbox.T
        bounds = rtbbox.aabb
        ray_o_rt = (ray_o - T) @ (R.T).T
        ray_d_rt = ray_d @ (R.T).T
        near, far, mask_at_box = AABBSampler.get_near_far(ray_o_rt, ray_d_rt, bounds, coord=coord)
        return near, far, mask_at_box
    
    def uniform_sample(self, ray_o, ray_d, coord, depth=None):
        near, far, mask_at_box = self.get_near_far(ray_o, ray_d, self.bounds, coord=coord)
        if depth is not None:
            #TODO:考虑最近和最远
            # 暂时只考虑修改near
            flag = mask_at_box & (depth > 0.05) & (depth<9999.)
            near[flag] = np.maximum(near[flag], depth[flag])
        # 返回的near, far是以mask_at_box为大小的
        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
        # 返回的near far是去掉长度的
        near = near[mask_at_box] / norm_d[mask_at_box, 0]
        far = far[mask_at_box] / norm_d[mask_at_box, 0]
        return near, far, mask_at_box

    def __call__(self, ray_o, ray_d, coord, depth=None):
        """calculate intersections with 3d bounding box"""
        return self.uniform_sample(ray_o, ray_d, coord, depth)

class AABBwMask(AABBSampler):
    def __init__(self, mask=None, label=None, dilate=True, rate_body=-1, **kwargs):
        super().__init__(**kwargs)
        self.instance = mask
        self.label = label
        self.dilate = dilate
        self.rate_body = rate_body
        if mask is not None:
            self.feature['coord_mask'] = mask

    def mask(self, K, R, T, H, W, **kwargs):
        mask_bounds = super().mask(K, R, T, H, W, **kwargs)
        if self.split != 'train' or self.rate_body < 0:
            return mask_bounds
        # mask_bounds: the mask of SMPL body
        mask_bounds = mask_bounds > 0
        if self.instance is not None:
            ys, xs = np.where(self.instance)
            bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
            mask_bounds = np.zeros_like(mask_bounds)
            padding = max(mask_bounds.shape[0]//50, 32)
            mask_bounds[bbox[1]-padding:bbox[3]+padding, bbox[0]-padding:bbox[2]+padding] = True
        # mask_out_body: the mask in the bounds and out of the human mask
        mask_out_body = mask_bounds^self.instance
        instance = self.instance.copy().astype(np.uint8)
        if self.dilate:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(instance.copy(), kernel)
            msk_dilate = cv2.dilate(instance.copy(), kernel)
            instance[(msk_dilate - msk_erode) == 1] = 0
            mask_out_body[(msk_dilate-msk_erode)==1] = 0

        size_body = instance.sum()
        size_outer = mask_out_body.sum()
        # 身体部分0.9, 身体以外的部分0.1
        rate_body = self.rate_body
        rate_outer = 1 - rate_body
        if size_body < 10 or size_outer < 10:
            return {'bound': {'mask': mask_out_body, 'rate': rate_outer}}

        rate_body = rate_body*(size_body +size_outer)/size_body
        rate_outer = rate_outer*(size_body+size_outer)/size_outer
        return {
            'body': {'mask': instance, 'rate': rate_body},
            'bound': {'mask': mask_out_body, 'rate': rate_outer}
        }
    
    def __call__(self, ray_o, ray_d, coord, depth=None):
        if self.label is not None:
            label = self.label[coord[:, 0], coord[:, 1]]
            label[label<0.1] = -1.
            self.feature['label'] = label
        if 'semantic' in self.feature_input.keys() and self.feature_input['semantic'] is not None:
            self.feature['semantic'] = self.feature_input['semantic'][coord[:, 0], coord[:, 1]]
        if 'R' in self.feature.keys() and 'bounds_canonical' in self.feature.keys():
            # sample the near far in canonical coordinate
            R = self.feature['R']
            T = self.feature['Th']
            bounds = self.feature['bounds_canonical']
            ray_o_rt = (ray_o - T) @ (R.T).T
            ray_d_rt = ray_d @ (R.T).T
            near, far, mask_at_box = self.get_near_far(ray_o_rt, ray_d_rt, bounds, coord=coord)
            # 返回的near, far是以mask_at_box为大小的
            norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
            # 返回的near far是去掉长度的
            near = near[mask_at_box] / norm_d[mask_at_box, 0]
            far = far[mask_at_box] / norm_d[mask_at_box, 0]
            return near, far, mask_at_box
        else:
            return super().__call__(ray_o, ray_d, coord, depth)

class TwoAABBSampler(BaseSampler):
    def __init__(self, split, bbox_inter, bbox_outer):
        super().__init__(split)
        self.bbox_inter = np.array(bbox_inter).astype(np.float32)
        self.bbox_outer = np.array(bbox_outer).astype(np.float32)
        self.bounds = self.bbox_outer

    def mask(self, K, R, T, H, W, **kwargs):
        mask = np.ones((H, W), dtype=np.uint8)
        self._mask = mask
        return mask
    
    def get_near_far(self, ray_o, ray_d, coord):
        near_inter, far_inter, mask_inter = AABBSampler.get_near_far(ray_o, ray_d, self.bbox_inter, coord)
        near_outer, far_outer, mask_outer = AABBSampler.get_near_far(ray_o, ray_d, self.bbox_outer, coord)
        mask_at_box = mask_inter & mask_outer & (far_inter < far_outer)
        # 返回的near, far是以mask_at_box为大小的
        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
        # 返回的near far是去掉长度的
        near = far_inter[mask_at_box] / norm_d[mask_at_box, 0]
        far = far_outer[mask_at_box] / norm_d[mask_at_box, 0]
        return near, far, mask_at_box

    def __call__(self, ray_o, ray_d, coord, depth=None):
        near, far, mask_at_box = self.get_near_far(ray_o, ray_d, coord)
        return near, far, mask_at_box

class PlaneSampler(AABBSampler):
    cache = {}
    def __init__(self, center, radius, zranges, xybounds=None, **kwargs):
        bounds = np.array([[center[0]-radius, center[1]-radius, center[2]+zranges[0]], [center[0]+radius, center[1]+radius, center[2]+zranges[1]]])
        super().__init__(bounds=bounds, **kwargs)
        self.center = center
        self.radius = radius
        self.xybounds = xybounds
        self.zranges = zranges
        self.feature = {'bounds': self.bounds}

    def mask(self, K, R, T, H, W, **kwargs):
        _KRT = tuple((K @ np.hstack([R, T])).astype(np.int32).reshape(-1).tolist())
        if _KRT in self.cache.keys():
            mask_bounds = self.cache[_KRT]
        else:
            mask_bounds = super().mask(K, R, T, H, W, **kwargs)
            self.cache[_KRT] = mask_bounds
        return mask_bounds
    
    def __call__(self, ray_o, ray_d, coord, depth=None):
        near, far, mask = super().__call__(ray_o, ray_d, coord, depth)
        # filter the ray out the xyranges
        if self.xybounds is not None:
            pts = ray_o[mask] + ray_d[mask] * near[:, None]
            mask_xy = (pts[:, 0] > self.xybounds[0])&(pts[:, 1] > self.xybounds[0])&(pts[:, 0] < self.xybounds[1])&(pts[:, 1] < self.xybounds[1])
            mask[mask] &= mask_xy
            near = near[mask_xy]
            far = far[mask_xy]
        return near, far, mask

class CylinderSampler(BaseSampler):
    cache = {}
    def __init__(self, center, split, zranges, radius=(3., 7), **cfg):
        super().__init__(split)
        self.radius = radius
        self.zranges = zranges
        # TODO:consider the center of the cylinder
        self.center = center
        self.bounds = np.array([[self.center[0] - self.radius[1],
                                self.center[1] - self.radius[1],
                                self.center[2] + self.zranges[0]],
                               [self.center[0] + self.radius[1], 
                                self.center[1] + self.radius[1], 
                                self.center[2] + self.zranges[1]]], dtype=np.float32)
        self.feature = {'bounds': self.bounds}

    def mask(self, K, R, T, H, W, ray_o, ray_d, **kwargs):
        _KRT = tuple((K @ np.hstack([R, T])).astype(np.int32).reshape(-1).tolist())
        if _KRT in self.cache.keys():
            mask, near, far = self.cache[_KRT]
        else:
            norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
            viewdirs = ray_d/norm_d
            near, far = self.get_near_far_cylinder(ray_o, ray_d, viewdirs, self.radius)
            near = near/norm_d[..., 0]
            far = far/norm_d[..., 0]
            # mask: near < far and the intersection is in the zranges
            zval_near = ray_o[..., 2] +ray_d[..., 2]*near
            mask = (near < far) & (zval_near > self.zranges[0]) & (zval_near < self.zranges[1])
            self.cache[_KRT] = (mask, near, far)
        self._mask = mask
        self.near = near
        self.far = far
        return mask

    @staticmethod
    def get_near_far_cylinder(ray_o, ray_d, viewdirs, radius):
        # 计算与圆柱的交点
        radius0, radius1 = radius
        # 1. 计算xy方向的单位向量
        ray_d_xy = ray_d[..., :2]
        viewdirs_xy = ray_d_xy/np.linalg.norm(ray_d_xy, axis=-1, keepdims=True)
        # d1: 相机中心到原点的向量在射线方向的投影
        d1 = - (viewdirs_xy * ray_o[..., :2]).sum(axis=-1)
        # d0_dir: 直线 x=0, y=0到射线的距离
        d_0_dir = np.sqrt(np.maximum((ray_o[..., :2]*ray_o[..., :2]).sum(axis=-1) - d1 * d1, 1e-5))
        # 计算与内圆交点：确保到射线的距离小于半径
        # assert d_0_dir.max() < radius0, d_0_dir.max()
        # 计算与圆的第二个交点
        dr0 = np.sqrt(np.clip(radius0*radius0 - d_0_dir*d_0_dir, 0., 1e5)) + d1
        dr1 = np.sqrt(np.clip(radius1*radius1 - d_0_dir*d_0_dir, 0., 1e5)) + d1
        # 现在这个距离是二维的，需要变成三维的
        # 由于计算的是时间t，所以这个除的时候，直接除以归一化xy平面的就好
        # 得到的值也是绝对时间
        norm_xy = np.linalg.norm(viewdirs[..., :2], axis=-1)
        dr0, dr1 = dr0/norm_xy, dr1/norm_xy
        return dr0, dr1

    def __call__(self, ray_o, ray_d, coord):
        near, far, mask = self.near[coord[:, 0], coord[:, 1]], self.far[coord[:, 0], coord[:, 1]], self._mask[coord[:, 0], coord[:, 1]]
        near, far = near[mask], far[mask]
        # 注意，这里都是当作背景来处理的，所以mask_at_box一定是全是True的
        return near, far, mask

def create_cameras_mean(cameras, camera_args):
    Told = np.stack([d['T'] for d in cameras])
    Rold = np.stack([d['R'] for d in cameras])
    Kold = np.stack([d['K'] for d in cameras])
    Cold = - np.einsum('bmn,bnp->bmp', Rold.transpose(0, 2, 1), Told)
    center = Cold.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(Cold - center, axis=1).mean()
    zmean = Rold[:, 2, 2].mean()
    xynorm = np.sqrt(1. - zmean**2)
    thetas = np.linspace(0., 2*np.pi, camera_args['allstep'])
    # 计算第一个相机对应的theta
    dir0 = Cold[0] - center[0]
    dir0[2, 0] = 0.
    dir0 = dir0 / np.linalg.norm(dir0)
    theta0 = np.arctan2(dir0[1,0], dir0[0,0]) + np.pi/2
    thetas += theta0
    sint = np.sin(thetas)
    cost = np.cos(thetas)
    R1 = np.stack([cost, sint, np.zeros_like(sint)]).T
    R3 = xynorm * np.stack([-sint, cost, np.zeros_like(sint)]).T
    R3[:, 2] = zmean
    R2 = - np.cross(R1, R3)
    Rnew = np.stack([R1, R2, R3], axis=1)
    # set locations
    loc = np.stack([radius * sint, -radius * cost, np.zeros_like(sint)], axis=1)[..., None] + center
    print('[sample] camera centers: ', center[0].T[0])
    print('[sample] camera radius: ', radius)
    print('[sample] camera start theta: ', theta0)
    Tnew = -np.einsum('bmn,bnp->bmp', Rnew, loc)
    K = Kold.mean(axis=0, keepdims=True).repeat(Tnew.shape[0], 0)
    return K, Rnew, Tnew

def create_center_radius(center, radius=5., up='y', ranges=[0, 360, 36], angle_x=0, **kwargs):
    center = np.array(center).reshape(1, 3)
    thetas = np.deg2rad(np.linspace(*ranges))
    st = np.sin(thetas)
    ct = np.cos(thetas)
    zero = np.zeros_like(st)
    Rotx = cv2.Rodrigues(np.deg2rad(angle_x) * np.array([1., 0., 0.]))[0]
    if up == 'z':
        center = np.stack([radius*ct, radius*st, zero], axis=1) + center
        R = np.stack([-st, ct, zero, zero, zero, zero-1, -ct, -st, zero], axis=-1)
    elif up == 'y':
        center = np.stack([radius*ct, zero, radius*st, ], axis=1) + center
        R = np.stack([
            +st,  zero,  -ct,
            zero, zero-1, zero, 
            -ct,  zero, -st], axis=-1)
    R = R.reshape(-1, 3, 3)
    R = np.einsum('ab,fbc->fac', Rotx, R)
    center = center.reshape(-1, 3, 1)
    T = - R @ center
    RT = np.dstack([R, T])
    return RT