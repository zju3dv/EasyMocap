'''
    @ Date: 2020-06-04 12:47:04
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-04-19 17:02:57
    @ Author: Qing Shuai
    @ Mail: s_q@zju.edu.cn
'''
from os.path import join
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from .hrnet import HRNet

COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
pairs = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]
def coco17tobody25(points2d):
    kpts = np.zeros((points2d.shape[0], 25, 3))
    kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
    kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
    # 需要交换一下
    # kpts = kpts[:, :, [1,0,2]]
    return kpts

# 生成高斯核
def generate_gauss(sigma):
    tmp_size = sigma * 3
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g, tmp_size

gauss = {}
for SIGMA in range(1, 5):
    gauss_kernel, gauss_radius = generate_gauss(SIGMA)
    gauss[SIGMA] = {
        'kernel': gauss_kernel,
        'radius': gauss_radius
    }

def box_to_center_scale(box, model_image_width, model_image_height, scale_factor=1.25):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = (box[0], box[1])
    top_right_corner = (box[2], box[3])
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    scale = scale * scale_factor
    return center, scale

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def batch_affine_transform(points, trans):
    points = np.hstack((points[:, :2], np.ones((points.shape[0], 1))))
    out = points @ trans.T
    return out

def transform_preds(coords, center, scale, rot, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    target_coords[:, :2] = batch_affine_transform(coords, trans)
    return target_coords

config_ = {'kintree': [[1, 0], [2, 0], [3, 1], [4, 2], [5, 0], [6, 0], [7, 5], [8, 6], [9, 7], [10, 8], [11, 5], [12, 6], [13, 11], [
    14, 12], [15, 13], [16, 14], [6, 5], [12, 11]], 'color': ['g', 'r', 'g', 'r', 'g', 'r', 'g', 'r', 'g', 'r', 'g', 'r', 'g', 'r', 'g', 'r', 'k', 'k']}
colors_table = {
    # colorblind/print/copy safe:
    '_blue': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [ 166/255.,  229/255.,  204/255.],
    '_mint2': [ 202/255.,  229/255.,  223/255.],
    '_green': [ 153/255.,  216/255.,  201/255.],
    '_green2': [ 171/255.,  221/255.,  164/255.],
    '_red': [ 251/255.,  128/255.,  114/255.],
    '_orange': [ 253/255.,  174/255.,  97/255.],
    '_yellow': [ 250/255.,  230/255.,  154/255.],
    'r':[255/255,0,0],
    'g':[0,255/255,0],
    'b':[0,0,255/255],
    'k':[0,0,0],
    'y':[255/255,255/255,0],
    'purple':[128/255,0,128/255]
}
for key, val in colors_table.items():
    colors_table[key] = tuple([int(val[2]*255), int(val[1]*255), int(val[0]*255)])

def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+2)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))
        resized_image_copy = resized_image.copy()
        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for ip in range(len(config_['kintree'])):
            src, dst = config_['kintree'][ip]
            c = config_['color'][ip]
            if maxvals[i][src] < 0.1 or maxvals[i][dst] < 0.1:
                continue
            plot_line(resized_image_copy, preds[i][src], preds[i][dst], colors_table[c], 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            mask = (heatmap > 0.1)[:,:,None]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = (colored_heatmap*0.7 + resized_image*0.3)*mask + resized_image*(1-mask)
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+2)
            width_end = heatmap_width * (j+2+1)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
        grid_image[height_begin:height_end, heatmap_width:heatmap_width+heatmap_width, :] = resized_image_copy
    cv2.imwrite(file_name, grid_image)
    
import math

def get_final_preds(batch_heatmaps, center, scale, rot=None, flip=None):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if True:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        if flip is not None:
            if flip[i]:
                coords[i, :, 0] = heatmap_width - 1 - coords[i, :, 0]
        if rot is None:
            _rot = 0
        else:
            _rot = rot[i]
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], _rot, [heatmap_width, heatmap_height]
        )
    return preds, maxvals

def get_gaussian_maps(net_out, keypoints, sigma):
    radius, kernel = gauss[sigma]['radius'], gauss[sigma]['kernel']
    weights = np.ones(net_out.shape, dtype=np.float32)
    for i in range(weights.shape[0]):
        for nj in range(weights.shape[1]):
            if keypoints[i][nj][2] < 0:
                weights[i][nj] = 0
                continue
            elif keypoints[i][nj][2] < 0.01:
                weights[i][nj] = 0
                continue
            weights[i][nj] = 0
            mu_x, mu_y = keypoints[i][nj][:2]
            mu_x, mu_y = int(mu_x + 0.5), int(mu_y + 0.5)
            # Usable gaussian range
            ul = [mu_x - radius, mu_y - radius]
            br = [mu_x + radius + 1, mu_y + radius + 1]
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], weights.shape[3]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], weights.shape[2]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], weights.shape[3])
            img_y = max(0, ul[1]), min(br[1], weights.shape[2])
            weights[i][nj][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                kernel[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return weights

humanId = 0

class SimpleHRNet:
    def __init__(self, c, nof_joints, checkpoint_path, device, resolution=(288, 384),):
        self.device = device
        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.max_batch_size = 64
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.model = HRNet(c=c, nof_joints=nof_joints).to(device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
    
    def __call__(self, image, bboxes, rot=0, net_out=False):
        # image: 
        images = torch.zeros((len(bboxes), 3, self.resolution[1], self.resolution[0]), device=self.device)  # (height, width)
        if len(bboxes) > 0:
            # pose estimation : for multiple people
            centers, scales, trans_all = [], [], []
            for box in bboxes:
                center, scale = box_to_center_scale(box, self.resolution[0], self.resolution[1])
                centers.append(center)
                scales.append(scale)
                trans = get_affine_transform(center, scale, rot=rot, output_size=self.resolution)
                trans_all.append(trans)
            for i, trans in enumerate(trans_all):
                # Crop smaller image of people
                model_input = cv2.warpAffine(
                    image, trans,
                    (int(self.resolution[0]), int(self.resolution[1])),
                    flags=cv2.INTER_LINEAR)
                # cv2.imshow('input', model_input)
                # cv2.waitKey(0)
                # hwc -> 1chw
                model_input = self.transform(model_input)#.unsqueeze(0)
                images[i] = model_input
            images = images.to(self.device) 
            with torch.no_grad():
                out = self.model(images)
            out = out.cpu().detach().numpy()
            if net_out:
                return out, trans_all, centers, scales, rot
            coords, max_val = get_final_preds(
                out,
                np.asarray(centers),
                np.asarray(scales),
                [rot for _ in range(out.shape[0])])
            pts = np.concatenate((coords, max_val), axis=2)
            return coco17tobody25(pts)
        else:
            return np.empty(0, 25, 3)
        
    def predict_with_previous(self, image, bboxes, keypoints, sigma):
        # (batch, nJoints, height, width)
        net_out, trans_all, centers, scales, rot = self.__call__(image, bboxes, net_out=True)
        keypoints = keypoints[:, COCO17_IN_BODY25]
        keypoints_rescale = keypoints.copy()
        for i in range(keypoints.shape[0]):
            keypoints_rescale[..., :2] = batch_affine_transform(keypoints[i], trans_all[i])/4
        weights = get_gaussian_maps(net_out, keypoints_rescale, sigma)        
        out = net_out * weights
        coords, max_val = get_final_preds(
            out,
            np.asarray(centers),
            np.asarray(scales),
            rot)
        pts = np.concatenate((coords, max_val), axis=2)
        return coco17tobody25(pts)

    def predict(self, image, detections, keypoints=None, ret_crop=False):
        if keypoints is not None:
            keypoints = keypoints[:, COCO17_IN_BODY25]
            kpts_rescale = [None for _ in range(len(keypoints))]
        boxes = []
        rotation = 0
        image_pose = image
        # image_pose = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if detections is not None:
            images = torch.zeros((len(detections), 3, self.resolution[1], self.resolution[0]), device=self.device)  # (height, width)
            # pose estimation : for multiple people
            centers = []
            scales = []
            for box in detections:
                center, scale = box_to_center_scale(box, self.resolution[0], self.resolution[1])
                centers.append(center)
                scales.append(scale)
            model_inputs = []
            for i, (center, scale) in enumerate(zip(centers, scales)):
                trans = get_affine_transform(center, scale, rotation, self.resolution)
                # Crop smaller image of people
                model_input = cv2.warpAffine(
                    image_pose,
                    trans,
                    (int(self.resolution[0]), int(self.resolution[1])),
                    flags=cv2.INTER_LINEAR)
                if keypoints is not None:
                    kpts_homo = keypoints[i].copy()
                    kpts_homo[:, 2] = 1
                    kpts_rescale[i] = (kpts_homo @ trans.T)/4
                # global humanId
                # cv2.imwrite('../output/debughrnet/person_{}.jpg'.format(humanId), model_input[:,:,[2,1,0]])
                # humanId += 1
                # hwc -> 1chw
                model_input = self.transform(model_input)#.unsqueeze(0)
                images[i] = model_input
        # torch.cuda.synchronize(self.device)

        # print(' - spending {:.2f}ms in preprocess.'.format(1000*(time.time() - start)))
        if images.shape[0] == 0:
            return np.empty((0, 25, 3))
        else:
            # start = time.time()
            images = images.to(self.device) 
            # torch.cuda.synchronize(self.device)

            # print(' - spending {:.2f}ms in copy to cuda.'.format(1000*(time.time() - start)))
            # start = time.time()
            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)
                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[1] // 4, self.resolution[0] // 4)
                    ).to(self.device)
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])
            # torch.cuda.synchronize(self.device)
            global humanId
            if keypoints is not None:
                filename = join('../output/debughrnet', '{:06d}.jpg'.format(humanId))
                humanId += 1
                # save_batch_heatmaps(images, out, filename)
                # 制造高斯核，默认为1
                weights = np.ones(out.shape, dtype=np.float32)
                for i in range(weights.shape[0]):
                    for nj in range(weights.shape[1]):
                        if keypoints[i][nj][2] < 0:
                            weights[i][nj] = 0
                            continue
                        elif keypoints[i][nj][2] < 0.01:
                            continue
                        weights[i][nj] = 0
                        mu_x, mu_y = kpts_rescale[i][nj]
                        mu_x, mu_y = int(mu_x + 0.5), int(mu_y + 0.5)
                        # Usable gaussian range
                        ul = [mu_x - gauss_radius, mu_y - gauss_radius]
                        br = [mu_x + gauss_radius + 1, mu_y + gauss_radius + 1]
                        # Usable gaussian range
                        g_x = max(0, -ul[0]), min(br[0], weights.shape[3]) - ul[0]
                        g_y = max(0, -ul[1]), min(br[1], weights.shape[2]) - ul[1]
                        # Image range
                        img_x = max(0, ul[0]), min(br[0], weights.shape[3])
                        img_y = max(0, ul[1]), min(br[1], weights.shape[2])
                        weights[i][nj][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                            gauss_kernel[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                filename = join('../output/debughrnet', '{:06d}.jpg'.format(humanId))
                humanId += 1
                # save_batch_heatmaps(images, torch.Tensor(weights), filename)
                out = out.cpu().detach().numpy()
                out = out * weights
                filename = join('../output/debughrnet', '{:06d}.jpg'.format(humanId))
                humanId += 1
                # save_batch_heatmaps(images, torch.Tensor(out), filename)
            else:
                out = out.cpu().detach().numpy()
            coords, max_val = get_final_preds(
                out,
                np.asarray(centers),
                np.asarray(scales))
            pts = np.concatenate((coords, max_val), axis=2)
            # torch.cuda.synchronize(self.device)
            # print(' - spending {:.2f}ms in postprocess.'.format(1000*(time.time() - start)))
            # print('')
            if ret_crop:
                return coco17tobody25(pts), images
            else:
                return coco17tobody25(pts)