'''
  @ Date: 2020-10-23 20:07:49
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-05 13:43:01
  @ FilePath: /EasyMocap/code/estimator/SPIN/spin_api.py
'''
"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
"""

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2

from .models import hmr

class constants:
    FOCAL_LENGTH = 5000.
    IMG_RES = 224

    # Mean and standard deviation for normalizing input image
    IMG_NORM_MEAN = [0.485, 0.456, 0.406]
    IMG_NORM_STD = [0.229, 0.224, 0.225]


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t
    
def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1
    
def crop(img, center, scale, res, rot=0, bias=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape) + bias

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    new_img = cv2.resize(new_img, (res[0], res[1]))
    return new_img
    
def process_image(img, bbox, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    img = img[:, :, ::-1].copy()
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    l, t, r, b = bbox[:4]
    center = [(l+r)/2, (t+b)/2]
    width = max(r-l, b-t)
    scale = width/200.0
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

def estimate_translation_np(S, joints_2d, joints_conf, K):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length
    f = np.array([K[0, 0], K[1, 1]])
    # optical center
    center = np.array([K[0, 2], K[1, 2]])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans
    
class SPIN:
    def __init__(self, SMPL_MEAN_PARAMS, checkpoint, device) -> None:
        model = hmr(SMPL_MEAN_PARAMS).to(device)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model'], strict=False)
        # Load SMPL model
        model.eval()
        self.model = model
        self.device = device
    
    def forward(self, img, bbox, use_rh_th=True):
        # Preprocess input image and generate predictions
        img, norm_img = process_image(img, bbox, input_res=constants.IMG_RES)
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = self.model(norm_img.to(self.device))
        results = {
            'shapes': pred_betas.detach().cpu().numpy()
        }
        rotmat = pred_rotmat[0].detach().cpu().numpy()
        poses = np.zeros((1, rotmat.shape[0]*3))
        for i in range(rotmat.shape[0]):
            p, _ = cv2.Rodrigues(rotmat[i])
            poses[0, 3*i:3*i+3] = p[:, 0]
        results['poses'] = poses
        if use_rh_th:
            body_params = {
                'poses': results['poses'],
                'shapes': results['shapes'],
                'Rh': results['poses'][:, :3].copy(),
                'Th': np.zeros((1, 3)),
            }
            body_params['Th'][0, 2] = 5
            body_params['poses'][:, :3] = 0
            results = body_params
        return results

def init_with_spin(body_model, spin_model, img, bbox, kpts, camera):
    body_params = spin_model.forward(img.copy(), bbox)
    body_params = body_model.check_params(body_params)
    # only use body joints to estimation translation
    nJoints = 15
    keypoints3d = body_model(return_verts=False, return_tensor=False, **body_params)[0]
    trans = estimate_translation_np(keypoints3d[:nJoints], kpts[:nJoints, :2], kpts[:nJoints, 2], camera['K'])
    body_params['Th'] += trans[None, :]
    # convert to world coordinate
    Rhold = cv2.Rodrigues(body_params['Rh'])[0]
    Thold = body_params['Th']
    Rh = camera['R'].T @ Rhold
    Th = (camera['R'].T @ (Thold.T - camera['T'])).T
    body_params['Th'] = Th
    body_params['Rh'] = cv2.Rodrigues(Rh)[0].reshape(1, 3)
    vertices = body_model(return_verts=True, return_tensor=False, **body_params)[0]
    keypoints3d = body_model(return_verts=False, return_tensor=False, **body_params)[0]
    results = {'body_params': body_params, 'vertices': vertices, 'keypoints3d': keypoints3d}
    return results
    
if __name__ == '__main__':
    pass