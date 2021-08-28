<!--
 * @Date: 2021-04-02 11:52:33
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-07-22 20:58:33
 * @FilePath: /EasyMocapRelease/doc/installation.md
-->
# EasyMocap - Installation

## 0. Download models

## 0.1 SMPL models

This step is the same as [smplx](https://github.com/vchoutas/smplx#model-loading).

To download the *SMPL* model go to [this](http://smpl.is.tue.mpg.de) (male and female models, version 1.0.0, 10 shape PCs) and [this](http://smplify.is.tue.mpg.de) (gender neutral model) project website and register to get access to the downloads section. 

To download the *SMPL+H* model go to [this project website](http://mano.is.tue.mpg.de) and register to get access to the downloads section. 

To download the *SMPL-X* model go to [this project website](https://smpl-x.is.tue.mpg.de) and register to get access to the downloads section. 

**Place them as following:**

```bash
data
└── smplx
    ├── J_regressor_body25.npy
    ├── J_regressor_body25_smplh.txt
    ├── J_regressor_body25_smplx.txt
    ├── J_regressor_mano_LEFT.txt
    ├── J_regressor_mano_RIGHT.txt
    ├── smpl
    │   ├── SMPL_FEMALE.pkl
    │   ├── SMPL_MALE.pkl
    │   └── SMPL_NEUTRAL.pkl
    ├── smplh
    │   ├── MANO_LEFT.pkl
    │   ├── MANO_RIGHT.pkl
    │   ├── SMPLH_FEMALE.pkl
    │   └── SMPLH_MALE.pkl
    └── smplx
        ├── SMPLX_FEMALE.pkl
        ├── SMPLX_MALE.pkl
        └── SMPLX_NEUTRAL.pkl
```

## 0.2 (Optional) SPIN model
This part is used in `1v1p*.py`. You can skip this step if you only use the multiple views dataset.

Download pretrained SPIN model [here](http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt) and place it to `data/models/spin_checkpoints.pt`.

Fetch the extra data [here](http://visiondata.cis.upenn.edu/spin/data.tar.gz) and place the `smpl_mean_params.npz` to `data/models/smpl_mean_params.npz`.

## 0.3 (Optional) 2D model

You can skip this step if you use openpose as your human keypoints detector.

Download [yolov4.weights]() and place it into `data/models/yolov4.weights`.

Download pretrained HRNet [weight]() and place it into `data/models/pose_hrnet_w48_384x288.pth`.

```bash
data
└── models
    ├── smpl_mean_params.npz
    ├── spin_checkpoint.pt
    ├── pose_hrnet_w48_384x288.pth
    └── yolov4.weights 
```

## 2. Requirements

- python>=3.6
- torch==1.4.0
- torchvision==0.5.0
- opencv-python
- [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html#python-installation): for visualization, or [pyrender for server without a screen](https://pyrender.readthedocs.io/en/latest/install/index.html#getting-pyrender-working-with-osmesa).
- chumpy: for loading SMPL model
- OpenPose[4]: for 2D pose

Some of python libraries can be found in `requirements.txt`. You can test different version of PyTorch.

## 3. Install

```bash
python3 setup.py develop --user
```