<!--
 * @Date: 2021-04-02 11:52:33
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-04-02 11:52:59
 * @FilePath: /EasyMocapRelease/doc/installation.md
-->
# EasyMocap - Installation

### 1. Download SMPL models

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

### 2. Requirements

- python>=3.6
- torch==1.4.0
- torchvision==0.5.0
- opencv-python
- [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html#python-installation): for visualization
- chumpy: for loading SMPL model
- OpenPose[4]: for 2D pose

Some of python libraries can be found in `requirements.txt`. You can test different version of PyTorch.