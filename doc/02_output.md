<!--
 * @Date: 2021-03-07 14:41:22
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-03-13 21:42:11
 * @FilePath: /EasyMocap/doc/02_output.md
-->
# EasyMocap Doc - Output
[En](Output) | [中文](#输出)

## Contents
1. [Json Format](#json-format)
2. [Export to .bvh](#export-to-bvh-format)

## Json Format
The results are saved in `json` format.
```bash
<output_root>
├── keypoints3d
│   ├── 000000.json
│   └── xxxxxx.json
└── smpl
    ├── 000000.jpg
    ├── 000000.json
    └── 000004.json
```
The data in `keypoints3d/000000.json` is a list, each element represents a human body.
```bash
{
    'id': <id>, # the person ID
    'keypoints3d': [[x0, y0, z0, c0], [x1, y1, z0, c1], ..., [xn, yn, zn, cn]], # x,y,z is the 3D coordinates, c means the confidence of this joint. If the c=0, it means this joint is invisible.
}
```
The definition of the joints is as [body25](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#pose-output-format-body_25).

The data in `smpl/000000.json` is also a list, each element represents the SMPL parameters which is slightly different from official model.
```bash
{
    "id": <id>,
    "Rh": <(1, 3)>,
    "Th": <(1, 3)>,
    "poses": <(1, 72/78/87)>,
    "expression": <(1, 10)>,
    "shapes": <(1, 10)>
}
```
If you use SMPL+H model, the poses contains `22x3+6+6`. We use `6` pca coefficients for each hand. `3(jaw, left eye, right eye)x3` poses of head are added for SMPL-X model.

### Attention (for SMPL/SMPL-X users)

**This parameter is a little different from original SMPL/SMPL-X parameters.**

We set the first 3 dimensions of `poses` to zero, and add a new parameter `Rh` to represents the global oritentation, the vertices of SMPL model V = RX(theta, beta) + T.
Please note that the paramter `Rh` is not equal to `global_orient` in SMPL-X model. We take this representation because that changing paramters to new coordinate system in origin is difficult(see [this link](https://www.dropbox.com/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?dl=0&rlkey=lotq1sh6wzkmyttisc05h0in0)).

In our representation, you can just use `R'@(RX + T) + T'` to convert the model, and the new global rotaion and translation is simply written as `R'@R` and `R'@T + T'`

To compute the joints locations from these parameters, please refer to `./code/vis_render.py`. The key steps are:
```python
# 0. load SMPL model
from smplmodel import load_model
body_model = load_model(args.gender, model_type=args.model)
# 1. load parameters
infos = dataset.read_smpl(nf*step)
# 2. compute joints
joints = body_model(return_verts=False, return_tensor=False, **info)[0]
# 3. compute vertices
vertices = body_model(return_verts=True, return_tensor=False, **info)[0]
```

## Export to bvh format
To export the SMPL results to bvh file, you need to download the SMPL-maya model from the website of SMPL. Place the `.fbx` model in `./data/smplx/SMPL_maya`, it may be like this:
```bash
└── smplx
    ├── smpl
    │   ├── SMPL_FEMALE.pkl
    │   ├── SMPL_MALE.pkl
    │   └── SMPL_NEUTRAL.pkl
    ├── SMPL_maya
    │   ├── basicModel_f_lbs_10_207_0_v1.0.2.fbx
    │   ├── basicModel_m_lbs_10_207_0_v1.0.2.fbx
    │   ├── joints_mat_v1.0.2.pkl
    │   ├── README.txt
    │   ├── release_notes_v1.0.2.txt
    │   └── SMPL_maya_plugin_v1.0.2.py
    └── smplx
```
The Blender is also needed. The `<path_to_output_smpl>` is usually `${out}/smpl`, which contanis the `000000.json, ...` of SMPL parameters.
```bash
BLENDER_PATH=<path_to_blender>/blender-2.79a-linux-glibc219-x86_64
${BLENDER_PATH}/blender -b -t 12 -P scripts/postprocess/convert2bvh.py -- <path_to_output_smpl> --o <output_path>
```
We have not implement the export of SMPL+H, SMPL-X model yet. If you are interested on it, feel free to create a pull request to us.

-----

# 输出
## Json格式
关键点重建的结果会输出到`${out}/keypoints3d`路径下
```bash
<out>
├── keypoints3d
│   ├── 000000.json
│   └── xxxxxx.json
└── skel
```
每个json里面是一个列表，包含了当前帧的所有人，列表里的每一个元素表示一个人，内容如下：
```json
{
    'id': <id>, # 表示人的跟踪的id
    'keypoints3d': [[x0, y0, z0, c0], [x1, y1, z0, c1], ..., [xn, yn, zn, cn]]: # (N, 4)，表示人的关键点坐标，c表示置信度，置信度为0则该关节点不可见
}
```
关键点的定义使用OpenPose的[BODY25格式](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#pose-output-format-body_25)

## 导出为bvh格式
这里使用Blender进行导出，测试的Blender版本为2.79。需要先下载SMPL的fbx模型
```bash
BLENDER_PATH=<path_to_blender>/blender-2.79a-linux-glibc219-x86_64
${BLENDER_PATH}/blender -b -t 12 -P scripts/postprocess/convert2bvh.py -- <path_to_output_smpl> --o <path_to_bvh>
```
