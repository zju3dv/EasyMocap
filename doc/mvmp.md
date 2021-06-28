<!--
 * @Date: 2021-06-28 14:09:50
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-06-28 19:28:14
 * @FilePath: /EasyMocapRelease/doc/mvmp.md
-->
# EasyMocap - mvmp

This code aims to solve the problem of reconstructing multiple persons from multiple calibrated cameras. The released code is an easy-to-use version. See [Advanced](#Advanced) for more details.

## 0. Preparation

Prepare your calibrated and synchronized system by yourself.

You can download our dataset [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/s_q_zju_edu_cn/EZFGgpK2Y6RBkPbGvny_PC0BIS08qJvxGYEHYopjhHX_TQ?e=LY3pgm).

```bash
├── intri.yml
├── extri.yml
├── annots
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   ├── 5
│   ├── 6
│   └── 7
└── videos
    ├── 0.mp4
    ├── 1.mp4
    ├── 2.mp4
    ├── 3.mp4
    ├── 4.mp4
    ├── 5.mp4
    ├── 6.mp4
    └── 7.mp4
```

Extract the images from videos:
```bash
data=/path/to/data
python3 scripts/preprocess/extract_video.py ${data} --no2d
```

## 1. Reconstucting human pose
This step will reconstruct the human pose in each frame.
```bash
python3 apps/demo/mvmp.py ${data} --out ${data}/output --annot annots --cfg config/exp/mvmp1f.yml --undis --vis_det --vis_repro
```

## 2. Recovering SMPL body model
First we should tract the human pose in each frame. This step will track and interpolate missing frames.
```bash
python3 apps/demo/auto_track.py ${data}/output ${data}/output-track --track3d
```

Then we can fit SMPL model to the tracked keyponts:

```bash
python3 apps/demo/smpl_from_keypoints.py ${data} --skel ${data}/output-track/keypoints3d --out ${data}/output-track/smpl --verbose --opts smooth_poses 1e1
```

To visualize the results, see [visualization tutorial](./doc/realtime_visualization.md)


## Advanced

For more complicated scenes, our lab has a real-time version of this algorithm, which can perform 3D reconstruction and tracking simultaneously.

If you want to use this part for commercial queries, please contact [Xiaowei Zhou](mailto:xwzhou@zju.edu.cn).



https://user-images.githubusercontent.com/22812405/123629197-968c0080-d846-11eb-8417-4e6d3a65466d.mp4

