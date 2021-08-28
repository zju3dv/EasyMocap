<!--
 * @Date: 2021-04-02 11:53:16
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-07-22 20:57:16
 * @FilePath: /EasyMocapRelease/doc/quickstart.md
-->
# Quick Start

First install this project following [install](./installation.md)

## Demo

We provide an example multiview dataset[[dropbox](https://www.dropbox.com/s/24mb7r921b1g9a7/zju-ls-feng.zip?dl=0)][[BaiduDisk](https://pan.baidu.com/s/1lvAopzYGCic3nauoQXjbPw)(vg1z)], which has 800 frames from 23 synchronized and calibrated cameras. After downloading the dataset, you can run the following example scripts.

```bash
data=path/to/data
# 0. extract the video to images
python3 scripts/preprocess/extract_video.py ${data} --handface
# 2.1 example for SMPL reconstruction
python3 apps/demo/mv1p.py ${data} --out ${data}/output/smpl --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --vis_smpl
# 2.2 example for SMPL-X reconstruction
python3 apps/demo/mv1p.py ${data} --out ${data}/output/smplx --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body bodyhandface --model smplx --gender male --vis_smpl
# 2.3 example for MANO reconstruction
#     MANO model is required for this part
python3 apps/demo/mv1p.py ${data} --out ${data}/output/manol --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body handl --model manol --gender male --vis_smpl
python3 apps/demo/mv1p.py ${data} --out ${data}/output/manor --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body handr --model manor --gender male --vis_smpl
```

# Demo On Your Dataset

## 0. Prepare Your Own Dataset

```bash
<seq>
├── intri.yml
├── extri.yml
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
    ├── 8.mp4
    └── 9.mp4
```

The input videos are placed in `videos/`.

Here `intri.yml` and `extri.yml` store the camera intrinsici and extrinsic parameters.

See [`apps/calibration/Readme`](../apps/calibration/Readme.md) for instruction of camera calibration.

See [`apps/calibration/camera_parameters`](../apps/calibration/camera_parameters.md) for the format of camera parameters.

### 1. Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

```bash
data=path/to/data
out=path/to/output
python3 scripts/preprocess/extract_video.py ${data} --openpose <openpose_path> --handface
```

- `--openpose`: specify the openpose path
- `--handface`: detect hands and face keypoints

### 2. Run the code

The input flags:

- `--undis`: use to undistort the images
- `--start, --end`: control the begin and end number of frames.

The output flags:

- `--vis_det`: visualize the detection
- `--vis_repro`: visualize the reprojection
- `--sub_vis`: use to specify the views to visualize. If not set, the code will use all views
- `--vis_smpl`: use to render the SMPL mesh to images.
- `--write_smpl_full`: use to write the full poses of the SMPL parameters

### 3. Output

Please refer to [output.md](../doc/02_output.md)