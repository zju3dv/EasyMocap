<!--
 * @Date: 2021-04-02 11:53:55
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-04-02 11:53:55
 * @FilePath: /EasyMocapRelease/doc/notquickstart.md
-->

### 0. Prepare Your Own Dataset

```bash
zju-ls-feng
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

Here `intri.yml` and `extri.yml` store the camera intrinsici and extrinsic parameters. For example, if the name of a video is `1.mp4`, then there must exist `K_1`, `dist_1` in `intri.yml`, and `R_1((3, 1), rotation vector of camera)`, `T_1(3, 1)` in `extri.yml`. The file format is following [OpenCV format](https://docs.opencv.org/master/dd/d74/tutorial_file_input_output_with_xml_yml.html).

### 1. Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

```bash
data=path/to/data
out=path/to/output
python3 scripts/preprocess/extract_video.py ${data} --openpose <openpose_path> --handface
```

- `--openpose`: specify the openpose path
- `--handface`: detect hands and face keypoints

### 2. Run the code

```bash
# 1. example for skeleton reconstruction
python3 code/demo_mv1pmf_skel.py ${data} --out ${out} --vis_det --vis_repro --undis --sub_vis 1 7 13 19
# 2. example for SMPL reconstruction
python3 code/demo_mv1pmf_smpl.py ${data} --out ${out} --end 300 --vis_smpl --undis --sub_vis 1 7 13 19
```

The input flags:

- `--undis`: use to undistort the images
- `--start, --end`: control the begin and end number of frames.

The output flags:

- `--vis_det`: visualize the detection
- `--vis_repro`: visualize the reprojection
- `--sub_vis`: use to specify the views to visualize. If not set, the code will use all views
- `--vis_smpl`: use to render the SMPL mesh to images.

### 3. Output

Please refer to [output.md](doc/02_output.md)