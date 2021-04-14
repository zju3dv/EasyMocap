<!--
 * @Date: 2021-03-02 16:14:48
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-03-27 21:56:34
 * @FilePath: /EasyMocap/scripts/calibration/Readme.md
-->
# Camera Calibration
Before reading this document, you should read the OpenCV-Python Tutorials of [Camera Calibration](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html) carefully.

## Some Tips
1. Use a chessboard as big as possible.
2. You must keep the same resolution during all the steps.

## 0. Prepare your chessboard

## 1. Record videos
Usually, we need to record two sets of videos, one for intrinsic parameters and one for extrinsic parameters.

First, you should record a video with your chessboard for each camera separately. The videos of each camera should be placed into the `<intri_data>/videos` directory. The following code will take the file name as the name of each camera.
```bash
<intri_data>
└── videos
    ├── 01.mp4
    ├── 02.mp4
    ├── ...
    └── xx.mp4
```

For the extrinsic parameters, you should place the chessboard pattern where it will be visible to all the cameras (on the floor for example) and then take a picture or a short video on all of the cameras.

```bash
<extri_data>
└── videos
    ├── 01.mp4
    ├── 02.mp4
    ├── ...
    └── xx.mp4
```

## 2. Detect the chessboard
For both intrinsic parameters and extrinsic parameters, we need detect the corners of the chessboard. So in this step, we first extract images from videos and second detect and write the corners.
```bash
# extrac 2d
python3 scripts/preprocess/extract_video.py ${data} --no2d
# detect chessboard
python3 apps/calibration/detect_chessboard.py ${data} --out ${data}/output/calibration --pattern 9,6 --grid 0.1
```
The results will be saved in `${data}/chessboard`, the visualization will be saved in `${data}/output/calibration`.

To specify your chessboard, add the option `--pattern`, `--grid`.

Repeat this step for `<intri_data>` and `<extri_data>`.

## 3. Intrinsic Parameter Calibration

```bash
python3 apps/calibration/calib_intri.py ${data} --step 5
```

## 4. Extrinsic Parameter Calibration
```
python3 apps/calibration/calib_extri.py ${extri} --intri ${intri}/output/intri.yml
```

## 5. (Optional)Bundle Adjustment

Coming soon

## 6. Check the calibration

1. Check the calibration results with chessboard:
```bash
python3 apps/calibration/check_calib.py ${extri} --out ${intri}/output --vis --show
```

Check the results with a cube.
```bash
python3 apps/calibration/check_calib.py ${extri} --out ${extri}/output --cube
```

2. (TODO) Check the calibration results with people.