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

## 1. Distortion and Intrinsic Parameter Calibration


## 2. Extrinsic Parameter Calibration
Prepare your images as following:
```bash
./data/examples/calibration
├── extri_images
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── 7.jpg
│   └── 8.jpg
└── intri.yml
```
The basename of the images must be same as the name of cameras in `intri.yml`.

```bash
python3 scripts/calibration/calib_extri.py -i ./data/examples/calibration/extri_images -o ./data/examples/calibration --debug
```
To specify your chessboard, add the option `--pattern`, `--grid`
```bash
python3 scripts/calibration/calib_extri.py -i ./data/examples/calibration/extri_images -o ./data/examples/calibration --debug --pattern 9,6 --grid 0.1
```
More details can be found in the code.