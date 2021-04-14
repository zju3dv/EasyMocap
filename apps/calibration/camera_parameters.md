<!--
 * @Date: 2021-04-13 16:49:12
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-04-13 16:51:16
 * @FilePath: /EasyMocapRelease/apps/calibration/camera_parameters.md
-->
# Camera Parameters Format

For example, if the name of a video is `1.mp4`, then there must exist `K_1`, `dist_1` in `intri.yml`, and `R_1((3, 1), rotation vector of camera)`, `T_1(3, 1)` in `extri.yml`. The file format is following [OpenCV format](https://docs.opencv.org/master/dd/d74/tutorial_file_input_output_with_xml_yml.html).

## Write/Read

See `easymocap/mytools/camera_utils.py`=>`write_camera`, `read_camera` functions.

## Conversion between different format

TODO

