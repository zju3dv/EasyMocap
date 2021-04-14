<!--
 * @Date: 2021-04-13 17:30:14
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-04-13 17:36:26
 * @FilePath: /EasyMocapRelease/apps/annotation/Readme.md
-->
# EasyMocap - Annotator

## Usage
### Example
To start with our annotator, you should take 1 minutes to learn it. First you can run our example script:

```bash
python3 apps/annotation/annot_example.py ${data}
```

#### Mouse
In this example, you can try the two basic operations: `click` and `move`.
- `click`: click the left mousekey. This operation is often used if you want to select something.
- `move`: press the left mousekey and drag the mouse. This operation is often used if you want to plot a line or move something.

#### Keyboard
We list some common keys:
|key|usage|
|----|----|
|`h`|help|
|`w`, `a`, `s`, `d`|switch the frame|
|`q`|quit|
|`p`|start/stop recording the frame|


## Annotate tracking

```bash
python3 apps/annotation/annot_track.py ${data}
```

- `click` the center of bbox to select a person.
- press `0-9` to set the person's ID
- `x` to delete the bbox
- `drag` the corner to reshape the bounding box
- `t`: tracking the person to previous frame

## Annotate vanishing line

```bash
python3 apps/annotation/annot_vanish.py ${data}
```

- `drag` to plot a line
- `X`, `Y`, `Z` to add this line to the set of vanishing lines.
- `k` to calculate the intrinsic matrix with vanishing points in dim x, y.
- `b` to calculating the vanishing point from human keypoints

## Annotate keypoints(coming soon)

## Annotate calibration board(coming soon)

## Define your annotator

