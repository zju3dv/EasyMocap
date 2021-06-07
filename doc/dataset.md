<!--
 * @Date: 2021-06-07 11:57:34
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-06-07 14:45:17
 * @FilePath: /EasyMocapRelease/doc/dataset.md
-->
# EasyMoCap - Dataset

For convenience, all of the data used by EasyMoCap share the same format.

## Input structure

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

You can use this commond to extract the videos to images:
```bash
python3 scripts/preprocess/extract_video.py ${data} --no2d
```

After this, the folder will be like:
```bash
<seq>
├── intri.yml
├── extri.yml
└── images
    ├── 1
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    ├── 2
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    ├── ...
    ├── ...
    ├── 8
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── 9
        ├── 000000.jpg
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
```

## 2D Pose

For each image, we record its 2D pose in a `json` file. For an image at `root/images/1/000000.jpg`, the 2D pose willl store at `root/annots/1/000000.json`. The content of the annotation file is:

```bash
{
    "filename": "images/0/000000.jpg",
    "height": <the height of image>,
    "width": <the width of image>,
    "annots:[
        {
            'personID': 0, # ID of person
            'bbox': [l, t, r, b, conf],
            'keypoints': [[x0, y0, c0], [x1, y1, c1], ..., [xn, yn, cn]],
            'area': <the area of bbox>
        },
        {
            'personID': 1, # ID of person
            'bbox': [l, t, r, b, conf],
            'keypoints': [[x0, y0, c0], [x1, y1, c1], ..., [xn, yn, cn]],
            'area': <the area of bbox>
        }
    ]
}
```

The definition of the `keypoints` is `body25`. If you want to use other definitions, you should add it to `easymocap/dataset/config.py`

## 3D Pose

```bash
[
    {
        'id': <id>, # the person ID
        'keypoints3d': [[x0, y0, z0, c0], [x1, y1, z0, c1], ..., [xn, yn, zn, cn]], # x,y,z is the 3D coordinates, c means the confidence of this joint. If the c=0, it means this joint is invisible.
    },
    {
        'id': <id>, # the person ID
        'keypoints3d': [[x0, y0, z0, c0], [x1, y1, z0, c1], ..., [xn, yn, zn, cn]], # x,y,z is the 3D coordinates, c means the confidence of this joint. If the c=0, it means this joint is invisible.
    }
]
```
