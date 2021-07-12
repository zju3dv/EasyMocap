<!--
 * @Date: 2021-06-07 11:57:34
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-07-12 20:21:27
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

If you use hand and face, the annot is defined as:
```bash
{
    "personID": i,
    "bbox": [l, t, r, b, conf],
    "keypoints": [[x0, y0, c0], [x1, y1, c1], ..., [xn, yn, cn]],
    "bbox_handl2d": [l, t, r, b, conf],
    "bbox_handr2d": [l, t, r, b, conf],
    "bbox_face2d": [l, t, r, b, conf],
    "handl2d": [[x0, y0, c0], [x1, y1, c1], ..., [xn, yn, cn]],
    "handr2d": [[x0, y0, c0], [x1, y1, c1], ..., [xn, yn, cn]],
    "face2d": [[x0, y0, c0], [x1, y1, c1], ..., [xn, yn, cn]]
}
```

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

The definition of the keypoints can be found in `easymocap/dataset/config.py`. We main use the following formats:
- body25: 25 keypoints of body
- bodyhand: 25 body + 21 left hand + 21 right hand
- bodyhandface: 25 body + 21 left hand + 21 right hand + 51 face keypoints