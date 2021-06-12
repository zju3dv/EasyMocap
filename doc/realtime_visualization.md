<!--
 * @Date: 2021-06-04 15:56:55
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-06-12 15:29:23
 * @FilePath: /EasyMocapRelease/doc/realtime_visualization.md
-->
# EasyMoCap -> Real-time Visualization

We are the first one to release a real-time visualization tool for both skeletons and SMPL/SMPL+H/SMPL-X/MANO models.

## Install

Please install `EasyMocap` first. This part requires `Open3D==0.9.0`:

```bash
python3 -m pip install open3d==0.9.0
```

## Open the server
Before any visualization, you should run a server:

```bash
# quick start:
python3 apps/vis/vis_server.py --cfg config/vis3d/o3d_scene.yml
# If you want to specify the host and port:
python3 apps/vis/vis_server.py --cfg config/vis3d/o3d_scene.yml host <your_ip_address> port <set_a_port>
```

This step will open the visualization window:

![](./assets/vis_server.png)

You can alternate the viewpoints free. The configuration file `config/vis/o3d_scene.yml` defines the scene and other properties. In the default setting, we define the xyz-axis in the origin, the bounding box of the scene and a chessboard in the ground.

## Send the data

If you are success to open the server, you can visualize your 3D data anywhere. We provide an example code:

```bash
python3 apps/vis/vis_client.py --path <path/to/your/keypoints3d> --host <previous_ip_address> --port <previous_port>
```

Take the `zju-ls-feng` results as example, you can show the skeleton in the main window:

![](./assets/vis_client.png)

## Embed this feature to your code

To add this visualization to your other code, you can follow these steps:

```bash
# 1. import the base client
from easymocap.socket.base_client import BaseSocketClient
# 2. set the ip address and port
client = BaseSocketClient(host, port)
# 3. send the data
client.send(data)
```

The format of data is:
```python
data = [
    {
        'id': 0,
        'keypoints3d': numpy.ndarray # (nJoints, 4) , (x, y, z, c) for each joint
    },
    {
        'id': 1,
        'keypoints3d': numpy.ndarray # (nJoints, 4)
    }
]
```

## Define your scene

In the configuration file, we main define the `body_model` and `scene`. You can replace them for your data.