'''
  @ Date: 2021-05-24 18:57:48
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-04 16:43:00
  @ FilePath: /EasyMocapRelease/apps/vis/vis_client.py
'''
import socket
import time
from easymocap.socket.base_client import BaseSocketClient
import os

def send_rand(client):
    import numpy as np
    for _ in range(1000):
        k3d = np.random.rand(25, 4)
        data = [
            {
                'id': 0,
                'keypoints3d': k3d
            }
        ]
        client.send(data)
        time.sleep(0.005)
    client.close()

def send_dir(client, path):
    from os.path import join
    from glob import glob
    from tqdm import tqdm
    from easymocap.mytools.reader import read_keypoints3d
    results = sorted(glob(join(path, '*.json')))
    for result in tqdm(results):
        data = read_keypoints3d(result)
        client.send(data)
        time.sleep(0.005)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='auto')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.host == 'auto':
        args.host = socket.gethostname()
    client = BaseSocketClient(args.host, args.port)

    if args.path is not None and os.path.isdir(args.path):
        send_dir(client, args.path)
    else:
        send_rand(client)