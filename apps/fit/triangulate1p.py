'''
  @ Date: 2022-03-07 14:33:33
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-21 15:27:52
  @ FilePath: /EasyMocapPublic/apps/fit/triangulate1p.py
'''
from easymocap.config.baseconfig import load_object, Config
from easymocap.mytools import Timer
from tqdm import tqdm
from easymocap.socket.base_client import BaseSocketClient
from easymocap.mytools.debug_utils import mywarn, run_cmd
import time
import numpy as np

def check_ip_port(address):
    ip, port = address.split(':')[:2]
    port = int(port)
    flag = port != -1
    return flag, ip, port

INDEX_HALF = [11,12,13,14,15,16,17,18,19, 20]
INDEX_HALF = sum([[3*i+d for d in range(3)] for i in INDEX_HALF], [])

def triangulate(triangulator, dataset, vis_client):
    for nf in tqdm(range(len(dataset)), desc='recon'):
        with Timer('require data', not args.timer):
            data = dataset[nf]
        with Timer('triangulate', not args.timer):
            results = triangulator(data)
        if vis_client is not None and results != -1 and args.half2total:
            results = [r.copy() for r in results]
            for res in results:
                root = np.zeros((1, 3))
                poses = np.zeros((1, 63))
                poses[:, INDEX_HALF] = res['poses']
                poses_full = np.hstack([root, poses, res['handl'], res['handr']])
                res['poses'] = poses_full
            results = {'annots': results}
            vis_client.to_euler(results)
            vis_client.send_str(results)
        elif vis_client is not None and results != -1:
            vis_client.send_any(results)
        if results != -1 and not args.no_write:
            dataset.write_all(results, data)
        elif results != -1:
            pass
        else:
            mywarn('No results in frame {}'.format(nf))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage='''
    This script helps to triangulate the keypoints.
''')
    for name in ['data', 'exp']:
        parser.add_argument('--cfg_{}'.format(name), type=str)
        parser.add_argument('--opt_{}'.format(name), type=str, nargs='+', default=[])
    parser.add_argument('--det2d', type=str, default='localhost:-1',
        help='detector server address')
    parser.add_argument('--vis3d', type=str, default='localhost:-1',
        help='vis3d server address')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--half2total', action='store_true')
    parser.add_argument('--timer', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # open the 2d detector
    flag_2d, ip_2d, port_2d = check_ip_port(args.det2d)
    # check the 3d visualizer
    flag_3d, ip_3d, port_3d = check_ip_port(args.vis3d)
    if flag_3d:
        try:
            vis_client = BaseSocketClient(args.vis3d)
        except:
            cmd = 'python3 apps/vis/vis_server.py --cfg config/vis3d/o3d_scene_manol.yml host {} port {}'.format(ip_3d, port_3d)
            run_cmd(cmd, bg=True)
            time.sleep(10)
    else:
        vis_client = None
    
    opt_data = args.opt_data
    if flag_2d:
        opt_data.extend(['args.host', [args.det2d]])

    cfg_data = Config.load(args.cfg_data, args.opt_data)
    cfg_exp = Config.load(args.cfg_exp, args.opt_exp)
    if args.debug:
        cfg_exp.args.debug = True
        print(cfg_data)
        print(cfg_exp)
    with Timer('Loading {}'.format(args.cfg_data)):
        dataset = load_object(cfg_data.module, cfg_data.args)
    
    triangulator = load_object(cfg_exp.module, cfg_exp.args)
    triangulate(triangulator, dataset, vis_client)