'''
  @ Date: 2021-05-24 18:51:58
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-04 17:00:15
  @ FilePath: /EasyMocapRelease/apps/vis/vis_server.py
'''
# socket server for 3D visualization
from easymocap.socket.o3d import VisOpen3DSocket
from easymocap.config.vis_socket import Config

def main(cfg):
    server = VisOpen3DSocket(cfg.host, cfg.port, cfg)
    while True:
        server.update()

if __name__ == "__main__":
    cfg = Config.load_from_args()
    main(cfg)