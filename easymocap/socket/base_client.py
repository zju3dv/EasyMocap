'''
  @ Date: 2021-05-25 13:39:07
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-16 14:42:35
  @ FilePath: /EasyMocap/easymocap/socket/base_client.py
'''
import socket
from .utils import encode_detect, encode_smpl

class BaseSocketClient:
    def __init__(self, host, port) -> None:
        if host == 'auto':
            host = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        self.s = s
    
    def send(self, data):
        val = encode_detect(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)
    
    def send_smpl(self, data):
        val = encode_smpl(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)
    
    def close(self):
        self.s.close()