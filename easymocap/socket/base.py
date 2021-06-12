'''
  @ Date: 2021-05-25 11:14:48
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-05 19:32:56
  @ FilePath: /EasyMocap/easymocap/socket/base.py
'''
import socket
import time
from threading import Thread
from queue import Queue

def log(x):
    from datetime import datetime
    time_now = datetime.now().strftime("%m-%d-%H:%M:%S.%f ")
    print(time_now + x)

class BaseSocket:
    def __init__(self, host, port, debug=False) -> None:
        # 创建 socket 对象
        print('[Info] server start')
        serversocket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((host, port))
        serversocket.listen(1)
        self.serversocket = serversocket
        self.queue = Queue()
        self.t = Thread(target=self.run)
        self.t.start()
        self.debug = debug
        self.disconnect = False
    
    @staticmethod
    def recvLine(sock):
        flag = True
        result = b''
        while not result.endswith(b'\n'):
            res = sock.recv(1)
            if not res:
                flag = False
                break
            result += res
        return flag, result.strip().decode('ascii')

    @staticmethod
    def recvAll(sock, l):
        l = int(l)
        result = b''
        while (len(result) < l):
            t = sock.recv(l - len(result))
            result += t
        return result.decode('ascii')

    def run(self):
        while True:
            clientsocket, addr = self.serversocket.accept()
            print("[Info] Connect: %s" % str(addr))
            self.disconnect = False
            while True:
                flag, l = self.recvLine(clientsocket)
                if not flag:
                    print("[Info] Disonnect: %s" % str(addr))
                    self.disconnect = True
                    break
                data = self.recvAll(clientsocket, l)
                if self.debug:log('[Info] Recv data')
                self.queue.put(data)
            clientsocket.close()
    
    def update(self):
        time.sleep(1)
        while not self.queue.empty():
            log('update')
            data = self.queue.get()
            self.main(data)
    
    def main(self, datas):
        print(datas)

    def __del__(self):
        self.serversocket.close()
        self.t.join()