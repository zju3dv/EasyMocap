'''
  @ Date: 2021-01-15 11:12:00
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-15 11:19:55
  @ FilePath: /EasyMocap/code/mytools/utils.py
'''
import time

class Timer:
    def __init__(self, name, silent=False):
        self.name = name
        self.silent = silent
    
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        end = time.time()
        if not self.silent:
            print('-> [{}]: {:.2f}s'.format(self.name, end-self.start))
