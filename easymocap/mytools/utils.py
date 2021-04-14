'''
  @ Date: 2021-01-15 11:12:00
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-08 21:07:48
  @ FilePath: /EasyMocap/code/mytools/utils.py
'''
import time
import tabulate
class Timer:
    records = {}
    @classmethod
    def report(cls):
        header = ['', 'Time(ms)']
        contents = []
        for key, val in cls.records.items():
            contents.append(['{:20s}'.format(key), '{:.2f}'.format(sum(val)/len(val))])
        print(tabulate.tabulate(contents, header, tablefmt='fancy_grid'))

    def __init__(self, name, silent=False):
        self.name = name
        self.silent = silent
        if name not in Timer.records.keys():
            Timer.records[name] = []
    
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        end = time.time()
        Timer.records[self.name].append((end-self.start)*1000)
        if not self.silent:
            print('-> [{:20s}]: {:5.1f}ms'.format(self.name, (end-self.start)*1000))
