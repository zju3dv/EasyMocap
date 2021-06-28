'''
  @ Date: 2021-01-15 11:12:00
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 21:07:29
  @ FilePath: /EasyMocapRelease/easymocap/mytools/utils.py
'''
import time
import tabulate
class Timer:
    records = {}
    tmp = None

    @classmethod
    def tic(cls):
        cls.tmp = time.time()
    @classmethod
    def toc(cls):
        res = (time.time() - cls.tmp) * 1000
        cls.tmp = None
        return res
    
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
            t = (end - self.start)*1000
            if t > 1000:
                print('-> [{:20s}]: {:5.1f}s'.format(self.name, t/1000))
            elif t > 1e3*60*60:
                print('-> [{:20s}]: {:5.1f}min'.format(self.name, t/1e3/60))
            else:
                print('-> [{:20s}]: {:5.1f}ms'.format(self.name, (end-self.start)*1000))
