'''
  @ Date: 2021-01-15 11:12:00
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-27 15:50:50
  @ FilePath: /EasyMocapPublic/easymocap/mytools/timer.py
'''
import time
import tabulate
class Timer:
    records = {}
    tmp = None
    indent = -1

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
        Timer.indent += 1

    def __exit__(self, exc_type, exc_value, exc_tb):
        end = time.time()
        Timer.records[self.name].append((end-self.start)*1000)
        indent = self.indent * '  '
        if not self.silent:
            t = (end - self.start)*1000
            if t > 1000:
                print('-> {}[{:20s}]: {:5.1f}s'.format(indent, self.name, t/1000))
            elif t > 1e3*60*60:
                print('-> {}[{:20s}]: {:5.1f}min'.format(indent, self.name, t/1e3/60))
            else:
                print('-> {}[{:20s}]: {:5.1f}ms'.format(indent, self.name, (end-self.start)*1000))
        Timer.indent -= 1

    @staticmethod
    def timer(name):
        from functools import wraps
        def decorator(func):
            @wraps(func)
            def wrapped_function(*args, **kwargs):
                with Timer(name):
                    ret = func(*args, **kwargs)
                return ret
            return wrapped_function
        return decorator

if __name__ == '__main__':
    @Timer.timer('testfunc')
    def dummyfunc():
        time.sleep(1)
    with Timer('level0'):
        with Timer('level1'):
            with Timer('level2'):
                time.sleep(1)
            time.sleep(1)
        time.sleep(1)
    dummyfunc()
    Timer.report()
            