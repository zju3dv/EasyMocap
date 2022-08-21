'''
  @ Date: 2022-02-14 14:54:50
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-06-14 18:07:19
  @ FilePath: /EasyMocapPublic/easymocap/mytools/debug_utils.py
'''
from termcolor import colored
import os
from os.path import join
import shutil
import subprocess
import time
import datetime

def toc():
    return time.time() * 1000

def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))

def log(text):
    myprint(text, 'info')

def log_time(text):
    strf = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(colored(strf, 'yellow'), colored(text, 'green'))

def mywarn(text):
    myprint(text, 'warn')

warning_infos = set()

def oncewarn(text):
    if text in warning_infos:
        return
    warning_infos.add(text)
    myprint(text, 'warn')


def myerror(text):
    myprint(text, 'error')

def run_cmd(cmd, verbo=True, bg=False):
    if verbo: myprint('[run] ' + cmd, 'run')
    if bg:
        args = cmd.split()
        print(args)
        p = subprocess.Popen(args)
        return [p]
    else:
        os.system(cmd)
        return []

def mkdir(path):
    if os.path.exists(path):
        return 0
    log('mkdir {}'.format(path))
    os.makedirs(path, exist_ok=True)

def cp(srcname, dstname):
    mkdir(join(os.path.dirname(dstname)))
    shutil.copyfile(srcname, dstname)

def print_table(header, contents):
    from tabulate import tabulate
    length = len(contents[0])
    tables = [[] for _ in range(length)]
    mean = ['Mean']
    for icnt, content in enumerate(contents):
        for i in range(length):
            if isinstance(content[i], float):
                tables[i].append('{:6.2f}'.format(content[i]))
            else:
                tables[i].append('{}'.format(content[i]))
        if icnt > 0:
            mean.append('{:6.2f}'.format(sum(content)/length))
    tables.append(mean)
    print(tabulate(tables, header, tablefmt='fancy_grid'))

def check_exists(path):
    flag1 = os.path.isfile(path) and os.path.exists(path)
    flag2 = os.path.isdir(path) and len(os.listdir(path)) >= 10
    return flag1 or flag2