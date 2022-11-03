'''
  @ Date: 2021-03-02 16:53:55
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-11-03 13:10:59
  @ FilePath: /EasyMocapRelease/setup.py
'''
from setuptools import setup

setup(
    name='easymocap',     
    version='0.2.1',   #
    description='Easy Human Motion Capture Toolbox',
    author='Qing Shuai', 
    author_email='s_q@zju.edu.cn',
    # test_suite='setup.test_all',
    packages=[
        'easymocap',
        'easymocap.dataset',
        'easymocap.smplmodel',
        'easymocap.pyfitting',
        'easymocap.mytools', 
        'easymocap.annotator',
        'easymocap.estimator'
    ],
    install_requires=[],
    data_files = []
)
