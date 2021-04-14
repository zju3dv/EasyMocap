'''
  @ Date: 2021-03-02 16:53:55
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-14 16:20:10
  @ FilePath: /EasyMocap/setup.py
'''
from setuptools import setup

setup(
    name='easymocap',     
    version='0.2',   #
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
