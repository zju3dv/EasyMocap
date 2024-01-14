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
        'easymocap.config',
        'easymocap.dataset',
        'easymocap.smplmodel',
        'easymocap.pyfitting',
        'easymocap.mytools', 
        'easymocap.annotator',
        'easymocap.estimator',
        'myeasymocap'
    ],
    entry_points={
        'console_scripts': [
            'emc=apps.mocap.run:main_entrypoint',
            # 'easymocap_calib=easymocap.mytools.entry:calib',
            # 'easymocap_tools=easymocap.mytools.entry:main',
            # 'extract_keypoints=easymocap.mytools.cmdtools.extract_keypoints:main'
        ],
    },
    install_requires=[],
    data_files = []
)

emc = "apps.mocap.run:main_entrypoint"
