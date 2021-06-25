'''
  @ Date: 2021-06-23 16:54:04
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 11:44:26
  @ FilePath: /EasyMocapRelease/easymocap/assignment/group.py
'''
import numpy as np
from ..mytools.reconstruction import projectN3
from ..mytools.file_utils import batch_bbox_from_pose

class Person:
    width = 1024
    height = 1024
    Pall = None
    def __init__(self, pid) -> None:
        self.id = pid
        self.age = []
        # property:
        self.info = {key:None for key in ['bbox', 'kptsRepro', 'keypoints3d', 'Vused']}

    def add(self, keypoints3d, Vused, **kwargs):
        self.keypoints3d = keypoints3d
        self.Vused = Vused

    def __str__(self) -> str:
        return '{}: {}'.format(self.id, self.Vused)

    @property
    def keypoints3d(self):
        return self.info['keypoints3d']
    
    @keypoints3d.setter
    def keypoints3d(self, k3d):
        kpts_repro = projectN3(k3d, self.Pall)
        bbox = batch_bbox_from_pose(kpts_repro, self.height, self.width)
        self.info['keypoints3d'] = k3d
        self.info['bbox'] = bbox
        self.info['kptsRepro'] = kpts_repro

    @property
    def bbox(self):
        return self.info['bbox']
    
    @property
    def kptsRepro(self):
        return self.info['kptsRepro']
        
class PeopleGroup(dict):
    def __init__(self, Pall, cfg) -> None:
        self.maxid = 0
        self.pids = []
        self.current = []
        self.dimGroups = []
        self.Pall = Pall
        Person.Pall = Pall
    
    def add(self, info):
        # self.current.append(info)
        pid = self.maxid
        people = Person(pid)
        people.add(**info)
        self.maxid += 1
        self[pid] = people

    def clear(self):
        self.pids = []
        self.maxid = 0
        super().clear()

    def __setitem__(self, pid, people) -> None:
        self.pids.append(pid)
        self.maxid += 1
        super().__setitem__(pid, people)

    @property
    def results(self):
        results = []
        for pid, people in self.items():
            result = {'id': pid, 'keypoints3d': people.keypoints3d}
            results.append(result)
        return results
    
    def __str__(self):
        res = '  PeopleDict {:6d}: {}\n'.format(Person.time, ' '.join(map(str, self.pids)))
        for pid in self.pids:
            res += '    {:3d}: {}\n'.format(pid, self[pid])
        return res