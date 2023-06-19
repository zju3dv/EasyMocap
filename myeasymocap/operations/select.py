from typing import Any
import numpy as np
import cv2

LOG_FILE = 'log_hand_select.txt'
LOG_LEVEL = 2 #0 2
FULL_LOG = (lambda x: print(x, file=open(LOG_FILE, 'a'))) if LOG_LEVEL > 1 else (lambda x: None)
LOG = (lambda x: print(x, file=open(LOG_FILE, 'a'))) if LOG_LEVEL > 0 else (lambda x: None)

def views_from_dimGroups(dimGroups):
    views = np.zeros(dimGroups[-1], dtype=np.int)
    for nv in range(len(dimGroups) - 1):
        views[dimGroups[nv]:dimGroups[nv+1]] = nv
    return views

class Select_Views:
    def __init__(self, camtoworld, handtype) -> None:
        self.camtoworld = camtoworld
        self.results = []
        self.DIST_MAX = 50
        self.threshold = 2
        self.handtype = handtype

        self.threshold2 = 0.3

        self.count = 0
        self.mode = 0 #[0,1] 0-sum  1-max&sum

    def cvt_Rh_Rot(self, Rh):
        import cv2
        RotList = []
        for i in range(Rh.shape[0]):
            RotList.append(cv2.Rodrigues(Rh[i])[0])
        return np.stack(RotList)

    def get_dis_Rh(self, Rh1, Rh2):
        rh_dis = (self.cvt_Rh_Rot(Rh1) - self.cvt_Rh_Rot(Rh2))**2
        return rh_dis.sum(axis=(1,2))

    def match_with_lastframe(self, lastpose, new_poses):
        # breakpoint()
        if self.mode==0:
            rh_dis = self.get_dis_Rh(np.array(new_poses)[:,:3], lastpose[None][:,:3])
            dis = ((np.array(new_poses)[:,3:] - lastpose[None][:,3:])**2).sum(axis=1)
            dis+=rh_dis
            minid = np.argmin(dis)
            return new_poses[minid], dis[minid], minid, dis
        else:
        # breakpoint()
            dis1 = ((np.array(new_poses) - lastpose[None])**2).sum(axis=1)
            dis2 = ((np.array(new_poses) - lastpose[None])**2).max(axis=1)
            dis = np.stack([dis2,dis1]).T
            val_idx = dis[:,0]<self.threshold2
            dis = dis[val_idx,:]
            if(len(dis)==0):
                dis = ((np.array(new_poses) - lastpose[None])**2).sum(axis=1)
                minid = np.argmin(dis1)
                mindis = dis[minid]
                return new_poses[minid], mindis, minid, dis

            else:
                minid = np.argmin(dis[:,1])
                mindis = dis[minid,1]
                # breakpoint()
                # minid = val_idx[minid]
                return np.array(new_poses)[val_idx,:][minid], mindis, minid, dis


        # breakpoint()
        # dis = ((np.array(new_poses) - lastpose[None])**2).sum(axis=1)
        # minid = np.argmin(dis)
        # return new_poses[minid], mindis, minid, dis
    
    def calculate_aff(self, poseslist, DIST_MAX):
        #TODO Rh的距离不能这么求，最好是转成Rot再求误差
        M = len(poseslist)
        distance = np.zeros((M, M), dtype=np.float32)
        for id0 in range(M):
            for id1 in range(id0+1,M):
                p0 = poseslist[id0]
                p1 = poseslist[id1]
                dis = ((p0-p1)**2).sum()
                distance[id0,id1]=dis
                distance[id1,id0]=dis
        DIST_MAX = max(DIST_MAX, distance.max())
        # breakpoint()
        # return distance
        for nv in range(M):
            distance[nv,nv]=DIST_MAX
        # for nv in range(nViews):
        #     distance[dimGroups[nv]:dimGroups[nv+1], dimGroups[nv]:dimGroups[nv+1]] = DIST_MAX
        distance -= np.eye(M) * DIST_MAX
        aff = (DIST_MAX - distance)/DIST_MAX
        aff = np.clip(aff, 0, 1)
        return aff
    
    def Hierarchical_Cluster(self, data,threshold=2):
        # import matplotlib.pyplot as plt
        # breakpoint()
        if(len(data)==1):
            return [[0]]
        import scipy.cluster.hierarchy as sch

        out = sch.linkage(data , method = 'ward')
        ret=[]
        vis=[]
        for i in range(len(data)):
            ret.append([i])
            vis.append(0)

        for i in range(out.shape[0]):
            if(out[i][2]>threshold):
                break
            id1 = int(out[i][0])
            id2 = int(out[i][1])
            vis[id1]=1
            vis[id2]=1
            vis.append(0)
            ret.append(ret[id1]+ret[id2])

        groups = []
        for i in range(len(ret)):
            if vis[i]==1:
                continue
            groups.append(ret[i])

        return groups
    
    def aff_to_groups(data, affinity, dimGroups, prev_id):
        sum1 = np.zeros((affinity.shape[0]))
        for i in range(len(dimGroups)-1):
            start, end = dimGroups[i], dimGroups[i+1]
            if end == start:continue
            sum1 += affinity[:, start:end].max(axis=-1)
        n2d = affinity.shape[0]
        nViews = len(dimGroups) - 1
        idx_zero = np.zeros(nViews, dtype=np.int) - 1
        views = views_from_dimGroups(dimGroups)
        # the assigned results of each person
        p2dAssigned = np.zeros(n2d, dtype=np.int) - 1
        visited = np.zeros(n2d, dtype=np.int)
        sortidx = np.argsort(-sum1)
        pid = 0
        k3dresults = []
        breakpoint()

        return k3dresults

        
    
    def __call__(self, posel , cameras, match3d_l):
        hand_list=[]
        # breakpoint()
        for pid in range(len(match3d_l)):
            dt = match3d_l[pid]

            Merge_list=[]
            Merge_list_rot = []

            if(isinstance(dt,int)):
                # TODO:处理-1的情况，也就是没有找到合适的匹配到的手
                # hand_list.append(np.zeros((48,)))
                Merge_list_rot.append(np.zeros((54,)))
                # continue
            else:
                for cid in range(len(dt['views'])):
                    nv = dt['views'][cid]
                    poseid = dt['indices'][cid]
                    pose = posel[nv][poseid].copy()

                    if self.camtoworld:
                        Rh = pose[:,:3].copy()
                        invR = np.linalg.inv(cameras['R'][nv])
                        Rh_m_old = np.matrix(cv2.Rodrigues(Rh)[0])
                        Rh_m_new = invR @ Rh_m_old
                        Rh = cv2.Rodrigues(Rh_m_new)[0]
                        Merge_list.append(np.hstack((Rh.reshape(3),pose[:,3:].reshape(-1))))
                        # breakpoint()
                        Merge_list_rot.append(np.hstack((np.array(Rh_m_new).reshape(-1),pose[:,3:].reshape(-1))))

                    else:
                        Merge_list.append(pose.reshape(-1))

                        Rh = pose[:,:3].copy()
                        Rh_m_old = np.matrix(cv2.Rodrigues(Rh)[0])
                        Merge_list_rot.append(np.hstack((np.array(Rh_m_old).reshape(-1),pose[:,3:].reshape(-1))))

                    #将坐标系转换，及视角选择完的pose整理成新的集合。

            
            # breakpoint()
            # self.count, pid, self.handtype, str(groups), (0,1)  0的话是,选了哪一组？ 1 xuanle怎么选择的
            #用层次聚类的方法进行视角的选择
            # groups = self.Hierarchical_Cluster(Merge_list, self.threshold)

            groups = self.Hierarchical_Cluster(Merge_list_rot, self.threshold)


            # #求亲和矩阵，即任意两个pose之间的距离。
            # affinity = self.calculate_aff(Merge_list,self.DIST_MAX)
            # N2D = affinity.shape[0]
            # prev_id = np.zeros(N2D) - 1
            # dims = [1]*N2D
            # dimGroups = np.cumsum([0] + dims)
            # groups = self.aff_to_groups(Merge_list, affinity, dimGroups, prev_id)
            # # #根据亲和矩阵进行分组，这里可以考虑将分组的结果Merge起来。
            # groups = []

            FULL_LOG('[select views] frame:{}, pid:{}, handtype:{}'.format(self.count, pid, self.handtype))
            FULL_LOG('[groups] groups:{}'.format(str(groups)))

            
            #合并分组结果
            new_poses = []
            for gp in groups:
                # merge_pose = np.array(Merge_list)[gp].mean(axis=0)
                merge_pose = np.array(Merge_list_rot)[gp].mean(axis=0)
                # breakpoint()


                Rot = merge_pose[:9].reshape((3,3))
                Rh = cv2.Rodrigues(Rot)[0]
                merge_pose = np.hstack((Rh.reshape(3),merge_pose[9:].reshape(-1)))

                new_poses.append(merge_pose)
            #多个组，求每个组和上一帧结果之间的距离。（找出上一帧匹配的手，和这帧对应的手）
            #根据该距离在多个组之间进行选择。选出距离更小的组。
            # if self.handtype == 'handr':
            #     breakpoint()
            if (len(self.results)>pid): # False and 
                # TODO 求与前一帧的距离，如果发现距离过大？则尝试重启跟踪？即选择视角最多的
                pose_, dis, minid, dis_ = self.match_with_lastframe(self.results[pid],new_poses)
                FULL_LOG('[select 0 ] minid:{}'.format(minid))
                FULL_LOG('[select 0 ] dis:{}'.format(str(dis_.tolist())))
                if isinstance(dt,int) or dis_.min()>10: # 没有合适的视角检测到手，或者所有视角检测到的都与上一帧差的很远
                    FULL_LOG('[select 0 ] las pose')
                    pose_ = self.results[pid].copy()
                else:
                    threshold_=0.3
                    if self.mode==1:
                        threshold_=1
                    if(dis>threshold_):# 超过一定阈值，假定上一帧不是很好，则这帧重选
                        array_len = np.array([len(gp) for gp in groups])
                        a_max = array_len.max()
                        d_max = 500
                        idx=0
                        for gid in range(array_len.shape[0]):
                            # breakpoint()
                            if array_len[gid]==a_max and dis_[gid]<d_max:
                                d_max = dis_[gid]
                                idx=gid
                        # dis_[array_len==a_max
                        # breakpoint()
                        # dis_
                        # idx=np.argmax([len(gp) for gp in groups])
                        pose_ = new_poses[idx].copy()

                        FULL_LOG('[select 0 ] max len(groups):{}\n'.format(idx))

                self.results[pid] = pose_.copy()
            else:
                #TODO如果没有前一帧的监督，一种可以用所有组的结果进行处理，另外就是可以用数量较多的组的结果
                #TODO 如果数量相同的有多组，需要进一步处理 比如根据aff求sum最大的？
                idx=np.argmax([len(gp) for gp in groups])
                pose_ = new_poses[idx].copy()
                self.results.append(pose_.copy())

                FULL_LOG('[select 1 ] max len(groups):{}\n'.format(idx))
            #将结果整理返回，有一组和身体id对应的左手或者右手的Pose集合（在世界坐标系下的），也可以返回Params ,看卡params是个list还是dict?

            hand_list.append(pose_)
        poses_ = np.stack(hand_list)
        Rh = poses_[:,:3].copy()
        poses_[:,:3] = 0
        params={
            'Rh':Rh,
            'Th':np.zeros_like(Rh),
            'poses':poses_,
            'shapes':np.zeros((Rh.shape[0],10)),
        }

        self.count+=1

        return {'params': params}

class Select_Views_handlr:
    def __init__(self, camtoworld) -> None:
        self.camtoworld = camtoworld
        self.model_l = Select_Views(camtoworld, 'handl')
        self.model_r = Select_Views(camtoworld, 'handr')

    def __call__(self, posel, poser, match3d_l, match3d_r, cameras) -> Any:
        params_l = self.model_l(posel, cameras, match3d_l)
        params_r = self.model_r(poser, cameras, match3d_r)
        return {'params_l':params_l['params'], 'params_r':params_r['params']}