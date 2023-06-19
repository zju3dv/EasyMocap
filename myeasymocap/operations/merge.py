import numpy as np
import cv2
import scipy
import torch

class MultilView_Merge:
    def __init__(self) -> None:
        pass
    def forward(self, data,ax=0):
        '''
        data - dict
        data[key] (nv,...)
        '''
        results={}
        for key in data.keys():
            results[key] = data[key].mean(axis=ax)
        return results


class Merge_hand(MultilView_Merge):
    def __init__(self, camtoworld) -> None:
        self.camtoworld = camtoworld
        # pass
    def __call__(self, posel , cameras, match3d_l):
        # ret = []
        # for nf in range(len(posel)):
        # breakpoint()
        hand_list=[]
        for pid in range(len(match3d_l)):
            dt = match3d_l[pid]
            if(isinstance(dt,int)):
                # TODO:处理-1的情况，也就是没有找到合适的匹配到的手
                hand_list.append(np.zeros((1,48)))
                break
            Merge_list=[]
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
                else:
                    Merge_list.append(pose)
            out = self.forward({'pose':np.stack(Merge_list)},0)
            
            hand_list.append(out['pose'])
        pose_ = np.stack(hand_list)
        Rh = pose_[:,:3].copy()
        pose_[:,:3] = 0
        params={
            'Rh':Rh,
            'Th':np.zeros_like(Rh),
            'poses':pose_,
            'shapes':np.zeros((Rh.shape[0],10)),
        }
        # ret.append(params)
        return {'params': params}

class Merge_handlr(Merge_hand):
    def __call__(self, posel, poser, cameras, match3d_l, match3d_r):
        params_l = super().__call__(posel, cameras, match3d_l)
        params_r = super().__call__(poser, cameras, match3d_r)
        # breakpoint()
        return {'params_l':params_l['params'], 'params_r':params_r['params']}
        # return {'params_l':params_l['params'], 'params_r':params_r['params'], 'params':params_l['params']}

class Merge_bodyandhand:
    def __init__(self, tmp) -> None:
        pass
    def get_R(self, poses, cfg, st):
        res = st.copy()
        for i in cfg:
            res = res @ cv2.Rodrigues(poses[i,:])[0]
        return  res
    def process_poses_mano(self, poses, hand_Rh, flag):
        if sum(flag) == 0:
            return poses 
        
        poses = poses.reshape((-1,3))
        cfg={'rt': [0,3,6,9],
            'r': [14,17,19],
            'l': [13,16,18]
        }
        RA = self.get_R(poses, cfg['rt'],np.eye(3))

        if flag[0] :
            RL = self.get_R(poses, cfg['l'],RA)
            tmppose = np.matrix(RL).I @ cv2.Rodrigues(np.array(hand_Rh[0]))[0]
            tmppose = cv2.Rodrigues(tmppose)[0]
            poses[20,:] = tmppose.reshape(3)

            e20 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(poses[20,:]).reshape(-1,3))
            e20 = e20.as_euler('ZYX', degrees=True)

           
            dt = scipy.spatial.transform.Rotation.from_euler('ZYX', np.array([0,0,e20[0,2]/2]), degrees=True)
            rot_dt = dt.as_matrix()
            rot18 = cv2.Rodrigues(poses[18,:])[0]
            rot18 = rot18@rot_dt
            vec18 = cv2.Rodrigues(rot18)[0].reshape((1,3))
            rot20 = cv2.Rodrigues(poses[20,:])[0]
            rot20 = np.linalg.inv(rot_dt) @ rot20
            vec20 = cv2.Rodrigues(rot20)[0].reshape((1,3))
            poses[20,:] = vec20
            poses[18,:] = vec18

            # e18 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(poses[18,:]).reshape(-1,3))
            # e18 = e18.as_euler('ZYX', degrees=True)
            # e20[0,2] =  e20[0,2]/2
            # e18[0,2] += e20[0,2]
            # e20 = scipy.spatial.transform.Rotation.from_euler('ZYX', e20, degrees=True)
            # e20 = e20.as_rotvec()
            # e18 = scipy.spatial.transform.Rotation.from_euler('ZYX', e18, degrees=True)
            # e18 = e18.as_rotvec()
            # poses[20,:] = e20
            # poses[18,:] = e18
        if flag[1] : #and sum(np.array(hand_Rh[1])!=0)>0:
            RR = self.get_R(poses, cfg['r'],RA)
            tmppose = np.matrix(RR).I @ cv2.Rodrigues(np.array(hand_Rh[1]))[0]
            tmppose = cv2.Rodrigues(tmppose)[0]
            poses[21,:] = tmppose.reshape(3)
            
            e21 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(poses[21,:]).reshape(-1,3))
            e21 = e21.as_euler('ZYX', degrees=True)

            dt = scipy.spatial.transform.Rotation.from_euler('ZYX', np.array([0,0,e21[0,2]/2]), degrees=True)
            rot_dt = dt.as_matrix()
            rot19 = cv2.Rodrigues(poses[19,:])[0]
            rot19 = rot19@rot_dt
            vec19 = cv2.Rodrigues(rot19)[0].reshape((1,3))
            rot21 = cv2.Rodrigues(poses[21,:])[0]
            rot21 = np.linalg.inv(rot_dt) @ rot21
            vec21 = cv2.Rodrigues(rot21)[0].reshape((1,3))
            poses[21,:] = vec21
            poses[19,:] = vec19

            # e19 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(poses[19,:]).reshape(-1,3))
            # e19 = e19.as_euler('ZYX', degrees=True)
            # e21[0,2] =  e21[0,2]/2
            # e19[0,2] += e21[0,2]
            # e21 = scipy.spatial.transform.Rotation.from_euler('ZYX', e21, degrees=True)
            # e21 = e21.as_rotvec()
            # e19 = scipy.spatial.transform.Rotation.from_euler('ZYX', e19, degrees=True)
            # e19 = e19.as_rotvec()
            # poses[21,:] = e21
            # poses[19,:] = e19

        return poses.reshape((1,-1))

    def merge_pose(self, bodypose,handlpose,handrpose):
        flag=[True,True]
        if abs(handlpose).sum()==0:
            flag[0]=False
        if abs(handrpose).sum()==0:
            flag[1]=False
                
        out_L = []
        pose = np.hstack((bodypose,handlpose[:,3:],handrpose[:,3:])) # (1,156)
        out_pose = self.process_poses_mano(pose, [handlpose[0,:3],handrpose[0,:3]], flag) # 如果没找到手，那么应该设置为全0 这里设置为false
        out_L.append(out_pose)
        return out_pose
    def __call__(self, params_l, params_r, params):
        # breakpoint()
        bz = params['Rh'].shape[0]
        ret = {
            'Rh':    np.zeros((bz,3),dtype=np.float32),
            'Th':    params['Th'],
            'poses': np.zeros((bz,156),dtype=np.float32),
            'shapes':np.zeros((bz,16),dtype=np.float32)
        }
        ret['shapes'][:,:10] = params['shapes']
        # breakpoint()
        #TODO for nframe nperson 
        for i in range(bz):
            inpose = np.zeros((1,66))
            inpose[:,3:] = params['poses'][i][:63].copy()
            inpose[:,:3] = params['Rh'][i].copy() # pose0:3有值 Rh  可能要合并

            handlpose = params_l['poses'][i].reshape((1,-1)).copy()
            handrpose = params_r['poses'][i].reshape((1,-1)).copy()
            handlpose[:,:3] = params_l['Rh'][i]
            handrpose[:,:3] = params_r['Rh'][i]

            out = self.merge_pose(inpose.reshape((1,-1)), handlpose, handrpose)
            ret['Rh'][i] = out[:,:3]
            ret['poses'][i,3:] = out[:,3:]
        return {'params_smplh': ret}