'''
  @ Date: 2021-06-04 20:47:38
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-04 21:50:53
  @ FilePath: /EasyMocapRelease/easymocap/affinity/matchSVT.py
'''
import numpy as np

def matchSVT(M_aff, dimGroups, M_constr, M_obs, control):
    max_iter = control['maxIter']
    w_rank = control['w_rank']
    tol = control['tol']
    X = M_aff
    N = X.shape[0]
    index_diag = np.arange(N)
    X[index_diag, index_diag] = 0.
    X = (X + X.T)/2
    Y = np.zeros((N, N))
    mu = 64
    W = control['w_sparse'] - X
    for iter_ in range(max_iter):
        X0 = X.copy()
        # update Q with SVT
        Q = 1.0/mu * Y + X
        U, s, VT = np.linalg.svd(Q)
        diagS = s - w_rank/mu
        diagS[diagS<0] = 0
        
        Q = U @ np.diag(diagS) @ VT
        # update X
        X = Q - (W + Y)/mu
        # project X
        for i in range(len(dimGroups)-1):
            ind1, ind2 = dimGroups[i], dimGroups[i + 1]
            X[ind1:ind2, ind1:ind2] = 0
        X[index_diag, index_diag] = 1.
        X[X < 0] = 0
        X[X > 1] = 1
        X = X * M_constr
        if False:
            pass
        
        X = (X + X.T)/2
        # update Y
        Y = Y + mu * (X - Q)
        pRes = np.linalg.norm(X - Q)/N
        dRes = mu * np.linalg.norm(X - X0)/N
        if control['log']:print('[Match] {}, Res = ({:.4f}, {:.4f}), mu = {}'.format(iter_, pRes, dRes, mu))

        if pRes < tol and dRes < tol:
            break
        
        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2
    return X