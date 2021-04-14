import numpy as np

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1))
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re

def align_by_pelvis(joints, names):
    l_id = names.index('LHip')
    r_id = names.index('RHip')
    pelvis = joints[[l_id, r_id], :].mean(axis=0, keepdims=True)
    return joints - pelvis

def keypoints_error(gt, est, names, use_align=False, joint_level=True):
    assert gt.shape[-1] == 4
    assert est.shape[-1] == 4
    isValid = est[..., -1] > 0
    isValidGT = gt[..., -1] > 0
    isValid_common = isValid * isValidGT
    est = est[..., :-1]
    gt = gt[..., :-1]
    dist = {}
    dist['abs'] = np.sqrt(((gt - est)**2).sum(axis=-1)) * 1000
    dist['pck@50'] = dist['abs'] < 50
    # dist['pck@100'] = dist['abs'] < 100
    # dist['pck@150'] = dist['abs'] < 0.15
    if use_align:
        l_id = names.index('LHip')
        r_id = names.index('RHip')
        assert isValid[l_id] and isValid[r_id]
        assert isValidGT[l_id] and isValidGT[r_id]
        # root align
        gt, est = align_by_pelvis(gt, names), align_by_pelvis(est, names)
        # Absolute error (MPJPE)
        dist['ra'] = np.sqrt(((est - gt) ** 2).sum(axis=-1)) * 1000
        # Reconstuction_error
        est_hat = compute_similarity_transform(est, gt)
        dist['pa'] = np.sqrt(((est_hat - gt) ** 2).sum(axis=-1)) * 1000
    result = {}
    for key in ['abs', 'ra', 'pa', 'pck@50', 'pck@100']:
        if key not in dist:
            continue
        result[key+'_mean'] = dist[key].mean()
        if joint_level:
            for i, name in enumerate(names):
                result[key+'_'+name] = dist[key][i]
    return result