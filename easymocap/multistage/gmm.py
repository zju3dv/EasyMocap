import pickle
import os
from os.path import join
import numpy as np
import torch
from .lossbase import LossBase

def create_prior_from_cmu(n_gaussians, epsilon=1e-15):
    """Load the gmm from the CMU motion database."""
    from os.path import dirname
    np_dtype = np.float32
    with open(join(dirname(__file__), 'gmm_%02d.pkl'%(n_gaussians)), 'rb') as f:
        gmm = pickle.load(f, encoding='latin1')
    if True:
        means = gmm['means'].astype(np_dtype)
        covs = gmm['covars'].astype(np_dtype)
        weights = gmm['weights'].astype(np_dtype)
    precisions = [np.linalg.inv(cov) for cov in covs]
    precisions = np.stack(precisions).astype(np_dtype)

    sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                        for c in gmm['covars']])
    const = (2 * np.pi)**(69 / 2.)

    nll_weights = np.asarray(gmm['weights'] / (const * (sqrdets / sqrdets.min())))
    cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
    return {
        'means': means,
        'covs': covs,
        'precisions': precisions,
        'nll_weights': -np.log(nll_weights[None]),
        'weights': weights,
        'pi_term': np.log(2*np.pi),
        'cov_dets': cov_dets
    }

class MaxMixturePrior(LossBase):
    def __init__(self, num_gaussians=8, epsilon=1e-16, use_merged=True,
        start=3, end=72):
        super(MaxMixturePrior, self).__init__()
        np_dtype = np.float32

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        data = create_prior_from_cmu(num_gaussians)
        self.start = start
        self.end = end
        for key, val in data.items():
            self.register_buffer(key, torch.tensor(val, dtype=torch.float32))

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, poses):
        poses = poses[..., self.start:self.end]
        diff_from_mean = poses.unsqueeze(dim=1) - self.means[None, :, :self.end-self.start]

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
            [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic + self.nll_weights
        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, poses, **kwargs):
        if self.use_merged:
            return self.merged_log_likelihood(poses).mean()
        else:
            return self.log_likelihood(poses).mean()

class MaxMixtureCompletePrior(object):
    """Prior density estimation."""
    prior = None
    mean_pose = None
    def __init__(self, n_gaussians=8, start=3, end=72):
        self.n_gaussians = n_gaussians
        self.start = start
        self.end = end
        if self.prior is None:
            self.prior = self.create_prior_from_cmu()

    def create_prior_from_cmu(self):
        """Load the gmm from the CMU motion database."""
        from os.path import dirname
        np_dtype = np.float32
        with open(join(dirname(__file__), 'gmm_%02d.pkl'%(self.n_gaussians)), 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')
        if True:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        self.means = means
        self.weights = weights
        self.mean_pose = weights.dot(means)

    def __call__(self, body_model, body_params, info):
        poses = body_params['poses']
        for nf in range(poses.shape[0]):
            poses[nf][self.start:self.end] = self.mean_pose[:self.end-self.start]
        return body_params

    def get_gmm_prior(self):
        """Getter implementation."""
        return self.prior

class GMMPrior(MaxMixturePrior):
    def __call__(self, pred, target):
        poses = pred['poses']
        poses = poses.reshape(-1, poses.shape[-1])
        if self.use_merged:
            return self.merged_log_likelihood(poses).mean()
        else:
            return self.log_likelihood(poses).mean()