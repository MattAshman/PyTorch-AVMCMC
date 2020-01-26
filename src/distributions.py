import pdb
import numpy as np
import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical

class GaussianMixture():
    """A representation of a mixture of Gaussian distribution.

    :locs: A list, means of the Gaussians.
    :scales: A list, covariance matrices of the Gaussians."""

    def __init__(self, locs, scales, mixing_coefs):
        
        assert len(locs) == len(scales) == len(mixing_coefs),"Number of means \
            and covariances not the same!"
        
        self.locs = locs
        self.scales = scales
        self.mixing_coefs = mixing_coefs
        self.num_mixtures = len(locs)
        
        dists = []
        for i in range(self.num_mixtures):
            dists.append(MultivariateNormal(locs[i], 
                torch.diag_embed(scales[i])))

        self.dists = dists

    def sample(self, num_samples):
        categorical = Categorical(self.mixing_coefs)
        samples = []

        for i in range(num_samples):
            dist_idx = categorical.sample()
            samples.append(self.dists[dist_idx].sample())

        return samples

    def log_prob(self, x):
        log_probs = []

        for i, dist in enumerate(self.dists):
            log_probs.append((torch.log(self.mixing_coefs[i]) +
                dist.log_prob(x)).unsqueeze_(1))

        log_probs = torch.cat(log_probs, dim=1)
        
        return torch.logsumexp(log_probs, dim=1)
