import pdb
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical

class NNGaussian(nn.Module):
    """A neural network for parameterising a multivariate Gaussian distribution.

    :dim_in: An int, dimension of the input variable.
    :dim_out: An int, dimension of the output variable.
    :dim_hidden: An int, number of hidden units per layer.
    :layers: An int, number of hidden layers."""

    def __init__(self, dim_in, dim_out, dim_hidden, layers):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.layers = layers

        self.fcs = nn.ModuleList()
        for i in range(layers+1):
            if i == 0:
                self.fcs.append(nn.Linear(dim_in, dim_hidden))
            elif i == layers:
                self.fcs.append(nn.Linear(dim_hidden, 2*dim_out))
            else:
                self.fcs.append(nn.Linear(dim_hidden, dim_hidden))

    def forward(self, x):
        for i in range(self.layers):
            x = F.tanh(self.fcs[i](x))

        x = self.fcs[-1](x)

        mu = x[:, :self.dim_out]
        sigma = torch.exp(x[:, self.dim_out:])

        dist = MultivariateNormal(mu, torch.diag_embed(sigma))

        return dist

class NNGaussianMixture(nn.Module):
    """A neural network for parameterising a mixture of multivariate Gaussian
    distributions.
    
    :dim_in: An int, dimension of the input variable.
    :dim_out: An int, dimension of the output variable.
    :dim_hidden: An int, number of hidden units per layer.
    :layers: An int, number of hidden layers.
    :num_mixtures: An int, number of Gaussian mixtures."""

    def __init__(self, dim_in, dim_out, dim_hidden, layers, num_mixtures):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.layers = layers
        self.num_mixtures = num_mixtures

        self.fcs = nn.ModuleList()
        for i in range(layers+1):
            if i == 0:
                self.fcs.append(nn.Linear(dim_in, dim_hidden))
            elif i == layers:
                self.fcs.append(nn.Linear(dim_hidden, 2*dim_out*mixes + 
                    num_mixtures))
            else:
                self.fcs.append(nn.Linear(dim_hidden, dim_hidden))

    def forward(self, x):
        for i in range(self.layers):
            x = F.tanh(self.fcs[i](x))

        x = self.fcs[-1](x)

        mus = x[:, :(self.dim_out*self.num_mixtures)]
        sigma = torch.exp(x[:, 
            (self.dim_out*self.num_mixtures):(2*self.dim_out*self.num_mixtures)])
        cat = F.softmax(x[:, (2*self.dim_out*self.num_mixtures):], dim=1)

        dists = []
        for j in range(self.num_mixtures):
            start = j * self.dim_out
            stop = (j + 1) * self.dim_out
            mu = mus[:, start:stop]
            sigma = sigmas[:, start:stop]
            dist = MultivariateNormal(mu, torch.diag_embed(sigma))
            dists.append(dist)

        return cat, dists

def gaussian_mix_sample(cat, dists):
    categorical = Categorical(cat)
    cat_samples = categorical.sample()

    all_samples = [dists[i].rsample() for i in range(len(dists))]
    samples = torch.empty_like(dists[0].mean)
    for i, idx in enumerate(cat_samples):
        samples[i] = all_samples[idx][i]

    return samples

def gaussian_mix_log_prob(x, cat, dists):
    log_probs = []
    for i, dist in enumerate(dists):
        log_probs.append((torch.log(cat[:, i]) +
            dist.log_prob(x)).unsqueeze_(1))

    log_probs = torch.cat(log_probs, dim=1)
    return torch.logsumexp(log_probs, dim=1)

class AVSampler():
    """A representation of the Auxiliary Variational Sampler.

    :dim_target: An int, dimension of the target distribution.
    :dim_aux: An int, dimension of the auxiliary distribution.
    :dim_hidden: An int, number of hidden units per layer.
    :layers: An int, number of hidden layers.
    :dist_target: A distribution, the target distribution to sample from.
    :perturb: A float, scales the random walk in auxiliary space."""

    def __init__(self, dim_target, dim_aux, dim_hidden, layers, dist_target,
            perturb):

        self.dim_target = dim_target
        self.dim_aux = dim_aux
        self.dim_hidden = dim_hidden
        self.layers = layers
        self.dist_target = dist_target
        self.perturb = perturb

        self.xtoa = NNGaussian(dim_target, dim_aux, dim_hidden, layers)
        self.atox = NNGaussian(dim_aux, dim_target, dim_hidden, layers)

    def train(self, epochs, num_samples, learning_rate=0.001):
        
        optimiser = optim.Adam((list(self.xtoa.parameters()) 
            + list(self.atox.parameters())), lr=learning_rate)

        self.losses = []

        for epoch in range(epochs):
            optimiser.zero_grad()
            loss = self._calc_loss(num_samples)
            loss.backward()
            optimiser.step()

            self.losses.append(loss.item())
            if (epoch % 50) == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

    def sample(self, num_samples, num_chains=1):
        x_samples = []
        a_samples = []

        with torch.no_grad():
            a = torch.randn(num_chains, self.dim_aux)
            q_xgiva = self.atox(a)

            x = q_xgiva.rsample()

            for sample in range(num_samples):
                p_agivx = self.xtoa.forward(x)
                a = p_agivx.rsample()
                ap = a + self.perturb * torch.randn(num_chains, self.dim_aux)

                q_xpgivap = self.atox.forward(ap)
                xp = q_xpgivap.rsample()
                
                alpha = self._calc_alpha(x, a, xp, ap, q_xpgivap, p_agivx)

                accept_prob = torch.min(torch.ones(num_chains), alpha)
                accept = torch.where(torch.rand(num_chains) < accept_prob,
                        torch.ones(num_chains), torch.zeros(num_chains))

                x = accept * xp + (1 - accept) * x
                a = accept * ap + (1 - accept) * a

                x_samples.append(x)
                a_samples.append(a)

        return x_samples, a_samples

    def _calc_loss(self, num_samples):
        q_a = MultivariateNormal(torch.zeros(num_samples, self.dim_aux),
            torch.diag_embed(torch.ones(num_samples, self.dim_aux)))

        a = q_a.sample()

        q_xgiva = self.atox.forward(a)
        x = q_xgiva.rsample()

        p_agivx = self.xtoa.forward(x)

        loss = (q_xgiva.log_prob(x) + q_a.log_prob(a) - 
                self.dist_target.log_prob(x) - p_agivx.log_prob(a))

        loss_mean = torch.mean(loss, axis=0)

        return loss_mean
    
    def _calc_alpha(self, x, a, xp, ap, q_xpgivap, p_agivx):
        p_apgivxp = self.xtoa.forward(xp)
        q_xgiva = self.atox.forward(a)

        alpha = torch.exp(self.dist_target.log_prob(xp) +
                p_apgivxp.log_prob(ap) + q_xgiva.log_prob(x) -
                self.dist_target.log_prob(x) - p_agivx.log_prob(a) -
                q_xpgivap.log_prob(xp))

        return alpha

class AVSamplerMix():
    """A representation of the Auxiliary Variational Sampler, using a mixture
    of Gaussians (parameterised by a neural network) as a variational 
    distribution.

    :dim_target: An int, dimension of the target distribution.
    :dim_aux: An int, dimension of the auxiliary distribution.
    :dim_hidden: An int, number of hidden units per layer.
    :layers: An int, number of hidden layers.
    :num_mixtures: An int, number Gaussian mixtures.   
    :dist_target: A distribution, the target distribution to sample from.
    :perturb: A float, scales the random walk in auxiliary space."""

    def __init__(self, dim_target, dim_aux, dim_hidden, layers, num_mixes, 
            dist_target, perturb):

        self.dim_target = dim_target
        self.dim_aux = dim_aux
        self.dim_hidden = dim_hidden
        self.layers = layers
        self.num_mixtures = num_mixtures
        self.dist_target = dist_target
        self.perturb = perturb

        self.xtoa = NNGaussianMix(dim_target, dim_aux, dim_hidden, layers,
                num_mixtures)
        self.atox = NNGaussian(dim_aux, dim_target, dim_hidden, layers)

    def train(self, epochs, num_samples, learning_rate=0.001):
        
        optimiser = optim.Adam((list(self.xtoa.parameters()) 
            + list(self.atox.parameters())), lr=learning_rate)

        self.losses = []

        for epoch in range(epochs):
            optimiser.zero_grad()
            loss = self._calc_loss(num_samples)
            loss.backward()
            optimiser.step()

            self.loss.append(loss.item())
            if (epoch % 50) == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

    def sample(self, num_samples, num_chains=1):
        x_samples = []
        a_samples = []

        with torch.no_grad():
            a = torch.randn(num_chains, self.dim_aux)
            q_xgiva = self.atox(a)
            x = q_xgiva.rsample()

            for sample in range(num_samples):
                mixes_agivx, dists_agivx = self.xtoa.forward(x)
                a = gaussian_mix_sample(mixes_agivx, dists_agivx)
                ap = a + self.perturb * torch.randn(num_chains, self.dim_aux)

                q_xpgivap = self.atox.forward(ap)
                xp = q_xpgivap.rsample()
                
                alpha = self._calc_alpha(x, a, xp, ap, q_xpgivap, mixes_agivx, 
                        dists_agivx)

                accept_prob = torch.min(torch.ones(num_chains), alpha)
                accept = torch.where(torch.rand(num_chains) < accept_prob,
                        torch.ones(num_chains), torch.zeros(num_chains))

                x = accept * xp + (1 - accept) * x
                a = accept * ap + (1 - accpet) * a

                x_samples.append(x)
                a_samples.append(a)

        return x_samples, a_samples

    def _calc_loss(self, num_samples):
        q_a = MultivariateNormal(torch.zeros(num_samples, self.dim_aux),
            torch.diag_embed(torch.ones(num_samples, self.dim_aux)))

        a = q_a.sample()

        q_xgiva = self.atox.forward(a)
        x = q_xgiva.rsample()

        mixes, dists = self.xtoa.forward(x)

        loss = (q_xgiva.log_prob(x) + q_a.log_prob(a) - 
                self.dist_target.log_prob(x) -
                gaussian_mix_log_prob(a, mixes, dists))

        loss_mean = torch.mean(loss, axis=0)

        return loss_mean
    
    def _calc_alpha(self, x, a, xp, ap, q_xpgivap, mixes_agivx, dists_agivx):
        mixes_apgivxp, dists_apgivxp = self.xtoa.forward(xp)
        q_xgiva = self.atox.forward(a)

        alpha = torch.exp(self.target_dist.log_prob(xp) +
                gaussian_mix_log_prob(ap, mixes_apgivxp, dists_apgivxp) + 
                q_xgiva.log_prob(x) - self.dist_target.log_prob(x) - 
                gaussian_mix_log_prob(a, mixes_agivx, dists_agivx) -
                q_xpgivap.log_prob(xp))

        return alpha

   
