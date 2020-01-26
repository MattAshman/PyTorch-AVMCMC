import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from sampler import AVSampler, AVSamplerMix
from distributions import GaussianMixture

def main(args):
    print('Setting up distribution.')
    locs = torch.tensor(args.means)
    scales = torch.tensor(args.scales)
    mixing_coefs = torch.tensor(args.mixing_coefs)  
    dist = GaussianMixture(locs, scales, mixing_coefs)
    
    print('Setting up sampler.')
    sampler = AVSampler(args.dim_target,
            args.dim_aux,
            args.dim_hidden,
            args.layers,
            dist,
            args.perturb)

    sampler.train(args.epochs, args.train_samples, args.learning_rate)

    samples, _ = sampler.sample(args.num_samples)
    samples = torch.cat(samples, dim=0)

    fig, ax = plt.subplots()
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--means', type=float, default=[[-10., 0.],[10., 0.]],
            help='Means of Gaussian mixtures.')
    parser.add_argument('--scales', type=float, default=[[1., 1.],[1., 1.]],
            help='Scales of Gaussian mixtures.')
    parser.add_argument('--mixing_coefs', type=float, default=[0.5, 0.5],
            help='Mixing coefficients of Gaussians.')
    parser.add_argument('--dim_target', type=int, default=2, 
            help='Dimensional of target distribution.')
    parser.add_argument('--dim_aux', type=int, default=1, 
            help='Dimension of auxiliary distribution.')
    parser.add_argument('--dim_hidden', type=int, default=50,
            help='Number of hidden units per layer.')
    parser.add_argument('--layers', type=int, default=1,
            help='Number of hidden layers.')
    parser.add_argument('--perturb', type=float, default=1.,
            help='Scale of auxiliary perturbation.')
    parser.add_argument('--epochs', type=int, default=2000,
            help='Number of training epochs.')
    parser.add_argument('--train_samples', type=int, default=200,
            help='Number of training samples used to estimate KL divergence')
    parser.add_argument('--learning_rate', type=float, default=0.001,
            help='Optimiser learning rate.')
    parser.add_argument('--num_samples', type=int, default=5000,
            help='Number of samples to generate.')

    args = parser.parse_args()
    main(args)
