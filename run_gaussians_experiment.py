import _pickle as cPickle
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.optim as optim
from src.losses import sliced_wasserstein_distance
from src.flows import *
from src.models import SWFlowModel
import argparse
import os
import logging
import numpy as np

def fit_gaussian(samples, hist):
    # samples - NxD, hist: N
    n, d = samples.shape
    mean = np.sum(samples * np.expand_dims(hist, 1), axis=0)
    p = samples - np.expand_dims(mean, 0)
    cov = np.zeros([d, d])
    for i in range(n):
        cov += hist[i] * np.outer(p[i, :], p[i, :])
    return mean, cov

def compare_gaussian_with_empirical(gt_mean, gt_cov, exp_hist, exp_samples):
    exp_mean, exp_cov = fit_gaussian(exp_samples, exp_hist)
    mean_diff = np.linalg.norm(gt_mean - exp_mean)
    cov_diff = np.linalg.norm(gt_cov - exp_cov)
    return {'mean_diff': mean_diff, 'cov_diff': cov_diff}

def main():
        device = torch.device("cuda")

        stats = {}
        rmd = {}
        for dim in range(2, 9, 2):
            #load barencter mean and cov
            print("Training for dim: " + str(dim))
            with open(r"src/gaussian/result/"+str(dim)+"d/00-gaussian_iterative.pkl", "rb") as input_file:
                gt_gaussian = cPickle.load(input_file)
                
            gt_mean = gt_gaussian['mean']
            gt_normal_A = gt_gaussian['normal_A']
            gt_cov = np.matmul(gt_normal_A, np.transpose(gt_normal_A))
            
            #load data mean and cov
            with open(r"src/gaussian/data/"+str(dim)+"d/gaussian_0.pkl", "rb") as input_file:
                param_1 = cPickle.load(input_file)
            with open(r"src/gaussian/data/"+str(dim)+"d/gaussian_1.pkl", "rb") as input_file:
                param_2 = cPickle.load(input_file)
                
            mean_1 = param_1['mean']
            normal_A_1 = param_1['normal_A']
            cov_1 = np.matmul(normal_A_1, np.transpose(normal_A_1))
            
            mean_2 = param_2['mean']
            normal_A_2 = param_2['normal_A']
            cov_2 = np.matmul(normal_A_2, np.transpose(normal_A_2))
            
            mean_1 = np.zeros(dim)
            cov_1 = np.eye(dim)
            
            mean_2 = np.zeros(dim) + 3
            cov_2 = np.eye(dim)
            
            n=10000

            data_1 = np.random.multivariate_normal(mean_1, cov_1, n)
            data_2 = np.random.multivariate_normal(mean_2, cov_2, n)
            
            print("_____Build the normalizing flows_____")
            flows = [ActNorm(dim=dim) for _ in range(4)]
            model1 = SWFlowModel(flows, device)
            optimizer1 = optim.Adam(model1.parameters(), lr=0.001)

            x = torch.from_numpy(data_1.astype(np.float32)).to(device)

            trueZ = torch.from_numpy(data_2.astype(np.float32)).to(device)
            
            lamb = 0.000
            gamma = 0.000

            for slices in np.arange(1500, 3200, 100):
                for i in range(1000):
                    optimizer1.zero_grad()
                    z, shatten, _ = model1(x)
                    sw = sliced_wasserstein_distance(z, trueZ, slices, 2, device).mean()
                    cost = torch.sum(model1.transport_cost(x))
                    loss = sw + lamb*cost + gamma*shatten
                    loss.backward()
                    optimizer1.step()
                print(f"Model "+str(dim)+" |\t" +
                    f"Slices: {slices}\t" +
                    f"SW: {sw:.5f}\t" +
                    f"shatten: {shatten.mean():.5f}\t" +
                    f"cost: {cost.mean():.5f}\t" +
                    f"loss: {loss:.5f}")
                if slices >= 2500: 
                    if lamb < 0.0008:
                        lamb += 0.00001
                    if gamma < 0.0008:
                        gamma += 0.00001
                        
            for i in range(1000):
                optimizer1.zero_grad()
                z, shatten, _ = model1(x)
                sw = sliced_wasserstein_distance(z, trueZ, slices, 2, device).mean()
                cost = torch.sum(model1.transport_cost(x))
                loss = sw + lamb*cost + gamma*shatten
                loss.backward()
                optimizer1.step()
            print(f"Model "+str(dim)+" |\t" +
                f"Slices: {slices}\t" +
                f"SW: {sw:.5f}\t" +
                f"shatten: {shatten.mean():.5f}\t" +
                f"cost: {cost.mean():.5f}\t" +
                f"loss: {loss:.5f}")
            
            data_1 = np.random.multivariate_normal(mean_1, cov_1, 10000)
            x = torch.from_numpy(data_1.astype(np.float32)).to(device)
                        
            exp_samples= model1.forward_barycenter(x, 2).detach().cpu().numpy()
            exp_hist = np.zeros(exp_samples.shape[0]) + 1/exp_samples.shape[0]
            
            xp_mean, xp_cov = fit_gaussian(exp_samples, exp_hist)
            tmp = compare_gaussian_with_empirical(gt_mean, gt_cov, exp_hist, exp_samples)
            lkh = multivariate_normal.pdf(exp_samples, gt_mean, gt_cov).mean()
            tmp["likelyhood"] = lkh
            print(tmp)

if __name__ == '__main__':
    main()
