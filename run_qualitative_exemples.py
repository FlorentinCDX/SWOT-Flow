import torch
import torch.nn as nn
import torch.optim as optim
from src.losses import sliced_wasserstein_distance
from src.flows import *
from src.models import SWFlowModel

import matplotlib.pyplot as plt
import argparse
import os
import logging
import numpy as np
from sklearn.datasets import make_circles

def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Optimal Transport Normilizing Flows')
    parser.add_argument('--outdir', default='images/', help='directory to output images')
    parser.add_argument('--epochs', type=int, default=700, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument("--nb_flows", default=4, type=int,
                        help='Number of successive flows transformation (default: 4)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default:10)')
    parser.add_argument('--data_size', type=int, default=2500, metavar='N',
                         help='Size of the datasets (default: 2500)')                        
    parser.add_argument('--dataset', type=str, default="circles", metavar='DT',
                         help='dataset on the x space (default: "circles")')  
                         
    args = parser.parse_args()
    # setup log printing 
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib.font_manager').disabled = True
    # determine device and device dep. args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Build the normalizing flows
    flows = [ActNorm(dim=2) for _ in range(args.nb_flows)]

    model1 = SWFlowModel(flows, device)
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr)

    if args.dataset:
        data_1, _ = make_circles(n_samples=args.data_size, factor=0.99999)
        data_1[:, 0] -= 1
        data_1[:, 1] -= 1
        data_2, _ = make_circles(n_samples=args.data_size, factor=0.99999)
        data_2[:, 0] += 1
        data_2[:, 1] += 1

    x = torch.from_numpy(data_1.astype(np.float32)).to(device)
    trueZ = torch.from_numpy(data_2.astype(np.float32)).to(device)
    lamb = 0.000
    gamma = 0.000

    for slices in np.arange(1500, 3000, 100):
        for i in range(args.epochs):
            optimizer1.zero_grad()
            z, shatten, _ = model1(x)
            sw = sliced_wasserstein_distance(z, trueZ, slices, 2, device).mean()
            cost = torch.sum(model1.transport_cost(x))
            loss = sw + lamb*cost + gamma*shatten
            loss.backward()
            optimizer1.step()
            if i % 100 == 0:
                print(f"Model 1 |\t" +
                    f"Slices: {slices}\t" +
                    f"Iter: {i}\t" +
                    f"SW: {sw:.5f}\t" +
                    f"shatten: {shatten.mean():.5f}\t" +
                    f"cost: {cost.mean():.5f}\t" +
                    f"loss: {loss:.5f}")
        if slices > 2500:
            if lamb < 0.00035:
                    lamb += 0.00001
            if gamma < 0.00035:
                gamma += 0.00001

    n = model1.nb_flows

    for i in range(n+1):
        dat = model1.forward_barycenter(x, i).detach().cpu().numpy()
        plt.axis([-2.1, 2.1, -2.1, 2.1])
        plt.axis('off')
        plt.scatter(dat[:, 0], dat[:, 1], s=1, color=plt.cm.Blues(0.3+i/4))
        plt.plot()
        
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(args.outdir + args.dataset + '_OT_ActNorm.pdf', format='pdf')

if __name__ == '__main__':
    main()
