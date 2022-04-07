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

def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Optimal Transport Normilizing Flows')
    parser.add_argument('--outdir', default='SWOT-Flows/images/', help='directory to output images')
    parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--hidden_dim', type=float, default=8, metavar='HD',
                        help='hidden dimention (default 256')
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
    # create output directory
    imagesdir = os.path.join(args.outdir, 'images')
    chkptdir = os.path.join(args.outdir, 'models')
    os.makedirs(imagesdir, exist_ok=True)
    os.makedirs(chkptdir, exist_ok=True)
    # determine device and device dep. args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

