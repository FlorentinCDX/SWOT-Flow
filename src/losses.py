import numpy as np

import torch
import torch.nn.functional as F

def rand_projections( embedding_dim, num_samples:int):
    """
    This function generates num_samples random samples from the latent space's unti sphere.r

    Args:
        embedding_dim (int): dimention of the embedding
        sum_samples (int): number of samples

    Return :
        torch.tensor: tensor of size (num_samples, embedding_dim)
    """
    projection = [w / np.sqrt((w**2).sum()) for w in np.random.normal(size=(num_samples, embedding_dim))]
    projection = np.array(projection)
    return torch.from_numpy(projection).type(torch.FloatTensor)


def sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2,
                                 device='cpu'):
    """
    Sliced Wasserstein distance between encoded samples and distribution samples

    Args:
        encoded_samples (torch.Tensor): tensor of encoded training samples
        distribution_samples (torch.Tensor): tensor drawn from the prior distribution
        num_projection (int): number of projections to approximate sliced wasserstein distance
        p (int): power of distance metric
        device (torch.device): torch device 'cpu' or 'cuda' gpu

    Return:
        torch.Tensor: Tensor of wasserstein distances of size (num_projections, 1)
    """
    embedding_dim = distribution_samples.size(1)

    projections = rand_projections(embedding_dim, num_projections).to(device)

    encoded_projections = encoded_samples.matmul(projections.transpose(0,1))
    
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))

    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])

    wasserstein_distance = torch.pow(wasserstein_distance, p)
    return wasserstein_distance.mean(1)