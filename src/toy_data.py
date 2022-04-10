import random
from itertools import islice
import numpy as np

from sklearn.datasets import make_circles, make_blobs

def built_circles(n_samples):
    """Generate points from to two circles
    
    Args:
        - data_size (int): number of points to generate per circle
    Returns:
        - data_1 (np.array): points from the first circle
        - data_2 (np.array): points from the second circle
    """
    data_1, _ = make_circles(n_samples=n_samples, factor=0.99999)
    data_1[:, 0] -= 1
    data_1[:, 1] -= 1
    data_2, _ = make_circles(n_samples=n_samples, factor=0.99999)
    data_2[:, 0] += 1
    data_2[:, 1] += 1
    return data_1, data_2

def built_blobs(n_samples):
    """Generate points from to blobs
    
    Args:
        - data_size (int): number of points to generate per blobs
    Returns:
        - data_1 (np.array): points from the first blobs
        - data_2 (np.array): points from the second blobs
    """
    data_1, _ = make_blobs(n_samples=n_samples, centers=[[-5, -5], [5, 5]])
    data_1 = data_1 / 5
    data_2, _ = make_blobs(n_samples=n_samples, centers=[[-5, 5], [5, -5]])
    data_2 = data_2 / 5
    return data_1, data_2

def uniform_triangle(u, v):
    while True:
        s = random.random()
        t = random.random()
        in_triangle = s + t <= 1
        p = s * u + t * v if in_triangle else (1 - s) * u + (1 - t) * v
        yield p

def built_triangles(n_samples):
    """Generate points from to random triangles

    Args:
        - data_size (int): number of points to generate per triangle 
    
    Returns:
        - points1 (np.array): points from the first triangle
        - points2 (np.array): points from the second triangle
    """
    triangle = np.array([
    [-2, -2],
    [0, -1.25],
    [-1.5, -0.2],
    ])

    it = uniform_triangle(
        triangle[1] - triangle[0],
        triangle[2] - triangle[0],
    )

    points1 = np.array(list(islice(it, 0, n_samples)))
    points1 += triangle[0]

    triangle2 = triangle+2

    it2 = uniform_triangle(
        triangle2[1] - triangle2[0],
        triangle2[2] - triangle2[0],
    )

    points2 = np.array(list(islice(it2, 0, n_samples)))
    points2 += triangle2[0]

    return points1, points2

def built_rectangles(n_samples):
    """Generate points from to random rectangles

    Args:
        - data_size (int): number of points to generate per rectangle 
    
    Returns:
        - points1 (np.array): points from the first rectangle
        - points2 (np.array): points from the second rectangle
    """
    data_1 = np.random.uniform(low=[0,0], high=[0.4,0.2], size=(n_samples, 2))
    data_2 = np.random.uniform(low=[0.8, 0.6], high=[1, 1], size=(n_samples, 2))
    return data_1, data_2

def build_disks(n_samples):
    """Generate points from to random disks

    Args:
        - data_size (int): number of points to generate per disks 
    
    Returns:
        - points1 (np.array): points from the first disks
        - points2 (np.array): points from the second disks
    """
    length = np.random.uniform(0.5, 1.5, n_samples)
    angle = np.pi * np.random.uniform(0, 2, n_samples)

    x = np.sqrt(length) * np.cos(angle) - 1.2
    y = np.sqrt(length) * np.sin(angle) - 1.2

    data_1 = np.stack((x, y), axis=1) 

    x = np.sqrt(length) * np.cos(angle) + 1.2
    y = np.sqrt(length) * np.sin(angle) + 1.2

    data_2 = np.stack((x, y), axis=1) 

    return data_1, data_2

def build_data(n_samples, data_type):
    """Generate points from data_type

    Args:
        - data_size (int): number of points to generate per rectangle 
        - data_type (str): type of data to generate
    
    Returns:
        - data1 (np.array): points from the first data
        - data2 (np.array): points from the second data
    """
    if data_type == 'circles':
        data1, data2 = built_circles(n_samples)
    elif data_type == 'blobs':
        data1, data2 = built_blobs(n_samples)
    elif data_type == 'triangles':
        data1, data2 = built_triangles(n_samples)
    elif data_type == 'rectangles':
        data1, data2 = built_rectangles(n_samples)
    elif data_type == 'disks':
        data1, data2 = build_disks(n_samples)
    else:
        raise ValueError('data_type must be one of the following: circles, blobs, triangles, rectangles, disks')
    return data1, data2