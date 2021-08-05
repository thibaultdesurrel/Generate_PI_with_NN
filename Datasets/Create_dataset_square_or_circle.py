#!/usr/bin/env python
# coding: utf-8

# # Create dataset square and save it

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

import gudhi as gd
import gudhi.representations

from tqdm import tqdm
import multiprocessing

from time import time

from sklearn.neighbors import KernelDensity
# ### Define the constants

# In[2]:


N_sets_train = 10000
N_sets_test = 200
N_points = 1000
PI_size = 50

size = 50
x_sample = np.linspace(-1, 1, size)
y_sample = np.linspace(-1, 1, size)
#xx, yy = np.meshgrid(x_sample, y_sample)
coord = np.array([[x, y] for x in x_sample for y in y_sample])
# In[11]:


def create_circle(N_points):
    X = np.empty([N_points, 2])
    r = 1
    for i in range(N_points):  # On fait un cercle
        theta = np.random.uniform() * 2 * np.pi
        X[i, 0], X[i, 1] = 0.9 * np.cos(theta), 0.9 * np.sin(theta)
    return X


def create_square(N_points):
    X = np.empty([N_points, 2])
    for i in range(N_points):
        X[i, 0], X[i, 1] = 0.9 * \
            (2 * np.random.uniform() - 1), 0.9 * (2 * np.random.uniform() - 1)
    return X


data_square = np.zeros((N_sets_train // 2, N_points, 2))
density_square_train = np.zeros((N_sets_train // 2, size * size))

data_circle = np.zeros((N_sets_train // 2, N_points, 2))
density_circle_train = np.zeros((N_sets_train // 2, size * size))


#PI_train = np.zeros((N_sets_train, PI_size * PI_size))


for i in tqdm(range(N_sets_train // 2), desc="Generating training dataset"):
    data_square[i] = create_square(N_points)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(data_square[i])
    density_square_train[i] = np.exp(kde.score_samples(coord))

    data_circle[i] = create_circle(N_points)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(data_circle[i])
    density_circle_train[i] = np.exp(kde.score_samples(coord))


data_train = np.concatenate((data_square, data_circle))
density_train = np.concatenate((density_square_train, density_circle_train))


PI = gd.representations.PersistenceImage(bandwidth=8e-2,
                                         weight=lambda x: x[1]**2.5,
                                         resolution=[PI_size, PI_size],
                                         im_range=[0, 0.5, 0, 0.5])


def compute_PI(i):
    # Compute its persistence image
    rcX = gd.AlphaComplex(points=data_train[i]).create_simplex_tree()
    dgmX = rcX.persistence()

    pi = PI.fit_transform([rcX.persistence_intervals_in_dimension(1)])
    return pi[0]


starttime = time()
pool = multiprocessing.Pool()
PI_train = pool.map(compute_PI, range(0, N_sets_train))
pool.close()
print('Time taken = {} seconds'.format(time() - starttime))

data_square = np.zeros((N_sets_test // 2, N_points, 2))
density_square_test = np.zeros((N_sets_test // 2, size * size))

data_circle = np.zeros((N_sets_test // 2, N_points, 2))
density_circle_test = np.zeros((N_sets_test // 2, size * size))


#PI_test = np.zeros((N_sets_test, PI_size * PI_size))


for i in tqdm(range(N_sets_test // 2), desc="Generating testing dataset"):
    data_square[i] = create_square(N_points)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(data_square[i])
    density_square_test[i] = np.exp(kde.score_samples(coord))

    data_circle[i] = create_circle(N_points)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(data_circle[i])
    density_circle_test[i] = np.exp(kde.score_samples(coord))


data_test = np.concatenate((data_square, data_circle))
density_test = np.concatenate((density_square_test, density_circle_test))


# In[ ]:


def compute_PI_test(i):
    # Compute its persistence image
    rcX = gd.AlphaComplex(points=data_test[i]).create_simplex_tree()
    dgmX = rcX.persistence()
    pi = PI.fit_transform([rcX.persistence_intervals_in_dimension(1)])
    return pi[0]


# In[ ]:


starttime = time()
pool = multiprocessing.Pool()
PI_test = pool.map(compute_PI_test, range(0, N_sets_test))
pool.close()
print('Time taken = {} seconds'.format(time() - starttime))


# ### Save data

# In[204]:


np.savez_compressed('PI_data_square_or_circle', data_train=data_train, density_train=density_train,
                    PI_train=PI_train, data_test=data_test, PI_test=PI_test, density_test=density_test)
