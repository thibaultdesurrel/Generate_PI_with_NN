#!/usr/bin/env python
# coding: utf-8

# # Create dataset  dynamical system and save it

# In[1]:

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
N_sets_test = 1000
N_points = 5000
PI_size = 50

size = 50
x_sample = np.linspace(0, 1, size)
y_sample = np.linspace(0, 1, size)
#xx, yy = np.meshgrid(x_sample, y_sample)
coord = np.array([[x, y] for x in x_sample for y in y_sample])

# In[3]:


def create_orbit(N_points, r):
    X = np.empty([N_points, 2])
    x, y = np.random.uniform(), np.random.uniform()
    for i in range(N_points):
        X[i, :] = [x, y]
        x = (X[i, 0] + r * X[i, 1] * (1 - X[i, 1])) % 1.
        y = (X[i, 1] + r * x * (1 - x)) % 1.

    return X


# ### Generate train dataset

# In[4]:


data_train = np.zeros((N_sets_train, N_points, 2))
density_train = np.zeros((N_sets_train, size * size))

r_list = [3.4, 4.1]

for i in tqdm(range(N_sets_train), desc='Generating train dataset : '):
    # Generate the orbit
    r = np.random.choice(r_list)
    data_train[i] = create_orbit(N_points, r)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(data_train[i])
    density_train[i] = np.exp(kde.score_samples(coord))

# In[39]:
PI = gd.representations.PersistenceImage(bandwidth=6e-3,
                                         weight=lambda x: x[1]**2.5,
                                         resolution=[50, 50], im_range=[0, 0.03, 0, 0.03])


def compute_PI_train(i):
    # Compute its persistence image
    #rcX = gd.RipsComplex(points=data_train[i],max_edge_length=max_edge_length).create_simplex_tree(max_dimension=2)
    rcX = gd.AlphaComplex(points=data_train[i]).create_simplex_tree()
    dgmX = rcX.persistence()

    # PI = gd.representations.PersistenceImage(bandwidth=5e-2,
    #                                         weight=lambda x: x[1]**3,
    #                                         resolution=[PI_size, PI_size],im_range=[0, 0.6, 0, 0.6])

    pi = PI.fit_transform([rcX.persistence_intervals_in_dimension(1)])
    return pi[0]


# In[42]:


starttime = time()
pool = multiprocessing.Pool()
PI_train = pool.map(compute_PI_train, range(0, N_sets_train))
pool.close()
print('Time taken = {} seconds'.format(time() - starttime))
print("Quantile 0.995 = ", np.quantile(PI_train, 0.995))
#PI_train /= np.max(PI_train)


# ### Generate test dataset

# In[5]:


data_test = np.zeros((N_sets_test, N_points, 2))
density_test = np.zeros((N_sets_test, size * size))

for i in tqdm(range(N_sets_test // 2), desc='Generating test dataset : '):
    # Generate the orbit
    r = 3.4
    data_test[i] = create_orbit(N_points, r)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(data_test[i])
    density_test[i] = np.exp(kde.score_samples(coord))

for i in tqdm(range(N_sets_test // 2, N_sets_test), desc='Generating test dataset : '):
    # Generate the orbit
    r = 4.1
    data_test[i] = create_orbit(N_points, r)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(data_test[i])
    density_test[i] = np.exp(kde.score_samples(coord))


# In[6]:


def compute_PI_test(i):
    # Compute its persistence image
    #rcX = gd.RipsComplex(points=data_test[i],max_edge_length=max_edge_length).create_simplex_tree(max_dimension=2)
    rcX = gd.AlphaComplex(points=data_test[i]).create_simplex_tree()
    dgmX = rcX.persistence()

    pi = PI.fit_transform([rcX.persistence_intervals_in_dimension(1)])
    return pi[0]


# In[29]:


starttime = time()
pool = multiprocessing.Pool()
PI_test = pool.map(compute_PI_test, range(0, N_sets_test))
pool.close()
print('Time taken = {} seconds'.format(time() - starttime))
print("Quantile 0.995 = ", np.quantile(PI_test, 0.995))
#PI_test /= np.quantile(PI_test,0.995)


# ### Save data

# In[47]:
np.savez_compressed('PI_data_1000_dynamical_alpha_density', data_train=data_train, density_train=density_train,
                    PI_train=PI_train, data_test=data_test, density_test=density_test, PI_test=PI_test)
