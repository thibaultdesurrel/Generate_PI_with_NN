#!/usr/bin/env python
# coding: utf-8

# # Create dataset circles and save it

# In[1]:


import numpy as np

from random import randint

import gudhi as gd
import gudhi.representations


from tqdm import tqdm
import multiprocessing
from time import time


# ### Define the constants

# In[81]:


N_sets_train = 2000
N_sets_test = 200
N_points = 500
PI_size = 50
m = 0.1
p = 2
dimension_max = 2
filtration_max = 0.7

# In[4]:


def create_circle(N_points, r, x_0, y_0, eps):
    X = []
    for i in range(N_points):  # On fait un cercle
        theta = np.random.uniform() * 2 * np.pi
        x = r * np.cos(theta) + x_0 + eps * (np.random.random() - 1 / 2)
        y = r * np.sin(theta) + y_0 + eps * (np.random.random() - 1 / 2)
        X.append([x, y])
    return np.array(X)


data_train = np.zeros((N_sets_train, N_points, 2))
r_train = np.zeros((N_sets_train,))
#PI_train = np.zeros((N_sets_train, PI_size * PI_size))


for i in tqdm(range(N_sets_train), desc="Generating training dataset"):
    eps = 0.5 * np.random.random()
    r = np.random.random()
    data_train[i] = create_circle(N_points, r, 0, 0, eps)
    r_train[i] = r


# In[80]:


def compute_PI(i):
    # Compute its persistence image
    rcX = gd.AlphaComplex(points=data_train[i]).create_simplex_tree()
    dgmX = rcX.persistence()

    PI = gd.representations.PersistenceImage(bandwidth=7e-2,
                                             weight=lambda x: x[1]**2,
                                             resolution=[50, 50], im_range=[0, 0.6, 0, 0.6])
    pi = PI.fit_transform([rcX.persistence_intervals_in_dimension(1)])
    return pi[0]


# In[200]:


starttime = time()
pool = multiprocessing.Pool()
PI_train = pool.map(compute_PI, range(0, N_sets_train))
pool.close()
print('Time taken = {} seconds'.format(time() - starttime))

# ### Generate test dataset

# In[201]:


data_test = np.zeros((N_sets_test, N_points, 2))
r_test = np.zeros((N_sets_test,))

for i in tqdm(range(N_sets_test), desc="Generating testing dataset"):
    eps = 0.5 * np.random.random()
    r = np.random.random()
    data_test[i] = create_circle(N_points, r, 0, 0, eps)
    r_test[i] = r_test
# In[ ]:


def compute_PI_test(i):
    # Compute its persistence image
    rcX = gd.AlphaComplex(points=data_test[i]).create_simplex_tree()
    dgmX = rcX.persistence()

    PI = gd.representations.PersistenceImage(bandwidth=7e-2,
                                             weight=lambda x: x[1]**2,
                                             resolution=[50, 50], im_range=[0, 0.6, 0, 0.6])
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


np.savez_compressed('PI_data_noisy_circle_reg', r_train=r_train, data_train=data_train,
                    PI_train=PI_train, data_test=data_test, PI_test=PI_test, r_test=r_test)
