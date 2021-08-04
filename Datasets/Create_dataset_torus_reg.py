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


# In[4]:


def create_torus(N_points, r, R):
    X = np.zeros((N_points, 3))
    for i in range(N_points):
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * 2 * np.pi
        X[i, 0] = (R + r * np.cos(theta)) * np.cos(phi)
        X[i, 1] = (R + r * np.cos(theta)) * np.sin(phi)
        X[i, 2] = r * np.sin(theta)
        #X[i] = [x,y,z]
    return np.array(X)


data_train = np.zeros((N_sets_train, N_points, 3))
r_train = np.zeros((N_sets_train,))
R_train = np.zeros((N_sets_train,))
#PI_train = np.zeros((N_sets_train, PI_size * PI_size))


for i in tqdm(range(N_sets_train), desc="Generating training dataset"):
    R = np.random.uniform(4, 6)
    r = np.random.uniform(R / 2, (R + 1) / 2)
    data_train[i] = create_torus(N_points, r, R)
    R_train[i] = R
    r_train[i] = r

# In[80]:
PI = gd.representations.PersistenceImage(bandwidth=4e-1,
                                         weight=lambda x: x[1]**2,
                                         resolution=[PI_size, PI_size],
                                         im_range=[0, 2.5, 0, 15])


def compute_PI(i):
    # Compute its persistence image
    rcX = gd.AlphaComplex(points=data_train[i]).create_simplex_tree()
    dgmX = rcX.persistence()

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


data_test = np.zeros((N_sets_test, N_points, 3))
r_test = np.zeros((N_sets_test,))
R_test = np.zeros((N_sets_test,))

for i in tqdm(range(N_sets_test), desc="Generating testing dataset"):
    R = np.random.uniform(4, 6)
    r = np.random.uniform(R / 2, (R + 1) / 2)
    data_test[i] = create_torus(N_points, r, R)
    r_test[i] = r
    R_test[i] = R
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


np.savez_compressed('PI_data_torus_reg', r_train=r_train, R_train=R_train, data_train=data_train,
                    PI_train=PI_train, data_test=data_test, PI_test=PI_test, r_test=r_test, R_test=R_test)
