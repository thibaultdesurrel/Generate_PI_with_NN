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


N_sets_train = 9999
N_sets_test = 999
N_points = 600
PI_size = 50
max_edge_length = 1


# In[4]:


def create_circle(N_points, r, x_0, y_0):
    X = []
    for i in range(N_points):  # On fait un cercle
        theta = np.random.uniform() * 2 * np.pi
        X.append([(r * np.cos(theta)) + x_0, (r * np.sin(theta) + y_0)])
    return np.array(X)


# In[5]:


def create_1_circle(N_points):
    r = 2
    x_0, y_0 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5

    return create_circle(N_points, r, x_0, y_0)

# In[54]:


def create_2_circle(N_points):
    r1 = 5
    r2 = 3

    #n_0 = randint(0, N_points)
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while(np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2) <= r1 + r2):
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    circle1 = create_circle(N_points // 2, r1, x_0, y_0)
    circle2 = create_circle(N_points - N_points // 2, r2, x_1, y_1)
    X = [0] * N_points
    X[:N_points // 2] = circle1
    X[N_points // 2:] = circle2
    np.random.shuffle(X)
    return np.array(X)


# In[55]:


def create_3_circle(N_points):
    #r0 = 3*np.random.random()
    #r1 = 3*np.random.random()
    #r2 = 3*np.random.random()
    r0 = 5
    r1 = 3
    r2 = 2
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while(np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2) <= r0 + r1):
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15

    x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while(np.sqrt((x_0 - x_2)**2 + (y_0 - y_2)**2) <= r0 + r2) or (np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2) <= r1 + r2):
        x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15

    circle0 = create_circle(N_points // 3, r0, x_0, y_0)
    circle1 = create_circle(N_points // 3, r1, x_1, y_1)
    circle2 = create_circle(N_points // 3, r2, x_2, y_2)

    X = [0] * N_points
    X[:N_points // 3] = circle0
    X[N_points // 3:2 * N_points // 3] = circle1
    X[2 * N_points // 3:] = circle2
    np.random.shuffle(X)
    return np.array(X)
# ### Generate train dataset

# In[78]:


data_train = np.zeros((N_sets_train, N_points, 2))
#PI_train = np.zeros((N_sets_train, PI_size * PI_size))


for i in tqdm(range(N_sets_train // 3)):
    # Generate the orbit
    data_train[i] = create_1_circle(N_points)
for i in tqdm(range(N_sets_train // 3, 2 * N_sets_train // 3)):
    # Generate the orbit
    data_train[i] = create_2_circle(N_points)
for i in tqdm(range(2 * N_sets_train // 3, N_sets_train)):
    # Generate the orbit
    data_train[i] = create_3_circle(N_points)

np.random.shuffle(data_train)
print("Train dataset generated")


# In[80]:


def compute_PI(i):
    # Compute its persistence image
    rcX = gd.AlphaComplex(points=data_train[i]).create_simplex_tree()
    dgmX = rcX.persistence()

    PI = gd.representations.PersistenceImage(bandwidth=1,
                                             weight=lambda x: 10 *
                                             np.tanh(x[1]),
                                             resolution=[PI_size, PI_size], im_range=[0, 5, 0, 25])
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

for i in tqdm(range(N_sets_test // 3)):
    # Generate the orbit
    data_test[i] = create_1_circle(N_points)
for i in tqdm(range(N_sets_test // 3, 2 * N_sets_test // 3)):
    # Generate the orbit
    data_test[i] = create_2_circle(N_points)
for i in tqdm(range(2 * N_sets_test // 3, N_sets_test)):
    # Generate the orbit
    data_test[i] = create_3_circle(N_points)

np.random.shuffle(data_test)
print("Test dataset generated")


# In[ ]:


def compute_PI_test(i):
    # Compute its persistence image
    rcX = gd.AlphaComplex(points=data_test[i]).create_simplex_tree()
    dgmX = rcX.persistence()

    PI = gd.representations.PersistenceImage(bandwidth=1,
                                             weight=lambda x: 10 *
                                             np.tanh(x[1]),
                                             resolution=[PI_size, PI_size], im_range=[0, 5, 0, 25])
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


np.savez_compressed('PI_data_multiple_circle', data_train=data_train,
                    PI_train=PI_train, data_test=data_test, PI_test=PI_test)
