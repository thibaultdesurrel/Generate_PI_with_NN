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

import velour2


# ### Define the constants

# In[81]:


N_sets_train = 999
N_sets_test = 99
N_points = 600
PI_size = 50
max_edge_length = 1
N_noise = 50
m = 0.1
p = 2
dimension_max = 2
filtration_max = 1


# In[4]:


def create_noisy_circle(N_points, r, x_0, y_0, taux):
    X = []
    for i in range(N_points):  # On fait un cercle
        # if np.random.random() < taux:
        # X.append([(2 * np.random.random() - 1) * r + x_0,
        #          (2 * np.random.random() - 1) * r + y_0])

        # else:
        theta = np.random.uniform() * 2 * np.pi
        X.append([(r * np.cos(theta)) + x_0, (r * np.sin(theta) + y_0)])
    return np.array(X)


# In[5]:


def create_1_circle(N_points):
    r = np.random.random()
    x_0, y_0 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    X = create_noisy_circle(N_points, r, x_0, y_0, 0.1)
    for i in range(N_noise):
        j = randint(0, N_points - 1)
        X[j] = [r * (2 * np.random.rand() - 1) + x_0,
                r * (2 * np.random.rand() - 1) + y_0]
    return np.array(X)
# In[54]:


def create_2_circle(N_points):
    r1 = 1
    r2 = 0.5

    #n_0 = randint(0,N_points)
    n_0 = N_points // 2
    x_0, y_0 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    x_1, y_1 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    while(np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2) <= r1 + r2):
        x_1, y_1 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    circle1 = create_noisy_circle(n_0, r1, x_0, y_0, 0.2)
    circle2 = create_noisy_circle(N_points - n_0, r2, x_1, y_1, 0.1)
    X = [0] * N_points
    X[:n_0] = circle1
    X[n_0:] = circle2
    np.random.shuffle(X)
    for i in range(N_noise):
        j = randint(0, N_points - 1)
        X[j] = [np.random.uniform(min(x_0 - r1, x_1 - r2), max(x_0 + r1, x_1 + r2)),
                np.random.uniform(min(y_0 - r1, y_1 - r2), max(y_0 + r1, y_1 + r2))]

    return np.array(X)


# In[55]:


def create_3_circle(N_points):
    #r0 = 3*np.random.random()
    #r1 = 3*np.random.random()
    #r2 = 3*np.random.random()
    r0 = 1
    r1 = 0.75
    r2 = 0.5
    x_0, y_0 = 15 * np.random.rand() - 7.5, 14 * np.random.rand() - 7.5
    x_1, y_1 = 15 * np.random.rand() - 7.5, 14 * np.random.rand() - 7.5
    while(np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2) <= r0 + r1):
        x_1, y_1 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5

    x_2, y_2 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    while(np.sqrt((x_0 - x_2)**2 + (y_0 - y_2)**2) <= r0 + r2) or (np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2) <= r1 + r2):
        x_2, y_2 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5

    circle0 = create_noisy_circle(N_points // 3, r0, x_0, y_0, 0.1)
    circle1 = create_noisy_circle(N_points // 3, r1, x_1, y_1, 0.1)
    circle2 = create_noisy_circle(N_points // 3, r2, x_2, y_2, 0.1)
    X = [0] * N_points
    X[:N_points // 3] = circle0
    X[N_points // 3:2 * N_points // 3] = circle1
    X[2 * N_points // 3:] = circle2
    np.random.shuffle(X)

    for i in range(N_noise):
        j = randint(0, N_points - 1)
        X[j] = [np.random.uniform(np.min([x_0 - r0, x_1 - r1, x_2 - r2]), np.max([x_0 + r0, x_1 + r1, x_2 + r2])),
                np.random.uniform(np.min([y_0 - r0, y_1 - r1, y_2 - r2]), np.max([y_0 + r0, y_1 + r1, y_2 + r2]))]

    return np.array(X)

# In[78]:


data_train = np.zeros((N_sets_train, N_points, 2))
#PI_train = np.zeros((N_sets_train, PI_size * PI_size))
label_train = np.zeros((N_sets_train,))

for i in tqdm(range(N_sets_train // 3)):
    # Generate the orbit
    data_train[i] = create_1_circle(N_points)
    label_train[i] = 1
for i in tqdm(range(N_sets_train // 3, 2 * N_sets_train // 3)):
    # Generate the orbit
    data_train[i] = create_2_circle(N_points)
    label_train[i] = 2
for i in tqdm(range(2 * N_sets_train // 3, N_sets_train)):
    # Generate the orbit
    data_train[i] = create_3_circle(N_points)
    label_train[i] = 3


shuffler = np.random.permutation(len(data_train))
label_train = label_train[shuffler]
data_train = data_train[shuffler]
print("Train dataset generated")


# In[80]:


def compute_PI(i):
    # Compute its persistence image
    st_DTM = velour2.DTMFiltration(
        data_train[i], m, p, dimension_max=dimension_max, filtration_max=filtration_max)

    dgmX = st_DTM.persistence()

    PI = gd.representations.PersistenceImage(bandwidth=4e-2,
                                             weight=lambda x: 10 *
                                             np.tanh(x[1]),
                                             resolution=[PI_size, PI_size], im_range=[0, 1, 0, 0.5])
    pi = PI.fit_transform([st_DTM.persistence_intervals_in_dimension(1)])
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
label_test = np.zeros((N_sets_train,))

for i in tqdm(range(N_sets_test // 3)):
    # Generate the orbit
    data_test[i] = create_1_circle(N_points)
    label_test[i] = 1
for i in tqdm(range(N_sets_test // 3, 2 * N_sets_test // 3)):
    # Generate the orbit
    data_test[i] = create_2_circle(N_points)
    label_test[i] = 2
for i in tqdm(range(2 * N_sets_test // 3, N_sets_test)):
    # Generate the orbit
    data_test[i] = create_3_circle(N_points)
    label_test[i] = 3

shuffler = np.random.permutation(len(data_test))
label_test = label_test[shuffler]
data_test = data_test[shuffler]
print("Test dataset generated")


# In[ ]:


def compute_PI_test(i):
    # Compute its persistence image
    st_DTM = velour2.DTMFiltration(
        data_test[i], m, p, dimension_max=dimension_max, filtration_max=filtration_max)

    dgmX = st_DTM.persistence()

    PI = gd.representations.PersistenceImage(bandwidth=4e-2,
                                             weight=lambda x: 10 *
                                             np.tanh(x[1]),
                                             resolution=[PI_size, PI_size], im_range=[0, 1, 0, 0.5])
    pi = PI.fit_transform([st_DTM.persistence_intervals_in_dimension(1)])
    return pi[0]


# In[ ]:


starttime = time()
pool = multiprocessing.Pool()
PI_test = pool.map(compute_PI_test, range(0, N_sets_test))
pool.close()
print('Time taken = {} seconds'.format(time() - starttime))


# ### Save data

# In[204]:


np.savez_compressed('PI_data_multiple_noisy_circle_classif', data_train=data_train, label_train=label_train,
                    PI_train=PI_train, label_test=label_test, PI_test=PI_test, data_test=data_test)
