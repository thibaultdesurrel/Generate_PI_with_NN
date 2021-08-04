#!/usr/bin/env python
# coding: utf-8

# # Create dataset circles and save it

# In[1]:


import numpy as np
from random import randint
N_noise = 50


def create_noisy_circle(N_points, r, x_0, y_0, taux):
    N_noise = 50
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


# ### Generate train dataset


def create_random_circle(N_points):
    N_noise = 50
    r = randint(1, 3)
    if r == 1:
        return create_1_circle(N_points)
    elif r == 2:
        return create_2_circle(N_points)
    elif r == 3:
        return create_3_circle(N_points)
