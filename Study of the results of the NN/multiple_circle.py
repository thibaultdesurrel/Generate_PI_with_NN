#!/usr/bin/env python
# coding: utf-8

# # Create dataset circles and save it

# In[1]:


import numpy as np
from random import randint


def create_circle(N_points, r, x_0, y_0):
    X = []
    for i in range(N_points):  # On fait un cercle
        theta = np.random.uniform() * 2 * np.pi
        X.append([(r * np.cos(theta)) + x_0, (r * np.sin(theta) + y_0)])
    return np.array(X)


# In[5]:

def create_1_circle(N_points):
    r = 5 * np.random.random()
    x_0, y_0 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5

    return create_circle(N_points, r, x_0, y_0)

# In[54]:


def create_2_circle(N_points):
    r1 = 5 * np.random.random()
    r2 = 5 * np.random.random()

    #n_0 = randint(0, N_points)
    x_0, y_0 = 20 * np.random.rand() - 10, 20 * np.random.rand() - 10
    x_1, y_1 = 20 * np.random.rand() - 10, 20 * np.random.rand() - 10
    while(np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2) <= r1 + r2):
        x_1, y_1 = 20 * np.random.rand() - 10, 20 * np.random.rand() - 10
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

 # Generate train dataset


def create_random_circle(N_points):
    r = randint(1, 3)
    if r == 1:
        return create_1_circle(N_points)
    elif r == 2:
        return create_2_circle(N_points)
    elif r == 3:
        return create_3_circle(N_points)
