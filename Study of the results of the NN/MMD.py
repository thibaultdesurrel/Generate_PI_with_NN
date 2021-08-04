import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd


def compute_mmd(X, Y, sigma):
    xx = np.matmul(X, X.transpose())
    yy = np.matmul(Y, Y.transpose())
    xy = np.matmul(X, Y.transpose())

    rx = np.broadcast_to(np.expand_dims(np.diag(xx), 0), xx.shape)
    ry = np.broadcast_to(np.expand_dims(np.diag(yy), 0), yy.shape)

    dxx = rx.transpose() + rx - 2. * xx
    dyy = ry.transpose() + rx - 2. * yy
    dxy = rx.transpose() + ry - 2. * xy
    # print(np.max(xx),np.max(xy),np.max(yy))
    XX = np.exp(dxx / (-2 * sigma**2))
    YY = np.exp(dyy / (-2 * sigma**2))
    XY = np.exp(dxy / (-2 * sigma**2))

    return (XX + YY - 2. * XY).mean()


def compute_pvalue(X, Y, n_permut, verbose):
    sigma = np.median(ssd.pdist(np.concatenate((X, Y)))) / 2
    our_mmd = compute_mmd(X, Y, sigma)
    both_distrib = np.concatenate((X, Y))
    mmds = []
    if verbose == 1:
        print("True MMD = ", our_mmd)

        for i in tqdm(range(n_permut)):
            xy = np.random.permutation(both_distrib)
            mmds.append(compute_mmd(xy[X.shape[0]:], xy[:X.shape[0]], sigma))
        mmds = np.array(mmds)
        plt.hist(mmds, bins=20)
    else:
        for i in range(n_permut):
            xy = np.random.permutation(both_distrib)
            mmds.append(compute_mmd(xy[X.shape[0]:], xy[:X.shape[0]], sigma))
        mmds = np.array(mmds)
    return (our_mmd < mmds).mean()
