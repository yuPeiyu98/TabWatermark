##### adding watermark #####

import argparse
import numpy as np
import os
import os.path as osp
import random
import seaborn as sns
import time

from matplotlib import pyplot as plt
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import chi2


def getGreenList(lo=0, hi=1, m=1000):
    """ return a list of tuple, representing the green list
    """

    waymarks = np.linspace(lo, hi, m + 1)

    green_list = []
    for i in range(0, m, 2):
        # select $delta_i^{2k}$ in equ.(2) with prob.
        if np.random.uniform() > .5:
            i += 1
        green_list.append([waymarks[i], waymarks[i + 1]])

    return green_list

def singleColumnWatermark(arr, green_list, m):
    arr_wm = arr.copy()

    for i in range(len(arr_wm)):
        # offset elem to [0, 1]
        e_flr = np.floor(arr_wm[i])
        e = arr_wm[i] - e_flr

        g = findNearestIntervalHash(e, green_list, m)

        if e > g[1] or e < g[0]:
            # if x[i] falls outside of the range, then
            # we re-sample the elem. from a uniform dist.
            e_wm = np.random.uniform(g[0], g[1]) 
            arr_wm[i] = e_wm + e_flr

        else:
            e_wm = e

        # print("elem: {}, int: {}, {}, elem_wm: {}, ofst: {}".format(e, g[0], g[1], e_wm, e_flr))

    return arr_wm

def findNearestInterval(e, green_list):
    """ return the nearest interval to the given element e
    """

    min_dist, min_indx = np.inf, -1

    # offset elem to [0, 1]
    e = e - np.floor(e)

    for i, intv in enumerate(green_list):
        cur_dist = np.abs(e - (intv[0] + intv[1]) / 2)

        if cur_dist < min_dist:
            min_dist = cur_dist
            min_indx = i

    return green_list[min_indx]

def findNearestIntervalHash(e, green_list, m):
    """ return the nearest interval to the given element e
    """

    min_dist, min_indx = np.inf, -1

    # offset elem to [0, 1]
    e = e - np.floor(e)

    # hashing
    idx_c = int(e // (2 / m))
    idx_l0, idx_r0 = max(0, idx_c - 1), min(idx_c + 1, len(green_list) - 1)
    idx_l1, idx_r1 = max(0, idx_c - 2), min(idx_c + 2, len(green_list) - 1)

    local_g_list = [green_list[idx_l1], green_list[idx_l0], green_list[idx_c], green_list[idx_r0], green_list[idx_r1]]
    for i, intv in enumerate(local_g_list):
        cur_dist = np.abs(e - (intv[0] + intv[1]) / 2)

        if cur_dist < min_dist:
            min_dist = cur_dist
            min_indx = i

    return local_g_list[min_indx]

def countElemInGreenList(arr, green_list, m):
    T = 0

    for e in arr:
        g = findNearestIntervalHash(e, green_list, m)

        e_ofst = e - np.floor(e)

        T += (e_ofst <= g[1] and e_ofst >= g[0])
        # print(g[0], g[1], e, e_ofst)

    return T

def binomTesting(arr, green_list, m):
    T = countElemInGreenList(arr, green_list, m)
    p = binom.cdf(T, len(arr), 0.5)
    print(T)
    return 1 - p

def gaussTesting(arr, green_list, m):
    T = countElemInGreenList(arr, green_list, m)

    mv = len(arr) / 2
    std = np.sqrt(len(arr)) / 2
    p = norm.cdf((T - mv) / std)
    print(T)
    return 1 - p    

def gaussMultTesting(arrs, green_lists, m):
    X = 0

    mv = len(arrs[0]) / 2
    std = np.sqrt(len(arrs[0])) / 2

    for arr, green_list in zip(arrs, green_lists):
        T = countElemInGreenList(arr, green_list, m)
        # print(T)

        T_c = (T - mv) / std
        X += T_c ** 2

    p = chi2.cdf(X, len(arrs))
    return 1 - p

def simulateData(n=10, m=1000):
    tab_list_wm = [np.random.randn(n) for __ in range(5)]
    tab_list_nwm = [np.random.randn(n) for __ in range(5)]
    # create green list
    g_list = getGreenList(m=m)

    print("############# n = {} #############".format(n))
    for tab in tab_list_nwm:
        print("Binom: {:.8f}, Gauss: {:.8f}".format(binomTesting(tab, g_list), gaussTesting(tab, g_list)))

    for tab in tab_list_wm:
        tab_wm = singleColumnWatermark(tab, g_list)
        print("Binom: {:.8f}, Gauss: {:.8f}".format(binomTesting(tab_wm, g_list), gaussTesting(tab_wm, g_list)))

def simulateMultiColData(n=10, c=10, m=1000):
    # create green list
    g_lists = [getGreenList(m=m) for __ in range(c)]

    tab_lists_no_wm = [np.random.randn(n) for __ in range(c)]
    tab_lists_wm_sing = [np.random.randn(n) for __ in range(c)]
    tab_lists_wm_all = [np.random.randn(n) for __ in range(c)] 

    print("############# n = {}, c = {} #############".format(n, c))
    print("no watermark w/ chi2 testing: {:.8f}".format(gaussMultTesting(tab_lists_no_wm, g_lists, m)))

    tab_lists_wm_sing[0] = singleColumnWatermark(tab_lists_wm_sing[0], g_lists[0], m)
    print("Single-column watermark w/ chi2 testing: {:.8f}".format(gaussMultTesting(tab_lists_wm_sing, g_lists, m)))

    for i, tab in enumerate(tab_lists_wm_all):
        tab_lists_wm_all[i] = singleColumnWatermark(tab, g_lists[i], m)
    print("All-column watermark w/ chi2 testing: {:.8f}".format(gaussMultTesting(tab_lists_wm_all, g_lists, m)))

def CovMultiColData(n=10, c=10, m=1000):
    # create green list
    g_lists = [getGreenList(m=m) for __ in range(c)]

    # tab_lists_no_wm = np.concatenate([np.random.randn(n, 1) * 10 for __ in range(c)], axis=1)
    tab_lists_no_wm = [np.random.randn(n, 1)]
    for k in range(c - 1):
        if k % 2 == 1:
            tab_lists_no_wm.append(tab_lists_no_wm[-1] * 1.1 + np.random.randn(n, 1))
        else:
            tab_lists_no_wm.append(tab_lists_no_wm[-1] / 1.1 + np.random.randn(n, 1))
    tab_lists_no_wm = np.concatenate(tab_lists_no_wm, axis=1)

    tab_lists_wm_all = [np.random.randn(n)]
    for k in range(c - 1):
        if k % 2 == 1:
            tab_lists_wm_all.append(tab_lists_wm_all[-1] * 1.1 + np.random.randn(n))
        else:
            tab_lists_wm_all.append(tab_lists_wm_all[-1] / 1.1 + np.random.randn(n))

    print("############# n = {}, c = {}, m = {} #############".format(n, c, m))
    corr_no_wm = np.corrcoef(tab_lists_no_wm, rowvar=False)
    cov_no_wm = np.cov(tab_lists_no_wm, rowvar=False)

    for i, tab in enumerate(tab_lists_wm_all):
        tab_lists_wm_all[i] = singleColumnWatermark(tab, g_lists[i], m).reshape(n, 1)
    tab_lists_wm_all = np.concatenate(tab_lists_wm_all, axis=1)
    corr_wm_all = np.corrcoef(tab_lists_wm_all, rowvar=False)
    cov_wm_all = np.cov(tab_lists_wm_all, rowvar=False)

    diff = corr_wm_all - corr_no_wm
    U, S, V = np.linalg.svd(diff, full_matrices=True)

    mask = np.triu(np.ones_like(diff, dtype=bool))
    # ===== plot corr_diff =====
    plt.figure()
    sns.heatmap(diff, mask=mask, cmap="YlGnBu", vmin=-1.0, vmax=1.0)
    plt.axis('off')
    plt.savefig('corr_diff_n{}_c{}_m{}.png'.format(n, c, m), dpi=1200)
    plt.close()

    # ===== plot corr_wm_all =====
    plt.figure()
    sns.heatmap(corr_wm_all, mask=mask, cmap="YlGnBu", vmin=-1.0, vmax=1.0)
    plt.axis('off')
    plt.savefig('corr_wm_all_n{}_c{}_m{}.png'.format(n, c, m), dpi=1200)
    plt.close()

    # ===== plot corr_no_wm =====
    plt.figure()
    sns.heatmap(corr_no_wm, mask=mask, cmap="YlGnBu", vmin=-1.0, vmax=1.0)
    plt.axis('off')
    plt.savefig('corr_no_wm_n{}_c{}_m{}.png'.format(n, c, m), dpi=1200)
    plt.close()

    print(cov_no_wm.shape, cov_wm_all.shape, S.shape)

    mse = np.mean(diff ** 2)
    max_indx = np.unravel_index(np.abs(diff).argmax(), diff.shape)
    print("MSE: {:.8f}, max abs.: {:.8f}, ({},{}), max sing. v.: {:.8f}".format(mse, np.abs(diff).max(), max_indx[0], max_indx[1], S.max()))

def main(args):
    # initialize random seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # for c_ in [10, 100, 1000]:
    #     for m_ in [1000, 5000, 10000]:
    #         CovMultiColData(n=10000, c=c_, m=m_)
    CovMultiColData(n=10000, c=1000, m=1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    # parser.add_argument('--dataset', type=str, default='cifar10')
    # parser.add_argument('--data_path', type=str, default='../../noise_mixture_nce/ncebm_torch/data', help='data path')
    
    # gridding parameters
    parser.add_argument('--m', type=int, default=1000, help='')

    args = parser.parse_args()
    main(args)
