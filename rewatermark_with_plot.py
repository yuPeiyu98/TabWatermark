##### adding watermark #####

import argparse
import numpy as np
import os
import os.path as osp
import random
import time

from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import chi2

from tqdm import tqdm

def getGreenList(gamma, lo=0, hi=1, m=1000):
    """ return a list of tuple, representing the green list
    """

    coarse_waymarks = np.linspace(lo, hi, m // 2 + 1)

    green_list = []
    for i in range(0, m // 2):
        # select $delta_i^{2k}$ in equ.(2) with prob. of \gamma
        if np.random.uniform() > .5:
            # pick the left part as the green list elem.
            l_ed_pt = coarse_waymarks[i]
            r_ed_pt = coarse_waymarks[i] + (coarse_waymarks[i + 1] - coarse_waymarks[i]) * gamma
        else:
            # pick the right part as the green list elem.
            l_ed_pt = coarse_waymarks[i] + (coarse_waymarks[i + 1] - coarse_waymarks[i]) * (1 - gamma)
            r_ed_pt = coarse_waymarks[i + 1]

        green_list.append([l_ed_pt, r_ed_pt])

    return green_list

""" 
    step 1: generate p green lists for a (n, p)-shaped arr.
    step 2: for each column, call `singleColumnWatermark(arr, green_list)`
"""

def singleColumnWatermark(arr, green_list):
    arr_wm = arr.copy()

    for i in range(len(arr_wm)):
        # offset elem to [0, 1]
        e_flr = np.floor(arr_wm[i])
        e = arr_wm[i] - e_flr

        g = findNearestInterval(e, green_list)

        if e > g[1] or e < g[0]:
            # if x[i] falls outside of the range, then
            # we re-sample the elem. from a uniform dist.
            arr_wm[i] = np.random.uniform(g[0], g[1]) + e_flr

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
    """ Count elements in the array falling within the green list intervals.
    """
    T = 0

    for e in arr:
        g = findNearestIntervalHash(e, green_list, m)

        e_ofst = e - np.floor(e)

        T += (e_ofst <= g[1] and e_ofst >= g[0])
        # print(g[0], g[1], e, e_ofst)

    return T

def binomTesting(arr, green_list, m, gamma):
    T = countElemInGreenList(arr, green_list, m)
    p = binom.cdf(T, len(arr), gamma)
    # print(T)
    return 1 - p

def gaussTesting(arr, green_list, m, gamma):
    T = countElemInGreenList(arr, green_list, m)

    mv = len(arr) * gamma
    std = np.sqrt(len(arr) * gamma * (1 - gamma))
    p = norm.cdf((T - mv) / std)
    # print(T)
    return 1 - p        

def gaussMultTesting(arrs, green_lists, m, gamma):
    """ The proposed method which performs hypothesis testing across multiple arrays with chi-squared distribution.
    """
    X = 0

    mv = len(arrs[0]) * gamma
    std = np.sqrt(len(arrs[0]) * gamma * (1 - gamma))

    for arr, green_list in zip(arrs, green_lists):
        T = countElemInGreenList(arr, green_list, m)
        # print(T)

        T_c = (T - mv) / std
        X += T_c ** 2

    p = chi2.cdf(X, len(arrs))
    return 1 - p

def simulateData(n=10, p=5, m=1000, gamma=0.5):
    tab_list_wm = [np.random.randn(n) for __ in range(p)]
    tab_list_nwm = [np.random.randn(n) for __ in range(p)]
    # create green list
    g_list = getGreenList(gamma=gamma, m=m)

    print("############# n = {}, p = {}, gamma = {} #############".format(n, p, gamma))
    print("--- No watermark ---")
    for tab in tab_list_nwm:
        print("Binom: {:.8f}, Gauss: {:.8f}".format(binomTesting(tab, g_list, m, gamma), gaussTesting(tab, g_list, m, gamma)))

    print("--- watermarked ---")
    for tab in tab_list_wm:
        tab_wm = singleColumnWatermark(tab, g_list)
        print("Binom: {:.8f}, Gauss: {:.8f}".format(binomTesting(tab_wm, g_list, m, gamma), gaussTesting(tab_wm, g_list, m, gamma)))

def simulateMultiColData(n=10, p=10, m=1000, gamma=0.5):
    # create green list
    g_lists = [getGreenList(gamma=gamma, m=m) for __ in range(p)]

    tab_lists_no_wm = [np.random.randn(n) for __ in range(p)]
    tab_lists_wm_sing = [np.random.randn(n) for __ in range(p)]
    tab_lists_wm_all = [np.random.randn(n) for __ in range(p)]

    print("############# n = {}, p = {}, gamma = {} #############".format(n, p, gamma))
    print("no watermark w/ chi2 testing: {:.8f}".format(gaussMultTesting(tab_lists_no_wm, g_lists, m, gamma)))

    tab_lists_wm_sing[0] = singleColumnWatermark(tab_lists_wm_sing[0], g_lists[0])
    print("Single-column watermark w/ chi2 testing: {:.8f}".format(gaussMultTesting(tab_lists_wm_sing, g_lists, m, gamma)))

    for i, tab in enumerate(tab_lists_wm_all):
        tab_lists_wm_all[i] = singleColumnWatermark(tab, g_lists[i])
    print("All-column watermark w/ chi2 testing: {:.8f}".format(gaussMultTesting(tab_lists_wm_all, g_lists, m, gamma)))

def simulateRewatermarkMultiColData(n=1000, p=10, m=1000, gamma=0.5, rem=100):
    # create green list
    g_lists = [getGreenList(gamma=gamma, m=m) for __ in range(p)]
    new_g_lists= [getGreenList(gamma=gamma, m=rem) for __ in range(p)]
    #print(g_lists[0])
    #print(new_g_lists[0])

    tab_lists_no_wm = [np.random.randn(n) for __ in range(p)]
    #tab_lists_wm_sing = [np.random.randn(n) for __ in range(p)]
    tab_lists_wm_all = [np.random.randn(n) for __ in range(p)]

    # print("############# n = {}, p = {}, gamma = {} #############".format(n, p, gamma))
    # print("no watermark w/ chi2 testing: {:.8f}".format(gaussMultTesting(tab_lists_no_wm, g_lists, m, gamma)))
    p_no_wm = gaussMultTesting(tab_lists_no_wm, g_lists, m, gamma)

    for i, tab in enumerate(tab_lists_wm_all):
        ###################################
        ###################################
        ##### with possible uniform offset
        ###################################
        ###################################
        u = np.random.uniform() 
        tab_lists_wm_all[i] = singleColumnWatermark(singleColumnWatermark(tab, g_lists[i]) + u, new_g_lists[i]) - u
    # print("All-column rewatermark w/ chi2 testing: {:.8f}".format(gaussMultTesting(tab_lists_wm_all, g_lists, m, gamma)))
    p_wm = gaussMultTesting(tab_lists_wm_all, g_lists, m, gamma)

    return p_no_wm, p_wm

from sklearn.metrics import roc_auc_score

def main(args):
    # initialize random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    ### simulate data
    # for n in [10, 100, 1000]:
    #     for gamma in [0.1, 0.3, 0.7, 0.9]:
    #         simulateData(n=n, m=args.m, gamma=gamma)

    alpha = .005
    N = 100

    # for n in [10, 100, 100]:
    #     for p in [10, 100, 1000]:
    for n in [10, 100, 1000]:
        for p in [10, 100, 1000]:
            p_no_wm_list, p_wm_list = [], []
            for __ in tqdm(range(N)):
                p_no_wm, p_wm = simulateRewatermarkMultiColData(n=n, p=p, m=args.m, rem=100, gamma=0.5)
                p_no_wm_list.append(p_no_wm)
                p_wm_list.append(p_wm)
            print("############# n = {}, p = {} #############".format(n, p))
            n_tp = (np.array(p_wm_list) < alpha).sum()
            n_tn = (np.array(p_no_wm_list) >= alpha).sum()
            print("TPR: {:.5f}, TNR: {:.5f}".format(n_tp / N, n_tn / N))
            y = np.concatenate([np.ones(N), np.zeros(N)])
            print('AUC: {:.5f}'.format(roc_auc_score(y, 1 - np.array(p_wm_list + p_no_wm_list))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    # gridding parameters
    parser.add_argument('--m', type=int, default=2000, help='')

    args = parser.parse_args()
    main(args)
