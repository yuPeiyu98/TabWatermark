##### adding watermark #####

import argparse
import numpy as np
import os
import os.path as osp
import random
import time

from scipy.stats import binom
from scipy.stats import norm

torch.multiprocessing.set_sharing_strategy('file_system')

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

def singleColumnWatermark(arr, green_list):
    arr_wm = arr.copy()

    for i in range(arr_wm):
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

def countElemInGreenList(arr, green_list):
    T = 0

    for e in arr:
        g = findNearestInterval(e, green_list)

        T += (e <= g[1] and e >= g[0])

    return T

def binomTesting(arr, green_list):
    T = countElemInGreenList(arr, green_list)
    p = binom.cdf(T, len(arr), 0.5)
    return 1 - p

def gaussTesting(arr, green_list):
    T = countElemInGreenList(arr, green_list)
    p = norm.cdf(T)
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

def main(args)
    # initialize random seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ### simulate data
    # n = 10, 5 wm tabs with 5 original tabs
    simulateData(n=10, m=args.m)

    # n = 100, 5 wm tabs with 5 original tabs
    simulateData(n=100, m=args.m)

    # n = 1000, 5 wm tabs with 5 original tabs
    simulateData(n=1000, m=args.m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    # parser.add_argument('--dataset', type=str, default='cifar10')
    # parser.add_argument('--data_path', type=str, default='../../noise_mixture_nce/ncebm_torch/data', help='data path')
    
    # gridding parameters
    parser.add_argument('--m', type=int, default=1000, help='')

    args = parser.parse_args()
    main(args)