import argparse
import numpy as np
import random

from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import chi2

import matplotlib.pyplot as plt


def getGreenList(lo=0, hi=1, m=1000):
    """ return a list of tuple, representing the green list """
    waymarks = np.linspace(lo, hi, m + 1)

    green_list = []
    for i in range(0, m, 2):
        # select $delta_i^{2k}$ in equ.(2) with prob.
        if np.random.uniform() > .5:
            i += 1
        green_list.append([waymarks[i], waymarks[i + 1]])

    return green_list


def findNearestIntervalHash(e, green_list, m):
    """ return the nearest interval to the given element e (local hashing) """
    min_dist, min_indx = np.inf, -1

    # offset elem to [0, 1]
    e = e - np.floor(e)

    # hashing
    idx_c = int(e // (2 / m))
    idx_l0, idx_r0 = max(0, idx_c - 1), min(idx_c + 1, len(green_list) - 1)
    idx_l1, idx_r1 = max(0, idx_c - 2), min(idx_c + 2, len(green_list) - 1)

    local_g_list = [
        green_list[idx_l1],
        green_list[idx_l0],
        green_list[idx_c],
        green_list[idx_r0],
        green_list[idx_r1],
    ]

    for i, intv in enumerate(local_g_list):
        cur_dist = np.abs(e - (intv[0] + intv[1]) / 2)
        if cur_dist < min_dist:
            min_dist = cur_dist
            min_indx = i

    return local_g_list[min_indx]


def singleColumnWatermark(arr, green_list, m):
    arr_wm = arr.copy()

    for i in range(len(arr_wm)):
        # offset elem to [0, 1]
        e_flr = np.floor(arr_wm[i])
        e = arr_wm[i] - e_flr

        g = findNearestIntervalHash(e, green_list, m)

        if e > g[1] or e < g[0]:
            # if x[i] falls outside of the range, then re-sample from uniform
            e_wm = np.random.uniform(g[0], g[1])
            arr_wm[i] = e_wm + e_flr

    return arr_wm


def countElemInGreenList(arr, green_list, m):
    T = 0
    for e in arr:
        g = findNearestIntervalHash(e, green_list, m)
        e_ofst = e - np.floor(e)
        T += (e_ofst <= g[1] and e_ofst >= g[0])
    return T


def binomTesting(arr, green_list, m):
    T = countElemInGreenList(arr, green_list, m)
    p = binom.cdf(T, len(arr), 0.5)
    return 1 - p


def gaussTesting(arr, green_list, m):
    T = countElemInGreenList(arr, green_list, m)
    mv = len(arr) / 2
    std = np.sqrt(len(arr)) / 2
    p = norm.cdf((T - mv) / std)
    return 1 - p


def gaussMultTesting(arrs, green_lists, m):
    X = 0.0
    mv = len(arrs[0]) / 2
    std = np.sqrt(len(arrs[0])) / 2

    for arr, green_list in zip(arrs, green_lists):
        T = countElemInGreenList(arr, green_list, m)
        T_c = (T - mv) / std
        X += T_c ** 2

    # p-value: 1 - CDF
    p = chi2.cdf(X, len(arrs))
    return 1 - p


# ----------------------------
# Figure 5(a)(b): point estimator
# ----------------------------
def simulateMultiColData_return(n=10, c=10, m=1000, T=1000, a=0.005, g_lists=None):
    """
    Return:
      det_one: detection rate for watermarking ONE column (col 0)
      det_all: detection rate for watermarking ALL columns
      nwm_acc: no-watermark accuracy (sanity check, should be ~1-a)
    """
    if g_lists is None:
        g_lists = [getGreenList(m=m) for _ in range(c)]

    chi_nwm, chi_sing_wm, chi_mult_wm = 0, 0, 0

    for _ in range(T):
        tab_lists_no_wm = [np.random.randn(n) for __ in range(c)]
        tab_lists_wm_sing = [np.random.randn(n) for __ in range(c)]
        tab_lists_wm_all = [np.random.randn(n) for __ in range(c)]

        # no watermark: accept null if pval > a
        chi_nwm += int(gaussMultTesting(tab_lists_no_wm, g_lists, m) > a)

        # one-col watermark: ONLY column 0
        tab_lists_wm_sing[0] = singleColumnWatermark(tab_lists_wm_sing[0], g_lists[0], m)
        chi_sing_wm += int(gaussMultTesting(tab_lists_wm_sing, g_lists, m) < a)

        # all-cols watermark
        for i, tab in enumerate(tab_lists_wm_all):
            tab_lists_wm_all[i] = singleColumnWatermark(tab, g_lists[i], m)
        chi_mult_wm += int(gaussMultTesting(tab_lists_wm_all, g_lists, m) < a)

    det_one = chi_sing_wm / T
    det_all = chi_mult_wm / T
    nwm_acc = chi_nwm / T
    return det_one, det_all, nwm_acc


def main(args):
    # Figure 5(a)(b) settings
    n_list = [10, 100, 1000]      # x-axis: log10n = 1,2,3
    p_list = [10, 100, 1000]      # legend: p (columns)
    alpha = 0.005                 # caption uses 0.005
    T = 1000                      # caption: 1000 independent samples
    runs = [1, 2, 3]              # caption: error bars over 3 runs

    # Fix keys per p across n and runs for stability (you can disable if needed)
    np.random.seed(0)
    random.seed(0)
    fixed_keys = {p: [getGreenList(m=args.m) for _ in range(p)] for p in p_list}

    # results[(panel,p,n)] -> list over runs
    results = {}
    for panel in ["one", "all", "nwm"]:
        for p in p_list:
            for n in n_list:
                results[(panel, p, n)] = []

    # run 3 times
    for seed in runs:
        np.random.seed(seed)
        random.seed(seed)

        for p in p_list:
            g_lists = fixed_keys[p]  # set to None if you want re-sampled keys per run
            for n in n_list:
                det_one, det_all, nwm_acc = simulateMultiColData_return(
                    n=n, c=p, m=args.m, T=T, a=alpha, g_lists=g_lists
                )
                results[("one", p, n)].append(det_one)
                results[("all", p, n)].append(det_all)
                results[("nwm", p, n)].append(nwm_acc)

                print(f"[seed={seed}] p={p}, n={n}: one={det_one:.4f}, all={det_all:.4f}, nwm={nwm_acc:.4f}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    def plot_panel(ax, panel_key, title):
        for p in p_list:
            xs, ys, yerr = [], [], []
            for n in n_list:
                vals = results[(panel_key, p, n)]
                xs.append(np.log10(n))
                ys.append(float(np.mean(vals)))
                yerr.append(float(np.std(vals)))
            ax.errorbar(xs, ys, yerr=yerr, marker='o', capsize=3, label=f"p={p}")
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("log_10n")
        ax.set_ylabel("det. rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend()

    plot_panel(axes[0], "one", "(a) wm. one col.")
    plot_panel(axes[1], "all", "(b) wm. all cols.")

    plt.tight_layout()
    plt.savefig("fig5_ab.png", dpi=300)
    plt.close()
    print("Saved fig5_ab.png")

    # sanity check
    print("\n(sanity) no-wm accuracy (should be ~1-alpha=0.995):")
    for p in p_list:
        for n in n_list:
            vals = results[("nwm", p, n)]
            print(f"p={p}, n={n}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed (kept for compatibility)')
    parser.add_argument('--m', type=int, default=10000, help='grid size m')
    args = parser.parse_args()
    main(args)
