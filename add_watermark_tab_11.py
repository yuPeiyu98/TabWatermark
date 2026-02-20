import argparse
import numpy as np
import random

from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import chi2

from sklearn.metrics import roc_auc_score


def getGreenList(lo=0, hi=1, m=1000):
    waymarks = np.linspace(lo, hi, m + 1)

    green_list = []
    for i in range(0, m, 2):
        if np.random.uniform() > 0.5:
            i += 1
        green_list.append([waymarks[i], waymarks[i + 1]])

    return green_list


def findNearestIntervalHash(e, green_list, m):
    min_dist, min_indx = np.inf, -1
    e = e - np.floor(e)

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
        e_flr = np.floor(arr_wm[i])
        e = arr_wm[i] - e_flr

        g = findNearestIntervalHash(e, green_list, m)

        if e > g[1] or e < g[0]:
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

    p = chi2.cdf(X, len(arrs))
    return 1 - p  # p-value style: smaller => more watermarked


def simulateMultiColData_return_scores(n=10, c=10, m=1000, T=30, g_lists=None):
    """
    Return arrays of detector p-values for:
      - watermark one column
      - watermark all columns
      - no watermark
    """
    if g_lists is None:
        g_lists = [getGreenList(m=m) for _ in range(c)]

    one_ps, all_ps, nwm_ps = [], [], []

    for _ in range(T):
        tab_lists_no_wm = [np.random.randn(n) for __ in range(c)]
        tab_lists_wm_sing = [np.random.randn(n) for __ in range(c)]
        tab_lists_wm_all = [np.random.randn(n) for __ in range(c)]

        nwm_ps.append(gaussMultTesting(tab_lists_no_wm, g_lists, m))

        tab_lists_wm_sing[0] = singleColumnWatermark(tab_lists_wm_sing[0], g_lists[0], m)
        one_ps.append(gaussMultTesting(tab_lists_wm_sing, g_lists, m))

        for i, tab in enumerate(tab_lists_wm_all):
            tab_lists_wm_all[i] = singleColumnWatermark(tab, g_lists[i], m)
        all_ps.append(gaussMultTesting(tab_lists_wm_all, g_lists, m))

    return np.array(one_ps), np.array(all_ps), np.array(nwm_ps)


# ---------------------------
# ONLY METRIC CALC CHANGED
# ---------------------------
def calc_tpr_tnr_auc(wm_pvals, nwm_pvals, alpha=0.005):
    """
    Metrics for Table 11:
      - AUC: use score = -pval (smaller p => more watermarked)
      - TPR/TNR: fixed operating point at p <= alpha
    """
    wm_pvals = np.asarray(wm_pvals)
    nwm_pvals = np.asarray(nwm_pvals)

    # ROC-AUC: higher score => more positive; p-values invert that, so use -p.
    y = np.concatenate([np.ones_like(wm_pvals), np.zeros_like(nwm_pvals)])
    s = np.concatenate([-wm_pvals, -nwm_pvals])
    auc = float(roc_auc_score(y, s))

    # Fixed p-value threshold
    tpr = float(np.mean(wm_pvals <= alpha))
    tnr = float(np.mean(nwm_pvals > alpha))
    return tpr, tnr, auc


def main(args):
    # Table 11 layout in your screenshot: n fixed to 100, p in {100, 1000, 10000}
    n = 100
    p_list = [100, 1000, 10000]

    # Keep your existing simulation settings (you said no need to run; this just defines defaults)
    T = 100
    runs = [1, 2, 3]

    np.random.seed(0)
    random.seed(0)
    fixed_keys = {p: [getGreenList(m=args.m) for _ in range(p)] for p in p_list}

    # Collect cells for the printed table
    row_tpr_tnr = []
    row_auc = []

    for p in p_list:
        g_lists = fixed_keys[p]

        all_all, nwm_all = [], []
        for seed in runs:
            np.random.seed(seed)
            random.seed(seed)

            _, all_p, nwm_p = simulateMultiColData_return_scores(
                n=n, c=p, m=args.m, T=T, g_lists=g_lists
            )
            all_all.append(all_p)
            nwm_all.append(nwm_p)

        all_p = np.concatenate(all_all)
        nwm_p = np.concatenate(nwm_all)

        tpr, tnr, auc = calc_tpr_tnr_auc(all_p, nwm_p, alpha=args.alpha)
        row_tpr_tnr.append(f"{tpr:.3f}/{tnr:.3f}")
        row_auc.append(f"{auc:.3f}")

    # Print in the same style as the paper table
    print("Table 11: Detection results on simulated high dimensional tables")
    print(f"{'n×p':>6}  {'100×100':>10}  {'100×1000':>10}  {'100×10000':>11}")
    print(f"{'TPR/TNR':>6}  {row_tpr_tnr[0]:>10}  {row_tpr_tnr[1]:>10}  {row_tpr_tnr[2]:>11}")
    print(f"{'AUC':>6}  {row_auc[0]:>10}  {row_auc[1]:>10}  {row_auc[2]:>11}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed (kept for compatibility)")
    parser.add_argument("--m", type=int, default=1000, help="grid size m")
    parser.add_argument("--alpha", type=float, default=0.005, help="p-value threshold for TPR/TNR")
    args = parser.parse_args()
    main(args)
