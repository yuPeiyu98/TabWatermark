import argparse
import numpy as np
import random
import time

from scipy.stats import chi2, rankdata


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


def gaussMultTesting(arrs, green_lists, m):
    X = 0
    mv = len(arrs[0]) / 2
    std = np.sqrt(len(arrs[0])) / 2

    for arr, green_list in zip(arrs, green_lists):
        T = countElemInGreenList(arr, green_list, m)
        T_c = (T - mv) / std
        X += T_c ** 2

    # one-sided p-value
    return float(chi2.sf(X, len(arrs)))


def roc_auc_from_scores(y_true, scores):
    """
    AUC via rank statistic (equiv. to Mann–Whitney U).
    y_true: 1 for positive (watermarked), 0 for negative (clean)
    scores: higher means more likely positive
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    pos = (y_true == 1)
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(scores, method="average")  # 1..N
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def simulateMultiColDataAttack_AUC(n=10, c=10, pro=0.90, sigma=1.0, m=1000, T=50):
    """
    Computes ROC-AUC BEFORE and AFTER attack for:
      - single-column watermark (only col 0 watermarked)
      - multi-column watermark (all columns watermarked)
    Scores are -pval from gaussMultTesting (smaller pval => more watermarked).
    """
    g_lists = [getGreenList(m=m) for __ in range(c)]

    # collect scores for ROC-AUC:
    # y=1: watermarked, y=0: clean
    y = []
    sing_scores_bf, sing_scores_af = [], []
    mult_scores_bf, mult_scores_af = [], []

    N = n  # mask is per-column in your original snippet
    ma = np.zeros(N, dtype=float)
    ma[: int(n * pro)] = 1.0

    for _ in range(T):
        # ---------- SINGLE-COLUMN watermark experiment ----------
        # negative (clean) table
        tab_clean = [np.random.randn(n) for __ in range(c)]
        # positive (wm on col 0) table
        tab_wm = [np.random.randn(n) for __ in range(c)]
        tab_wm[0] = singleColumnWatermark(tab_wm[0], g_lists[0], m)

        # before attack scores
        p_clean_bf = gaussMultTesting(tab_clean, g_lists, m)
        p_wm_bf = gaussMultTesting(tab_wm, g_lists, m)

        sing_scores_bf += [-p_clean_bf, -p_wm_bf]
        y += [0, 1]

        # after attack: attack ONLY column 0 (as in your snippet)
        tab_clean_af = [arr.copy() for arr in tab_clean]
        tab_wm_af = [arr.copy() for arr in tab_wm]
        tab_clean_af[0] += ma * np.random.randn(n) * sigma   # sigma is STD = s
        tab_wm_af[0] += ma * np.random.randn(n) * sigma

        p_clean_af = gaussMultTesting(tab_clean_af, g_lists, m)
        p_wm_af = gaussMultTesting(tab_wm_af, g_lists, m)

        sing_scores_af += [-p_clean_af, -p_wm_af]

        # ---------- MULTI-COLUMN watermark experiment (kept for completeness) ----------
        # tab_clean2 = [np.random.randn(n) for __ in range(c)]
        # tab_wm_all = [np.random.randn(n) for __ in range(c)]
        # for i in range(c):
        #     tab_wm_all[i] = singleColumnWatermark(tab_wm_all[i], g_lists[i], m)

        # p_clean2_bf = gaussMultTesting(tab_clean2, g_lists, m)
        # p_wm_all_bf = gaussMultTesting(tab_wm_all, g_lists, m)

        # mult_scores_bf += [-p_clean2_bf, -p_wm_all_bf]

        # tab_clean2_af = [arr.copy() for arr in tab_clean2]
        # tab_wm_all_af = [arr.copy() for arr in tab_wm_all]
        # for i in range(c):
        #     tab_clean2_af[i] += ma * np.random.randn(n) * sigma
        #     tab_wm_all_af[i] += ma * np.random.randn(n) * sigma

        # p_clean2_af = gaussMultTesting(tab_clean2_af, g_lists, m)
        # p_wm_all_af = gaussMultTesting(tab_wm_all_af, g_lists, m)

        # mult_scores_af += [-p_clean2_af, -p_wm_all_af]

    # AUCs
    # sing_auc_bf = roc_auc_from_scores(y, sing_scores_bf)
    sing_auc_af = roc_auc_from_scores(y, sing_scores_af)
    # mult_auc_bf = roc_auc_from_scores(y, mult_scores_bf)
    # mult_auc_af = roc_auc_from_scores(y, mult_scores_af)

    # return sing_auc_bf, sing_auc_af, mult_auc_bf, mult_auc_af
    return None, sing_auc_af, None, None


def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std())


def main(args):
    n = args.n          # 5000
    c = args.p          # 100 columns

    pro_list = [0.50, 0.75, 0.90, 0.95]   # proportion modified
    s_list = [0.001, 0.1, 1.0, 10.0]      # s is STD

    T = args.T
    run_seeds = [1, 2, 3]

    # store single-col AFTER-attack AUCs (the Table 17 target)
    sing_after_auc = {(pr, s): [] for pr in pro_list for s in s_list}

    t0 = time.time()
    print(f"\n[Table17] single-column WM, ROC-AUC after attack")
    print(f"n={n}, c(p)={c}, m={args.m}, T={T}")
    print("-" * 70)

    for sd in run_seeds:
        np.random.seed(sd)
        random.seed(sd)

        for pr in pro_list:
            for s in s_list:
                _, sing_af_auc, _, _ = simulateMultiColDataAttack_AUC(
                    n=n, c=c, pro=pr, sigma=s, m=args.m, T=T
                )
                sing_after_auc[(pr, s)].append(sing_af_auc)

    print("Elapsed {:.2f}s".format(time.time() - t0))
    print("-" * 70)

    # Print table: rows p (pro), cols s (STD)
    colw = 10
    header = ["p \\ s"] + [f"{s:g}" for s in s_list]
    print("\n" + "".join(h.rjust(colw) for h in header))

    for pr in pro_list:
        row = [f"{pr:.2f}"]
        for s in s_list:
            m_, _ = mean_std(sing_after_auc[(pr, s)])
            row.append(f"{m_:.3f}")
        print("".join(x.rjust(colw) for x in row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--p", type=int, default=100)   # number of columns
    parser.add_argument("--m", type=int, default=1000)
    parser.add_argument("--T", type=int, default=30)
    args = parser.parse_args()
    main(args)
