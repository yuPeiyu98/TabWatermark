import argparse
import numpy as np
import random
import time

from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns


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

    p = chi2.cdf(X, len(arrs))
    return 1 - p


def simulateMultiColDataAttack(n=10, c=10, pro=0.90, sigma=1.0, m=1000, T=50, a=0.005):
    g_lists = [getGreenList(m=m) for __ in range(c)]

    sing_wm_bf, sing_wm_af, mult_wm_bf, mult_wm_af = 0, 0, 0, 0

    for __ in range(T):
        tab_lists_wm_sing = [np.random.randn(n) for __ in range(c)]
        tab_lists_wm_all = [np.random.randn(n) for __ in range(c)]

        tab_lists_wm_sing[0] = singleColumnWatermark(tab_lists_wm_sing[0], g_lists[0], m)
        sing_wm_bf += int(gaussMultTesting(tab_lists_wm_sing, g_lists, m) < a)

        ma = np.zeros_like(tab_lists_wm_sing[0])
        ma[:int(n * pro)] = 1
        tab_lists_wm_sing[0] += ma * np.random.randn(n) * sigma

        sing_wm_af += int(gaussMultTesting(tab_lists_wm_sing, g_lists, m) < a)

        for i, tab in enumerate(tab_lists_wm_all):
            tab_lists_wm_all[i] = singleColumnWatermark(tab, g_lists[i], m)

        mult_wm_bf += int(gaussMultTesting(tab_lists_wm_all, g_lists, m) < a)

        for i, tab in enumerate(tab_lists_wm_all):
            tab_lists_wm_all[i] += ma * np.random.randn(n) * sigma

        mult_wm_af += int(gaussMultTesting(tab_lists_wm_all, g_lists, m) < a)

    sing_bf = sing_wm_bf / T
    sing_af = sing_wm_af / T
    mult_bf = mult_wm_bf / T
    mult_af = mult_wm_af / T

    return sing_bf, sing_af, mult_bf, mult_af


def mean_std(x):
    x = np.array(x, dtype=float)
    return float(x.mean()), float(x.std())


def main(args):
    sns.set_theme(style="whitegrid", context="notebook")
    sns.set_context("notebook", font_scale=1.1)

    n = args.n
    p = args.p

    pro_list = [0.5, 0.75, 0.9, 0.95]
    alpha = 0.005

    logvar_list = [-3, -1, 0, 1]
    sigma_list = [10 ** (lv) for lv in logvar_list]

    T = args.T
    run_seeds = [1, 2, 3]

    sing_after = {(prp, lv): [] for prp in pro_list for lv in logvar_list}
    mult_after = {(prp, lv): [] for prp in pro_list for lv in logvar_list}

    t0 = time.time()

    print(f"\n[Fig5(c)(d)] n={n}, p={p}, m={args.m}, T={T}")
    print("logvar =", logvar_list)
    print("sigma  =", [float(s) for s in sigma_list])
    print("runs   =", run_seeds)
    print("-" * 70)

    for sd in run_seeds:
        np.random.seed(sd)
        random.seed(sd)

        print(f"\n[Run seed = {sd}]")

        for prp in pro_list:
            print(f"  prp={prp}:")

            for lv, sig in zip(logvar_list, sigma_list):
                _, sing_af, _, mult_af = simulateMultiColDataAttack(
                    n=n,
                    c=p,
                    pro=prp,
                    sigma=sig,
                    m=args.m,
                    T=T,
                    a=alpha
                )

                sing_after[(prp, lv)].append(sing_af)
                mult_after[(prp, lv)].append(mult_af)

                print(
                    f"    logvar={lv:>2}  sigma={sig:.6g}  "
                    f"sing_af={sing_af:.4f}  mult_af={mult_af:.4f}"
                )

    print("\nDone. Elapsed {:.2f}s".format(time.time() - t0))
    print("-" * 70)

    palette = sns.color_palette("deep", n_colors=len(pro_list))

    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, prp in enumerate(pro_list):
        ys, yerr = [], []

        for lv in logvar_list:
            m_, sd_ = mean_std(sing_after[(prp, lv)])
            ys.append(m_)
            yerr.append(sd_)

        ax.errorbar(
            logvar_list,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=2,
            markersize=4,
            capsize=2,
            color=palette[idx],
            label=f"prp={prp}"
        )

    ax.set_title("(c) wm. one col. (atck)")
    ax.set_xlabel("log-variance")
    ax.set_ylabel("det. rate")
    ax.set_xticks(logvar_list)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig("fig5_c_paper_setting.png", dpi=300)
    plt.close()

    print("Saved fig5_c_paper_setting.png")

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    for idx, prp in enumerate(pro_list):
        ys, yerr = [], []

        for lv in logvar_list:
            m_, sd_ = mean_std(mult_after[(prp, lv)])
            ys.append(m_)
            yerr.append(sd_)

        ax.errorbar(
            logvar_list,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=2,
            markersize=4,
            capsize=2,
            color=palette[idx],
            label=f"prp={prp}"
        )

    ax.set_title("(d) wm. all cols. (atck)")
    ax.set_xlabel("log-variance")
    ax.set_ylabel("det. rate")
    ax.set_xticks(logvar_list)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig("fig5_d_paper_setting.png", dpi=300)
    plt.close()

    print("Saved fig5_d_paper_setting.png")

    print("\nSummary (mean±std):")

    for prp in pro_list:
        print(f"  prp={prp}:")

        for lv in logvar_list:
            m_s, sd_s = mean_std(sing_after[(prp, lv)])
            m_m, sd_m = mean_std(mult_after[(prp, lv)])

            print(
                f"    logvar={lv:>2}: "
                f"sing={m_s:.4f}±{sd_s:.4f} "
                f"mult={m_m:.4f}±{sd_m:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument("--m", type=int, default=1000)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    main(args)
