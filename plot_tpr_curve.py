##### adding watermark #####

import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import random
import seaborn as sns
import time

from matplotlib import pyplot as plt
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import chi2

def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    # tok = "chi_mult_col_tpr"
    tok = "chi_sing_col_tpr"
    # tok = "chi_sing_atck_bf"
    # tok = "chi_mult_col_tpr"

    df = dict({'type': [], 'log_10n': [], 'rate': []})
    df['type'] += ["p=10"] * 3 + ["p=100"] * 3 + ["p=1000"] * 3
    # df['log10n'] += [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3] * 2
    df['log_10n'] += [1, 2, 3] * 3
    # df['rate'] += [0.984, 0.991, 0.993, 0.995, 0.991, 0.990, 0.996, 0.996, 0.993] + [1] * 9
    df['rate'] += [0.083, 1.000, 1.000, 0.013, 1.000, 1.000, 0.006, 0.331, 1.000]
    # df['rate'] += [0.997, 0.995, 0.993, 0.996, 0.994, 0.993, 0.996, 0.997, 0.995]
    # df['rate'] += [1.000] * 9

    # df['rate'] += [1.000] * 9
    # df['rate'] += [1.000] * 8 + [1.000, 0.66, 0.64, 0.68] + [0.98, 0.02, 0.12, 0.02]

    df = pd.DataFrame(df)
 
    fig = plt.figure(figsize=(8,6))
    sns.set_style("darkgrid")

    palette = sns.color_palette("mako_r", 4)
    ax = sns.lineplot(
            data=df, x="log_10n", y="rate", hue="type", 
            linewidth=2, palette=palette)
    handles, labels = ax.get_legend_handles_labels()
    x_major_locator = plt.MultipleLocator(1)
    ax.set_xlabel("log_10n", fontsize=18)
    ax.set_ylabel("rate", fontsize=18)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylim(-0.1, 1.1)
    # ax.set_ylim(0.98, 1.)

    plt.legend(handles=handles, labels=labels, fontsize=12)
    plt.tight_layout()
    plt.savefig('{}.png'.format(tok))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    # parser.add_argument('--dataset', type=str, default='cifar10')
    # parser.add_argument('--data_path', type=str, default='../../noise_mixture_nce/ncebm_torch/data', help='data path')
    
    # gridding parameters
    parser.add_argument('--m', type=int, default=1000, help='')

    args = parser.parse_args()
    main(args)
