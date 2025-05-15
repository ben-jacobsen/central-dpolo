"""
    File: toeplitz_experiments.py
    Author: Ben Jacobsen
    Purpose: Playing around with different matrix factorizations
"""

import argparse
import os
from functools import cache
from itertools import pairwise

import matplotlib
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pandas as pd
import scipy.linalg as la
import seaborn as sns
from numpy.fft import fft, ifft
from scipy.sparse.linalg import svds

import sequences as seq
import singular_approximation as sa

matplotlib.use("qtagg")
sns.set_style("darkgrid")
sns.set_context('paper')
sns.set(rc={"lines.linewidth": 2.5, "figure.dpi": 160, "savefig.dpi": 160})
sns.set(font_scale=0.7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', default=100, type=int)
    parser.add_argument('--gammas', default=[-0.55], type=float, nargs='*')
    parser.add_argument('--deltas', default=[0], type=float, nargs='*')
    parser.add_argument('--no-cache', action="store_true")
    parser.add_argument('--tol', type=float, default=0)
    parser.add_argument('--coefficients', action="store_true")
    parser.add_argument('--variance', action="store_true")

    args = parser.parse_args()
    T = args.T
    sa.config['cache'] = not args.no_cache

    if not args.no_cache and not os.path.isdir('seq_cache'):
        os.mkdir('seq_cache')

    # delta = float(sa.optimize_sensitivity(-1 / 2, -1 / 2 - s))

    strategies = []
    strategies.append(seq.Hybrid(
        at= seq.Anytime(-1/2, -0.55, delta=0), 
        w=0.25, init_chunk=2, exponential=False))
    strategies.append(seq.Hybrid(
        at= seq.Independent(), 
        w=0.25, init_chunk=2, exponential=False))
    strategies.append(seq.Anytime(-1/2, -0.51, delta=0,
                                  tol=1e-4, asym_order=6))
    for gamma in args.gammas:
        for delta in args.deltas:
            strategies.append(
                seq.Anytime(-1 / 2,
                            gamma,
                            delta=delta,
                            tol=args.tol,
                            asym_order=4))
    # strategies.append(seq.BinaryMechanism(T))
    # strategies.append(seq.Hybrid(
    #     at= seq.Anytime(-1/2, -0.55, delta=0),
    #     bounded = seq.SmoothBinary(),
    #     w=0.25, init_chunk=2, exponential=False))
    strategies.append(seq.SmoothBinary(T))
    # strategies.append(seq.DoublingTrick(100, T, exponential=True))

    # Opt always goes last
    strategies.append(seq.Opt())


    num_plots = args.variance + args.coefficients

    fig = plt.figure(layout='constrained')
    orig_axs = fig.subplot_mosaic([[str(j) for j in range(num_plots)] + ["legend"]],
                                  width_ratios = [1.35] * num_plots + [1])
    orig_axs['legend'].axis("off")

    kwarg_list = []
    for strategy, color in zip(strategies, sns.color_palette("Dark2", len(strategies)-1) + ['black']):
        if isinstance(strategy, seq.Anytime):
            ls = '-'
            lw = 1.8
            alpha = 1
        else:
            ls = '-' # '--'
            lw = 1.3
            alpha = 1

        kwarg_list.append( {'label': strategy.name, 'color': color, 'ls': ls, 'lw': lw,
                            'alpha': alpha})

    i = 0
    if args.variance:
        variance_plot(T, strategies, kwarg_list, orig_axs[str(i)])
        i += 1
    if args.coefficients:
        coefficient_plot(T, strategies, kwarg_list, orig_axs[str(i)])


    # handles, labels = axs[0, 1].get_legend_handles_labels()
    # for ax in axs.reshape(-1):
    #     ax.get_legend().remove()
    # fig.legend(handles, labels, loc='outside right upper')
    # plt.legend()
    # fig.set_size_inches(w=3.25, h=2.5)
    h, l = orig_axs[str(0)].get_legend_handles_labels()
    k = int(np.log2(T))
    for i in range(num_plots):
        orig_axs[str(i)].set_xticks([2**j for j in range(k-4*(k//4), k+1, k//4)])
        try:
            orig_axs[str(i)].get_legend().remove()
        except AttributeError:
            pass
    orig_axs["legend"].legend(h, l, ncol=1, loc='center left', frameon=False, 
                              fontsize=8, borderpad=0.1, handlelength=1.5,
                              handletextpad=0.4,
                              borderaxespad=0)
    bbox_px = orig_axs["legend"].get_legend().get_window_extent()
    dpi = fig.dpi
    # legend_height = max(bbox_px.height, 360)
    # legend_width = bbox_px.width
    # fig.set_size_inches(w = (num_plots * legend_height + legend_width + 50 ) / dpi, 
    #                     h= legend_height/dpi)
    fig.set_size_inches(w=6, h=2)
    plt.show()



@cache
def plotting_points(T):
    steps = np.arange(1, T + 1)
    scale = 6
    view = np.arange(min(T, 2**scale))
    i = 1
    while T - 2**i > view[-1]:
        next_chunk = np.arange(view[-1] + 2**i, min(T, 2**(i+scale)), 2**i)
        view = np.concatenate((view, next_chunk), axis=None)
        i += 1
    return steps, view



def coefficient_plot(T, strategies, kwargs, ax, relative=False):
    ax.set_xscale('log', base=2)
    ax.set_xlabel("t", fontsize=8, labelpad=0)
    steps, view = plotting_points(T)


    for strategy, kw in zip(strategies, kwargs):
        if isinstance(strategy, seq.Sequence):
            l = strategy.first_k_left(T)
            r = strategy.first_k(T)

            if relative:
                base = seq.Opt().first_k(T)
                l /= base
                r /= base
                ax.set_yscale('linear')
                # ax.set_title('Ratio')
                ax.set_ylabel("Coefficient Ratio", fontsize=8, labelpad=0)
            else:
                ax.set_yscale('log', base=2)
                # ax.set_title('Matrix Coefficients')
                ax.set_ylabel("Matrix Coefficients", fontsize=8, labelpad=0)


            sns.lineplot(x=steps[view], y=r[view], ax=ax, **kw)
            sns.lineplot(x=steps[view],
                         y=l[view],
                         ax=ax,
                         # label=strategy.name + " (L)",
                         # ls='-.',
                         color=kw['color'])


def variance_plot(T, strategies, kwargs, ax):
    ax.set_xscale('log', base=2)
    ax.set_xlabel("t", fontsize=8, labelpad=0)
    ax.set_ylabel("Variance", fontsize=8, labelpad=0)
    # ax.set_title('Mechanism Variance')
    # ax.set_title('Variance')
    steps, view = plotting_points(T)

    for strategy, kw in zip(strategies, kwargs):
        sns.lineplot(x=steps[view], y = (strategy.noise_schedule(T)**2)[view],
                     ax=ax, **kw)
















@cache
def opt(k, alpha=1 / 2):
    k = int(k)
    if k < 1:
        return 0
    if k == 1:
        return 1
    return (1 - alpha / (k - 1)) * opt(k - 1, alpha)


def fourier(k):
    v = 1 / (2 * np.sqrt(k) * np.log(k + np.e - 1)) - np.sqrt(k) / (
        (k + np.e - 1) * np.power(np.log(k + np.e - 1), 2))
    return v / v[0]


def fourier2(k):
    k = np.append(k, k[-1] + 1)
    vec = np.sqrt(k) / np.log(k + np.e - 1)
    d = np.diff(vec)
    return d / d[0]


def back_to_front(k):
    T = len(k)
    vec = np.sqrt(2 * np.log(k + np.e - 1) / k)
    L_inv = la.solve_toeplitz((vec, np.zeros_like(vec)), np.eye(T))
    return np.cumsum(L_inv[:, 0])


def back_to_front2(k):
    T = len(k)
    vec = 1 / (2 * np.power(k + 1, 3 / 2)) * (1 / np.sqrt(np.log(k + 1)) -
                                              np.sqrt(np.log(k + 1)))
    vec /= vec[0]
    R = la.solve_toeplitz((vec, np.zeros_like(vec)), np.eye(T))
    print(k)
    print(vec)
    print(R)
    return R[:, 0]


@cache
def geom_test(k):
    if k == 1:
        return 1
    return (1 - 1 / k) * geom_test(k - 1)


def error_estimation(mat):
    _, s, _ = svds(mat, k=1, which='SM')
    return 1 / s[0]


def fast_inv_ltt(a):
    n = len(a)
    eps = 10**(-5 / n)
    vec = np.power(eps, np.arange(n))
    return (ifft(1 / fft(a * vec)) / vec).real


if __name__ == "__main__":
    main()
