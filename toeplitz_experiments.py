"""
    File: toeplitz_experiments.py
    Author: Ben Jacobsen
    Purpose: Playing around with different matrix factorizations
"""

import argparse
import os
from functools import cache
from itertools import pairwise

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

sns.set_theme()
sns.set_context('talk')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', default=100, type=int)
    parser.add_argument('--gammas', default=[-0.55], type=float, nargs='+')
    parser.add_argument('--deltas', default=[0], type=float, nargs='+')
    parser.add_argument('--no-cache', action="store_true")
    parser.add_argument('--tol', type=float, default=1e-4)
    args = parser.parse_args()
    T = args.T
    sa.config['cache'] = not args.no_cache

    if not args.no_cache and not os.path.isdir('seq_cache'):
        os.mkdir('seq_cache')

    # delta = float(sa.optimize_sensitivity(-1 / 2, -1 / 2 - s))

    strategies = []

    for gamma in args.gammas:
        for delta in args.deltas:
            strategies.append(
                seq.Anytime(-1 / 2,
                            gamma,
                            delta=delta,
                            tol=args.tol,
                            asym_order=4))

    strategies.append(seq.Opt())
    strategies.append(seq.DoublingTrick(100, T, exponential=True))
    # strategies.append(seq.Hybrid(w=seq.Hybrid.optimize_weight(T/100), init_chunk=100, exponential=False))
    strategies.append(seq.Hybrid(w=0.5, init_chunk=100, exponential=True))

    fig, axs = plt.subplots(2, 2, layout='constrained')
    axs[0, 0].set_title("Sensitivity")
    axs[1, 0].set_title("Standard Error")
    axs[0, 1].set_title("Est. Real Variance")
    axs[1, 1].set_title("Coefficients")
    axs[1, 1].set_yscale('log')
    for ax in axs.reshape(-1):
        ax.set_xscale('log')
        # ax.set_yscale('linear')
    axs[0, 0].sharey(axs[1, 0])
    # axs[0, 1].sharey(axs[1, 0])

    steps = np.arange(1, T + 1)
    scale = 6
    view = np.arange(min(T, 2**6))
    i = 1
    while T - 2**i > view[-1]:
        next_chunk = np.arange(view[-1] + 2**i, min(T, 2**(i+scale)), 2**i)
        view = np.concatenate((view, next_chunk), axis=None)
        i += 1

    # view = view[:T]

    for strategy, color in zip(strategies, sns.color_palette()):
        print('-' * 80)
        print(strategy.name)
        print('-' * 80)
        if isinstance(strategy, seq.Opt):
            color = 'black'
            ls = ':'
        else:
            ls = '-'

        kwargs = {'label': strategy.name, 'color': color, 'ls': ls}

        if isinstance(strategy, seq.Sequence):
            sensitivity = strategy.sensitivity()
            l = strategy.first_k_left(T)
            r = strategy.first_k(T)
            # mid = (l[0] + r[0]) / 2
            print(f"Sensitivity: {sensitivity}")
            print(l[view])
            print(r[view])
            se = strategy.standard_error(T)
            print(se[-1])

            sns.lineplot(x=steps[view],
                         y=strategy.smooth_sensitivity()[view],
                         ax=axs[0, 0],
                         **kwargs)
            sns.lineplot(x=steps[view], y=se[view], ax=axs[1, 0], **kwargs)

            sns.lineplot(x=steps[view], y=r[view], ax=axs[1, 1], **kwargs)
            # sns.lineplot(x=steps[view],
            #              y=l[view],
            #              ax=axs[1, 1],
            #              label=strategy.name + " (L)",
            #              ls=':',
            #              color=color)

        print(strategy.noise_schedule(T)[view])
        sns.lineplot(
            x=steps[view],
            y=(strategy.noise_schedule(T) / seq.Opt().noise_schedule(T))
            [view],  # / np.pow(np.log(steps[view]), 1 / 2 - gamma),  # se[view],
            ax=axs[0, 1],
            **kwargs)

    # sns.lineplot(x=steps[view],
    #              y=np.sqrt(1 + np.pow(np.log(steps[view]), 2 + 2 * s) /
    #                        (2 + 2 * s) / np.pi),
    #              ax=axs[1, 0],
    #              label=f"Expected SE (gamma={-1/2-s})")
    handles, labels = axs[0, 1].get_legend_handles_labels()
    for ax in axs.reshape(-1):
        ax.get_legend().remove()
    fig.legend(handles, labels, loc='outside right upper')
    # plt.legend()
    plt.show()


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
