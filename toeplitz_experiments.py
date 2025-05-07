"""
    File: toeplitz_experiments.py
    Author: Ben Jacobsen
    Purpose: Playing around with different matrix factorizations
"""

import argparse
from functools import cache
from itertools import pairwise

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pandas as pd
import scipy.linalg as la
import seaborn as sns
from numpy.fft import *
from scipy.sparse.linalg import svds

import sequences as seq
import singular_approximation as sa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', default=100, type=int)
    parser.add_argument('-s', default=0.01, type=float)
    args = parser.parse_args()
    T = args.T
    s = args.s

    # A = la.toeplitz(np.ones(T), np.zeros(T))

    # delta = float(sa.optimize_sensitivity(-1 / 2, -1 / 2 - s))

    strategies = [
        seq.Opt(T),
    ]

    for delta in np.linspace(0, (1 / 2 + s) * 6 / 5, 4):
        gamma = -1 / 2 - s
        strategies.append(
            seq.Anytime(-1 / 2, gamma, delta, tol=1e-4, asym_order=4))

    # strategies.append(
    #     seq.DoublingTrick(T // 2, seq.DoublingTrick.optimal_ratio(), T))

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

    steps = np.arange(1, T + 1)
    view = np.arange(1, min(T, 1000))
    if T > 1000:
        res = T // 1000
        view = np.concatenate((view, np.arange(1000, T, res)), axis=None)

    for strategy, color in zip(strategies, sns.color_palette()):
        print('-' * 80)
        print(strategy.name)
        print('-' * 80)
        if isinstance(strategy, seq.Sequence):
            sensitivity = strategy.sensitivity()
            l = strategy.first_k_left(T)
            r = strategy.first_k(T)
            mid = (l[0] + r[0]) / 2
            print(f"Sensitivity: {sensitivity}")
            print(l[:10])
            print(r[:10])
            se = strategy.standard_error(T)
            print(se[-1])

            sns.lineplot(x=steps[view],
                         y=strategy.smooth_sensitivity()[view],
                         ax=axs[0, 0],
                         label=strategy.name,
                         color=color)
            sns.lineplot(x=steps[view],
                         y=se[view],
                         ax=axs[1, 0],
                         label=strategy.name,
                         color=color)

            sns.lineplot(x=steps[view],
                         y=r[view] / r[0] * mid,
                         ax=axs[1, 1],
                         label=strategy.name + " (R)",
                         color=color)
            sns.lineplot(x=steps[view],
                         y=l[view] / l[0] * mid,
                         ax=axs[1, 1],
                         label=strategy.name + " (L)",
                         ls=':',
                         color=color)

        print(strategy.noise_schedule(T)[view])
        sns.lineplot(
            x=steps[view],
            y=strategy.noise_schedule(T)[view],  # se[view],
            ax=axs[0, 1],
            label=strategy.name)

    # sns.lineplot(x=steps[view],
    #              y=np.sqrt(1 + np.pow(np.log(steps[view]), 2 + 2 * s) /
    #                        (2 + 2 * s) / np.pi),
    #              ax=axs[1, 0],
    #              label=f"Expected SE (gamma={-1/2-s})")
    plt.legend()
    plt.show()

    # df = pd.DataFrame.from_dict(cols, orient='columns')
    # print(df)
    # for c1, c2 in pairwise(df.columns):
    #     print('-' * 80)
    #     name = f"{c1}*{c2}"
    #     print(name)
    #     coeffs = np.convolve(df[c1], df[c2])[:T]
    #     l1 = np.cumsum(np.abs(coeffs))
    #     sns.lineplot(l1, label=name)
    #     df[name] = l1
    # # target = 'log_decay_inverse*optimal'
    # # estimate = np.power((np.log(np.arange(1, T + 1))), 1 + s)
    # # estimate -= estimate[0]
    # # estimate /= estimate[-1]
    # # estimate *= (df[target][T - 1] - df[target][0])
    # # estimate += df[target][0]
    # # sns.lineplot(estimate, label="log experiment")
    # plt.xscale('log')
    # plt.legend()
    # plt.show()


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
