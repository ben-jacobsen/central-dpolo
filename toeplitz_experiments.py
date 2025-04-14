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

import singular_approximation as sa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', default=100, type=int)
    parser.add_argument('-s', default=0.05, type=float)
    args = parser.parse_args()
    T = args.T
    s = args.s

    # A = la.toeplitz(np.ones(T), np.zeros(T))

    strategies = {
        #'geometric': lambda k: q**(k - 1),
        # '1/sqrt(n)': lambda k: 1 / np.sqrt(k*np.pi),
        # '1/(sqrt(n) log(n))': lambda k: 1 / (np.sqrt(k) * np.log(k + 1)),
        # 'log_decay': lambda k: 1 / (np.sqrt(
        #     (k) * np.log(k + np.e - 1)**(1 + s))),
        # 'offset 3': lambda k: 1 / (np.sqrt((k + 3) * np.log(k + 3)**1.01)),
        # 'offset 5': lambda k: 1 / (np.sqrt((k + 5) * np.log(k + 5)**1.01)),
        # 'decaying optimal': lambda k: opt(k) / np.power(np.log(k + np.e - 1), 1/2),
        # 'geom_perturbation': lambda k: opt(k) - q * opt(k - 1),
        # 'fourier2': fourier2,
        # 'fourier': fourier,
        'optimal':
        opt,
        # 'opt_geom': lambda k: opt(k) * alpha**(k-1),
        # 'opt_shifted': lambda k: opt(k, 1/2+s)
        'log_decay':
        lambda k: opt(k) * np.maximum(1, np.log(np.log(4 * k - 3))) / np.power(
            np.maximum(1, np.log(4 * k - 3)), 1 / 2 + s),
        f'singular approx (gamma={-1/2-s})':
        lambda k: sa.combined_estimate(k, -1 / 2 - s, delta=0, tol=0.01),
        # f'loglog approx (gamma={-1/2-s}, delta=-gamma)':
        # lambda k: sa.combined_estimate(
        #     k, -1 / 2 - s, delta=1 / 2 + s, tol=0.01),
        f'loglog approx (gamma={-1/2-s}, delta=1)':
        lambda k: sa.combined_estimate(k, -1 / 2 - s, delta=1, tol=0.01),
    }

    print(f"opt_shifted sensitivity bound: {2**(s+1/2)*(s+1/2)/(np.pi*s)}")
    print(f"opt_shifted lower bound: {2**(s)*(s+1/2)/(np.pi*s)}")

    cols = {}

    # geometric
    fig, axs = plt.subplots(2, 2, layout='constrained')
    axs[0, 0].set_title("Sensitivity")
    axs[1, 0].set_title("Standard Error")
    axs[0, 1].set_title("Total Variance")
    axs[1, 1].set_title("Coefficients")
    axs[1, 1].set_yscale('log')
    for ax in axs.reshape(-1):
        ax.set_xscale('log')
    axs[0, 0].sharey(axs[1, 0])

    steps = np.arange(1, T + 1)
    for name, f in strategies.items():
        print('-' * 80)
        print(name)
        print('-' * 80)
        try:
            r = f(steps)
        except TypeError:
            r = np.array([f(k) for k in steps])

        sensitivity = np.sqrt(np.cumsum(np.power(r, 2)))
        max_sens = sensitivity[-1]
        print(f"Sensitivity: {max_sens}")

        #R = la.toeplitz(r, np.zeros_like(r)) # / max_sens
        # L = A @ la.inv(R)
        # R_inv = la.solve_toeplitz((R.T[:, 0], R.T[0, :]), np.eye(T)).T
        #L = A @ R_inv
        r_inv = fast_inv_ltt(r)
        l = np.cumsum(r_inv)

        cols[name] = r
        cols[name + "_inverse"] = r_inv

        #print(error_estimation(R))
        se = np.sqrt(np.cumsum(np.power(l, 2)))

        print(l[:10])
        print(r[:10])

        print(se[-1])

        # L_est = r * (np.log(np.arange(2, T + 2))**(1.01 / 2))
        mid = (l[0] + r[0]) / 2
        view = np.arange(1, min(T, 1000))
        if T > 1000:
            res = T // 1000
            view = np.concatenate((view, np.arange(1000, T, res)), axis=None)

        sns.lineplot(x=steps[view],
                     y=sensitivity[view],
                     ax=axs[0, 0],
                     label=name)
        sns.lineplot(x=steps[view], y=se[view], ax=axs[1, 0], label=name)
        sns.lineplot(x=steps[view],
                     y=sensitivity[view] * se[view],
                     ax=axs[0, 1],
                     label=name)
        sns.lineplot(x=steps[view],
                     y=r[view] / r[0] * mid,
                     ax=axs[1, 1],
                     label=name + " (R)")
        sns.lineplot(x=steps[view],
                     y=l[view] / l[0] * mid,
                     ax=axs[1, 1],
                     label=name + " (L)",
                     ls=':')
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
