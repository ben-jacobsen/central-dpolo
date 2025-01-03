"""
    File: toeplitz_experiments.py
    Author: Ben Jacobsen
    Purpose: Playing around with different matrix factorizations
"""

import argparse
from functools import cache
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la
import seaborn as sns
from scipy.sparse.linalg import svds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', default=100, type=int)
    args = parser.parse_args()
    T = args.T
    q = 1 / 100
    s = 0.01

    A = la.toeplitz(np.ones(T), np.zeros(T))

    strategies = {
        #'geometric': lambda k: q**(k - 1),
        #'1/n': lambda k: 1 / k,
        # '1/(sqrt(n) log(n))': lambda k: 1 / (np.sqrt(k) * np.log(k + 1)),
        'log_decay': lambda k: 1 / (np.sqrt(
            (k + 1) * np.log(k + np.e)**(1 + s))),
        # 'offset 3': lambda k: 1 / (np.sqrt((k + 3) * np.log(k + 3)**1.01)),
        # 'offset 5': lambda k: 1 / (np.sqrt((k + 5) * np.log(k + 5)**1.01)),
        #'decaying optimal': lambda k: opt(k) / np.sqrt(np.log(k + 3))
        # 'geom_perturbation': lambda k: opt(k) - q * opt(k - 1),
        'fourier': fourier,
        'optimal': opt,
    }

    cols = {}

    # geometric
    fig, axs = plt.subplots(3, layout='constrained')
    axs[0].set_title("Sensitivity")
    axs[1].set_title("Variance")
    axs[2].set_title("Coefficients")
    axs[2].set_yscale('log')
    for ax in axs:
        ax.set_xscale('log')
    for name, f in strategies.items():
        print('-' * 80)
        print(name)
        print('-' * 80)
        try:
            r = f(np.arange(1, T + 1))
        except TypeError:
            r = [f(k) for k in np.arange(1, T + 1)]

        sensitivity = np.sqrt(np.cumsum(np.power(r, 2)))
        max_sens = sensitivity[-1]

        R = la.toeplitz(r, np.zeros_like(r)) / max_sens
        # L = A @ la.inv(R)
        R_inv = la.solve_toeplitz((R.T[:, 0], R.T[0, :]), np.eye(T)).T
        L = A @ R_inv

        cols[name] = R[:, 0]
        cols[name + "_inverse"] = R_inv[:, 0]

        #print(error_estimation(R))
        # print(L)
        # print(R)
        variance = np.sum(np.power(L, 2), axis=1)

        print(variance[-1])

        L_est = r * (np.log(np.arange(2, T + 2))**(1.01 / 2))
        mid = (L[0, 0] + R[0, 0]) / 2

        sns.lineplot(sensitivity / max_sens, ax=axs[0], label=name)
        sns.lineplot(variance, ax=axs[1], label=name)
        sns.lineplot(R[:, 0] / R[0, 0] * mid, ax=axs[2], label=name + " (R)")
        sns.lineplot(L[:, 0] / L[0, 0] * mid,
                     ax=axs[2],
                     label=name + " (L)",
                     ls=':')
        # if name == 'offset 1':
        #     sns.lineplot(L_est / L_est[0] * mid,
        #                  ax=axs[2],
        #                  label="Log speculation",
        #                  ls=':',
        #                  color='black')
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
def opt(k):
    k = int(k)
    if k < 1:
        return 0
    if k == 1:
        return 1
    return (1 - 1 / (2 * k - 2)) * opt(k - 1)


@cache
def fourier(k):
    return 1 / (2 * np.sqrt(k) * np.log(k + np.e - 1)) - np.sqrt(k) / (
        (k + np.e - 1) * np.power(np.log(k + np.e - 1), 2))


@cache
def geom_test(k):
    if k == 1:
        return 1
    return (1 - 1 / k) * geom_test(k - 1)


def error_estimation(mat):
    _, s, _ = svds(mat, k=1, which='SM')
    return 1 / s[0]


def inv_ltt(coeffs):
    inv_coeffs = [1 / coeffs[0]]


if __name__ == "__main__":
    main()
