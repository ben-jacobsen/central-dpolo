"""
    File: toeplitz_experiments.py
    Author: Ben Jacobsen
    Purpose: Playing around with different matrix factorizations
"""

import argparse
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', default=100, type=int)
    args = parser.parse_args()
    T = args.T
    q = 1 - 1 / np.sqrt(T)

    A = la.toeplitz(np.ones(T), np.zeros(T))

    strategies = {
        'geometric': lambda k: q**(k - 1),
        'adaptive geometric': geom_test,
        '1/n': lambda k: 1 / k,
        '1/(sqrt(n) log(n))': lambda k: 1 / (np.sqrt(k) * np.log(k + 1)),
        #'offset 1': lambda k: 1 / (np.sqrt(k + 1) * np.log(k + 1)),
        'offset 3': lambda k: 1 / (np.sqrt(k + 3) * np.log(k + 3)),
        #'offset 5': lambda k: 1 / (np.sqrt(k + 5) * np.log(k + 5)),
        'optimal': opt
    }

    # geometric
    fig, axs = plt.subplots(2, layout='constrained')
    axs[0].set_title("Sensitivity")
    axs[1].set_title("Variance")
    for name, f in strategies.items():
        print('-' * 80)
        print(name)
        print('-' * 80)
        r = [f(k) for k in range(1, T + 1)]
        sensitivity = np.sqrt(np.cumsum(np.power(r, 2)))
        max_sens = sensitivity[-1]

        R = la.toeplitz(r, np.zeros_like(r)) / max_sens
        L = A @ la.inv(R)

        #print(L)
        #print(R)
        variance = np.sum(np.power(L, 2), axis=1)

        print(variance[-1])

        sns.lineplot(sensitivity / max_sens, ax=axs[0], label=name)
        sns.lineplot(variance, ax=axs[1], label=name)
    plt.legend()
    plt.show()


@cache
def opt(k):
    if k == 1:
        return 1
    return (1 - 1 / (2 * k - 2)) * opt(k - 1)


@cache
def geom_test(k):
    if k == 1:
        return 1
    return (1 - 1 / k) * geom_test(k - 1)


if __name__ == "__main__":
    main()
