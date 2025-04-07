"""
    File: singular_approximation.py
    Author: Ben Jacobsen
    Purpose: Evaluate different numerical strategies for computing the Taylor
        expansions of meromorphic functions of the form:

        g(x) = (1-z)^\alpha ((1/z)\log(1/(1-z)))^\gamma

    Around the point z=0. In this library, alpha is always assumed to be -1/2. 
"""

import argparse
from math import isclose

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import seaborn as sns
from scipy.signal import fftconvolve
from scipy.special import binom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=-0.55)
    parser.add_argument('-K', type=int, default=2)
    args = parser.parse_args()

    mp.prec = 53

    T = args.T
    gamma = args.gamma
    K = args.K

    coeffs = {
        # 'Function Values':
        # [singular_function(z, gamma) for z in np.linspace(-1e-14, 1e-14, 11)],
        # 'Direct Estimation': np.array(direct_estimation(T, gamma),
        #                               dtype=float),
        # 'Asymptotic Expansion':
        # np.array(asymptotic_expansion(T, gamma), dtype=float),
        # f'Order {K} Expansion':
        # np.array(full_computation(T, gamma, K), dtype=float),
        'Combined': np.array(combined_estimate(T, gamma), dtype=float),
        'Combined Inverse': np.array(combined_estimate(T, -gamma),
                                     dtype=float),
    }
    coeffs['Combined Product'] = fftconvolve(coeffs['Combined'],
                                             coeffs['Combined Inverse'],
                                             mode='full')[:T + 1]

    for k in np.arange(1, K + 1):
        coeffs[f'Order {k} Expansion'] = np.array(full_computation(
            T, gamma, k),
                                                  dtype=float)

    for k, v in coeffs.items():
        pretty_print(v, k)
        sns.lineplot(v, label=k)

    # pretty_print(coeffs[f'Order {K} Expansion'] / coeffs['Direct Estimation'],
    #              'Asymptotic Ratio')

    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.show()


def pretty_print(xs, label):
    l = len(label)
    r = (80 - (l + 2)) // 2
    print('#' * (80 - l % 2))
    print('#' * r + ' ' + label + ' ' + '#' * r)
    print('#' * (80 - l % 2))
    print(xs)


def singular_function(z, gamma, alpha=-1 / 2):
    if isclose(z, 0, abs_tol=1e-20):
        return 1
    return mp.power(1 - z, alpha) * mp.power((mp.log(1 / (1 - z)) / z), gamma)


def direct_estimation(T, gamma, alpha=-1 / 2):
    """
    This approach disregards all of the theorems we have about functions of
    this form and uses off-the-shelf numerical differentiation to approximate
    the Taylor coefficients
    """
    f = lambda z: singular_function(z, gamma, alpha)
    return mp.taylor(f, 0, T, direction=1,
                     h=1e-10)  #, radius=0.99, method='quad')


def asymptotic_expansion(T, gamma, alpha=-1 / 2):
    """
    This approach directly applies the asymptotic expansion from the ~-transfer
    theorem, disregarding error on lower order terms
    """
    coeffs = np.ones(T + 1)
    coeffs[1:] = np.array([
        mp.power(n, (-alpha - 1)) / mp.gamma(-alpha) for n in range(1, T + 1)
    ])
    coeffs[2:] *= np.pow(np.log(np.arange(2, T + 1)), gamma)
    return coeffs


def full_computation(T, gamma, K=1, alpha=-1 / 2):
    """
    This approaches employs the more complex asymptotic expression which includes
    powers of 1/(log(n)) and constant factors related to derivatives of the
    reciprocal gamma function (which must themselves be estimated numerically).
    The expansion is truncated to terms of order K or less.
    """
    baseline = asymptotic_expansion(T, gamma, alpha)
    scale = np.ones(T + 1)
    ks = np.arange(1, K + 1)
    # ignore terms where log(n) < 1
    ns = np.arange(3, T + 1)
    log_powers = np.array([np.pow(np.log(ns), -k) for k in ks])
    coeffs = expansion_coeff(gamma, ks, alpha)
    print(coeffs)
    scale[3:] = scale[3:] + coeffs @ log_powers
    return baseline * scale


def expansion_coeff(gamma, k, alpha=-1 / 2):
    """
    Numerically estimates the coefficient of (1/log(n))^k in the 
    full_computation method
    """
    return binom(gamma, k) * mp.gamma(-alpha) * rgamma_derivatives(k, alpha)


def rgamma_derivatives(k, alpha=-1 / 2):
    """
    Numerically computes the kth derivative(s) of the reciprocal gamma 
    function, evaluated at -alpha
    """
    print(k)
    try:
        ds = np.array(list(mp.diffs(mp.rgamma, -alpha, max(k))))
        print(ds)
        return ds[1:]
    except IndexError:
        return mp.diff(mp.rgamma, -alpha, k)


def combined_estimate(T, gamma):
    """
    This method numerically computes the first several terms before switching to
    the truncated asymptotic expansion
    """
    if T <= 50:
        return direct_estimation(T, gamma)
    else:
        coeff = full_computation(T, gamma)
        t = 50
        manual = direct_estimation(t, gamma)
        while (2 * t < T and abs(manual[t] / coeff[t] - 1) > 0.001):
            t *= 2
            manual = direct_estimation(t, gamma)
        coeff[:t + 1] = manual
        return coeff


if __name__ == "__main__":
    main()
