"""
    File: singular_approximation.py
    Author: Ben Jacobsen
    Purpose: Evaluate different numerical strategies for computing the Taylor
        expansions of meromorphic functions of the form:

        g(x) = (1-z)^\alpha ((1/z)\log(1/(1-z)))^\gamma

    Around the point z=0. In this library, alpha is always assumed to be -1/2. 
"""

import argparse
from math import factorial, isclose

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import seaborn as sns
import sympy as sym
from scipy.signal import fftconvolve
from scipy.special import binom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=-0.55)
    parser.add_argument('-K', type=int, default=1)
    parser.add_argument('--delta', type=float, default=1)
    args = parser.parse_args()

    mp.prec = 53

    T = args.T
    gamma = args.gamma
    K = args.K
    delta = args.delta

    coeffs = {
        # 'Function Values':
        # [singular_function(z, gamma) for z in np.linspace(-1e-14, 1e-14, 11)],
        'Direct Estimation':
        np.array(direct_estimation(T, gamma, delta=delta), dtype=float),
        'Asymptotic Expansion':
        np.array(asymptotic_expansion(T, gamma, delta=delta), dtype=float),
        # f'Order {K} Expansion':
        # np.array(full_computation(T, gamma, K), dtype=float),
        # 'Combined':
        # np.array(combined_estimate(T, gamma), dtype=float),
        # 'Combined Inverse':
        # np.array(combined_estimate(T, -gamma), dtype=float),
    }
    # coeffs['Combined Product'] = fftconvolve(coeffs['Combined'],
    # coeffs['Combined Inverse'],
    # mode='full')[:T + 1]

    for k in np.arange(1, K + 1):
        coeffs[f'Order {k} Expansion'] = np.array(full_computation(
            T, gamma, k, delta=delta),
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


def l(x):
    return mp.log(1 / (1 - x)) / x


def singular_function(z, gamma, alpha=-1 / 2, delta=0):
    # seems to be some numerical instability extremely close to the origin
    if isclose(abs(z), 0, abs_tol=1e-20):
        return 1
    return mp.power(2, delta) * mp.power(1 - z, alpha) * mp.power(
        mp.log(l(z)) / z, delta) * mp.power(l(z), gamma)


def direct_estimation(T, gamma, alpha=-1 / 2, delta=0):
    """
    This approach disregards all of the theorems we have about functions of
    this form and uses off-the-shelf numerical differentiation to approximate
    the Taylor coefficients
    """
    if isinstance(T, np.ndarray):
        T = T.size
    f = lambda z: singular_function(z, gamma, alpha, delta)
    coeffs = mp.taylor(f, 0, T, direction=1)
    # , h=1e-10), radius=0.99, method='quad')
    return np.array([c.real for c in coeffs], dtype=float)


def asymptotic_expansion(T, gamma, alpha=-1 / 2, delta=0):
    """
    This approach directly applies the asymptotic expansion from the ~-transfer
    theorem, disregarding error on lower order terms
    """
    if isinstance(T, np.ndarray):
        T = T.size
    coeffs = np.ones(T + 1)
    coeffs[1:] = np.pow(np.arange(1, T + 1),
                        (-alpha - 1)) / mp.gamma(-alpha) * 2**delta
    coeffs[2:] *= np.pow(np.log(np.arange(2, T + 1)), gamma)
    coeffs[3:] *= np.pow(np.log(np.log(np.arange(3, T + 1))), delta)
    return np.array(coeffs, dtype=float)


def full_computation(T, gamma, K=1, alpha=-1 / 2, delta=0):
    """
    This approaches employs the more complex asymptotic expression which includes
    powers of 1/(log(n)) and constant factors related to derivatives of the
    reciprocal gamma function (which must themselves be estimated numerically).
    The expansion is truncated to terms of order K or less.
    """
    baseline = asymptotic_expansion(T, gamma, alpha, delta)
    scale = np.ones(T + 1)
    # ignore terms where log(n) < 1
    if K > 0:
        ks = np.arange(1, K + 1)
        ns = np.arange(3, T + 1)
        coeffs = expansion_coeff(gamma, ks, ns, alpha=alpha, delta=delta)
        scale[3:] = scale[3:] + coeffs
    return baseline * scale


def expansion_coeff(gamma, ks, ns, alpha=-1 / 2, delta=0):
    """
    Numerically estimates the coefficient of log(log(n))(1/(log(log(n))log(n)))^k in the 
    full_computation method
    """
    if delta == 0:
        log_powers = np.array([np.pow(np.log(ns), -k) for k in ks])
        coeffs = np.pow(-1, ks) * binom(
            gamma, ks) * mp.gamma(-alpha) * rgamma_derivatives(ks, alpha)
        return coeffs @ log_powers
    else:
        ind_part = mp.gamma(-alpha) * rgamma_derivatives(ks, alpha)
        log_powers = np.array(
            [np.power(np.log(ns) * np.log(np.log(ns)), -k) for k in ks])

        # TODO: is there a good way to translate the symbolic function to
        # vectorized functions?
        if delta == int(delta):
            raise ValueError("delta must not be in the set 1,2,3,...")

        u = sym.Symbol('u')
        x = sym.Symbol('x')

        f = (1 - x * u)**gamma * (1 - (1 / x) * sym.log(1 - x * u))**delta

        funcs = [sym.diff(f, u, k) / factorial(k) for k in ks]

        eks = np.array(
            [[fk.evalf(subs={
                u: 0,
                x: np.log(np.log(n))
            }) for n in ns] for fk in funcs],
            dtype=float)

        log_powers *= eks

        return ind_part @ log_powers


def rgamma_derivatives(k, alpha=-1 / 2):
    """
    Numerically computes the kth derivative(s) of the reciprocal gamma 
    function with negated argument
    """
    print(k)
    f = lambda x: mp.rgamma(-x)
    try:
        ds = np.array(list(mp.diffs(f, alpha, max(k))))
        print(ds)
        return ds[1:]
    except IndexError:
        return mp.diff(f, alpha, k)


def combined_estimate(T, gamma, alpha=-1 / 2, delta=0, tol=1e-2):
    """
    This method numerically computes the first several terms before switching to
    the truncated asymptotic expansion
    """
    try:  # allow T to be a range of time steps
        T = T[-1]
    except IndexError:
        pass
    t = 100
    if T <= t:
        return direct_estimation(T, gamma, alpha=alpha, delta=delta)
    else:
        print(t)
        if delta > (1 - gamma) / 2:  # heuristic
            k = 0
        else:
            k = 1
        coeff = full_computation(T, gamma, alpha=alpha, delta=delta, K=k)
        manual = direct_estimation(t, gamma, alpha=alpha, delta=delta)
        rel_err = abs(manual[t] / coeff[t] - 1)
        while (t < T and rel_err > tol):
            print(f"\tRelative Error: {manual[t] / coeff[t]:.3f}")
            t = min(T, int(np.ceil(t * rel_err / tol)))
            print(f"\ttrying t={t}")
            manual = direct_estimation(t, gamma, alpha=alpha, delta=delta)
            rel_err = abs(manual[t] / coeff[t] - 1)
        print(f"\tRelative Error: {manual[t] / coeff[t]:.3f}")
        coeff[:t + 1] = manual
        return coeff


if __name__ == "__main__":
    main()
