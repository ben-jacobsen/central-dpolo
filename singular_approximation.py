"""
    File: singular_approximation.py
    Author: Ben Jacobsen
    Purpose: Evaluate different numerical strategies for computing the Taylor
        expansions of meromorphic functions of the form:

        g(x) = (1-z)^\alpha ((1/z)\log(1/(1-z)))^\gamma

    Around the point z=0. In this library, alpha is always assumed to be -1/2. 
"""
"""
    To prove optimal choice of delta: show that this minimizes
    the difference l_t - l_{t-1} uniformly for all t > t0
    and fixed gamma

    Visually, can compute optimal choice of delta for a fixed
    T and then show this converges to -6/5 for any fixed
    gamma as T increases

    also: calc 1, show that -6/5 gives desired 2nd coeff


    can i use flint to speed up the inversion process?
    flint.ctx.cap=1e6
    s = flint.arb_series([1]*1e6)
    x = flint.arb_series([0,1])
    ((2 * (s.log() / x).log() / x).__pow__(0.66) * (s.log()/x).__pow__(-0.55) * s.sqrt()).coeffs()
"""

import argparse
import os
import warnings
from functools import cache
from math import factorial, isclose

import flint
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import seaborn as sns
import sympy as sym
from scipy.optimize import minimize_scalar
from scipy.signal import convolve, fftconvolve
from scipy.special import binom

config = {'cache': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=-0.55)
    parser.add_argument('-K', type=int, default=1)
    parser.add_argument('--delta', type=float, default=0)
    parser.add_argument('--no-cache', action="store_true")
    args = parser.parse_args()

    mp.prec = 53

    T = args.T
    gamma = args.gamma
    K = args.K
    delta = args.delta
    config['cache'] = not args.no_cache

    coeffs = {
        # 'Function Values':
        # [singular_function(z, gamma) for z in np.linspace(-1e-14, 1e-14, 11)],
        # 'Direct Estimation':
        # np.array(direct_estimation(T, gamma, delta=delta), dtype=float),
        'Asymptotic Expansion':
        np.array(asymptotic_expansion(T, gamma, delta=delta), dtype=float),
        'Exact Convolution':
        exact_convolution(T + 1, gamma),
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
        print(f"\tRatio: {(coeffs[k] / coeffs['Exact Convolution'])[-1]}")
        sns.lineplot(v, label=k)

    # pretty_print(coeffs[f'Order {K} Expansion'] / coeffs['Exact Convolution'],
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


def exact_convolution(T, gamma, alpha=-1 / 2, delta=0):
    """
    Uses recurrence formulas to derive the exact Taylor coefficients of each
    component of the singular function separately, and then computes their
    Cauchy product in O(T log T) time and O(T) space with FFTs

    Returns a sequence of size T, i.e. all coefficients up to x^(T-1)
    """
    if isinstance(T, np.ndarray):
        T = len(T)

    poly_part = poly_coeffs(T - 1, alpha)
    log_part = log_coeffs(T - 1, gamma)

    if delta == 0:
        return convolve(poly_part, log_part, mode='full')[:T]
    else:
        loglog_part = loglog_coeffs(T - 1, delta)
        return convolve(convolve(poly_part, log_part, mode='full')[:T],
                        loglog_part,
                        mode='full')[:T]


@cache
def poly_coeffs(n, alpha=-1 / 2):
    """
    Returns the first n+1 coefficients in the Taylor expansion of
        f(x) = (1-x)^alpha
    """
    return np.array([poly_coeff_single(k, alpha) for k in range(n + 1)])


@cache
def poly_coeff_single(n, alpha=-1 / 2):
    """
    Recursively computes the coefficient of x^n in the Taylor expansion of
        f(x) = (1-x)^alpha
    """
    if n == 0:
        return 1
    return (1 - (1 + alpha) / n) * poly_coeff_single(n - 1, alpha)


def log_coeffs(n, gamma=1):
    """
    Returns the first n+1 coefficients in the Taylor expansion of
        f(x) = (1/x) ln(1/(1-x))
    """
    name = f"log_{n}_{gamma}.npy"
    if gamma != 1:
        if config['cache']:
            try:
                with open(os.path.join("seq_cache", name), 'rb') as f:
                    return np.load(f)
            except FileNotFoundError:
                pass

        # a = power_coeffs_explicit(1 / np.arange(1, n + 2), gamma)
        a = power_coeffs_implicit(1 / np.arange(1, n + 2), gamma)
        if config['cache']:
            with open(os.path.join("seq_cache", name), 'wb') as f:
                np.save(f, a)
        return a
    else:
        return 1 / np.arange(1, n + 2)


def loglog_coeffs(n, delta=1):
    """
    Returns the first n+1 coefficients in the Taylor expansion of
        g(x) = 2 * (1/x) ln( f(x) )

    where
        f(x) = (1/x) ln(1/(1-x))

    Uses the fact that:

        d/dx (x g(x)) = 2 f'(x) / f(x)
        (x g(x))(0) = 0

    to reduce problem to series inversion + convolution + term-by-term
        differentiation and integration
    """
    name = f"loglog_{n}_{delta}.npy"
    if config['cache']:
        try:
            with open(os.path.join("seq_cache", name), 'rb') as f:
                return np.load(f)
        except FileNotFoundError:
            pass
    if delta == 1:
        f_coeffs = log_coeffs(n + 1)
        f_prime_coeffs = f_coeffs[1:] * np.arange(1, n + 2)
        # rf_coeffs = power_coeffs_explicit(f_coeffs, -1)
        rf_coeffs = power_coeffs_implicit(f_coeffs, -1)

        xg_prime_coeffs = 2 * convolve(f_prime_coeffs, rf_coeffs)[:n + 1]
        # integrate both sides, then divide by x
        a = xg_prime_coeffs / np.arange(1, n + 2)
    else:
        # a = power_coeffs_explicit(loglog_coeffs(n, 1), delta)
        a = power_coeffs_implicit(loglog_coeffs(n, 1), delta)

    if config['cache']:
        with open(os.path.join("seq_cache", name), 'wb') as f:
            np.save(f, a)
    return a


def binary_components(k):
    bin_str = bin(k)[2:]
    comps = []
    for i, b in enumerate(bin_str[::-1]):
        if b == '1':
            comps.append(i)
    return comps


def power_coeffs_explicit(a_coeffs, p):
    """
    Given:
        a_coeffs: a sequence of n+1 real numbers, the first non-zero, 
            interpreted as Taylor coefficients of some function f(x), 
        p: any real or complex number

    Computes and returns the first n+1 Taylor coefficients of the function
        f(x)^p using the recurrence relation presented in:
        https://www.jstor.org/stable/2318904?seq=1
    """
    if p == 1:
        return a_coeffs

    a_coeffs = np.array(a_coeffs)
    n = len(a_coeffs) - 1
    a0 = a_coeffs[0]

    b_coeffs = np.zeros(n + 1)
    b_coeffs[0] = a0**p

    conv_array = np.zeros(n)

    for i in range(1, n + 1):
        conv_array[:i] = a_coeffs[1:i + 1] * b_coeffs[i - 1::-1]
        s1 = (p + 1) * np.dot(np.arange(1, i + 1), conv_array[:i])
        # conv_array[:i])
        s2 = i * np.sum(conv_array[:i])

        # s1 = (p + 1) * convolve(np.arange(1, i + 1) * a_coeffs[1:i + 1],
        #                         b_coeffs[:i],
        #                         mode='valid')[0]
        # s2 = i * convolve(a_coeffs[1:i + 1], b_coeffs[:i], mode='valid')[0]
        b_coeffs[i] = (1 / (i * a0)) * (s1 - s2)

    return b_coeffs


def power_coeffs_implicit(a_coeffs, p):
    """
    Given:
        a_coeffs: a sequence of n+1 real numbers, the first non-zero,
            interpreted as Taylor coefficients of some function f(x)
        p: any real or complex number

        Computes and returns the first n+1 Taylor coefficients of f(x)^p
            using the definition f(x)^p = exp(p log(f(x))) 
    """
    a_coeffs = a_coeffs.tolist()
    flint.ctx.cap = len(a_coeffs)
    # TODO: might be interesting to track precision
    return np.array(
        flint.arb_series(a_coeffs).__pow__(p).coeffs()).astype(float)


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


def asymptotic_expansion(T, gamma, alpha=-1 / 2, delta=0, single_coeff=False):
    """
    This approach directly applies the asymptotic expansion from the ~-transfer
    theorem, disregarding error on lower order terms
    """
    if isinstance(T, np.ndarray):
        T = T.size

    if single_coeff:
        coeffs = np.array([1.0])
        vals = np.array([T])
    else:
        coeffs = np.ones(T + 1).astype(float)
        vals = np.arange(T + 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        poly_part = np.pow(vals,
                           (-alpha - 1)) / float(mp.gamma(-alpha)) * 2**delta
        log_part = np.pow(np.log(vals), gamma)
        loglog_part = np.pow(np.log(np.log(vals)), delta)

    n1 = np.argmax((poly_part > 0) * np.isfinite(poly_part))
    n2 = np.argmax((log_part > 0) * np.isfinite(log_part))
    n3 = np.argmax((loglog_part > 0) * np.isfinite(loglog_part))

    coeffs[n1:] = poly_part[n1:]
    coeffs[n2:] *= log_part[n2:]
    coeffs[n3:] *= loglog_part[n3:]

    # coeffs[1:] = np.pow(np.arange(1, T + 1),
    #                     (-alpha - 1)) / mp.gamma(-alpha) * 2**delta
    # coeffs[2:] *= np.pow(np.log(np.arange(2, T + 1)), gamma)
    # coeffs[3:] *= np.pow(np.log(np.log(np.arange(3, T + 1))), delta)
    return np.array(coeffs, dtype=float)


def full_computation(T, gamma, K=1, alpha=-1 / 2, delta=0, single_coeff=False):
    """
    This approaches employs the more complex asymptotic expression which includes
    powers of 1/(log(n)) and constant factors related to derivatives of the
    reciprocal gamma function (which must themselves be estimated numerically).
    The expansion is truncated to terms of order K or less.
    """
    baseline = asymptotic_expansion(T,
                                    gamma,
                                    alpha,
                                    delta,
                                    single_coeff=single_coeff)
    if single_coeff:
        scale = np.ones(1)
        ns = np.array([T])
    else:
        scale = np.ones(T + 1)
        ns = np.arange(T + 1)
    # ignore terms where log(n) < 1
    if K > 0:
        ks = np.arange(1, K + 1)
        cutoff = np.argmax(ns >= 3)
        coeffs = expansion_coeff(gamma,
                                 ks,
                                 ns[cutoff:],
                                 alpha=alpha,
                                 delta=delta)
        scale[cutoff:] = scale[cutoff:] + coeffs
    return baseline * scale


def expansion_coeff(gamma, ks, ns, alpha=-1 / 2, delta=0):
    """
    Numerically estimates the coefficient of log(log(n))(1/(log(log(n))log(n)))^k in the 
    full_computation method
    """
    if delta == 0:
        log_powers = np.array([np.pow(np.log(ns), -k) for k in ks])
        coeffs = np.pow(-1, ks) * binom(
            gamma, ks) * mp.gamma(-alpha) * rgamma_derivatives(max(ks), alpha)
        return coeffs @ log_powers
    else:
        ind_part = mp.gamma(-alpha) * rgamma_derivatives(max(ks), alpha)
        log_powers = np.array(
            [np.power(np.log(ns) * np.log(np.log(ns)), -k) for k in ks])

        if delta == int(delta):
            raise ValueError("delta must not be in the set 1,2,3,...")

        u = sym.Symbol('u')
        x = sym.Symbol('x')

        f = (1 - x * u)**gamma * (1 - (1 / x) * sym.log(1 - x * u))**delta

        funcs = [
            sym.lambdify([x, u],
                         sym.diff(f, u, k) / factorial(k)) for k in ks
        ]

        eks = [fk(np.log(np.log(ns)), 0) for fk in funcs]

        # [fk.evalf(subs={
        # u: 0,
        # x: np.log(np.log(n))
        # }) for n in ns] for fk in funcs],
        # dtype=float)

        log_powers *= eks

        return ind_part @ log_powers


@cache
def rgamma_derivatives(k, alpha=-1 / 2):
    """
    Numerically computes the kth derivative(s) of the reciprocal gamma 
    function with negated argument
    """
    f = lambda x: mp.rgamma(-x)
    try:
        ds = np.array(list(mp.diffs(f, alpha, k)))
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


def f1(w):
    """
    Contribution of the polynomial part
    """
    return (2 * mp.cos(w))**2


def f2(w):
    """
    Contribution of the logarithmic part
    """
    return (1 / 4) * mp.log(f1(w))**2 + w**2


def f3(w):
    """
    (first) contribution of the iterated logarithmic part
    """
    return (1 / 4) * mp.log(f2(w))**2


def f4(w):
    """
    (second) contribution of the iterated logarithmic part
    """
    # if w >= mp.pi / 2:
    #     return 0
    return (mp.atan2(w, -mp.log(2 * mp.cos(w))) +
            mp.atan2(-mp.sin(2 * w), -mp.cos(2 * w)))**2


def sens_function(alpha, gamma, delta):
    """
    Using Parseval's theorem, we can evaluate the infinite sum 
    corresponding to the squared Taylor coefficients of our singular function
    by integration. This returns the integrand for the interval 0 to pi/2. 
    """
    # TODO: check for constant factor error with change of variables
    return lambda w: 2 / np.pi * f1(w)**alpha * f2(w)**gamma * (4 * (f3(
        w) + f4(w)))**delta


@cache
def compute_sensitivity(alpha, gamma, delta):
    """
    Actually integrate the function from above
    """
    return np.sqrt(
        mp.quad(sens_function(alpha, gamma, delta), [0, mp.pi / 3, mp.pi / 2]))


@cache
def compute_sensitivity_direct(alpha, gamma, delta):
    """
    Unsimplified version, for reference. Seems to be less numerically stable,
    esp. when delta is large.
    """
    return np.sqrt(
        mp.quad(
            lambda z: mp.fabs(
                singular_function(
                    mp.exp(1j * z), gamma, alpha=alpha, delta=delta))**2 /
            (2 * mp.pi), [-mp.pi, 0, mp.pi]))


def optimize_variance(T, alpha, gamma):
    """
    What is the optimal value of delta for fixed alpha, gamma, *and* T?
    """
    return minimize_scalar(
        lambda delta: float(compute_sensitivity(alpha, gamma, delta)) * np.
        sqrt(np.sum(
            exact_convolution(T, -gamma, alpha=alpha, delta=-delta)**2)),
        bounds=(-1, 1))
    # (full_computation(
    #     T * 2, -gamma, K=4, alpha=alpha, delta=-delta, single_coeff=True)**
    #  2 / full_computation(
    #      T, -gamma, K=4, alpha=alpha, delta=-delta, single_coeff=True)**2)[
    #          0],


def optimize_sensitivity(alpha, gamma):
    """
    What is the optimal value of delta for fixed alpha and gamma?
    """

    def weight(w):
        return mp.log(f3(w) + f4(w))

    def g1(delta):
        return mp.quad(
            lambda w: sens_function(alpha, gamma, delta)(w) * weight(w),
            [0, mp.pi / 3])

    def g2(delta):
        return mp.quad(
            lambda w: sens_function(alpha, gamma, delta)(w) * weight(w),
            [mp.pi / 3, mp.pi / 2])

    return mp.findroot(lambda delta: g1(delta) + g2(delta), 0, verbose=True)


if __name__ == "__main__":
    main()
