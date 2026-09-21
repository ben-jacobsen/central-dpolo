"""
    File: singular_approximation.py
    Author: Ben Jacobsen
    Purpose: Evaluate different numerical strategies for computing the Taylor
        expansions of meromorphic functions of the form:

        g(x) = (1-z)^\alpha ((1/z)\log(1/(1-z)))^\gamma

    Around the point z=0. In this library, alpha is always assumed to be -1/2. 
"""

import argparse
import os
import warnings
from copy import copy
from functools import cache
from math import factorial, isclose

import flint
import matplotlib
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import seaborn as sns
import sympy as sym
from scipy.optimize import minimize_scalar
from scipy.signal import convolve, fftconvolve
from scipy.special import binom

# FLINT uses a single worker by default.  Series exponentiation is one of the
# more expensive operations in this module, so use the CPUs available to the
# process unless the caller supplies an explicit override.
try:
    _available_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    _available_cpus = os.cpu_count() or 1
_flint_threads = int(os.environ.get("FLINT_THREADS", _available_cpus))
if _flint_threads < 1:
    raise ValueError("FLINT_THREADS must be a positive integer")
flint.ctx.threads = _flint_threads

config = {'cache': True}
sns.set_style("darkgrid")
sns.set_context('paper')
sns.set(rc={"lines.linewidth": 3, "figure.dpi": 160, "savefig.dpi": 160})
sns.set(font_scale=0.7)


def _cached_prefix(prefix, n, parameter):
    """Load the smallest cached prefix covering coefficients through ``n``."""
    cache_dir = "seq_cache"
    parameter_text = str(parameter)
    suffix = f"_{parameter_text}.npy"
    candidates = []
    try:
        names = os.listdir(cache_dir)
    except FileNotFoundError:
        return None

    for name in names:
        if not (name.startswith(f"{prefix}_") and name.endswith(suffix)):
            continue
        length_text = name[len(prefix) + 1:-len(suffix)]
        try:
            cached_n = int(length_text)
        except ValueError:
            continue
        if cached_n >= n:
            candidates.append((cached_n, name))

    if not candidates:
        return None
    _, name = min(candidates)
    with open(os.path.join(cache_dir, name), 'rb') as stream:
        return np.load(stream)[:n + 1]


def _cached_length(prefix, parameters):
    """Return the largest cached coefficient index for a parameter tuple."""
    suffix = "_" + "_".join(str(parameter) for parameter in parameters) + ".npy"
    try:
        names = os.listdir("seq_cache")
    except FileNotFoundError:
        return 0

    largest = 0
    for name in names:
        if not (name.startswith(f"{prefix}_") and name.endswith(suffix)):
            continue
        length_text = name[len(prefix) + 1:-len(suffix)]
        try:
            largest = max(largest, int(length_text) + 1)
        except ValueError:
            continue
    return largest


def _cached_prefix_multi(prefix, n, parameters):
    """Load the smallest cached prefix covering ``n`` for multiple parameters."""
    suffix = "_" + "_".join(str(parameter) for parameter in parameters) + ".npy"
    candidates = []
    try:
        names = os.listdir("seq_cache")
    except FileNotFoundError:
        return None

    for name in names:
        if not (name.startswith(f"{prefix}_") and name.endswith(suffix)):
            continue
        length_text = name[len(prefix) + 1:-len(suffix)]
        try:
            cached_n = int(length_text)
        except ValueError:
            continue
        if cached_n >= n:
            candidates.append((cached_n, name))

    if not candidates:
        return None
    _, name = min(candidates)
    with open(os.path.join("seq_cache", name), 'rb') as stream:
        return np.load(stream)[:n + 1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=-0.55)
    parser.add_argument('-K', type=int, default=1)
    parser.add_argument('--deltas', type=float, default=[0], nargs='+')
    parser.add_argument('--no-cache', action="store_true")
    parser.add_argument('--relative', action="store_true")
    args = parser.parse_args()

    mp.prec = 53

    T = args.T
    gamma = args.gamma
    K = args.K
    deltas = args.deltas
    config['cache'] = not args.no_cache

    fig, ax = plt.subplots(layout="constrained")

    coeffs = {}
    for delta, ls in zip(deltas, ['-', '--', ':', '-.']):
        kwargs = {'ls': ls, 'ax': ax}

        basename = f'Exact δ={delta:.2f}'
        coeffs[basename] = (copy(kwargs | {'color': 'black'}), exact_convolution(T + 1, gamma, delta=delta))

        for k, color in zip(np.arange(2, K + 1, 2), sns.color_palette("Dark2", K//2)):
            kwargs['color'] = color

            dname = "0" if delta == 0 else ("-γ" if delta == -gamma else f"{delta:.2f}")
            name = f'δ={dname},K={k}'
            val = np.array(full_computation(
                T, gamma, k, delta=delta), dtype=float)
            if args.relative:
                val = np.abs(val - coeffs[basename][1]) / coeffs[basename][1]
            coeffs[name] = (copy(kwargs), val)


    for key, (kwargs, val) in coeffs.items():
        # pretty_print(v, k)
        # print(f"\tRatio: {(coeffs[k] / coeffs['Exact'])[-1]}")
        if args.relative:
            if key[:5] != 'Exact':
                sns.lineplot(val, label=key, **kwargs)
            # plt.ylim([-0.05, 0.05])
        else:
            sns.lineplot(val, label=key, **kwargs)

    # pretty_print(coeffs[f'Order {K} Expansion'] / coeffs['Exact Convolution'],
    #              'Asymptotic Ratio')

        # plt.hlines(y=[0], xmin=0, xmax=T+1, colors=['black'], linestyle=[':'])
    if args.relative:
        ax.set_ylim([1e-7, 1])
        # plt.legend(loc="lower left")
    else:
        ax.set_ylim([1/(2*np.sqrt(T * np.log(T))), 0.1])
        # plt.legend(loc="upper right")
    ax.set_yscale('log', base=2)
    ax.set_xlim((10, T))
    ax.set_xscale('log', base=2)
    # plt.grid()
    ax.legend(ncol=2, loc='lower left', framealpha=0.6, handlelength=1.5, columnspacing=1, labelspacing=0.25, fontsize=8)
    ax.set_xlabel("t", fontsize=8, labelpad=0)
    ax.set_ylabel("Relative Error", fontsize=8, labelpad=0)
    fig.set_size_inches(w=2.5, h=2)
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

    if config['cache']:
        os.makedirs("seq_cache", exist_ok=True)
    name = f"sequence_{T - 1}_{alpha}_{gamma}_{delta}.npy"
    if config['cache']:
        try:
            with open(os.path.join("seq_cache", name), 'rb') as stream:
                return np.load(stream)[:T]
        except FileNotFoundError:
            cached = _cached_prefix_multi(
                "sequence", T - 1, (alpha, gamma, delta))
            if cached is not None:
                return cached

    poly_part = poly_coeffs(T - 1, alpha)
    log_part = log_coeffs(T - 1, gamma)

    if delta == 0:
        result = _truncated_convolution(poly_part, log_part, T)
    else:
        loglog_part = loglog_coeffs(T - 1, delta)
        first_product = _truncated_convolution(poly_part, log_part, T)
        result = _truncated_convolution(first_product, loglog_part, T)

    if config['cache']:
        os.makedirs("seq_cache", exist_ok=True)
        with open(os.path.join("seq_cache", name), 'wb') as stream:
            np.save(stream, result)
    return result


def cached_convolution_prefix(gamma, alpha=-1 / 2, delta=0):
    """Return the largest exact-convolution prefix currently cached."""
    if not config['cache']:
        return 0

    final = _cached_length("sequence", (alpha, gamma, delta))
    component_limits = [_cached_length("poly", (alpha,))]
    if gamma != 1:
        component_limits.append(_cached_length("log", (gamma,)))
    if delta != 0:
        component_limits.append(_cached_length("loglog", (delta,)))
    return max(final, min(component_limits))


def _truncated_convolution(left, right, length):
    """Return the first ``length`` terms of a linear convolution.

    SciPy's full convolution result is a view when sliced, so returning a
    slice directly would keep the complete ``len(left) + len(right) - 1``
    array alive.  Copying the requested prefix lets the full temporary be
    released before another large convolution is started.
    """
    full = convolve(left, right, mode='full')
    prefix = np.array(full[:length], copy=True)
    del full
    return prefix


@cache
def poly_coeffs(n, alpha=-1 / 2):
    """
    Returns the first n+1 coefficients in the Taylor expansion of
        f(x) = (1-x)^alpha
    """
    name = f"poly_{n}_{alpha}.npy"
    if config['cache']:
        try:
            with open(os.path.join("seq_cache", name), 'rb') as stream:
                return np.load(stream)[:n + 1]
        except FileNotFoundError:
            cached = _cached_prefix("poly", n, alpha)
            if cached is not None:
                return cached

    coefficients = np.empty(n + 1, dtype=float)
    coefficients[0] = 1.0
    for k in range(1, n + 1):
        coefficients[k] = coefficients[k - 1] * (1 - (1 + alpha) / k)
    if config['cache']:
        with open(os.path.join("seq_cache", name), 'wb') as stream:
            np.save(stream, coefficients)
    return coefficients


def poly_coeff_single(n, alpha=-1 / 2):
    """
    Computes the coefficient of x^n in the Taylor expansion of
        f(x) = (1-x)^alpha
    """
    coefficient = 1.0
    for k in range(1, n + 1):
        coefficient *= 1 - (1 + alpha) / k
    return coefficient


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
                    return np.load(f)[:n + 1]
            except FileNotFoundError:
                cached = _cached_prefix("log", n, gamma)
                if cached is not None:
                    return cached

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
                return np.load(f)[:n + 1]
        except FileNotFoundError:
            cached = _cached_prefix("loglog", n, delta)
            if cached is not None:
                return cached
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
    # python-flint 0.6 accepts a Python list here.  Release the NumPy source
    # array before constructing the Arb series so both representations do not
    # remain live throughout the expensive power operation.
    coeff_list = np.asarray(a_coeffs).tolist()
    del a_coeffs
    flint.ctx.cap = len(coeff_list)
    # TODO: might be interesting to track precision
    series = flint.arb_series(coeff_list)
    del coeff_list
    result = series.__pow__(p)
    del series
    return np.array(result.coeffs()).astype(float)


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
        coeffs = (np.pow(-1, ks) * binom(gamma, ks) *
                  float(mp.gamma(-alpha)) *
                  np.asarray(rgamma_derivatives(max(ks), alpha),
                             dtype=float))
        return coeffs @ log_powers
    else:
        ind_part = (float(mp.gamma(-alpha)) *
                    np.asarray(rgamma_derivatives(max(ks), alpha),
                               dtype=float))
        log_powers = np.array(
            [np.power(np.log(ns) * np.log(np.log(ns)), -k) for k in ks])

        if delta == int(delta):
            raise ValueError("delta must not be in the set 1,2,3,...")

        x_values = np.log(np.log(ns))
        funcs = [_expansion_function(int(k)) for k in ks]
        eks = [fk(x_values, gamma, delta) for fk in funcs]

        # [fk.evalf(subs={
        # u: 0,
        # x: np.log(np.log(n))
        # }) for n in ns] for fk in funcs],
        # dtype=float)

        log_powers *= eks

        return ind_part @ log_powers


_ASYM_U = sym.Symbol('u')
_ASYM_X = sym.Symbol('x')
_ASYM_GAMMA = sym.Symbol('gamma')
_ASYM_DELTA = sym.Symbol('delta')


@cache
def _expansion_function(k):
    """Return a cached numerical evaluator for the kth delta correction."""
    base = ((1 - _ASYM_X * _ASYM_U)**_ASYM_GAMMA *
            (1 + sym.log(1 - _ASYM_X * _ASYM_U) / _ASYM_X)**_ASYM_DELTA)
    expression = sym.diff(base, _ASYM_U, k).subs(_ASYM_U, 0) / factorial(k)
    return sym.lambdify((_ASYM_X, _ASYM_GAMMA, _ASYM_DELTA), expression,
                        modules='numpy')


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



if __name__ == "__main__":
    main()
