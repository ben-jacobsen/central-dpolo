"""
    File: toeplitz_experiments.py
    Author: Ben Jacobsen
    Purpose: Playing around with different matrix factorizations
"""

import argparse
import os
from functools import cache
from itertools import pairwise
from time import perf_counter

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

sns.set_style("darkgrid")
sns.set_context('paper')
sns.set(rc={"lines.linewidth": 2.5, "figure.dpi": 160, "savefig.dpi": 160})
sns.set(font_scale=0.85)

fontsize = 8
HYBRID_PLOT_MULTIPLIER = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', default=2**16, type=int)
    parser.add_argument('--N', type=float, default=np.inf,
                        help='privacy horizon for Anytime (default: inf)')
    parser.add_argument('--plot-points', type=int, default=100,
                        help='maximum number of logarithmically spaced plot points')
    parser.add_argument('--n-hats', default=[2**12], type=int, nargs='*')
    parser.add_argument('--gammas', default=[], type=float, nargs='*')
    parser.add_argument('--deltas', default=[0], type=float, nargs='*')
    parser.add_argument('--no-cache', action="store_true")
    parser.add_argument('--tol', type=float, default=0)
    parser.add_argument('--coefficients', action="store_true")
    parser.add_argument('--variance', action="store_true")
    parser.add_argument('--verbose', action="store_true",
                        help='print timing updates while building and plotting')

    args = parser.parse_args()
    T = args.T
    if T < 1 or args.plot_points < 2:
        parser.error('T must be positive and --plot-points must be at least 2')
    if not args.variance and not args.coefficients:
        parser.error('select --variance and/or --coefficients')
    N = args.N
    if np.isnan(N) or N < T or (np.isfinite(N) and not N.is_integer()):
        parser.error('N must be an integer at least T, or inf')
    N = int(N) if np.isfinite(N) else np.inf
    if any(n < 1 or n > N for n in args.n_hats):
        parser.error('each n-hat must be positive and no greater than N')
    sa.config['cache'] = not args.no_cache

    if not args.no_cache and not os.path.isdir('seq_cache'):
        os.mkdir('seq_cache')

    strategies = []

    def add_strategy(strategy, started=None):
        if started is None:
            started = perf_counter()
        strategies.append(strategy)
        if args.verbose:
            print(f"Added strategy {strategy.name} ({perf_counter() - started:.3f}s)",
                  flush=True)
    # strategies.append(seq.Hybrid(
    #     at= seq.Anytime(-1/2, -0.55, delta=0),
    #     w=0.25, init_chunk=2, exponential=False))
    add_strategy(seq.Hybrid(
        at= seq.Independent(),
        w=0.25, init_chunk=2, exponential=False))
    # strategies.append(seq.Anytime(-1/2, -0.51, delta=0,
                                  # tol=1e-4, asym_order=6))
    for gamma in args.gammas:
        for delta in args.deltas:
            add_strategy(
                seq.Anytime(-1 / 2,
                            gamma,
                            N=N,
                            delta=delta,
                            tol=args.tol,
                            asym_order=4))

    for n_hat in args.n_hats:
        started = perf_counter()
        strategy = seq.Anytime.init_optimized(
            n_hat, N=N, cache=not args.no_cache,
            kwargs={'tol': args.tol, 'asym_order': 4})
        add_strategy(strategy, started=started)
    # strategies.append(seq.BinaryMechanism(T))
    # strategies.append(seq.Hybrid(
    #     at= seq.Anytime(-1/2, -0.55, delta=0),
    #     bounded = seq.SmoothBinary(2),
    #     w=0.25, init_chunk=2, exponential=False))
    add_strategy(seq.SmoothBinary(T))
    # strategies.append(seq.DoublingTrick(100, T, exponential=True))

    # Opt always goes last
    baseline = seq.Opt(N=T)
    # baseline.name = f"Sqrt decomposition (N={T})"
    add_strategy(baseline)


    num_plots = args.variance + args.coefficients

    fig = plt.figure(layout='constrained')
    # orig_axs = fig.subplot_mosaic([[str(j) for j in range(num_plots)] + ["legend"]],
    #                               width_ratios = [1.35] * num_plots + [1])
    # orig_axs['legend'].axis("off")

    if num_plots > 0:
        orig_axs = fig.subplot_mosaic([
            [str(j) for j in range(num_plots)],
            ["legend"] * num_plots,
            ],
            height_ratios=[1, 0.3])
        orig_axs['legend'].axis("off")
    else:
        orig_axs = {}


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
        variance_plot(T, strategies, kwarg_list, orig_axs[str(i)],
                      points=args.plot_points, verbose=args.verbose)
        i += 1
    if args.coefficients:
        coefficient_plot(T, strategies, kwarg_list, orig_axs[str(i)],
                         points=args.plot_points, verbose=args.verbose)


    # handles, labels = axs[0, 1].get_legend_handles_labels()
    # for ax in axs.reshape(-1):
    #     ax.get_legend().remove()
    # fig.legend(handles, labels, loc='outside right upper')
    # plt.legend()
    # fig.set_size_inches(w=3.25, h=2.5)
    h, l = orig_axs[str(0)].get_legend_handles_labels()
    k = int(np.log2(T))
    for i in range(num_plots):
        orig_axs[str(i)].set_xticks([2**j for j in range(0, k+1, max(1, k//4))])
        try:
            orig_axs[str(i)].get_legend().remove()
        except AttributeError:
            pass
    orig_axs["legend"].legend(h, l, ncol=3, loc='center', frameon=False,
                              fontsize=fontsize, borderpad=0.1, handlelength=1.5,
                              handletextpad=0.3,
                              borderaxespad=0)
    bbox_px = orig_axs["legend"].get_legend().get_window_extent()
    dpi = fig.dpi
    # legend_height = max(bbox_px.height, 360)
    # legend_width = bbox_px.width
    # fig.set_size_inches(w = (num_plots * legend_height + legend_width + 50 ) / dpi,
    #                     h= legend_height/dpi)
    width = 2.5 * (args.variance + args.coefficients)
    fig.set_size_inches(w=width, h=2.75)
    plt.show()



def plotting_points(T, points=100, mechanisms=()):
    """One-based sample times, with memory independent of the horizon."""
    if T < 1 or points < 2:
        raise ValueError("T must be positive and points must be at least 2")
    samples = np.rint(np.geomspace(1, T, min(T, points))).astype(np.int64)
    required = [mechanism.required_plot_times(T) for mechanism in mechanisms]
    if required:
        samples = np.concatenate((samples, *required))
    return np.unique(samples)



def coefficient_plot(T, strategies, kwargs, ax, relative=False, points=100,
                     verbose=False):
    ax.set_xscale('log', base=2)
    ax.set_xlabel("t", fontsize=8, labelpad=0)
    times = plotting_points(T, points, strategies)


    view = times - 1
    for strategy, kw in zip(strategies, kwargs):
        if isinstance(strategy, seq.Sequence):
            l = strategy.first_k_left(T)
            r = strategy.first_k(T)

            if relative:
                base = seq.Opt().first_k(T)
                l = l / base
                r = r / base
                ax.set_yscale('linear')
                # ax.set_title('Ratio')
                ax.set_ylabel("Coefficient Ratio", fontsize=8, labelpad=0)
            else:
                ax.set_yscale('log', base=2)
                # ax.set_title('Matrix Coefficients')
                ax.set_ylabel("Matrix Coefficients", fontsize=8, labelpad=0)


            sns.lineplot(x=times, y=r[view], ax=ax, **kw)
            sns.lineplot(x=times,
                         y=l[view],
                         ax=ax,
                         # label=strategy.name + " (L)",
                         # ls='-.',
                         color=kw['color'])
            if verbose:
                print(f"Plotted coefficients for {strategy.name} "
                      f"({len(times)} points)", flush=True)


def variance_plot(T, strategies, kwargs, ax, points=100, verbose=False):
    ax.set_xscale('log', base=2)
    ax.set_xlabel("t", fontsize=fontsize, labelpad=0)
    ax.set_ylabel("Variance", fontsize=fontsize, labelpad=0)
    # ax.set_title('Mechanism Variance')
    # ax.set_title('Variance')
    for strategy, kw in zip(strategies, kwargs):
        strategy_points = points
        if isinstance(strategy, seq.Hybrid):
            strategy_points *= HYBRID_PLOT_MULTIPLIER
        times = plotting_points(T, strategy_points, [strategy])
        started = perf_counter()
        variances = strategy.variances_at(times)
        sns.lineplot(x=times, y=variances,
                     ax=ax, **kw)
        if verbose:
            print(f"Plotted variance for {strategy.name} "
                  f"({len(times)} points, {perf_counter() - started:.3f}s)",
                  flush=True)




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
