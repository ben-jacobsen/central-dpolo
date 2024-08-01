"""
    File: meb.py
    Author: Ben Jacobsen
    Purpose: Explore different strategies from approximating the minimum
        enclosing ball of matrices wrt the spectral norm
"""

import argparse

import matplotlib.pyplot as plt
import miniball
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", type=int, default=10)
    parser.add_argument("-m", type=int, default=25)
    parser.add_argument("-n", type=int, default=16)
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()

    gen = np.random.default_rng()
    T = args.T
    m = args.m
    n = args.n
    # mats = gen.multivariate_normal(mean=np.zeros(m),
    #                                cov=np.eye(m),
    #                                size=(n, T)).reshape(T, m, n)
    avg_norm_base = 0
    avg_norm_mean = 0
    avg_norm_meb = 0
    # avg_norm_lb = 0
    avg_norm_spec = 0
    for _ in tqdm(range(args.num_iters)):
        ps = 1 - gen.uniform(0, 1, size=(m, n))**4
        mats = gen.uniform(0, 1, size=(T, m, n))
        #mats = (mats > ps).astype(int)
        proj_mat = np.eye(m) - 1 / m**2 * np.ones(m) @ np.ones(m).T
        reduced_mats = proj_mat @ mats

        transpose = np.swapaxes(reduced_mats, 1, 2)
        am = arithmetic_mean(transpose).T
        fmeb = frobenius_meb(transpose).T
        # lb = strict_lb(mats)
        smeb = spectral_meb(reduced_mats, num_iters=300, debug=False)

        avg_norm_base += spectral_norm(reduced_mats) / args.num_iters
        avg_norm_mean += spectral_norm(reduced_mats - am) / args.num_iters
        avg_norm_meb += spectral_norm(reduced_mats - fmeb) / args.num_iters
        # avg_norm_lb += spectral_norm(proj_mat @ (mats - lb)) / args.num_iters
        avg_norm_spec += spectral_norm(reduced_mats - smeb) / args.num_iters

    print("Base:", avg_norm_base)
    print("AM:", avg_norm_mean)
    print("FMEB:", avg_norm_meb)
    # print(avg_norm_lb)
    print("SMEB:", avg_norm_spec)

    # bases = get_bases(mats)
    # basis = grassmann_common_subspace(bases)
    # P = basis @ basis.T

    # print("Centered")
    # print(spectral_norm(mats - arithmetic_mean(np.swapaxes(P @ mats, 1, 2)).T))
    # print(spectral_norm(mats - frobenius_meb(np.swapaxes(P @ mats, 1, 2)).T))


def strict_lb(mats):
    return np.min(mats, axis=0)


def spectral_norm(mats):
    return np.max(np.linalg.svdvals(mats))


def arithmetic_mean(mats):
    return np.average(mats, axis=0)


def harmonic_mean(mats):
    return np.linalg.pinv(np.average(np.linalg.pinv(mats), axis=0))


def frobenius_meb(mats):
    T, n, m = mats.shape
    return np.array(
        miniball.Miniball(np.array([mat.flatten()
                                    for mat in mats])).center()).reshape(n, m)


def get_bases(mats):
    return np.linalg.qr(mats).Q


def grassmann_common_subspace(bases, tol=1e-2):
    c = bases[0]
    t = 1
    err = tol + 1
    while err > tol:
        t += 1
        ds = [grassmann_distance(c, b) for b in bases]
        furthest = bases[np.argmax(ds)]
        U, D, V = np.linalg.svd(furthest.T @ c)
        s0 = c @ V.T
        s1 = furthest @ U
        theta = np.arccos(np.clip(D, 0, 1))
        delta = 1 / t
        gamma_del = np.diag(np.cos(delta * theta))
        sigma_del = np.diag(np.sin(delta * theta))
        gamma_1 = np.diag(np.cos(theta))
        sigma_1 = np.diag(np.sin(theta))

        c = s0 @ gamma_del + (
            s1 - s0 @ gamma_1) @ np.linalg.pinv(sigma_1) @ sigma_del

        ind = np.argpartition(ds, -2)[-2:]
        err = np.abs(ds[ind[0]] - ds[ind[1]])

    return c


def grassmann_distance(basis1, basis2):
    _, S, _ = np.linalg.svd(basis1.T @ basis2)
    pr_angles = np.arccos(np.clip(S, 0, 1))
    return np.sqrt(np.sum(pr_angles**2))


def spectral_meb(mats, num_iters=100, debug=False):
    """
    subgradient-based algorithm for exactly solving the spectral meb
    """
    x = arithmetic_mean(mats)  # initial guess
    best_x = np.copy(x)
    best_val = spectral_norm(mats - x)

    if debug:
        print("INIT:")
        print(best_val)

    for t in range(num_iters):
        step_size = 1 / np.sqrt(t + 1)

        U, D, V = np.linalg.svd(mats - x, full_matrices=False)
        ixs = np.argwhere(np.isclose(np.abs(D), np.max(np.abs(D))))
        for i, j in ixs:
            x += step_size * np.sign(D[i][j]) * np.outer(
                U[i, :, j], V[i, j, :])

        val = spectral_norm(mats - x)
        if val < best_val:
            best_val = val
            np.copyto(best_x, x)
        if debug:
            print(f"ITERATION {t+1}:")
            print(val)

    return best_x


if __name__ == "__main__":
    main()
