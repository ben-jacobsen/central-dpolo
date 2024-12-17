"""
    File: anytime_matrix.py
    Author: Ben Jacobsen
    Purpose: Implements and evaluates an anytime version of the matrix method
        for continual counting from 
        
        https://epubs.siam.org/doi/pdf/10.1137/1.9781611977554.ch183
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=int, default=10)
    args = parser.parse_args()
    T = args.T

    column = first_column(T)
    """
    for offset in range(args.T):
        variances = variance_schedule(column, offset=offset)
        print(f"{offset}: {privacy_schedule(column, variances)[-1]}")
    """
    variances = variance_schedule(column)
    print(column[-5:])
    print(variances[-5:])
    print(privacy_schedule(column, variances)[-5:])
    print(final_noise_variance(column, variances)[-5:])


def first_column(T):
    out = [1]
    for k in range(1, T):
        out.append((1 - 1 / (2 * k)) * out[-1])
    return out


def variance(t, k):
    return np.power(np.log(1 + t), 1 + k)


def variance_schedule(column, k=1, eta2=1, offset=0):
    T = len(column)
    steps = np.arange(1 + offset, T + 1 + offset)
    return eta2 * variance(steps, k)


def privacy_schedule(column, variances):
    mu2s = np.power(column, 2) / variances
    return np.sqrt(np.cumsum(mu2s))


def final_noise_variance(column, variances):
    T = len(column)
    return np.convolve(np.power(column, 2), variances)[:T]


if __name__ == "__main__":
    main()
