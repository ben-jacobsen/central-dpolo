"""
    File: FILENAME
    Author: AUTHOR
    Purpose: PURPOSE
"""

import mpmath as mp
import numpy as np
from tqdm import tqdm

import sequences as s
import singular_approximation as sa
import toeplitz_experiments as te

gen = np.random.default_rng()
NUM_ITERS = 25


def test_sensitivity_polynomial():
    for _ in tqdm(range(NUM_ITERS)):
        alpha = -1 / 2 + gen.random()
        gamma = 0
        delta = 0

        at = s.Anytime(alpha, gamma, delta)

        sensitivity = at.sensitivity()
        empirical_lb = at.smooth_sensitivity(1000)[-1]

        assert sensitivity >= empirical_lb
        assert mp.almosteq(sensitivity,
                           sa.compute_sensitivity_direct(alpha, gamma, delta),
                           rel_eps=1 + 1e-10)


def test_sensitivity_log():
    for _ in tqdm(range(NUM_ITERS)):
        alpha = -1 / 2 + gen.random()
        gamma = 2 * gen.random() - 1
        delta = 0

        at = s.Anytime(alpha, gamma, delta)

        sensitivity = at.sensitivity()
        empirical_lb = at.smooth_sensitivity(1000)[-1]

        assert sensitivity >= empirical_lb
        assert mp.almosteq(sensitivity,
                           sa.compute_sensitivity_direct(alpha, gamma, delta),
                           rel_eps=1 + 1e-10)


def test_sensitivity_iterated_log():
    for _ in tqdm(range(NUM_ITERS)):
        alpha = -1 / 2 + gen.random()
        gamma = 2 * gen.random() - 1
        delta = 2 * gen.random() - 1

        at = s.Anytime(alpha, gamma, delta)

        sensitivity = at.sensitivity()
        empirical_lb = at.smooth_sensitivity(1000)[-1]

        assert sensitivity >= empirical_lb
        assert mp.almosteq(sensitivity,
                           sa.compute_sensitivity_direct(alpha, gamma, delta),
                           rel_eps=1 + 1e-10)
