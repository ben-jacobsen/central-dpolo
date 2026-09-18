"""
    File: test_sequences.py
    Author: AUTHOR
    Purpose: PURPOSE
"""

from itertools import combinations

import mpmath as mp
import numpy as np
import pytest

import radial_sensitivity as rs
import sequences as s
import singular_approximation as sa
import toeplitz_experiments as te

RADIAL_N_VALUES = [2**n for n in [10, 14, 18]]
CONVERGENT_PARAMS = [(-0.5, -0.6), (-0.51, 0), (-0.51, 0.51)]
DIVERGENT_PARAMS = [(-c, -d) for c, d in CONVERGENT_PARAMS]
TEST_DPS = 40


def exact_coefficients(T, gamma, alpha=-1 / 2, delta=0):
    """
    Compute coefficients using singular_approximation.exact_convolution without
    reading or writing the on-disk sequence cache.

    Disabling the cache makes these tests independent of the contents of
    seq_cache/.
    """
    old_cache = sa.config['cache']
    sa.config['cache'] = False
    try:
        return sa.exact_convolution(T, gamma, alpha=alpha, delta=delta)
    finally:
        sa.config['cache'] = old_cache


def exact_partial_sum(N, gamma, alpha=-1 / 2, delta=0):
    """
    Inclusive sum of squared coefficients through z^N.
    """
    coeffs = exact_coefficients(N + 1, gamma, alpha=alpha, delta=delta)
    return np.sum(coeffs**2)


@pytest.mark.parametrize("N", RADIAL_N_VALUES)
@pytest.mark.parametrize("gamma, delta", CONVERGENT_PARAMS + DIVERGENT_PARAMS)
@pytest.mark.parametrize("r", [0.5, 0.9, 0.99])
def test_damped_norm_matches_exact_coefficients(N, gamma, delta, r):
    """
    Compare radial Parseval integration with a directly weighted coefficient
    sum through N.
    """
    r = mp.mpf(r)

    coeffs = exact_coefficients(N + 1, gamma, delta=delta)
    weights = np.power(float(r), 2 * np.arange(N + 1))
    expected = np.sum(weights * coeffs**2)

    with mp.workdps(TEST_DPS):
        actual = rs.damped_norm_sq(r, gamma, delta=delta)

    assert np.isclose(
        float(actual),
        expected,
        rtol=1e-6,
        atol=1e-12,
    ), (float(actual), expected)


@pytest.mark.parametrize("N", RADIAL_N_VALUES)
@pytest.mark.parametrize("gamma, delta", CONVERGENT_PARAMS + DIVERGENT_PARAMS)
def test_finite_upper_bound_bounds_exact_partial_sum(N, gamma, delta):
    """
    Check the Chernoff/radial inequality over exponentially spaced horizons
    against coefficients computed independently by exact_convolution.
    """
    # Keep r reasonably close to one without making r^(-2N) enormous.
    # The bound is valid for every r in (0,1), so this is only a numerical
    # convenience for the test.
    r = mp.exp(-mp.mpf(1) / (N + 1))
    expected = exact_partial_sum(N, gamma, delta=delta)

    with mp.workdps(TEST_DPS):
        bound = rs.finite_upper_bound(N, r, gamma, delta=delta)

    assert float(bound) >= expected


@pytest.mark.parametrize("N", RADIAL_N_VALUES)
@pytest.mark.parametrize("gamma, delta", CONVERGENT_PARAMS + DIVERGENT_PARAMS)
def test_partial_sum_matches_exact_convolution(N, gamma, delta):
    """
    Validate the Stehfest inverse-Laplace reconstruction over exponentially
    spaced horizons against explicit coefficient calculations.

    partial_sum(N, ...) is inclusive through z^N, hence the N + 1 passed to
    exact_convolution.

    Degree 12 empirically gives errors around 1e-6--1e-5 at small horizons
    and substantially smaller errors for moderate N.  The tolerance is
    deliberately numerical rather than machine-level.
    """
    expected = exact_partial_sum(N, gamma, delta=delta)

    with mp.workdps(TEST_DPS):
        actual = rs.partial_sum(N, gamma, delta=delta, degree=12)

    assert np.isclose(
        float(actual),
        expected,
        rtol=2e-4,
        atol=1e-8,
    )


def test_full_norm_convergence_region():
    """
    For alpha=-1/2 the full squared sensitivity is finite precisely for

        gamma < -1/2,

    or gamma=-1/2 with delta<-1/2.
    """
    assert mp.isinf(rs.full_norm_sq(-0.49, delta=-10))
    assert mp.isinf(rs.full_norm_sq(-0.50, delta=-0.49))

    with mp.workdps(TEST_DPS):
        value = rs.full_norm_sq(-0.50, delta=-0.60, tail_cutoff=100)

    assert mp.isfinite(value)
    assert value > 0


@pytest.mark.parametrize("gamma, delta", CONVERGENT_PARAMS)
@pytest.mark.parametrize("S", [2**n for n in range(10, 21, 2)])
def test_full_norm_tail_cutoff_stability(gamma, delta, S):
    """
    The endpoint-tail approximation should be more or less invariant
    to different choices for the tail cutoff
    """
    with mp.workdps(TEST_DPS):
        vals = [rs.full_norm_sq(gamma, delta=delta, tail_cutoff=S)]

    for (v1, v2) in combinations(vals, 2):
        assert np.isclose(
            float(v1),
            float(v2),
            rtol=2e-4,
            atol=1e-10,
        )


@pytest.mark.parametrize("N", RADIAL_N_VALUES)
def test_full_norm_is_above_finite_partial_sum(N):
    """
    Every finite coefficient sum must lie below the full squared sensitivity.
    Check this over the same exponentially spaced horizons used by the other
    radial tests.
    """
    gamma = -1.0
    delta = 0.0

    expected_lower_bound = exact_partial_sum(N, gamma, delta=delta)

    with mp.workdps(40):
        full = rs.full_norm_sq(gamma, delta=delta, tail_cutoff=150)

    assert float(full) >= expected_lower_bound
