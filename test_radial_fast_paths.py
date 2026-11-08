"""Numerical regressions for the damped norm's cheaper evaluation paths."""
import mpmath as mp
import numpy as np
import pytest

import radial_sensitivity as rs
import singular_approximation as sa


@pytest.mark.parametrize('alpha,gamma,delta', [
    (-0.5, -0.7, 0), (-0.5, -0.51, 0.51),
])
def test_real_integrand_matches_original_generating_function(alpha, gamma, delta):
    with mp.workdps(30):
        r = mp.mpf('0.93')
        reference = mp.quad(lambda theta: abs(sa.singular_function(
            r * mp.exp(1j * theta), gamma, alpha, delta))**2,
            [0, mp.pi / 3, mp.pi]) / mp.pi
        actual = rs.damped_norm_sq(r, gamma, alpha, delta)
        assert mp.almosteq(actual, reference, rel_eps=mp.mpf('1e-25'))


@pytest.mark.parametrize('radius', ['0.1', '0.999'])
def test_sqrt_norm_matches_independent_quadrature(radius):
    with mp.workdps(35):
        r = mp.mpf(radius)
        reference = mp.quad(lambda t: 1 / abs(1 - r * mp.exp(mp.j*t)),
                            [0, mp.pi/3, mp.pi]) / mp.pi
        actual = rs.damped_norm_sq(r, 0)
        assert mp.almosteq(actual, reference, rel_eps=mp.mpf('1e-25'))


@pytest.mark.parametrize('p', [20, 40])
def test_sqrt_partial_sum_matches_independent_laplace_reference(p):
    # This exercises the stable complementary-elliptic-integral path at large
    # horizons without requiring a particular quadrature implementation.
    with mp.workdps(60):
        x = mp.mpf(2**p) - mp.mpf('0.5')
        def reference_transform(q):
            return 2 * mp.ellipk(mp.exp(-q)) / (mp.pi * q)
        expected = mp.invertlaplace(reference_transform, x,
                                    method='stehfest', degree=24)
    actual = rs.partial_sum(2**p - 1, 0)
    assert np.isclose(float(actual), float(expected), rtol=1e-8)


def test_sqrt_partial_sum_matches_explicit_coefficients():
    count = 2**16
    coefficients = np.concatenate(([1.0], np.cumprod(
        1 - 0.5 / np.arange(1, count))))
    expected = np.sum(coefficients**2)
    assert np.isclose(float(rs.partial_sum(count - 1, 0)), expected, rtol=2e-6)
