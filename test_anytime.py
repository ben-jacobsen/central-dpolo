"""
    NAME: test_anytime.py
    AUTHOR: Ben Jacobsen
    PURPOSE: Verifies that the instance methods defined in sequences.py
      return the same answers that would have been obtained by directly
      calling the corresponding static method in radial_sensitivity.py
"""

import mpmath as mp
import numpy as np
import pytest

import radial_sensitivity as rs
import sequences as s

TEST_DPS = 40
STEEHFEST_RTOL = 2e-4
SMALL_N = [32, 64, 128]
LARGE_N = [2**n for n in [8, 12, 16]]
CONVERGENT_PARAMS = [(-0.5, -0.6), (-0.51, 0), (-0.51, 0.51)]
DIVERGENT_PARAMS = [(-c, -d) for c, d in CONVERGENT_PARAMS]


def left_parameters(alpha, gamma, delta):
    """Parameters of L(z) when L(z) R(z) = 1/(1-z)."""
    return -1 - alpha, -gamma, -delta


def radial_right_norm(k, alpha, gamma, delta):
    """l2 norm of the first k right coefficients."""
    if k <= 0:
        return 0.0
    return float(mp.sqrt(rs.partial_sum(k - 1, gamma, alpha, delta)))


def radial_left_norm(k, alpha, gamma, delta):
    """l2 norm of the first k left coefficients."""
    if k <= 0:
        return 0.0
    alpha_L, gamma_L, delta_L = left_parameters(alpha, gamma, delta)
    return float(mp.sqrt(rs.partial_sum(k - 1, gamma_L, alpha_L, delta_L)))


@pytest.mark.parametrize("N", [8, 32, 128, 512])
@pytest.mark.parametrize("gamma, delta", CONVERGENT_PARAMS)
def test_anytime_finite_sensitivity_matches_radial_small_N(N, gamma, delta):
    """
    For small N, Anytime computes sensitivity from exact coefficients while
    radial_sensitivity uses inverse Laplace reconstruction.  They should agree
    to the numerical accuracy expected from degree-12 Stehfest inversion.

    N is a number of coefficients/timesteps in Anytime, so the corresponding
    inclusive index passed to radial_sensitivity.partial_sum is N - 1.
    """
    alpha = -0.5
    at = s.Anytime(alpha, gamma, N=N, delta=delta)

    with mp.workdps(TEST_DPS):
        expected = radial_right_norm(N, alpha, gamma, delta)
        actual = at.sensitivity()

    assert np.isclose(actual, expected, rtol=STEEHFEST_RTOL, atol=1e-10)


@pytest.mark.parametrize("N", LARGE_N)
@pytest.mark.parametrize("gamma, delta", CONVERGENT_PARAMS)
def test_anytime_finite_sensitivity_matches_radial_large_N(N, gamma, delta):
    """
    Above the exact-coefficient threshold, Anytime itself delegates to
    radial_sensitivity.  This test is therefore intentionally strict: its main
    purpose is to catch argument and off-by-one mistakes in that delegation.
    """
    alpha = -0.5
    at = s.Anytime(alpha, gamma, N=N, delta=delta)

    with mp.workdps(TEST_DPS):
        expected = radial_right_norm(N, alpha, gamma, delta)
        actual = at.sensitivity()

    assert np.isclose(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("gamma, delta", CONVERGENT_PARAMS)
def test_anytime_infinite_sensitivity_matches_full_norm(gamma, delta):
    alpha = -0.5
    at = s.Anytime(alpha, gamma, N=mp.inf, delta=delta)

    with mp.workdps(TEST_DPS):
        expected = float(mp.sqrt(rs.full_norm_sq(gamma, alpha, delta)))
        actual = at.sensitivity()

    assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("k", [8, 32, 128, 512])
@pytest.mark.parametrize("gamma, delta", CONVERGENT_PARAMS)
def test_anytime_standard_error_matches_radial_small_k(k, gamma, delta):
    """
    The non-cumulative standard error is the l2 norm of the first k left
    coefficients.  At small k, Anytime obtains this from exact coefficients;
    compare it against the inverse-Laplace reference calculation.
    """
    alpha = -0.5
    at = s.Anytime(alpha, gamma, N=k, delta=delta)

    with mp.workdps(TEST_DPS):
        expected = radial_left_norm(k, alpha, gamma, delta)
        actual = at.standard_error(k, cumulative=False)

    assert np.isclose(actual, expected, rtol=STEEHFEST_RTOL, atol=1e-10)


def test_anytime_standard_error_matches_radial_large_k():
    """
    This exercises the radial_sensitivity branch of standard_error directly.
    It is particularly useful for detecting accidental use of the right-hand
    parameters in place of the reciprocal left-hand parameters.
    """
    alpha = -0.5
    gamma = -0.8
    delta = 1.1
    k = 2**11
    at = s.Anytime(alpha, gamma, N=k, delta=delta)

    with mp.workdps(TEST_DPS):
        expected = radial_left_norm(k, alpha, gamma, delta)
        actual = at.standard_error(k, cumulative=False)

    assert np.isclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_anytime_cumulative_standard_error_matches_radial_checkpoints():
    """
    Check several entries of the cumulative error curve against independent
    partial-sum evaluations.  Only a few checkpoints are used because each
    inverse Laplace transform is relatively expensive.
    """
    alpha = -0.5
    gamma = -0.51
    delta = 0.51
    k = max(SMALL_N)
    checkpoints = np.array(SMALL_N) - 1

    at = s.Anytime(alpha, gamma, N=k, delta=delta)
    actual = at.standard_error(k, cumulative=True)

    with mp.workdps(TEST_DPS):
        expected = np.array([
            radial_left_norm(int(j + 1), alpha, gamma, delta)
            for j in checkpoints
        ])

    assert np.allclose(
        actual[checkpoints],
        expected,
        rtol=STEEHFEST_RTOL,
        atol=1e-10,
    )


def test_anytime_smooth_sensitivity_matches_radial_checkpoints():
    """Check the right-hand cumulative norm curve at selected times."""
    alpha = -0.5
    gamma = 0.51
    delta = -0.51
    k = max(SMALL_N)
    checkpoints = np.array(SMALL_N) - 1

    at = s.Anytime(alpha, gamma, N=k, delta=delta)
    actual = at.smooth_sensitivity(k)

    with mp.workdps(TEST_DPS):
        expected = np.array([
            radial_right_norm(int(j + 1), alpha, gamma, delta)
            for j in checkpoints
        ])

    assert np.allclose(
        actual[checkpoints],
        expected,
        rtol=STEEHFEST_RTOL,
        atol=1e-10,
    )


def test_anytime_noise_schedule_endpoint_matches_radial():
    """
    At time k, the ordinary schedule should be horizon sensitivity times the
    left norm through k, while the smooth schedule should use the right norm
    only through k.
    """
    alpha = -0.5
    gamma = -0.51
    delta = 0.51
    N = 2**10
    k = 2**6

    at = s.Anytime(alpha, gamma, N=N, delta=delta)

    with mp.workdps(TEST_DPS):
        left = radial_left_norm(k, alpha, gamma, delta)
        horizon_right = radial_right_norm(N, alpha, gamma, delta)
        local_right = radial_right_norm(k, alpha, gamma, delta)

    ordinary = at.noise_schedule(k, smooth=False)[-1]
    smooth = at.noise_schedule(k, smooth=True)[-1]

    assert np.isclose(
        ordinary,
        horizon_right * left,
        rtol=3e-4,
        atol=1e-10,
    )
    assert np.isclose(
        smooth,
        local_right * left,
        rtol=3e-4,
        atol=1e-10,
    )


def test_anytime_init_optimized_uses_optimizer_result(monkeypatch):
    """
    init_optimized should simply transfer the optimized gamma and delta into
    the returned Anytime instance.  Mocking the expensive optimizer makes this
    a pure API/parameter-mapping test.
    """
    expected_gamma = -1.25
    expected_delta = 1.75

    def fake_optimize_parameters(n_hat, N):
        assert n_hat == 123
        assert N == 456
        return {
            "gamma": expected_gamma,
            "delta": expected_delta,
        }

    monkeypatch.setattr(rs, "optimize_parameters", fake_optimize_parameters)

    at = s.Anytime.init_optimized(123, N=456, short_name=False)

    assert isinstance(at, s.Anytime)
    assert at.N == 456
    assert at.alpha == -0.5
    assert at.gamma == expected_gamma
    assert at.delta == expected_delta
