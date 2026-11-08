"""Horizon calibration and sparse evaluation of the public mechanism API."""
import numpy as np
import pytest

import sequences as s
import singular_approximation as sa
from toeplitz_experiments import plotting_points


@pytest.fixture(autouse=True)
def no_coefficient_cache(monkeypatch):
    monkeypatch.setitem(sa.config, 'cache', False)


@pytest.mark.parametrize('factory', [
    lambda: s.Anytime(-0.5, -0.7, N=128),
    lambda: s.Opt(N=128),
    s.Independent,
    lambda: s.BinaryMechanism(128),
    lambda: s.SmoothBinary(128),
    lambda: s.DoublingTrick(2, 128),
    lambda: s.Hybrid(at=s.Independent()),
    lambda: s.Hybrid(at=s.Independent(), bounded=s.SmoothBinary(2)),
])
def test_sparse_matches_dense_and_preserves_order(factory):
    mechanism = factory()
    assert isinstance(mechanism, s.Mechanism)
    times = np.array([63, 1, 16, 2, 63, 128, 3, 7, 15, 31])
    expected = mechanism.noise_schedule(128)[times - 1]**2
    np.testing.assert_allclose(mechanism.variances_at(times), expected, rtol=3e-4)
    np.testing.assert_allclose(mechanism.variances_at(times[:4]), expected[:4], rtol=3e-4)
    assert mechanism.variances_at([]).shape == (0,)


def test_fixed_horizon_prefix_invariance():
    mechanism = s.Anytime(-0.5, -0.7, N=128)
    expected = mechanism.noise_schedule(16)
    np.testing.assert_allclose(mechanism.noise_schedule(128)[:16], expected, rtol=3e-4)


@pytest.mark.parametrize('times', [[0], [-1], [1.5], [np.inf], [[1]], [True], [129]])
def test_invalid_times(times):
    with pytest.raises(ValueError):
        s.Opt(N=128).variances_at(times)


def test_large_sparse_variance_matches_independent_coefficients():
    times = np.array([1024, 1025, 4096])
    mechanism = s.Anytime(-0.5, -0.7, N=8192, delta=0.2)
    right = sa.exact_convolution(8192, -0.7, delta=0.2)
    left = sa.exact_convolution(4096, 0.7, delta=-0.2)
    expected = np.sum(right**2) * np.cumsum(left**2)[times - 1]
    np.testing.assert_allclose(mechanism.variances_at(times), expected, rtol=3e-4)
    assert mechanism.size <= 2**20


def test_doubling_accumulates_each_finished_chunk_once():
    mechanism = s.DoublingTrick(2, 30)
    endpoint_variances = [s.Opt(N=n).variances_at([n])[0] for n in [2, 4, 8, 16]]
    np.testing.assert_allclose(mechanism.variances_at([2, 6, 14, 30]), np.cumsum(endpoint_variances))


def test_hybrid_does_not_recalibrate_supplied_bounded_object():
    bounded = s.Opt(N=4)
    mechanism = s.Hybrid(at=s.Independent(), bounded=bounded)
    assert np.all(np.isfinite(mechanism.variances_at([1, 30, 128])))
    assert bounded.N == 4
    assert bounded.size == 0


def test_plot_grid_is_sparse_and_includes_endpoints():
    times = plotting_points(2**40)
    assert times[0] == 1 and times[-1] == 2**40
    assert np.all(np.diff(times) > 0)


def test_hybrid_required_plot_times_include_segment_boundaries():
    hybrid = s.Hybrid(at=s.Independent(), init_chunk=2, ratio=2)
    np.testing.assert_array_equal(
        hybrid.required_plot_times(20),
        [1, 2, 3, 4, 6, 7, 8, 14, 15, 16, 20],
    )


def test_exponential_hybrid_required_plot_times():
    hybrid = s.Hybrid(at=s.Independent(), init_chunk=2, exponential=True)
    np.testing.assert_array_equal(
        hybrid.required_plot_times(30),
        [1, 2, 3, 4, 6, 7, 8, 22, 23, 24, 30],
    )


def test_plot_grid_unions_hybrid_boundaries():
    hybrid = s.Hybrid(at=s.Independent(), init_chunk=2, ratio=2)
    times = plotting_points(20, points=4, mechanisms=[hybrid])
    for boundary in [1, 3, 7, 15, 20]:
        assert boundary in times


