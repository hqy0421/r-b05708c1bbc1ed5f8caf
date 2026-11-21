"""Tests for randomized quantile residuals of Poisson count models."""

import numpy as np
import pytest
from scipy.stats import norm, poisson

from pygam import PoissonGAM
from pygam.residuals import (
    QuantileResidualResult,
    RegionTailSummary,
    _row_uniforms,
    poisson_randomized_quantile_residuals,
    tail_exceedance_by_region,
)


@pytest.fixture
def poisson_sample():
    rng = np.random.default_rng(42)
    n = 5000
    mu = rng.uniform(0.2, 30.0, n)
    y = rng.poisson(mu)
    return y, mu


def test_residuals_lie_inside_poisson_cdf_jump_intervals(poisson_sample):
    y, mu = poisson_sample
    result = poisson_randomized_quantile_residuals(y, mu, seed=7)
    u = norm.cdf(result.residuals)
    lower = poisson.cdf(y - 1, mu)
    upper = poisson.cdf(y, mu)
    assert np.all(u > lower - 1e-12)
    assert np.all(u <= upper + 1e-12)


def test_residuals_are_standard_normal_under_correct_model(poisson_sample):
    y, mu = poisson_sample
    result = poisson_randomized_quantile_residuals(y, mu, seed=7)
    assert result.residuals.shape == y.shape
    assert np.isfinite(result.residuals).all()
    assert abs(result.residuals.mean()) < 0.05
    assert abs(result.residuals.std() - 1.0) < 0.05


def test_residuals_stable_under_reordering(poisson_sample):
    y, mu = poisson_sample
    n = len(y)
    result = poisson_randomized_quantile_residuals(y, mu, seed=3)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    shuffled = poisson_randomized_quantile_residuals(
        y[perm], mu[perm], obs_ids=perm, seed=3
    )
    assert np.allclose(shuffled.residuals[np.argsort(perm)], result.residuals)


def test_residuals_stable_under_batching(poisson_sample):
    y, mu = poisson_sample
    n = len(y)
    ids = np.arange(n)
    full = poisson_randomized_quantile_residuals(y, mu, obs_ids=ids, seed=11)

    cut = n // 3
    part_one = poisson_randomized_quantile_residuals(
        y[:cut], mu[:cut], obs_ids=ids[:cut], seed=11
    )
    part_two = poisson_randomized_quantile_residuals(
        y[cut:], mu[cut:], obs_ids=ids[cut:], seed=11
    )
    batched = np.concatenate([part_one.residuals, part_two.residuals])
    assert np.allclose(batched, full.residuals)


def test_default_obs_ids_match_row_positions(poisson_sample):
    y, mu = poisson_sample
    explicit = poisson_randomized_quantile_residuals(
        y, mu, obs_ids=np.arange(len(y)), seed=5
    )
    default = poisson_randomized_quantile_residuals(y, mu, seed=5)
    assert np.allclose(explicit.residuals, default.residuals)


def test_different_seeds_give_different_draws(poisson_sample):
    y, mu = poisson_sample
    first = poisson_randomized_quantile_residuals(y, mu, seed=0).residuals
    second = poisson_randomized_quantile_residuals(y, mu, seed=1).residuals
    assert not np.allclose(first, second)


def test_string_obs_ids_are_supported_and_stable(poisson_sample):
    y, mu = poisson_sample
    ids = np.array([f"device-{i:05d}" for i in range(len(y))])
    first = poisson_randomized_quantile_residuals(y, mu, obs_ids=ids, seed=2)
    second = poisson_randomized_quantile_residuals(y, mu, obs_ids=ids, seed=2)
    assert np.allclose(first.residuals, second.residuals)
    assert np.isfinite(first.residuals).all()


def test_uniforms_are_inside_open_unit_interval():
    uniforms = _row_uniforms(np.arange(1000), seed=9)
    assert uniforms.shape == (1000,)
    assert np.all(uniforms > 0.0)
    assert np.all(uniforms < 1.0)


def test_duplicate_obs_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        poisson_randomized_quantile_residuals(
            y=np.array([0, 1]), mu=np.array([1.0, 1.0]), obs_ids=np.array([7, 7])
        )


def test_extreme_tails_keep_finite_differentiated_residuals():
    # lower tail: zero count under huge exposure
    mild = poisson_randomized_quantile_residuals(
        y=np.array([0]), mu=np.array([1000.0]), obs_ids=[0]
    ).residuals[0]
    extreme = poisson_randomized_quantile_residuals(
        y=np.array([0]), mu=np.array([1e6]), obs_ids=[0]
    ).residuals[0]
    ultra = poisson_randomized_quantile_residuals(
        y=np.array([0]), mu=np.array([1e8]), obs_ids=[0]
    ).residuals[0]
    assert np.isfinite([mild, extreme, ultra]).all()
    assert mild > extreme > ultra

    # upper tail: counts far above the expectation, beyond scipy's
    # representable log survival range
    upper_mild = poisson_randomized_quantile_residuals(
        y=np.array([5000]), mu=np.array([1000.0]), obs_ids=[0]
    ).residuals[0]
    upper_extreme = poisson_randomized_quantile_residuals(
        y=np.array([8000]), mu=np.array([1000.0]), obs_ids=[0]
    ).residuals[0]
    assert np.isfinite([upper_mild, upper_extreme]).all()
    assert upper_extreme > upper_mild


def test_zero_exposure_zero_count_is_excluded():
    result = poisson_randomized_quantile_residuals(
        y=np.array([0, 3, 2]),
        mu=np.array([0.5, 0.5, 0.5]),
        exposure=np.array([0.0, 1.0, 2.0]),
        obs_ids=[10, 11, 12],
    )
    assert np.isnan(result.residuals[0])
    assert np.isfinite(result.residuals[1:]).all()
    assert result.excluded_zero_exposure.tolist() == [0]
    assert result.expected_counts.tolist() == [0.0, 0.5, 1.0]
    assert result.valid_mask.tolist() == [False, True, True]


def test_zero_exposure_nonzero_count_raises():
    with pytest.raises(ValueError, match="zero exposure"):
        poisson_randomized_quantile_residuals(
            y=np.array([1]),
            mu=np.array([0.5]),
            exposure=np.array([0.0]),
            obs_ids=[0],
        )


def test_exposure_scales_expected_counts(poisson_sample):
    y, mu = poisson_sample
    exposure = np.linspace(0.5, 5.0, len(y))
    result = poisson_randomized_quantile_residuals(y, mu, exposure=exposure, seed=1)
    assert np.allclose(
        result.expected_counts[result.valid_mask], (mu * exposure)[result.valid_mask]
    )


@pytest.mark.parametrize(
    "bad_y, bad_mu",
    [
        (np.array([-1]), np.array([1.0])),
        (np.array([0.5]), np.array([1.0])),
        (np.array([1]), np.array([0.0])),
        (np.array([1]), np.array([np.nan])),
    ],
)
def test_input_validation(bad_y, bad_mu):
    with pytest.raises(ValueError):
        poisson_randomized_quantile_residuals(y=bad_y, mu=bad_mu, obs_ids=[0])


def test_seed_validation():
    with pytest.raises(ValueError):
        poisson_randomized_quantile_residuals(
            y=np.array([1]), mu=np.array([1.0]), seed=-1, obs_ids=[0]
        )
    with pytest.raises(ValueError):
        poisson_randomized_quantile_residuals(
            y=np.array([1]), mu=np.array([1.0]), seed=1.5, obs_ids=[0]
        )


def test_tail_exceedance_matches_manual_calculation(poisson_sample):
    y, mu = poisson_sample
    result = poisson_randomized_quantile_residuals(y, mu, seed=4)
    rng = np.random.default_rng(1)
    regions = rng.integers(0, 3, len(y))
    weights = rng.uniform(0.1, 5.0, len(y))
    threshold = norm.ppf(0.975)

    summaries = tail_exceedance_by_region(
        result, regions, weights=weights, tail_level=0.05
    )

    assert set(summaries) == {0, 1, 2, "ALL"}
    for label in [0, 1, 2]:
        mask = regions == label
        total = weights[mask].sum()
        lower = weights[mask & (result.residuals <= -threshold)].sum() / total
        upper = weights[mask & (result.residuals >= threshold)].sum() / total
        assert np.isclose(summaries[label].lower_tail_exceedance, lower)
        assert np.isclose(summaries[label].upper_tail_exceedance, upper)
        assert summaries[label].n_rows == mask.sum()

    assert summaries["ALL"].n_rows == len(y)
    assert np.isclose(
        summaries["ALL"].total_weight,
        sum(summaries[k].total_weight for k in [0, 1, 2]),
    )


def test_tail_exceedance_excludes_zero_exposure_rows():
    y = np.array([0, 5, 2, 8])
    mu = np.array([0.5, 4.0, 1.0, 6.0])
    exposure = np.array([0.0, 1.0, 1.0, 1.0])
    regions = np.array(["a", "a", "b", "b"])
    result = poisson_randomized_quantile_residuals(
        y, mu, exposure=exposure, obs_ids=np.arange(4)
    )
    summaries = tail_exceedance_by_region(result, regions, weights=np.ones(4))
    assert summaries["a"].n_rows == 1
    assert summaries["a"].n_excluded_zero_exposure == 1
    assert summaries["b"].n_rows == 2
    assert summaries["b"].n_excluded_zero_exposure == 0
    assert summaries["ALL"].n_rows == 3
    assert summaries["ALL"].n_excluded_zero_exposure == 1


def test_default_weights_are_expected_counts(poisson_sample):
    y, mu = poisson_sample
    exposure = np.linspace(1.0, 4.0, len(y))
    result = poisson_randomized_quantile_residuals(y, mu, exposure=exposure, seed=2)
    regions = np.full(len(y), "zone")
    summaries = tail_exceedance_by_region(result, regions)
    threshold = norm.ppf(0.975)
    weights = result.expected_counts
    mask = result.valid_mask
    expected = (
        weights[mask & (result.residuals >= threshold)].sum() / weights[mask].sum()
    )
    assert np.isclose(summaries["zone"].upper_tail_exceedance, expected)


def test_tail_level_validation(poisson_sample):
    y, mu = poisson_sample
    result = poisson_randomized_quantile_residuals(y, mu, seed=0)
    regions = np.zeros(len(y), dtype=int)
    with pytest.raises(ValueError):
        tail_exceedance_by_region(result, regions, tail_level=0.0)
    with pytest.raises(ValueError):
        tail_exceedance_by_region(result, regions, tail_level=1.0)


def test_summary_to_dict_and_repr():
    summary = RegionTailSummary(
        region="north",
        n_rows=10,
        total_weight=20.0,
        lower_tail_exceedance=0.1,
        upper_tail_exceedance=0.2,
    )
    assert summary.to_dict()["region"] == "north"
    assert summary["upper_tail_exceedance"] == 0.2
    assert "north" in repr(summary)

    result = QuantileResidualResult(
        residuals=np.array([np.nan, 0.5]),
        excluded_zero_exposure=np.array([0]),
        expected_counts=np.array([0.0, 2.0]),
        obs_ids=np.array([0, 1]),
    )
    assert result.valid_mask.tolist() == [False, True]
    assert "n_excluded_zero_exposure=1" in repr(result)


def test_poisson_gam_end_to_end():
    rng = np.random.default_rng(7)
    n = 800
    X = rng.uniform(0, 10, size=(n, 1))
    rate = np.exp(0.1 * X.ravel() - 1.0)
    exposure = rng.uniform(0.5, 3.0, n)
    y = rng.poisson(rate * exposure)
    regions = rng.choice(["north", "south"], size=n)

    gam = PoissonGAM().fit(X, y, exposure=exposure)
    result, summaries = gam.region_tail_exceedance(
        X, y, regions, exposure=exposure, obs_ids=np.arange(n), seed=0
    )

    assert isinstance(result, QuantileResidualResult)
    assert result.residuals.shape == (n,)
    assert np.isfinite(result.residuals).all()
    # expected counts come from the fitted model and the per-row exposure
    assert np.allclose(result.expected_counts, gam.predict(X, exposure=exposure))
    assert set(summaries) == {"north", "south", "ALL"}
    for summary in summaries.values():
        assert 0.0 <= summary.lower_tail_exceedance <= 1.0
        assert 0.0 <= summary.upper_tail_exceedance <= 1.0

    # method-level call agrees with the functional API
    direct = gam.randomized_quantile_residuals(
        X, y, exposure=exposure, obs_ids=np.arange(n), seed=0
    )
    assert np.allclose(direct.residuals, result.residuals)


def test_poisson_gam_residuals_require_fit():
    gam = PoissonGAM()
    with pytest.raises(AttributeError):
        gam.randomized_quantile_residuals(X=np.array([[1.0]]), y=np.array([1]))
