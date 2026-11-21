"""Randomized quantile residuals for discrete GAMs.

Randomized quantile residuals (Dunn & Smyth, 1996) turn the discrete
response of a Poisson model into a continuous, standard-normal residual.
For an observation with count ``y`` and expected count ``mu``, a point is
drawn uniformly from the jump interval of the Poisson cumulative
distribution function at ``y`` and mapped through the inverse standard
normal CDF.

The uniform draw for each row is a deterministic function of a stable
observation identifier and a user seed, so reordering the rows or
processing the data in batches reproduces exactly the same residual for
the same observation.  Extreme tails are evaluated in log probability
space and are never flattened against a fixed quantile cap.
"""

import hashlib

import numpy as np
from scipy.special import ndtri_exp
from scipy.stats import poisson

__all__ = [
    "QuantileResidualResult",
    "RegionTailSummary",
    "poisson_randomized_quantile_residuals",
    "tail_exceedance_by_region",
]


def _inverse_log_normal_tail(log_p):
    """Return positive ``z`` solving ``log(Phi(-z)) = log_p`` element-wise.

    Uses the log-domain inverse normal CDF while it stays finite and falls
    back to a Mills-ratio fixed point beyond scipy's representable range,
    so the most extreme tail probabilities still keep differentiated,
    finite quantiles instead of collapsing to infinity.
    """
    log_p = np.asarray(log_p, dtype=np.float64)
    # log of a tail probability must be strictly negative; values at zero
    # only arise from rounding when the probability is indistinguishable
    # from one, in which case the corresponding tail is vanishingly small.
    log_p = np.minimum(log_p, -np.finfo(np.float64).tiny)
    z = -ndtri_exp(log_p)
    overflow = ~np.isfinite(z)
    if not overflow.any():
        return z

    # Mills ratio: log Phi(-z) ~ -z^2/2 - log z - 0.5 log(2 pi),
    # solved by fixed point iteration on z.
    z[overflow] = np.sqrt(np.maximum(-2.0 * log_p[overflow], 1.0))
    for _ in range(100):
        previous = z[overflow]
        z[overflow] = np.sqrt(
            -2.0 * (log_p[overflow] + np.log(previous) + 0.5 * np.log(2.0 * np.pi))
        )
        if np.all(np.abs(z[overflow] - previous) <= 1e-12 * z[overflow]):
            break
    return z


def _splitmix64(z):
    """Apply the SplitMix64 finalizer to a uint64 array element-wise."""
    z = np.asarray(z, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _uint64_to_open_unit_interval(bits):
    """Map uint64 values to uniform doubles strictly inside ``(0, 1)``."""
    # take the leading 53 random bits and shift onto (0, 1)
    floats = (bits >> np.uint64(11)).astype(np.float64) + 0.5
    return floats / float(1 << 53)


def _row_uniforms(obs_ids, seed):
    """Draw one stable uniform in ``(0, 1)`` for each observation id.

    The mapping is a pure function of ``(obs_id, seed)`` and contains no
    loop-carried state, so rows can be reordered or batched freely.

    Parameters
    ----------
    obs_ids : array-like of shape (n, )
        stable per-row identifiers. integer and floating point ids are
        hashed arithmetically; other dtypes (e.g. strings) are hashed with
        SHA-256.  ids must be unique.
    seed : int
        non-negative integer seed.

    Returns
    -------
    uniforms : np.array of shape (n, ) with values strictly in ``(0, 1)``
    """
    ids = np.asarray(obs_ids)
    if ids.ndim != 1:
        raise ValueError(
            f"obs_ids must be 1-dimensional, but found {ids.ndim} dimensions"
        )
    if ids.shape[0] != np.unique(ids).shape[0]:
        raise ValueError("obs_ids must be unique, but found duplicate identifiers")

    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, but found {seed!r}")
    seed_mix = _splitmix64(np.uint64(int(seed)))

    if np.issubdtype(ids.dtype, np.integer):
        # reinterpret signed ids by their bit pattern so negative ids hash too
        base = ids.astype(np.int64).view(np.uint64) ^ seed_mix
        bits = _splitmix64(base)
    elif np.issubdtype(ids.dtype, np.floating):
        base = ids.astype(np.float64).view(np.uint64) ^ seed_mix
        bits = _splitmix64(base)
    else:
        bits = np.empty(ids.shape[0], dtype=np.uint64)
        for i, obs_id in enumerate(ids.tolist()):
            digest = hashlib.sha256(f"{seed}|{obs_id!r}".encode()).digest()
            bits[i] = int.from_bytes(digest[:8], byteorder="little", signed=False)
        bits = _splitmix64(bits ^ seed_mix)

    uniforms = _uint64_to_open_unit_interval(bits)
    # defensive clamp: the construction already guarantees an open interval
    return np.minimum(
        np.maximum(uniforms, np.finfo(np.float64).tiny), 1.0 - np.finfo(np.float64).eps
    )


def _as_count_array(y, n, name="y"):
    """Validate and flatten a non-negative integer count vector."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, but found {y.shape[0]}")
    if not np.isfinite(y).all():
        raise ValueError(f"{name} must not contain NaN or infinite values")
    if np.any(y < 0.0) or np.any(y != np.floor(y)):
        raise ValueError(f"{name} must contain non-negative integer counts")
    return y


def poisson_randomized_quantile_residuals(y, mu, exposure=None, obs_ids=None, seed=0):
    r"""Compute randomized quantile residuals under a Poisson count model.

    For each row the expected count is the model rate multiplied by the
    row exposure.  Given the observed count ``y`` and expected count
    ``mu``, a uniform point ``u`` is drawn from the jump interval of the
    Poisson CDF at ``y``,

    .. math::

        F(\\mu; y-1) < u \\le F(\\mu; y),

    and the residual is ``Phi^{-1}(u)`` where ``Phi`` is the standard
    normal CDF.  All interval endpoints and the inverse normal transform
    are evaluated in log probability space, so observations in either
    extreme tail keep a finite, differentiated residual instead of being
    flattened against a quantile cap.

    Parameters
    ----------
    y : array-like of shape (n, )
        observed non-negative integer counts.
    mu : array-like of shape (n, )
        expected counts.  when ``exposure`` is given these are the model
        rates (per unit exposure); otherwise they are the expected counts
        directly.
    exposure : array-like of shape (n, ) or None, default: None
        per-row exposure.  ``None`` is equivalent to an exposure of one
        for every row.  rows with zero exposure and zero count receive a
        ``NaN`` residual and are reported separately; zero exposure with a
        non-zero count raises a ``ValueError``.
    obs_ids : array-like of shape (n, ) or None, default: None
        stable per-row identifiers controlling the uniform draws.
        ``None`` uses the row positions ``0, ..., n-1``.  the residual for
        a given id is unaffected by row ordering or batching.
    seed : int, default: 0
        non-negative integer seed combined with every observation id.

    Returns
    -------
    result : QuantileResidualResult
        object holding the per-row residuals and bookkeeping for rows
        without exposure.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    n = mu.shape[0]
    if not np.isfinite(mu).all():
        raise ValueError("mu must not contain NaN or infinite values")
    if np.any(mu <= 0.0):
        raise ValueError("mu must contain strictly positive expected counts/rates")

    y = _as_count_array(y, n)

    if exposure is None:
        exposure = np.ones(n, dtype=np.float64)
    else:
        exposure = np.asarray(exposure, dtype=np.float64).ravel()
        if exposure.shape[0] != n:
            raise ValueError(
                f"exposure must have length {n}, but found {exposure.shape[0]}"
            )
        if not np.isfinite(exposure).all() or np.any(exposure < 0.0):
            raise ValueError("exposure must contain finite, non-negative values")

    if obs_ids is None:
        obs_ids = np.arange(n, dtype=np.int64)
    else:
        obs_ids = np.asarray(obs_ids).ravel()
        if obs_ids.shape[0] != n:
            raise ValueError(
                f"obs_ids must have length {n}, but found {obs_ids.shape[0]}"
            )

    zero_exposure = exposure == 0.0
    if np.any(zero_exposure & (y > 0.0)):
        bad = np.flatnonzero(zero_exposure & (y > 0.0))
        raise ValueError(
            "non-zero counts with zero exposure are impossible under a Poisson "
            f"count model; offending row indices: {bad.tolist()}"
        )

    residuals = np.full(n, np.nan, dtype=np.float64)
    valid = ~zero_exposure
    excluded = np.flatnonzero(zero_exposure).astype(np.int64)

    if np.any(valid):
        expected_counts = mu[valid] * exposure[valid]
        counts = y[valid].astype(np.int64)

        # randomized uniforms, stable under reordering and batching
        uniforms_all = _row_uniforms(obs_ids, seed)
        uniforms = uniforms_all[valid]

        # endpoints of the Poisson CDF jump interval, in log space.
        # logpmf stays accurate even when logcdf/logsf underflow to -inf.
        log_lower = poisson.logcdf(counts - 1, expected_counts)
        log_upper = poisson.logcdf(counts, expected_counts)
        log_width = poisson.logpmf(counts, expected_counts)

        # build the randomized point in log probability space, branching on
        # which tail the interval sits in, so neither extreme tail is
        # truncated to a fixed quantile.
        lower_tail = log_upper <= np.log(0.5)
        residuals_valid = np.empty(counts.shape[0], dtype=np.float64)

        if np.any(lower_tail):
            # log of u = F(y-1) + U * (F(y) - F(y-1)); when F(y-1)
            # underflowed to -inf the first term is exactly zero.
            lt = lower_tail
            addition = np.log(uniforms[lt]) + log_width[lt]
            log_u = np.where(
                np.isneginf(log_lower[lt]),
                addition,
                np.logaddexp(log_lower[lt], addition),
            )
            residuals_valid[lt] = -_inverse_log_normal_tail(log_u)

        if np.any(~lower_tail):
            # upper tail: work on the survival side, where v = 1 - u is
            # uniform on (S(y), S(y-1)) and S(y-1) - S(y) = pmf(y),
            # so v = S(y) + U * pmf(y).
            ut = ~lower_tail
            log_sf_upper = poisson.logsf(counts[ut], expected_counts[ut])
            log_pmf_ut = log_width[ut]
            underflow = np.isneginf(log_sf_upper)
            # when S(y) underflowed its term is exactly zero, so avoid
            # evaluating the -inf + finite expression altogether
            log_v = np.empty(ut.sum(), dtype=np.float64)
            log_v[underflow] = np.log(uniforms[ut][underflow]) + log_pmf_ut[underflow]
            finite_sf = ~underflow
            log_v[finite_sf] = log_sf_upper[finite_sf] + np.log1p(
                uniforms[ut][finite_sf]
                * np.exp(log_pmf_ut[finite_sf] - log_sf_upper[finite_sf])
            )
            residuals_valid[ut] = _inverse_log_normal_tail(log_v)

        residuals[valid] = residuals_valid

    return QuantileResidualResult(
        residuals=residuals,
        excluded_zero_exposure=excluded,
        expected_counts=np.where(zero_exposure, 0.0, mu * exposure),
        obs_ids=obs_ids,
    )


def tail_exceedance_by_region(result, regions, weights=None, tail_level=0.05):
    """Weighted upper- and lower-tail exceedance fractions by region.

    For each region the fraction of residual weight outside the symmetric
    standard-normal thresholds ``z = ±Phi^{-1}(1 - tail_level / 2)`` is
    computed separately for the lower and upper tail.  Rows excluded for
    zero exposure (residual ``NaN``) never enter the denominators.

    Parameters
    ----------
    result : QuantileResidualResult
        output of :func:`poisson_randomized_quantile_residuals`.
    regions : array-like of shape (n, )
        region label of every row; numeric or string labels are allowed.
    weights : array-like of shape (n, ) or None, default: None
        row weights used in the weighted fractions.  ``None`` weights each
        valid row by its expected count (model rate times exposure); pass an
        array of ones for unweighted fractions.
    tail_level : float, default: 0.05
        two-sided tail level; each tail cuts at ``tail_level / 2``.
        must be strictly between 0 and 1.

    Returns
    -------
    summaries : dict[str | int | float, RegionTailSummary]
        one :class:`RegionTailSummary` per region plus an ``"ALL"`` entry
        pooling every valid row.
    """
    if not 0.0 < tail_level < 1.0:
        raise ValueError(
            f"tail_level must be strictly between 0 and 1, but found {tail_level}"
        )

    residuals = np.asarray(result.residuals, dtype=np.float64).ravel()
    n = residuals.shape[0]
    regions = np.asarray(regions).ravel()
    if regions.shape[0] != n:
        raise ValueError(f"regions must have length {n}, but found {regions.shape[0]}")

    if weights is None:
        weights = np.asarray(result.expected_counts, dtype=np.float64).ravel()
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.shape[0] != n:
        raise ValueError(f"weights must have length {n}, but found {weights.shape[0]}")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("weights must contain finite, non-negative values")

    threshold = float(_inverse_log_normal_tail(np.array([np.log(tail_level / 2.0)]))[0])
    valid = np.isfinite(residuals)

    summaries = {}
    labels = list(np.unique(regions))

    for label in labels:
        mask = valid & (regions == label)

        row_count = int(np.count_nonzero(mask))
        total_weight = float(weights[mask].sum())
        excluded_count = int(np.count_nonzero(~valid & (regions == label)))

        if total_weight > 0.0:
            lower_weight = float(weights[mask & (residuals <= -threshold)].sum())
            upper_weight = float(weights[mask & (residuals >= threshold)].sum())
            lower_fraction = lower_weight / total_weight
            upper_fraction = upper_weight / total_weight
        else:
            lower_fraction = np.nan
            upper_fraction = np.nan

        summaries[label] = RegionTailSummary(
            region=label,
            n_rows=row_count,
            total_weight=total_weight,
            lower_tail_exceedance=lower_fraction,
            upper_tail_exceedance=upper_fraction,
            n_excluded_zero_exposure=excluded_count,
        )

    all_total_weight = float(weights[valid].sum())
    if all_total_weight > 0.0:
        all_lower = (
            float(weights[valid & (residuals <= -threshold)].sum()) / all_total_weight
        )
        all_upper = (
            float(weights[valid & (residuals >= threshold)].sum()) / all_total_weight
        )
    else:
        all_lower = np.nan
        all_upper = np.nan

    summaries["ALL"] = RegionTailSummary(
        region="ALL",
        n_rows=int(np.count_nonzero(valid)),
        total_weight=all_total_weight,
        lower_tail_exceedance=all_lower,
        upper_tail_exceedance=all_upper,
        n_excluded_zero_exposure=int(np.count_nonzero(~valid)),
    )

    return summaries


class RegionTailSummary:
    """Weighted tail-exceedance statistics for one region.

    Parameters
    ----------
    region : str or int or float
        region label.
    n_rows : int
        number of valid rows (with finite residuals) in the region.
    total_weight : float
        sum of the row weights over valid rows.
    lower_tail_exceedance : float
        weighted fraction of rows with residual below the lower threshold.
    upper_tail_exceedance : float
        weighted fraction of rows above the upper threshold.
    n_excluded_zero_exposure : int
        number of rows excluded because they had zero exposure and zero count.
    """

    def __init__(
        self,
        region,
        n_rows,
        total_weight,
        lower_tail_exceedance,
        upper_tail_exceedance,
        n_excluded_zero_exposure=0,
    ):
        self.region = region
        self.n_rows = n_rows
        self.total_weight = total_weight
        self.lower_tail_exceedance = lower_tail_exceedance
        self.upper_tail_exceedance = upper_tail_exceedance
        self.n_excluded_zero_exposure = n_excluded_zero_exposure

    def to_dict(self):
        """Return the summary as a plain dictionary."""
        return {
            "region": self.region,
            "n_rows": self.n_rows,
            "total_weight": self.total_weight,
            "lower_tail_exceedance": self.lower_tail_exceedance,
            "upper_tail_exceedance": self.upper_tail_exceedance,
            "n_excluded_zero_exposure": self.n_excluded_zero_exposure,
        }

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __eq__(self, other):
        if isinstance(other, RegionTailSummary):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def __repr__(self):
        return (
            "RegionTailSummary("
            f"region={self.region!r}, n_rows={self.n_rows}, "
            f"total_weight={self.total_weight:.6g}, "
            f"lower_tail_exceedance={self.lower_tail_exceedance:.6g}, "
            f"upper_tail_exceedance={self.upper_tail_exceedance:.6g}, "
            f"n_excluded_zero_exposure={self.n_excluded_zero_exposure})"
        )


class QuantileResidualResult:
    """Per-row randomized quantile residuals and exposure bookkeeping.

    Parameters
    ----------
    residuals : np.array of shape (n, )
        standard-normal randomized quantile residuals in original row
        order.  rows with zero exposure are ``NaN``.
    excluded_zero_exposure : np.array of integer row indices
        rows that had zero exposure (and therefore zero count); these are
        not included in any region summary.
    expected_counts : np.array of shape (n, )
        model rate multiplied by exposure for every row (zero on the
        excluded rows).
    obs_ids : np.array of shape (n, )
        observation identifiers used to generate the uniform draws.
    """

    def __init__(self, residuals, excluded_zero_exposure, expected_counts, obs_ids):
        self.residuals = np.asarray(residuals, dtype=np.float64).ravel()
        self.excluded_zero_exposure = np.asarray(
            excluded_zero_exposure, dtype=np.int64
        ).ravel()
        self.expected_counts = np.asarray(expected_counts, dtype=np.float64).ravel()
        self.obs_ids = np.asarray(obs_ids).ravel()

    @property
    def valid_mask(self):
        """Boolean mask of rows that enter the residual summaries."""
        return np.isfinite(self.residuals)

    def to_dict(self):
        """Return the result as a plain dictionary."""
        return {
            "residuals": self.residuals,
            "excluded_zero_exposure": self.excluded_zero_exposure,
            "expected_counts": self.expected_counts,
            "obs_ids": self.obs_ids,
        }

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __repr__(self):
        n = self.residuals.shape[0]
        return (
            "QuantileResidualResult("
            f"n_rows={n}, "
            f"n_excluded_zero_exposure={self.excluded_zero_exposure.shape[0]})"
        )
