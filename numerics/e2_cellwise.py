#!/usr/bin/env python3
"""Cellwise numerical illustration for minimax calibration experiments.

Exact curves are Binomial mixtures of Beta order-statistic moments.  Monte
Carlo draws cell counts and conditional Beta variables only; no ``(X,Y)`` data
or score learner is simulated.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc
from scipy.stats import beta as beta_distribution
from scipy.stats import binom, norm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAPPAS = (1.0, 3.0, 9.0)
DEFAULT_K_VALUES = (23, 64, 256)
P_VALUES = (1, 2, 8)
TAIL_TOLERANCE = 1e-50
Z975 = float(norm.ppf(0.975))

K_COLORS = {23: "#9ecae1", 64: "#3182bd", 256: "#08519c", 1024: "#08306b"}
LINESTYLES = ("-", "--", "-.", ":")


def carrier_rank(n: np.ndarray | int, alpha: float) -> np.ndarray:
    """Return ``ceil((n+1)(1-alpha))`` without floating boundary errors."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    counts = np.asarray(n, dtype=np.int64)
    sample_size = counts + 1
    target = np.longdouble(1.0) - np.longdouble(alpha)
    product = sample_size.astype(np.longdouble) * target
    rank = np.ceil(product).astype(np.int64)

    # Correct products close enough to an integer that rounding could change
    # the ceiling.  Exact arithmetic is needed only on this sparse subset.
    nearest = np.rint(product)
    ambiguous = np.abs(product - nearest) <= 2.0 * np.spacing(np.abs(product))
    if np.any(ambiguous):
        numerator, denominator = alpha.as_integer_ratio()
        flat_size = sample_size.reshape(-1)
        flat_rank = rank.reshape(-1)
        for index in np.flatnonzero(ambiguous.reshape(-1)):
            size = int(flat_size[index])
            flat_rank[index] = size - (size * numerator) // denominator
    return rank


def _beta_shifted_moment(
    a: np.ndarray,
    total: np.ndarray,
    target: float,
    order: int,
) -> np.ndarray:
    """Compute ``E[(B-target)^order]`` by Beta-Stein recurrence."""
    if order < 0:
        raise ValueError("order must be nonnegative")
    a_float = np.asarray(a, dtype=float)
    total_float = np.asarray(total, dtype=float)
    previous = np.ones_like(a_float)
    if order == 0:
        return previous
    current = a_float / total_float - target
    if order == 1:
        return current
    for r in range(1, order):
        following = (
            r * target * (1.0 - target) * previous
            + (r * (1.0 - 2.0 * target) + a_float - total_float * target)
            * current
        ) / (total_float + r)
        previous, current = current, following
    return current


def _conditional_absolute_moments(
    n: np.ndarray,
    alpha: float,
    p_values: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Return ``r_n^(p)=E|B_{n,k_n}-(1-alpha)|^p`` for integer p."""
    counts = np.asarray(n, dtype=np.int64)
    target = 1.0 - alpha
    rank = carrier_rank(counts, alpha)
    interior = rank <= counts
    a = rank[interior]
    total = counts[interior] + 1
    output: dict[int, np.ndarray] = {}

    for p in p_values:
        if p != 1 and (p < 2 or p % 2):
            raise ValueError("exact moments support p=1 and positive even integers")
        conditional = np.full(counts.shape, alpha**p, dtype=float)
        if np.any(interior):
            if p == 1:
                b = counts[interior] + 1 - a
                mean = a / total
                value = (
                    2.0 * target * betainc(a, b, target)
                    - 2.0 * mean * betainc(a + 1, b, target)
                    + mean
                    - target
                )
            else:
                value = _beta_shifted_moment(a, total, target, p)
            conditional[interior] = np.maximum(value, 0.0)
        output[p] = conditional
    return output


def _lower_binomial_quantile(m: int, probability: float) -> int:
    log_tolerance = math.log(TAIL_TOLERANCE)
    lower = -1
    upper = min(m, max(0, int(math.ceil(m * probability))))
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if float(binom.logcdf(midpoint, m, probability)) >= log_tolerance:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _upper_binomial_quantile(m: int, probability: float) -> int:
    log_tolerance = math.log(TAIL_TOLERANCE)
    lower = min(m - 1, max(0, int(math.floor(m * probability))))
    step = max(1, int(math.ceil(math.sqrt(m * probability * (1.0 - probability)))))
    upper = min(m, lower + step)
    while upper < m and float(binom.logsf(upper, m, probability)) > log_tolerance:
        step *= 2
        upper = min(m, lower + step)
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if float(binom.logsf(midpoint, m, probability)) <= log_tolerance:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _truncated_binomial_support(
    m: int,
    probability: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    lower = _lower_binomial_quantile(m, probability)
    upper = _upper_binomial_quantile(m, probability)
    n = np.arange(lower, upper + 1, dtype=np.int64)
    weights = binom.pmf(n, m, probability)
    lower_tail = float(binom.cdf(lower - 1, m, probability)) if lower else 0.0
    upper_tail = float(binom.sf(upper, m, probability)) if upper < m else 0.0
    omitted = lower_tail + upper_tail
    mass_error = abs(float(np.sum(weights)) + omitted - 1.0)
    if omitted > 2.1 * TAIL_TOLERANCE:
        raise ArithmeticError("truncated Binomial support exceeds tail tolerance")
    if mass_error > 5e-11:
        raise ArithmeticError(f"truncated Binomial mass error {mass_error:.3e}")
    return n, weights, omitted


def exact_pth_means(
    alpha: float,
    kappa: float,
    K: int,
    m: int,
    p_values: Iterable[int] = P_VALUES,
) -> dict[int, dict[str, float]]:
    """Compute ``M_p={E[L_p^p]}^(1/p)`` by marginal Binomial--Beta identity."""
    requested = tuple(dict.fromkeys(int(p) for p in p_values))
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if kappa <= 0.0 or K < 1 or m < 1:
        raise ValueError("require kappa>0, K>=1, and m>=1")
    probability = 1.0 / ((1.0 + kappa) * K)
    n, weights, omitted = _truncated_binomial_support(m, probability)
    conditional = _conditional_absolute_moments(n, alpha, requested)
    cap = max(alpha, 1.0 - alpha)
    output: dict[int, dict[str, float]] = {}
    for p in requested:
        moment = float(np.dot(weights, conditional[p]))
        output[p] = {
            "moment": moment,
            "root_moment": moment ** (1.0 / p),
            "tail_probability": omitted,
            "moment_error_bound": omitted * cap**p,
        }
    return output


def _mean_and_se(total: float, total_square: float, size: int) -> tuple[float, float]:
    mean = total / size
    centered = max(total_square - size * mean * mean, 0.0)
    return mean, math.sqrt(centered / ((size - 1) * size))


def _keyed_rng(seed: int, kappa: float, K: int, m: int, tag: int) -> np.random.Generator:
    key = int(round(kappa * 1_000_000.0))
    return np.random.default_rng(np.random.SeedSequence([seed, key, K, m, tag]))


def cellwise_mc(
    *,
    alpha: float,
    kappa: float,
    K: int,
    m: int,
    repetitions: int,
    rng: np.random.Generator,
    p_values: Iterable[int] = P_VALUES,
    include_infinity: bool = False,
    batch_size: int | None = None,
) -> dict[int | str, dict[str, float]]:
    """Estimate ``E[L_p]`` from multinomial counts and conditional Beta draws."""
    requested = tuple(dict.fromkeys(int(p) for p in p_values))
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if kappa <= 0.0 or K < 1 or m < 1 or repetitions < 2:
        raise ValueError("require kappa>0, K>=1, m>=1, and repetitions>=2")
    if any(p < 1 for p in requested):
        raise ValueError("p values must be positive")
    if batch_size is None:
        batch_size = max(64, min(4096, 2_000_000 // K))

    one_plus_kappa = 1.0 + kappa
    probability = 1.0 / (one_plus_kappa * K)
    probabilities = np.r_[
        np.full(K, probability), 1.0 - 1.0 / one_plus_kappa
    ]
    lp_sum = {p: 0.0 for p in requested}
    lp_square_sum = {p: 0.0 for p in requested}
    power_sum = {p: 0.0 for p in requested}
    power_square_sum = {p: 0.0 for p in requested}
    infinity_sum = 0.0
    infinity_square_sum = 0.0
    target = 1.0 - alpha

    generated = 0
    while generated < repetitions:
        batch = min(batch_size, repetitions - generated)
        counts = rng.multinomial(m, probabilities, size=batch)[:, :K]
        rank = carrier_rank(counts, alpha)
        pit = np.ones(counts.shape, dtype=float)
        interior = rank <= counts
        pit[interior] = rng.beta(
            rank[interior], counts[interior] + 1 - rank[interior]
        )
        losses = np.abs(pit - target)

        for p in requested:
            profile_power = np.mean(losses**p, axis=1)
            profile_lp = profile_power ** (1.0 / p)
            lp_sum[p] += float(np.sum(profile_lp))
            lp_square_sum[p] += float(np.dot(profile_lp, profile_lp))
            power_sum[p] += float(np.sum(profile_power))
            power_square_sum[p] += float(np.dot(profile_power, profile_power))
        if include_infinity:
            profile_infinity = np.max(losses, axis=1)
            infinity_sum += float(np.sum(profile_infinity))
            infinity_square_sum += float(np.dot(profile_infinity, profile_infinity))
        generated += batch

    output: dict[int | str, dict[str, float]] = {}
    for p in requested:
        mean_lp, mean_lp_se = _mean_and_se(lp_sum[p], lp_square_sum[p], repetitions)
        mean_power, mean_power_se = _mean_and_se(
            power_sum[p], power_square_sum[p], repetitions
        )
        sample_root = mean_power ** (1.0 / p)
        if mean_lp > sample_root + 2e-14:
            raise AssertionError(f"sample Jensen inequality failed at p={p}")
        derivative = sample_root / (p * mean_power) if mean_power > 0.0 else 0.0
        output[p] = {
            "mean_lp": mean_lp,
            "se": mean_lp_se,
            "ci_low": max(0.0, mean_lp - Z975 * mean_lp_se),
            "ci_high": mean_lp + Z975 * mean_lp_se,
            "sample_moment": mean_power,
            "sample_moment_se": mean_power_se,
            "sample_root_moment": sample_root,
            "sample_root_se": derivative * mean_power_se,
        }
    if include_infinity:
        mean, se = _mean_and_se(infinity_sum, infinity_square_sum, repetitions)
        output["inf"] = {
            "mean_lp": mean,
            "se": se,
            "ci_low": max(0.0, mean - Z975 * se),
            "ci_high": mean + Z975 * se,
        }
    return output


def effective_validity_boundary(alpha: float) -> float:
    return alpha * (1.0 - alpha) / (4.0 * min(alpha, 1.0 - alpha) ** 2)


def fano_expected_constant(alpha: float, K: int) -> float:
    probability = max(0.0, 1.0 - 2.0 * math.exp(-K / 32.0))
    return probability * math.sqrt(alpha * (1.0 - alpha)) / 64.0


def _effective_grid(maximum: float, points: int) -> np.ndarray:
    return np.geomspace(0.5, maximum, points)


def compute_exact_grid(
    *,
    alpha: float,
    kappas: tuple[float, ...],
    K_values: tuple[int, ...],
    desired_effective: np.ndarray,
) -> dict[tuple[float, int], dict[str, object]]:
    data: dict[tuple[float, int], dict[str, object]] = {}
    for kappa in kappas:
        for K in K_values:
            product = (1.0 + kappa) * K
            m_grid = np.maximum(1, np.rint(product * desired_effective).astype(np.int64))
            values = {p: [] for p in P_VALUES}
            for m in m_grid:
                exact = exact_pth_means(alpha, kappa, K, int(m))
                for p in P_VALUES:
                    values[p].append(exact[p]["root_moment"])
            data[(kappa, K)] = {
                "m": m_grid,
                "x": m_grid / product,
                "root": {p: np.asarray(values[p]) for p in P_VALUES},
            }
    return data


def compute_mc_grid(
    *,
    alpha: float,
    kappa: float,
    K: int,
    desired_effective: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    product = (1.0 + kappa) * K
    m_grid = np.maximum(1, np.rint(product * desired_effective).astype(np.int64))
    values = {
        p: {"mean": [], "se": [], "low": [], "high": [], "sample_root": [], "sample_root_se": []}
        for p in P_VALUES
    }
    for m in m_grid:
        result = cellwise_mc(
            alpha=alpha,
            kappa=kappa,
            K=K,
            m=int(m),
            repetitions=repetitions,
            rng=_keyed_rng(seed, kappa, K, int(m), 2),
        )
        for p in P_VALUES:
            values[p]["mean"].append(result[p]["mean_lp"])
            values[p]["se"].append(result[p]["se"])
            values[p]["low"].append(result[p]["ci_low"])
            values[p]["high"].append(result[p]["ci_high"])
            values[p]["sample_root"].append(result[p]["sample_root_moment"])
            values[p]["sample_root_se"].append(result[p]["sample_root_se"])
    return {
        "m": m_grid,
        "x": m_grid / product,
        "values": {
            p: {name: np.asarray(entries) for name, entries in values[p].items()}
            for p in P_VALUES
        },
    }


def compute_mc_grids(
    *,
    alpha: float,
    kappas: tuple[float, ...],
    K_values: tuple[int, ...],
    desired_effective: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[tuple[float, int], dict[str, object]]:
    """Compute Monte Carlo curves on the full ``(kappa, K)`` grid."""
    return {
        (kappa, K): compute_mc_grid(
            alpha=alpha,
            kappa=kappa,
            K=K,
            desired_effective=desired_effective,
            repetitions=repetitions,
            seed=seed,
        )
        for kappa in kappas
        for K in K_values
    }


def compute_infinity_plateau(
    *,
    alpha: float,
    K_values: tuple[int, ...],
    effective_size: float,
    repetitions: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    kappa = 3.0
    means, errors, actual = [], [], []
    for K in K_values:
        product = (1.0 + kappa) * K
        m = max(1, int(round(product * effective_size)))
        x = m / product
        result = cellwise_mc(
            alpha=alpha,
            kappa=kappa,
            K=K,
            m=m,
            repetitions=repetitions,
            rng=_keyed_rng(seed, kappa, K, m, 3),
            p_values=(),
            include_infinity=True,
        )["inf"]
        actual.append(x)
        means.append(math.sqrt(x) * result["mean_lp"])
        errors.append(math.sqrt(x) * Z975 * result["se"])
    K_array = np.asarray(K_values, dtype=float)
    values = np.asarray(means)
    reference = np.sqrt(np.log(K_array))
    coefficient = float(np.dot(reference, values) / np.dot(reference, reference))
    return {
        "K": K_array,
        "effective": np.asarray(actual),
        "mean": values,
        "error": np.asarray(errors),
        "coefficient": coefficient,
    }


def _check_mc_against_exact(
    exact: dict[int, dict[str, float]],
    mc: dict[int | str, dict[str, float]],
    *,
    check_moment_identity: bool = True,
) -> float:
    absolute_tolerance = 5e-5
    worst_standardized_excess = -math.inf
    for p in P_VALUES:
        exact_root = exact[p]["root_moment"]
        excess = mc[p]["mean_lp"] - exact_root
        se = mc[p]["se"]
        excess_beyond_tolerance = excess - absolute_tolerance
        if excess_beyond_tolerance <= 0.0:
            standardized = 0.0
        else:
            standardized = excess_beyond_tolerance / max(se, np.finfo(float).tiny)
        worst_standardized_excess = max(worst_standardized_excess, standardized)
        if excess > 5.0 * se + absolute_tolerance:
            raise AssertionError(f"MC E[L_p] exceeds exact M_p at p={p}: z={standardized:.2f}")
        if check_moment_identity:
            moment_error = abs(mc[p]["sample_root_moment"] - exact_root)
            tolerance = 6.0 * mc[p]["sample_root_se"] + 3e-5
            if moment_error > tolerance:
                raise AssertionError(
                    f"MC pth moment misses exact identity at p={p}: "
                    f"error={moment_error:.3e}, tolerance={tolerance:.3e}"
                )
    return worst_standardized_excess


def run_checks() -> dict[str, float]:
    """Check exact moments, MC marginal identity, Jensen, and fixed seed."""
    alpha, kappa, K = 0.1, 3.0, 23
    if int(carrier_rank(np.array([2]), 0.3333333333333333)[0]) != 3:
        raise AssertionError("rank failed below a floating integral boundary")
    if int(carrier_rank(np.array([9]), np.nextafter(0.1, 0.0))[0]) != 10:
        raise AssertionError("rank failed for alpha immediately below 0.1")
    n = 31
    rank = int(carrier_rank(np.array([n]), alpha)[0])
    b = n + 1 - rank
    closed = _conditional_absolute_moments(np.array([n]), alpha, P_VALUES)
    for p in P_VALUES:
        numerical, _ = quad(
            lambda value: abs(value - (1.0 - alpha)) ** p
            * beta_distribution.pdf(value, rank, b),
            0.0,
            1.0,
            points=[1.0 - alpha],
            epsabs=2e-13,
        )
        if not math.isclose(float(closed[p][0]), numerical, rel_tol=2e-9, abs_tol=2e-13):
            raise AssertionError(f"conditional Beta moment fails quadrature at p={p}")

    m = int((1.0 + kappa) * K * 20)
    exact = exact_pth_means(alpha, kappa, K, m)
    mc = cellwise_mc(
        alpha=alpha,
        kappa=kappa,
        K=K,
        m=m,
        repetitions=5_000,
        rng=_keyed_rng(20260826, kappa, K, m, 7),
    )
    worst = _check_mc_against_exact(exact, mc)
    if abs(mc[1]["mean_lp"] - mc[1]["sample_root_moment"]) > 2e-14:
        raise AssertionError("p=1 Jensen equality failed")
    return {
        "worst_jensen_excess_beyond_tolerance_z": worst,
        "p2_exact": exact[2]["root_moment"],
        "p2_mc": mc[2]["mean_lp"],
    }


def _setup_plotting():
    cache = Path(tempfile.gettempdir()) / "global-cqr-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.titlesize": 9.3,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def make_figure(
    *,
    alpha: float,
    kappas: tuple[float, ...],
    K_values: tuple[int, ...],
    maximum_effective_size: float,
    points: int,
    repetitions: int,
    infinity_repetitions: int,
    infinity_effective_size: float,
    seed: int,
    output: Path,
) -> dict[str, float]:
    """Compute E2 and write vector PDF."""
    desired = _effective_grid(maximum_effective_size, points)
    exact = compute_exact_grid(
        alpha=alpha,
        kappas=kappas,
        K_values=K_values,
        desired_effective=desired,
    )
    mc = compute_mc_grids(
        alpha=alpha,
        kappas=kappas,
        K_values=K_values,
        desired_effective=desired,
        repetitions=repetitions,
        seed=seed,
    )
    linf_K = tuple(sorted(set(K_values + (1024,))))
    infinity = compute_infinity_plateau(
        alpha=alpha,
        K_values=linf_K,
        effective_size=infinity_effective_size,
        repetitions=infinity_repetitions,
        seed=seed,
    )

    worst_jensen_z = -math.inf
    raw_jensen_crossings = 0
    for key, simulation in mc.items():
        exact_data = exact[key]
        for index, _ in enumerate(simulation["m"]):
            one_exact = {
                p: {"root_moment": float(exact_data["root"][p][index])}
                for p in P_VALUES
            }
            one_mc = {
                p: {
                    "mean_lp": float(simulation["values"][p]["mean"][index]),
                    "se": float(simulation["values"][p]["se"][index]),
                    "sample_root_moment": float(
                        simulation["values"][p]["sample_root"][index]
                    ),
                    "sample_root_se": float(
                        simulation["values"][p]["sample_root_se"][index]
                    ),
                }
                for p in P_VALUES
            }
            raw_jensen_crossings += sum(
                one_mc[p]["mean_lp"] > one_exact[p]["root_moment"]
                for p in P_VALUES
            )
            worst_jensen_z = max(
                worst_jensen_z,
                _check_mc_against_exact(
                    one_exact,
                    one_mc,
                    check_moment_identity=False,
                ),
            )

    plt = _setup_plotting()
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(7.35, 5.75))
    grid = fig.add_gridspec(2, 3)
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.09,
        top=0.80,
        wspace=0.40,
        hspace=0.58,
    )
    ax_raw = fig.add_subplot(grid[0, :2])
    ax_infinity = fig.add_subplot(grid[0, 2])
    normalized_axes = tuple(fig.add_subplot(grid[1, index]) for index in range(3))

    fallback_colors = plt.cm.Blues(np.linspace(0.40, 0.90, len(K_values)))
    colors = {K: K_COLORS.get(K, fallback_colors[index]) for index, K in enumerate(K_values)}
    styles = {kappa: LINESTYLES[index % len(LINESTYLES)] for index, kappa in enumerate(kappas)}
    marker_values = ("o", "s", "^")
    markers = {
        kappa: marker_values[index % len(marker_values)]
        for index, kappa in enumerate(kappas)
    }
    for (kappa, K), data in exact.items():
        simulation = mc[(kappa, K)]
        ax_raw.plot(
            data["x"],
            data["root"][2],
            color="0.20",
            ls=styles[kappa],
            lw=1.05,
            alpha=0.62,
        )
        ax_raw.fill_between(
            simulation["x"],
            simulation["values"][2]["low"],
            simulation["values"][2]["high"],
            color=colors[K],
            alpha=0.08,
            linewidth=0.0,
        )
        ax_raw.plot(
            simulation["x"],
            simulation["values"][2]["mean"],
            ls="none",
            marker=markers[kappa],
            color=colors[K],
            markerfacecolor="white",
            markeredgewidth=0.65,
            ms=2.3,
            alpha=0.82,
        )

        for p, ax in zip(P_VALUES, normalized_axes):
            exact_scale = np.sqrt(data["x"])
            mc_scale = np.sqrt(simulation["x"])
            ax.plot(
                data["x"],
                exact_scale * data["root"][p],
                color="0.20",
                ls=styles[kappa],
                lw=1.0,
                alpha=0.62,
            )
            ax.fill_between(
                simulation["x"],
                mc_scale * simulation["values"][p]["low"],
                mc_scale * simulation["values"][p]["high"],
                color=colors[K],
                alpha=0.08,
                linewidth=0.0,
            )
            ax.plot(
                simulation["x"],
                mc_scale * simulation["values"][p]["mean"],
                ls="none",
                marker=markers[kappa],
                color=colors[K],
                markerfacecolor="white",
                markeredgewidth=0.60,
                ms=2.1,
                alpha=0.80,
            )

    boundary = effective_validity_boundary(alpha)
    if maximum_effective_size > boundary:
        floor_x = np.geomspace(
            boundary * (1.0 + 1e-8), maximum_effective_size, 250
        )
        for K in K_values:
            ax_raw.plot(
                floor_x,
                fano_expected_constant(alpha, K) / np.sqrt(floor_x),
                color=colors[K],
                lw=0.9,
                ls=(0, (1.0, 1.5)),
                alpha=0.75,
            )
    ax_raw.set_xscale("log")
    ax_raw.set_yscale("log")
    ax_raw.grid(which="major", color="0.90", lw=0.5)
    ax_raw.set_xlabel(r"effective cell size $\lambda=m/((1+\kappa)K)$")
    ax_raw.set_ylabel(r"exact $M_2$ and MC mean $L_2$")
    ax_raw.set_title(fr"Cellwise risk, $p=2$, $\alpha={alpha:g}$")

    panel_titles = (
        r"$p=1$",
        r"$p=2$",
        r"$p=8$",
    )
    for label, title, p, ax in zip(
        ("(b1)", "(b2)", "(b3)"), panel_titles, P_VALUES, normalized_axes
    ):
        ax.set_xscale("log")
        ax.grid(which="major", color="0.90", lw=0.5)
        ax.set_xlabel(r"effective cell size $\lambda$")
        ax.set_title(title)
        gaussian_moment = (
            2.0 ** (0.5 * p)
            * math.gamma(0.5 * (p + 1))
            / math.sqrt(math.pi)
        )
        plateau = math.sqrt(alpha * (1.0 - alpha)) * gaussian_moment ** (1.0 / p)
        ax.axhline(plateau, color="0.25", lw=0.85, ls=":", alpha=0.85)
        ax.text(-0.20, 1.04, label, transform=ax.transAxes, fontsize=9.3)
    normalized_axes[0].set_ylabel(
        r"exact $\sqrt{\lambda}M_p$ and MC mean $\sqrt{\lambda}L_p$"
    )

    K_array = infinity["K"]
    mean = infinity["mean"]
    error = infinity["error"]
    coefficient = float(infinity["coefficient"])
    ax_infinity.errorbar(
        K_array,
        mean,
        yerr=error,
        fmt="o",
        color="#5E3C99",
        markerfacecolor="white",
        capsize=2.0,
        lw=1.0,
        label="MC",
    )
    K_fit = np.geomspace(float(np.min(K_array)), float(np.max(K_array)), 200)
    ax_infinity.plot(
        K_fit,
        coefficient * np.sqrt(np.log(K_fit)),
        color="#E66101",
        lw=1.4,
        label=r"$c\sqrt{\log K}$ fit",
    )
    ax_infinity.set_xscale("log", base=2)
    ax_infinity.grid(which="major", color="0.90", lw=0.5)
    ax_infinity.set_xlabel(r"number of cells $K$")
    ax_infinity.set_ylabel(r"$\sqrt{\lambda}\,\mathbb{E}L_\infty$")
    ax_infinity.set_title(
        fr"$p=\infty$"
        + "\n"
        + fr"$\kappa=3$, $\lambda\approx{infinity_effective_size:g}$"
    )
    ax_infinity.legend(frameon=False, loc="upper left")

    ax_raw.text(-0.10, 1.04, "(a)", transform=ax_raw.transAxes, fontsize=9.5)
    ax_infinity.text(-0.22, 1.04, "(c)", transform=ax_infinity.transAxes, fontsize=9.5)
    quantity_handles = [
        Line2D([], [], color="0.2", lw=1.3, label=r"exact $M_p$"),
        Line2D(
            [], [], color=colors[K_values[len(K_values) // 2]], marker="o",
            markerfacecolor="white", lw=0,
            label=r"MC mean $L_p$ (95% CI band)",
        ),
        Line2D(
            [], [], color="0.25", lw=0.85, ls=":",
            label="Gaussian reference",
        ),
        Line2D(
            [], [], color=colors[K_values[len(K_values) // 2]],
            lw=1.0, ls=(0, (1.0, 1.5)), label="Fano lower bound",
        ),
    ]
    K_handles = [
        Line2D(
            [], [], color=colors[K], lw=2.2, label=fr"${K}$",
        )
        for K in K_values
    ]
    kappa_handles = [
        Line2D(
            [], [], color="0.35", lw=1.2, ls=styles[kappa],
            marker=markers[kappa], markerfacecolor="white",
            label=fr"${kappa:g}$",
        )
        for kappa in kappas
    ]
    fig.legend(
        handles=quantity_handles,
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.995),
        title="quantity",
        title_fontsize=7.4,
        columnspacing=0.8,
        handlelength=1.8,
    )
    fig.legend(
        handles=K_handles,
        frameon=False,
        ncol=len(K_handles),
        loc="upper center",
        bbox_to_anchor=(0.68, 0.995),
        title=r"$K$ (color)",
        title_fontsize=7.4,
        columnspacing=0.8,
        handlelength=1.5,
    )
    fig.legend(
        handles=kappa_handles,
        frameon=False,
        ncol=len(kappa_handles),
        loc="upper right",
        bbox_to_anchor=(0.99, 0.995),
        title=r"$\kappa$ (style/marker)",
        title_fontsize=7.4,
        columnspacing=0.8,
        handlelength=1.5,
    )
    # Keep nonzero strokes at least 1 pt after scaling to the SIAM text width.
    from matplotlib.collections import Collection
    from matplotlib.spines import Spine

    fig.canvas.draw()
    minimum_linewidth = 1.01 * fig.get_figwidth() / 6.151
    for artist in fig.findobj():
        if isinstance(artist, Line2D):
            if artist.get_linewidth() > 0:
                artist.set_linewidth(max(artist.get_linewidth(), minimum_linewidth))
            if artist.get_markeredgewidth() > 0:
                artist.set_markeredgewidth(
                    max(artist.get_markeredgewidth(), minimum_linewidth)
                )
        elif isinstance(artist, Collection):
            widths = np.asarray(artist.get_linewidths(), dtype=float)
            artist.set_linewidths(np.where(
                widths > 0, np.maximum(widths, minimum_linewidth), 0.0
            ))
        elif isinstance(artist, Spine):
            artist.set_linewidth(max(artist.get_linewidth(), minimum_linewidth))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return {
        "worst_jensen_excess_beyond_tolerance_z": worst_jensen_z,
        "raw_jensen_crossings": float(raw_jensen_crossings),
        "mc_grid_size": float(len(mc)),
        "linf_fit_coefficient": coefficient,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, nargs="+", default=list(DEFAULT_KAPPAS))
    parser.add_argument("--K", dest="K_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--linf-reps", type=int, default=10_000)
    parser.add_argument("--max-effective", type=float, default=5_000.0)
    parser.add_argument("--linf-effective", type=float, default=10_000.0)
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "figure" / "e2_cellwise.pdf")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kappas = tuple(dict.fromkeys(float(value) for value in args.kappa))
    K_values = tuple(dict.fromkeys(int(value) for value in args.K_values))
    if any(value <= 0.0 for value in kappas):
        raise ValueError("all kappa values must be positive")
    if any(value < 23 for value in K_values):
        raise ValueError("K must be at least 23 for the cellwise construction")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if args.points < 3 or args.max_effective <= 0.0 or args.linf_effective <= 0.0:
        raise ValueError("invalid grid parameters")
    if not args.quick and (args.reps < 10_000 or args.linf_reps < 10_000):
        raise ValueError("full E2 requires at least 10,000 repetitions per point")

    if args.check_only:
        diagnostics = run_checks()
        print("E2 checks PASS:", ", ".join(f"{key}={value:.6g}" for key, value in diagnostics.items()))
        return 0

    maximum = min(args.max_effective, 100.0) if args.quick else args.max_effective
    infinity_effective = min(args.linf_effective, 300.0) if args.quick else args.linf_effective
    points = min(args.points, 8) if args.quick else args.points
    repetitions = min(args.reps, 1_000) if args.quick else args.reps
    infinity_repetitions = min(args.linf_reps, 1_000) if args.quick else args.linf_reps
    started = time.perf_counter()
    diagnostics = make_figure(
        alpha=args.alpha,
        kappas=kappas,
        K_values=K_values,
        maximum_effective_size=maximum,
        points=points,
        repetitions=repetitions,
        infinity_repetitions=infinity_repetitions,
        infinity_effective_size=infinity_effective,
        seed=args.seed,
        output=args.output,
    )
    elapsed = time.perf_counter() - started
    print(
        f"wrote {args.output.resolve()} in {elapsed:.2f}s; "
        f"worst Jensen excess beyond tolerance z="
        f"{diagnostics['worst_jensen_excess_beyond_tolerance_z']:.2f}; "
        f"Linf fit c={diagnostics['linf_fit_coefficient']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
