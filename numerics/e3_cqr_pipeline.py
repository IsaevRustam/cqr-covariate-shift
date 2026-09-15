#!/usr/bin/env python3
"""End-to-end CQR experiment under a known one-dimensional covariate shift.

Training and calibration covariates are uniform on [0, 1].  Target covariates
follow an exponential tilt whose chi-square divergence is calibrated to the
requested kappa.  Conditional coverage and interval length are integrated
exactly in Y through the Gaussian CDF; Monte Carlo is used only for source
training and calibration samples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy
from scipy.optimize import brentq, linprog
from scipy.sparse import csr_matrix, eye, hstack
from scipy.special import ndtr
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAPPAS = (1.0, 3.0)
DEFAULT_M_GRID = (128, 192, 256, 384, 512)
DEFAULT_N_GRID = (128, 256, 512, 1024, 2048)
DEFAULT_SEED = 20260826
Z975 = float(norm.ppf(0.975))

RULES = ("exact", "population", "unweighted")
METRICS = ("coverage", "length")
LEARNERS = ("mlp", "linear")


@dataclass(frozen=True)
class EndpointModel:
    learner: str
    parameters: tuple[np.ndarray, ...]

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(x, dtype=float).reshape(-1)
        if self.learner == "linear":
            (coefficients,) = self.parameters
            design = np.column_stack((np.ones_like(values), values))
            raw = design @ coefficients.T
        elif self.learner == "mlp":
            input_weight, hidden_bias, output_weight, skip_weight, output_bias = (
                self.parameters
            )
            standardized = 2.0 * values[:, None] - 1.0
            hidden = np.tanh(standardized @ input_weight + hidden_bias)
            raw = hidden @ output_weight + standardized @ skip_weight + output_bias
        else:
            raise ValueError(f"unknown learner {self.learner!r}")
        return np.minimum(raw[:, 0], raw[:, 1]), np.maximum(raw[:, 0], raw[:, 1])


@dataclass(frozen=True)
class TargetGrid:
    x: np.ndarray
    quadrature_weights: np.ndarray
    likelihood_ratio: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    oracle_low: np.ndarray
    oracle_high: np.ndarray
    oracle_length: np.ndarray
    nominal_coverage: float
    theta: float
    kappa: float


def conditional_mean(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * math.pi * np.asarray(x, dtype=float))


def conditional_scale(x: np.ndarray) -> np.ndarray:
    return 0.5 + 0.25 * np.cos(2.0 * math.pi * np.asarray(x, dtype=float))


def one_plus_chi_square(theta: float) -> float:
    """Return E_P[w_theta(X)^2] for P=Unif[0,1]."""
    if theta < 0.0:
        raise ValueError("theta must be nonnegative")
    if theta < 1e-6:
        return 1.0 + theta * theta / 12.0
    return 0.5 * theta / math.tanh(0.5 * theta)


def theta_from_kappa(kappa: float) -> float:
    """Solve (theta/2)coth(theta/2)=1+kappa for the positive tilt."""
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    upper = max(8.0, 2.5 * (1.0 + kappa))
    while one_plus_chi_square(upper) < 1.0 + kappa:
        upper *= 2.0
    return float(
        brentq(
            lambda value: one_plus_chi_square(value) - (1.0 + kappa),
            1e-9,
            upper,
            xtol=2e-14,
            rtol=2e-14,
        )
    )


def likelihood_ratio(x: np.ndarray, theta: float) -> np.ndarray:
    """Density q_theta/p, evaluated without forming exp(theta)."""
    values = np.asarray(x, dtype=float)
    return theta * np.exp(theta * (values - 1.0)) / (-math.expm1(-theta))


def split_rank(m: int, alpha: float) -> int:
    """Return ceil((m+1)(1-alpha)) using the CLI decimal value exactly."""
    if m < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("require m>=1 and alpha in (0,1)")
    alpha_fraction = Fraction(str(float(alpha)))
    target = (1 - alpha_fraction) * (m + 1)
    return (target.numerator + target.denominator - 1) // target.denominator


def fit_endpoint_model(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    learner: str,
    hidden_units: int,
    mlp_steps: int,
    rng: np.random.Generator,
) -> EndpointModel:
    """Fit source quantiles by pinball loss, then sort them at prediction."""
    values_x = np.asarray(x, dtype=float).reshape(-1)
    values_y = np.asarray(y, dtype=float).reshape(-1)
    if values_x.size != values_y.size or values_x.size < 2:
        raise ValueError("x and y must have the same length, at least two")
    if hidden_units < 1:
        raise ValueError("hidden_units must be positive")
    if learner == "mlp":
        if mlp_steps < 1:
            raise ValueError("mlp_steps must be positive")
        standardized = 2.0 * values_x[:, None] - 1.0
        taus = np.array([0.5 * alpha, 1.0 - 0.5 * alpha])
        input_weight = rng.normal(0.0, 1.25, size=(1, hidden_units))
        hidden_bias = rng.uniform(-1.0, 1.0, size=hidden_units)
        output_weight = rng.normal(0.0, 0.05, size=(hidden_units, 2))
        skip_weight = np.zeros((1, 2))
        output_bias = np.quantile(values_y, taus)
        parameters = [
            input_weight,
            hidden_bias,
            output_weight,
            skip_weight,
            output_bias,
        ]
        first_moment = [np.zeros_like(value) for value in parameters]
        second_moment = [np.zeros_like(value) for value in parameters]
        batch_size = min(256, values_x.size)
        beta1, beta2 = 0.9, 0.999
        weight_decay = 1e-5
        for step in range(1, mlp_steps + 1):
            if batch_size == values_x.size:
                index = slice(None)
            else:
                index = rng.integers(0, values_x.size, size=batch_size)
            batch_x = standardized[index]
            batch_y = values_y[index]
            hidden = np.tanh(batch_x @ input_weight + hidden_bias)
            prediction = hidden @ output_weight + batch_x @ skip_weight + output_bias
            residual = batch_y[:, None] - prediction
            prediction_gradient = ((residual < 0.0) - taus) / batch_y.size
            gradients = [
                batch_x.T
                @ ((prediction_gradient @ output_weight.T) * (1.0 - hidden * hidden))
                + weight_decay * input_weight,
                np.sum(
                    (prediction_gradient @ output_weight.T) * (1.0 - hidden * hidden),
                    axis=0,
                ),
                hidden.T @ prediction_gradient + weight_decay * output_weight,
                batch_x.T @ prediction_gradient + weight_decay * skip_weight,
                np.sum(prediction_gradient, axis=0),
            ]
            gradient_norm = math.sqrt(
                sum(float(np.sum(gradient * gradient)) for gradient in gradients)
            )
            if gradient_norm > 5.0:
                gradients = [gradient * (5.0 / gradient_norm) for gradient in gradients]
            cosine = 0.5 * (1.0 + math.cos(math.pi * step / mlp_steps))
            learning_rate = 0.002 + 0.018 * cosine
            for position, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                first_moment[position] *= beta1
                first_moment[position] += (1.0 - beta1) * gradient
                second_moment[position] *= beta2
                second_moment[position] += (1.0 - beta2) * gradient * gradient
                corrected_first = first_moment[position] / (1.0 - beta1**step)
                corrected_second = second_moment[position] / (1.0 - beta2**step)
                parameter -= learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )
        return EndpointModel("mlp", tuple(parameters))
    if learner != "linear":
        raise ValueError(f"unknown learner {learner!r}")

    design = np.column_stack((np.ones_like(values_x), values_x))
    sample_size, dimension = design.shape
    identity = eye(sample_size, format="csr")
    equality = hstack(
        (csr_matrix(design), identity, -identity),
        format="csr",
    )
    bounds = [(None, None)] * dimension + [(0.0, None)] * (2 * sample_size)
    coefficients = []
    for tau in (0.5 * alpha, 1.0 - 0.5 * alpha):
        objective = np.r_[
            np.zeros(dimension),
            np.full(sample_size, tau / sample_size),
            np.full(sample_size, (1.0 - tau) / sample_size),
        ]
        result = linprog(
            objective,
            A_eq=equality,
            b_eq=values_y,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        if not result.success:
            raise RuntimeError(
                f"pinball fit failed for learner={learner}, tau={tau:g}: "
                f"{result.message}"
            )
        coefficients.append(result.x[:dimension])
    return EndpointModel("linear", (np.vstack(coefficients),))


def _source_draw(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.random(size)
    y = conditional_mean(x) + conditional_scale(x) * rng.standard_normal(size)
    return x, y


def make_target_grid(
    *, alpha: float, kappa: float, points: int
) -> TargetGrid:
    if points < 51 or points % 2 == 0:
        raise ValueError("x-grid points must be odd and at least 51")
    theta = theta_from_kappa(kappa)
    x = np.linspace(0.0, 1.0, points)
    ratio = likelihood_ratio(x, theta)
    trap = np.ones(points)
    trap[[0, -1]] = 0.5
    quadrature_weights = trap * ratio
    quadrature_weights /= np.sum(quadrature_weights)
    mu = conditional_mean(x)
    sigma = conditional_scale(x)
    z = float(norm.ppf(1.0 - 0.5 * alpha))
    oracle_low = mu - z * sigma
    oracle_high = mu + z * sigma
    return TargetGrid(
        x=x,
        quadrature_weights=quadrature_weights,
        likelihood_ratio=ratio,
        mu=mu,
        sigma=sigma,
        oracle_low=oracle_low,
        oracle_high=oracle_high,
        oracle_length=oracle_high - oracle_low,
        nominal_coverage=1.0 - alpha,
        theta=theta,
        kappa=kappa,
    )


def conformity_scores(
    low: np.ndarray, high: np.ndarray, y: np.ndarray
) -> np.ndarray:
    return np.maximum(low - y, y - high)


def _threshold_at_mass(
    sorted_scores: np.ndarray,
    cumulative_weights: np.ndarray,
    required_mass: np.ndarray | float,
) -> np.ndarray:
    required = np.asarray(required_mass, dtype=float)
    indices = np.searchsorted(cumulative_weights, required, side="left")
    output = np.full(required.shape, np.inf, dtype=float)
    finite = indices < sorted_scores.size
    output[finite] = sorted_scores[indices[finite]]
    return output


def exact_weighted_threshold(
    sorted_scores: np.ndarray,
    sorted_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    cumulative = np.cumsum(sorted_weights)
    required = (1.0 - alpha) * (cumulative[-1] + test_weights)
    return _threshold_at_mass(sorted_scores, cumulative, required)


def population_normalized_threshold(
    sorted_scores: np.ndarray,
    sorted_weights: np.ndarray,
    alpha: float,
) -> float:
    cumulative = np.cumsum(sorted_weights)
    rank = split_rank(sorted_scores.size, alpha)
    return float(_threshold_at_mass(sorted_scores, cumulative, float(rank)))


def unweighted_threshold(sorted_scores: np.ndarray, alpha: float) -> float:
    rank = split_rank(sorted_scores.size, alpha)
    if rank > sorted_scores.size:
        return math.inf
    return float(sorted_scores[rank - 1])


def evaluate_interval(
    *,
    endpoint_low: np.ndarray,
    endpoint_high: np.ndarray,
    threshold: np.ndarray | float,
    target: TargetGrid,
) -> dict[str, float]:
    q = np.broadcast_to(np.asarray(threshold, dtype=float), endpoint_low.shape)
    finite = np.isfinite(q)
    coverage = np.ones_like(endpoint_low)
    length = np.full_like(endpoint_low, np.inf)
    if np.any(finite):
        lower = endpoint_low[finite] - q[finite]
        upper = endpoint_high[finite] + q[finite]
        nonempty = lower <= upper
        finite_coverage = np.zeros_like(lower)
        if np.any(nonempty):
            x_index = np.flatnonzero(finite)[nonempty]
            finite_coverage[nonempty] = (
                ndtr((upper[nonempty] - target.mu[x_index]) / target.sigma[x_index])
                - ndtr((lower[nonempty] - target.mu[x_index]) / target.sigma[x_index])
            )
        coverage[finite] = finite_coverage
        length[finite] = np.maximum(upper - lower, 0.0)

    marginal_coverage = float(np.dot(target.quadrature_weights, coverage))
    coverage_l2 = float(
        math.sqrt(
            np.dot(
                target.quadrature_weights,
                (coverage - target.nominal_coverage) ** 2,
            )
        )
    )
    if np.any(~np.isfinite(length)):
        length_l2 = math.inf
    else:
        length_l2 = float(
            math.sqrt(
                np.dot(
                    target.quadrature_weights,
                    (length - target.oracle_length) ** 2,
                )
            )
        )
    return {
        "marginal_coverage": marginal_coverage,
        "coverage_l2": coverage_l2,
        "length_l2": length_l2,
    }


def _summary(
    values: np.ndarray,
    *,
    coverage: bool = False,
    empirical: bool = False,
) -> dict[str, np.ndarray]:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("summary expects at least two repetitions")
    if empirical:
        return {
            "center": np.median(values, axis=0),
            "low": np.quantile(values, 0.025, axis=0, method="nearest"),
            "high": np.quantile(values, 0.975, axis=0, method="nearest"),
        }
    mean = np.mean(values, axis=0)
    se = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    low = mean - Z975 * se
    high = mean + Z975 * se
    if coverage:
        low = np.maximum(low, 0.0)
        high = np.minimum(high, 1.0)
    else:
        low = np.maximum(low, np.finfo(float).tiny)
    return {"center": mean, "se": se, "low": low, "high": high}


def _rng(seed: int, panel: int, repetition: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, panel, repetition]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _resolved_config(
    *,
    alpha: float,
    kappas: tuple[float, ...],
    seed: int,
    repetitions: int,
    m_grid: tuple[int, ...],
    n_grid: tuple[int, ...],
    n_fixed: int,
    n_validity: int,
    m_fixed: int,
    x_points: int,
    hidden_units: int,
    mlp_steps: int,
    quick: bool,
) -> dict[str, object]:
    return {
        "alpha": alpha,
        "kappa": list(kappas),
        "seed": seed,
        "repetitions": repetitions,
        "m_grid": list(m_grid),
        "n_grid": list(n_grid),
        "n_fixed": n_fixed,
        "n_validity": n_validity,
        "m_fixed": m_fixed,
        "x_points": x_points,
        "hidden_units": hidden_units,
        "mlp_steps": mlp_steps,
        "quick": quick,
    }


def _rng_streams() -> dict[str, str]:
    return {
        "panels_a_b_data": "SeedSequence([seed, 1, repetition])",
        "panel_b_optimizer": "SeedSequence([seed, 11, 0])",
        "panel_c_data": "SeedSequence([seed, 2, repetition])",
        "panel_c_optimizer": "SeedSequence([seed, 21, n])",
    }


def write_results_archive(
    *,
    path: Path,
    config: dict[str, object],
    runtime_seconds: float,
    diagnostics: dict[str, float],
    coverage_samples: dict[tuple[str, float], np.ndarray],
    m_metric_samples: dict[tuple[str, float], np.ndarray],
    n_metric_samples: dict[tuple[str, str], np.ndarray],
) -> None:
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "resolved_config": config,
        "rng_streams": _rng_streams(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "provenance": {
            "git_head": _git_head(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "simulation_runtime_seconds": runtime_seconds,
        "diagnostics": diagnostics,
    }
    kappas = tuple(float(value) for value in config["kappa"])
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))
    }
    for kappa_index, kappa in enumerate(kappas):
        for rule in RULES:
            arrays[f"coverage_{rule}_kappa_{kappa_index}"] = coverage_samples[
                (rule, kappa)
            ]
        for metric in METRICS:
            arrays[f"m_metric_{metric}_kappa_{kappa_index}"] = m_metric_samples[
                (metric, kappa)
            ]
    for learner in LEARNERS:
        for metric in METRICS:
            arrays[f"n_metric_{learner}_{metric}"] = n_metric_samples[
                (learner, metric)
            ]

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validate_saved_config(config: object) -> dict[str, object]:
    if not isinstance(config, dict):
        raise ValueError("results archive has no configuration object")
    required = {
        "alpha",
        "kappa",
        "seed",
        "repetitions",
        "m_grid",
        "n_grid",
        "n_fixed",
        "n_validity",
        "m_fixed",
        "x_points",
        "hidden_units",
        "mlp_steps",
        "quick",
    }
    if not required.issubset(config):
        raise ValueError("results archive configuration is incomplete")

    alpha = config["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("saved alpha must be numeric")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("saved alpha must lie in (0,1)")

    kappas = config["kappa"]
    if not isinstance(kappas, list) or not kappas:
        raise ValueError("saved kappa must be a nonempty list")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in kappas
    ):
        raise ValueError("saved kappa values must be finite and positive")
    if len({float(value) for value in kappas}) != len(kappas):
        raise ValueError("saved kappa values must be unique")

    for name, minimum in (
        ("repetitions", 2),
        ("n_fixed", 2),
        ("n_validity", 2),
        ("m_fixed", 1),
        ("x_points", 3),
        ("hidden_units", 1),
        ("mlp_steps", 1),
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"saved {name} must be an integer at least {minimum}")
    if int(config["x_points"]) % 2 == 0:
        raise ValueError("saved x_points must be odd")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("saved seed must be a nonnegative integer")
    if not isinstance(config["quick"], bool):
        raise ValueError("saved quick flag must be Boolean")

    for name, minimum in (("m_grid", 1), ("n_grid", 2)):
        values = config[name]
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"saved {name} must contain at least two points")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            for value in values
        ):
            raise ValueError(f"saved {name} contains invalid values")
        if values != sorted(set(values)):
            raise ValueError(f"saved {name} must be strictly increasing")
    return config


def load_results_archive(
    path: Path,
) -> tuple[
    dict[str, object],
    dict[tuple[str, float], np.ndarray],
    dict[tuple[str, float], np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
    with np.load(path, allow_pickle=False) as archive:
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if not isinstance(metadata, dict):
                raise ValueError("results metadata must be an object")
            if metadata.get("schema_version") != 1:
                raise ValueError("unsupported E3 results schema")
            config = _validate_saved_config(metadata.get("resolved_config"))
            diagnostics = metadata.get("diagnostics")
            if not isinstance(diagnostics, dict) or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in diagnostics.values()
            ):
                raise ValueError("results archive has no diagnostics object")
            kappas = tuple(float(value) for value in config["kappa"])
            repetitions = int(config["repetitions"])
            m_count = len(config["m_grid"])
            n_count = len(config["n_grid"])
            coverage_samples = {
                (rule, kappa): np.asarray(
                    archive[f"coverage_{rule}_kappa_{kappa_index}"], dtype=float
                )
                for kappa_index, kappa in enumerate(kappas)
                for rule in RULES
            }
            m_metric_samples = {
                (metric, kappa): np.asarray(
                    archive[f"m_metric_{metric}_kappa_{kappa_index}"], dtype=float
                )
                for kappa_index, kappa in enumerate(kappas)
                for metric in METRICS
            }
            n_metric_samples = {
                (learner, metric): np.asarray(
                    archive[f"n_metric_{learner}_{metric}"], dtype=float
                )
                for learner in LEARNERS
                for metric in METRICS
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid E3 results archive: {path}") from error

    expected_m_shape = (repetitions, m_count)
    expected_n_shape = (repetitions, n_count)
    if any(values.shape != expected_m_shape for values in coverage_samples.values()):
        raise ValueError("coverage arrays have incompatible shapes")
    if any(values.shape != expected_m_shape for values in m_metric_samples.values()):
        raise ValueError("panel-(b) arrays have incompatible shapes")
    if any(values.shape != expected_n_shape for values in n_metric_samples.values()):
        raise ValueError("panel-(c) arrays have incompatible shapes")
    all_arrays = (
        *coverage_samples.values(),
        *m_metric_samples.values(),
        *n_metric_samples.values(),
    )
    if not all(np.all(np.isfinite(values)) for values in all_arrays):
        raise ValueError("results archive contains nonfinite values")
    tolerance = 16.0 * np.finfo(float).eps
    if any(
        np.any((values < -tolerance) | (values > 1.0 + tolerance))
        for values in coverage_samples.values()
    ):
        raise ValueError("coverage arrays contain values outside [0,1]")
    if any(np.any(values < -tolerance) for values in m_metric_samples.values()):
        raise ValueError("panel-(b) arrays contain negative errors")
    if any(np.any(values < -tolerance) for values in n_metric_samples.values()):
        raise ValueError("panel-(c) arrays contain negative errors")
    return metadata, coverage_samples, m_metric_samples, n_metric_samples


def write_run_manifest(
    *,
    path: Path,
    output: Path,
    alpha: float,
    kappas: tuple[float, ...],
    seed: int,
    repetitions: int,
    m_grid: tuple[int, ...],
    n_grid: tuple[int, ...],
    n_fixed: int,
    n_validity: int,
    m_fixed: int,
    x_points: int,
    hidden_units: int,
    mlp_steps: int,
    quick: bool,
    runtime_seconds: float,
    diagnostics: dict[str, float],
    results: Path | None,
    execution_mode: str,
) -> None:
    import matplotlib

    script_path = Path(__file__).resolve()
    output_path = output.resolve()
    payload = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": execution_mode,
        "command": [sys.executable, *sys.argv],
        "resolved_config": _resolved_config(
            alpha=alpha,
            kappas=kappas,
            seed=seed,
            repetitions=repetitions,
            m_grid=m_grid,
            n_grid=n_grid,
            n_fixed=n_fixed,
            n_validity=n_validity,
            m_fixed=m_fixed,
            x_points=x_points,
            hidden_units=hidden_units,
            mlp_steps=mlp_steps,
            quick=quick,
        ),
        "rng_streams": _rng_streams(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "provenance": {
            "git_head": _git_head(),
            "script": str(script_path),
            "script_sha256": _sha256(script_path),
            "pdf": str(output_path),
            "pdf_sha256": _sha256(output_path),
            "results": str(results.resolve()) if results is not None else None,
            "results_sha256": _sha256(results.resolve()) if results is not None else None,
        },
        "runtime_seconds": runtime_seconds,
        "diagnostics": diagnostics,
    }
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def simulate_m_panels(
    *,
    alpha: float,
    kappas: tuple[float, ...],
    m_grid: tuple[int, ...],
    n_fixed: int,
    n_validity: int,
    repetitions: int,
    x_points: int,
    hidden_units: int,
    mlp_steps: int,
    seed: int,
) -> tuple[
    dict[tuple[str, float], dict[str, np.ndarray]],
    dict[tuple[str, float], dict[str, np.ndarray]],
    dict[str, float],
    dict[tuple[str, float], np.ndarray],
    dict[tuple[str, float], np.ndarray],
]:
    targets = {
        kappa: make_target_grid(alpha=alpha, kappa=kappa, points=x_points)
        for kappa in kappas
    }
    coverage_samples = {
        (rule, kappa): np.empty((repetitions, len(m_grid)))
        for rule in RULES
        for kappa in kappas
    }
    metric_samples = {
        (metric, kappa): np.empty((repetitions, len(m_grid)))
        for metric in METRICS
        for kappa in kappas
    }
    population_infinite = 0
    exact_infinite = 0
    maximum_m = max(m_grid)

    for repetition in range(repetitions):
        rng = _rng(seed, 1, repetition)
        train_x, train_y = _source_draw(max(n_fixed, n_validity), rng)
        mlp_model = fit_endpoint_model(
            train_x[:n_fixed],
            train_y[:n_fixed],
            alpha,
            "mlp",
            hidden_units,
            mlp_steps,
            _rng(seed, 11, 0),
        )
        linear_model = fit_endpoint_model(
            train_x[:n_validity],
            train_y[:n_validity],
            alpha,
            "linear",
            hidden_units,
            mlp_steps,
            _rng(seed, 12, repetition),
        )
        grid_x = next(iter(targets.values())).x
        mlp_grid_low, mlp_grid_high = mlp_model.predict(grid_x)
        linear_grid_low, linear_grid_high = linear_model.predict(grid_x)
        calibration_x, calibration_y = _source_draw(maximum_m, rng)
        mlp_cal_low, mlp_cal_high = mlp_model.predict(calibration_x)
        linear_cal_low, linear_cal_high = linear_model.predict(calibration_x)
        mlp_scores = conformity_scores(mlp_cal_low, mlp_cal_high, calibration_y)
        linear_scores = conformity_scores(
            linear_cal_low, linear_cal_high, calibration_y
        )

        for m_index, m in enumerate(m_grid):
            linear_order = np.argsort(linear_scores[:m], kind="mergesort")
            linear_sorted_scores = linear_scores[:m][linear_order]
            mlp_order = np.argsort(mlp_scores[:m], kind="mergesort")
            mlp_sorted_scores = mlp_scores[:m][mlp_order]
            unweighted_q = unweighted_threshold(linear_sorted_scores, alpha)
            for kappa in kappas:
                target = targets[kappa]
                linear_sorted_weights = likelihood_ratio(
                    calibration_x[:m][linear_order], target.theta
                )
                mlp_sorted_weights = likelihood_ratio(
                    calibration_x[:m][mlp_order], target.theta
                )
                coverage_exact_q = exact_weighted_threshold(
                    linear_sorted_scores,
                    linear_sorted_weights,
                    target.likelihood_ratio,
                    alpha,
                )
                population_q = population_normalized_threshold(
                    linear_sorted_scores, linear_sorted_weights, alpha
                )
                coverage_exact_result = evaluate_interval(
                    endpoint_low=linear_grid_low,
                    endpoint_high=linear_grid_high,
                    threshold=coverage_exact_q,
                    target=target,
                )
                population_result = evaluate_interval(
                    endpoint_low=linear_grid_low,
                    endpoint_high=linear_grid_high,
                    threshold=population_q,
                    target=target,
                )
                unweighted_result = evaluate_interval(
                    endpoint_low=linear_grid_low,
                    endpoint_high=linear_grid_high,
                    threshold=unweighted_q,
                    target=target,
                )
                metric_exact_q = exact_weighted_threshold(
                    mlp_sorted_scores,
                    mlp_sorted_weights,
                    target.likelihood_ratio,
                    alpha,
                )
                metric_exact_result = evaluate_interval(
                    endpoint_low=mlp_grid_low,
                    endpoint_high=mlp_grid_high,
                    threshold=metric_exact_q,
                    target=target,
                )
                coverage_samples[("exact", kappa)][repetition, m_index] = (
                    coverage_exact_result["marginal_coverage"]
                )
                coverage_samples[("population", kappa)][repetition, m_index] = (
                    population_result["marginal_coverage"]
                )
                coverage_samples[("unweighted", kappa)][repetition, m_index] = (
                    unweighted_result["marginal_coverage"]
                )
                metric_samples[("coverage", kappa)][repetition, m_index] = (
                    metric_exact_result["coverage_l2"]
                )
                metric_samples[("length", kappa)][repetition, m_index] = (
                    metric_exact_result["length_l2"]
                )
                population_infinite += int(math.isinf(population_q))
                if not np.all(np.isfinite(metric_exact_q)):
                    exact_infinite += 1

    if exact_infinite:
        raise RuntimeError(
            f"exact weighted threshold was infinite in {exact_infinite} panel-(b) fits; "
            "increase the calibration grid"
        )
    if not all(np.all(np.isfinite(values)) for values in metric_samples.values()):
        raise RuntimeError("nonfinite exact-weighted L2 functional")

    coverage_summary = {
        key: _summary(values, coverage=True)
        for key, values in coverage_samples.items()
    }
    metric_summary = {
        key: _summary(values, empirical=True) for key, values in metric_samples.items()
    }

    validity_z = math.inf
    for kappa in kappas:
        summary = coverage_summary[("exact", kappa)]
        deficits = (1.0 - alpha) - summary["center"]
        positive = summary["se"] > 0.0
        if np.any(positive):
            validity_z = min(
                validity_z,
                float(np.min(-deficits[positive] / summary["se"][positive])),
            )
        if np.any(deficits > 4.0 * summary["se"] + 2e-3):
            raise AssertionError(
                f"exact weighted mean coverage is incompatible with validity at kappa={kappa:g}"
            )
    diagnostics = {
        "population_infinite_thresholds": float(population_infinite),
        "exact_infinite_thresholds": float(exact_infinite),
        "minimum_weighted_validity_z": float(validity_z),
    }
    largest_kappa = max(kappas)
    unweighted_upper = float(
        coverage_summary[("unweighted", largest_kappa)]["high"][-1]
    )
    diagnostics["unweighted_upper_at_largest_m"] = unweighted_upper
    log_m = np.log(np.asarray(m_grid, dtype=float))
    for metric in METRICS:
        for kappa in kappas:
            slope = np.polyfit(
                log_m,
                np.log(metric_summary[(metric, kappa)]["center"]),
                deg=1,
            )[0]
            diagnostics[f"panel_b_slope_{metric}_kappa_{kappa:g}"] = float(slope)
    return (
        coverage_summary,
        metric_summary,
        diagnostics,
        coverage_samples,
        metric_samples,
    )


def simulate_n_panel(
    *,
    alpha: float,
    kappa: float,
    n_grid: tuple[int, ...],
    m_fixed: int,
    repetitions: int,
    x_points: int,
    hidden_units: int,
    mlp_steps: int,
    seed: int,
) -> tuple[
    dict[tuple[str, str], dict[str, np.ndarray]],
    dict[str, float],
    dict[tuple[str, str], np.ndarray],
]:
    target = make_target_grid(alpha=alpha, kappa=kappa, points=x_points)
    samples = {
        (learner, metric): np.empty((repetitions, len(n_grid)))
        for learner in LEARNERS
        for metric in METRICS
    }
    exact_infinite = 0
    maximum_n = max(n_grid)

    for repetition in range(repetitions):
        rng = _rng(seed, 2, repetition)
        train_x, train_y = _source_draw(maximum_n, rng)
        calibration_x, calibration_y = _source_draw(m_fixed, rng)
        calibration_weights = likelihood_ratio(calibration_x, target.theta)
        for n_index, n in enumerate(n_grid):
            for learner in LEARNERS:
                model = fit_endpoint_model(
                    train_x[:n],
                    train_y[:n],
                    alpha,
                    learner,
                    hidden_units,
                    mlp_steps,
                    np.random.default_rng(
                        np.random.SeedSequence([seed, 21, n])
                    ),
                )
                grid_low, grid_high = model.predict(target.x)
                cal_low, cal_high = model.predict(calibration_x)
                scores = conformity_scores(cal_low, cal_high, calibration_y)
                order = np.argsort(scores, kind="mergesort")
                exact_q = exact_weighted_threshold(
                    scores[order],
                    calibration_weights[order],
                    target.likelihood_ratio,
                    alpha,
                )
                if not np.all(np.isfinite(exact_q)):
                    exact_infinite += 1
                    continue
                result = evaluate_interval(
                    endpoint_low=grid_low,
                    endpoint_high=grid_high,
                    threshold=exact_q,
                    target=target,
                )
                samples[(learner, "coverage")][repetition, n_index] = result[
                    "coverage_l2"
                ]
                samples[(learner, "length")][repetition, n_index] = result[
                    "length_l2"
                ]

    if exact_infinite:
        raise RuntimeError(
            f"exact weighted threshold was infinite in {exact_infinite} panel-(c) fits; "
            "increase --m-fixed"
        )
    if not all(np.all(np.isfinite(values)) for values in samples.values()):
        raise RuntimeError("nonfinite panel-(c) functional")
    summaries = {key: _summary(values, empirical=True) for key, values in samples.items()}
    diagnostics = {"exact_infinite_thresholds": float(exact_infinite)}
    log_n = np.log(np.asarray(n_grid, dtype=float))
    for learner in LEARNERS:
        for metric in METRICS:
            diagnostics[f"slope_{learner}_{metric}"] = float(
                np.polyfit(
                    log_n,
                    np.log(summaries[(learner, metric)]["center"]),
                    deg=1,
                )[0]
            )
    return summaries, diagnostics, samples


def run_checks() -> dict[str, float]:
    alpha = 0.1
    worst_moment_error = 0.0
    for kappa in DEFAULT_KAPPAS:
        target = make_target_grid(alpha=alpha, kappa=kappa, points=2001)
        numerical_second_moment = float(
            np.dot(target.quadrature_weights, target.likelihood_ratio)
        )
        worst_moment_error = max(
            worst_moment_error,
            abs(numerical_second_moment - (1.0 + kappa)),
        )
        oracle_result = evaluate_interval(
            endpoint_low=target.oracle_low,
            endpoint_high=target.oracle_high,
            threshold=0.0,
            target=target,
        )
        if abs(oracle_result["marginal_coverage"] - (1.0 - alpha)) > 2e-12:
            raise AssertionError("Gaussian oracle coverage check failed")
        if oracle_result["length_l2"] > 2e-14:
            raise AssertionError("Gaussian oracle length check failed")
    if worst_moment_error > 2e-4:
        raise AssertionError("tilt second moment disagrees with 1+kappa")

    if split_rank(9, 0.1) != 9 or split_rank(2, 1.0 / 3.0) != 3:
        raise AssertionError("split conformal rank check failed")
    scores = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0, 1.0])
    exact = exact_weighted_threshold(scores, weights, np.array([1.0]), 0.5)
    if exact.shape != (1,) or exact[0] != 1.0:
        raise AssertionError("exact weighted threshold check failed")

    rng = np.random.default_rng(314159)
    train_x, train_y = _source_draw(80, rng)
    for learner in LEARNERS:
        model = fit_endpoint_model(
            train_x,
            train_y,
            alpha,
            learner,
            6,
            300,
            np.random.default_rng(271828),
        )
        low, high = model.predict(np.linspace(0.0, 1.0, 101))
        if not np.all(np.isfinite(low)) or not np.all(low <= high):
            raise AssertionError(f"endpoint sorting check failed for {learner}")
    return {
        "worst_tilt_second_moment_error": worst_moment_error,
        "theta_kappa_1": theta_from_kappa(1.0),
        "theta_kappa_3": theta_from_kappa(3.0),
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
            "font.size": 8.3,
            "axes.titlesize": 9.1,
            "axes.labelsize": 8.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _plot_band(ax, x, summary, *, color, linestyle, marker, label):
    ax.fill_between(
        x,
        summary["low"],
        summary["high"],
        color=color,
        alpha=0.08,
        linewidth=0.0,
    )
    ax.plot(
        x,
        summary["center"],
        color=color,
        linestyle=linestyle,
        marker=marker,
        markerfacecolor="white",
        markeredgewidth=0.65,
        markersize=3.0,
        linewidth=1.15,
        label=label,
    )


def make_figure(
    *,
    alpha: float,
    kappas: tuple[float, ...],
    m_grid: tuple[int, ...],
    n_grid: tuple[int, ...],
    n_fixed: int,
    n_validity: int,
    m_fixed: int,
    repetitions: int,
    x_points: int,
    hidden_units: int,
    mlp_steps: int,
    seed: int,
    output: Path,
    saved_results: tuple[
        dict[tuple[str, float], np.ndarray],
        dict[tuple[str, float], np.ndarray],
        dict[tuple[str, str], np.ndarray],
    ]
    | None = None,
    saved_diagnostics: dict[str, float] | None = None,
) -> tuple[
    dict[str, float],
    dict[tuple[str, float], np.ndarray],
    dict[tuple[str, float], np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
    c_kappa = max(kappas)
    if saved_results is None:
        (
            coverage,
            m_metrics,
            m_diagnostics,
            coverage_samples,
            m_metric_samples,
        ) = simulate_m_panels(
            alpha=alpha,
            kappas=kappas,
            m_grid=m_grid,
            n_fixed=n_fixed,
            n_validity=n_validity,
            repetitions=repetitions,
            x_points=x_points,
            hidden_units=hidden_units,
            mlp_steps=mlp_steps,
            seed=seed,
        )
        n_metrics, n_diagnostics, n_metric_samples = simulate_n_panel(
            alpha=alpha,
            kappa=c_kappa,
            n_grid=n_grid,
            m_fixed=m_fixed,
            repetitions=repetitions,
            x_points=x_points,
            hidden_units=hidden_units,
            mlp_steps=mlp_steps,
            seed=seed,
        )
        diagnostics = dict(m_diagnostics)
        diagnostics.update(
            {
                "panel_c_exact_infinite_thresholds": n_diagnostics[
                    "exact_infinite_thresholds"
                ],
                "panel_c_kappa": c_kappa,
            }
        )
        for name, value in n_diagnostics.items():
            if name != "exact_infinite_thresholds":
                diagnostics[f"panel_c_{name}"] = value
    else:
        coverage_samples, m_metric_samples, n_metric_samples = saved_results
        coverage = {
            key: _summary(values, coverage=True)
            for key, values in coverage_samples.items()
        }
        m_metrics = {
            key: _summary(values, empirical=True)
            for key, values in m_metric_samples.items()
        }
        n_metrics = {
            key: _summary(values, empirical=True)
            for key, values in n_metric_samples.items()
        }
        diagnostics = dict(saved_diagnostics or {})

    plt = _setup_plotting()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.65))
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.34, top=0.91, wspace=0.43)
    ax_coverage, ax_m, ax_n = axes
    m_values = np.asarray(m_grid, dtype=float)
    n_values = np.asarray(n_grid, dtype=float)

    rule_colors = {
        "exact": "#2166ac",
        "population": "#e08214",
        "unweighted": "#737373",
    }
    rule_labels = {
        "exact": "exact weighted",
        "population": "population-normalized",
        "unweighted": "unweighted",
    }
    kappa_styles = ("-", "--", "-.", ":")
    kappa_markers = ("o", "s", "^", "D")
    for rule in RULES:
        for kappa_index, kappa in enumerate(kappas):
            _plot_band(
                ax_coverage,
                m_values,
                coverage[(rule, kappa)],
                color=rule_colors[rule],
                linestyle=kappa_styles[kappa_index % len(kappa_styles)],
                marker=kappa_markers[kappa_index % len(kappa_markers)],
                label="_nolegend_",
            )
    ax_coverage.axhline(1.0 - alpha, color="0.15", linestyle=":", linewidth=0.9)
    ax_coverage.set_xscale("log", base=2)
    ax_coverage.set_xlabel(r"calibration size $m$")
    ax_coverage.set_ylabel("target marginal coverage")
    ax_coverage.set_title(fr"(a) Validity, linear QR" + "\n" + fr"$n={n_validity}$")
    all_coverage_lows = [coverage[key]["low"] for key in coverage]
    ax_coverage.set_ylim(max(0.0, min(map(np.min, all_coverage_lows)) - 0.02), 1.005)
    rule_legend = ax_coverage.legend(
        handles=[
            Line2D([], [], color=rule_colors[rule], linewidth=1.3, label=rule_labels[rule])
            for rule in RULES
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.34),
        ncol=1,
        handlelength=1.7,
        borderaxespad=0.0,
    )
    ax_coverage.add_artist(rule_legend)
    ax_coverage.legend(
        handles=[
            Line2D(
                [],
                [],
                color="0.35",
                linestyle=kappa_styles[index % len(kappa_styles)],
                marker=kappa_markers[index % len(kappa_markers)],
                markerfacecolor="white",
                linewidth=1.2,
                label=fr"$\kappa={kappa:g}$",
            )
            for index, kappa in enumerate(kappas)
        ]
        + [
            Line2D(
                [],
                [],
                color="0.15",
                linestyle=":",
                linewidth=0.9,
                label=r"nominal $1-\alpha$",
            )
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.78, -0.34),
        ncol=1,
        handlelength=1.7,
        borderaxespad=0.0,
    )

    kappa_colors = ("#2166ac", "#b2182b", "#4d9221", "#762a83")
    metric_styles = {"coverage": ("-", "o"), "length": ("--", "s")}
    metric_labels = {
        "coverage": r"coverage $L^2$",
        "length": r"length $L^2$",
    }
    for kappa_index, kappa in enumerate(kappas):
        for metric in METRICS:
            linestyle, marker = metric_styles[metric]
            _plot_band(
                ax_m,
                m_values,
                m_metrics[(metric, kappa)],
                color=kappa_colors[kappa_index % len(kappa_colors)],
                linestyle=linestyle,
                marker=marker,
                label="_nolegend_",
            )
    reference_anchor = m_metrics[("coverage", kappas[0])]["center"][0]
    reference = reference_anchor * np.sqrt(m_values[0] / m_values)
    ax_m.plot(
        m_values,
        reference,
        color="0.25",
        linestyle=":",
        linewidth=1.0,
        label="_nolegend_",
    )
    ax_m.set_xscale("log", base=2)
    ax_m.set_yscale("log")
    ax_m.set_xlabel(r"calibration size $m$")
    ax_m.set_ylabel(r"target $L^2(Q_X)$ error")
    ax_m.set_title(
        fr"(b) Exact weighted, pinball MLP" + "\n" + fr"$n={n_fixed}$"
    )
    metric_legend = ax_m.legend(
        handles=[
            Line2D(
                [],
                [],
                color="0.35",
                linestyle=metric_styles[metric][0],
                marker=metric_styles[metric][1],
                markerfacecolor="white",
                linewidth=1.2,
                label=metric_labels[metric],
            )
            for metric in METRICS
        ]
        + [
            Line2D(
                [],
                [],
                color="0.25",
                linestyle=":",
                linewidth=1.0,
                label=r"$m^{-1/2}$ reference",
            )
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.34),
        ncol=1,
        handlelength=1.7,
        borderaxespad=0.0,
    )
    ax_m.add_artist(metric_legend)
    ax_m.legend(
        handles=[
            Line2D(
                [],
                [],
                color=kappa_colors[index % len(kappa_colors)],
                linewidth=1.3,
                label=fr"$\kappa={kappa:g}$",
            )
            for index, kappa in enumerate(kappas)
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.02, -0.34),
        ncol=1,
        handlelength=1.7,
        borderaxespad=0.0,
    )

    learner_colors = {"mlp": "#2166ac", "linear": "#e08214"}
    learner_labels = {"mlp": "pinball MLP", "linear": "linear QR"}
    for learner in LEARNERS:
        for metric in METRICS:
            linestyle, marker = metric_styles[metric]
            _plot_band(
                ax_n,
                n_values,
                n_metrics[(learner, metric)],
                color=learner_colors[learner],
                linestyle=linestyle,
                marker=marker,
                label="_nolegend_",
            )
    ax_n.set_xscale("log", base=2)
    ax_n.set_yscale("log")
    ax_n.set_xlabel(r"training size $n$")
    ax_n.set_ylabel(r"target $L^2(Q_X)$ error")
    ax_n.set_title(
        fr"(c) Training-size effect"
        + "\n"
        + fr"$m={m_fixed}$, $\kappa={c_kappa:g}$"
    )
    learner_legend = ax_n.legend(
        handles=[
            Line2D(
                [],
                [],
                color=learner_colors[learner],
                linewidth=1.3,
                label=learner_labels[learner],
            )
            for learner in LEARNERS
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.34),
        ncol=1,
        handlelength=1.7,
        borderaxespad=0.0,
    )
    ax_n.add_artist(learner_legend)
    ax_n.legend(
        handles=[
            Line2D(
                [],
                [],
                color="0.35",
                linestyle=metric_styles[metric][0],
                marker=metric_styles[metric][1],
                markerfacecolor="white",
                linewidth=1.2,
                label=metric_labels[metric],
            )
            for metric in METRICS
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.02, -0.34),
        ncol=1,
        handlelength=1.7,
        borderaxespad=0.0,
    )

    for ax in axes:
        ax.grid(which="major", color="0.90", linewidth=0.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    for kappa in kappas:
        target = make_target_grid(alpha=alpha, kappa=kappa, points=x_points)
        diagnostics[f"theta_kappa_{kappa:g}"] = target.theta
        diagnostics[f"wmax_kappa_{kappa:g}"] = float(
            likelihood_ratio(np.array([1.0]), target.theta)[0]
        )
    return (
        diagnostics,
        coverage_samples,
        m_metric_samples,
        n_metric_samples,
    )


def _validate_artifact_paths(args: argparse.Namespace) -> None:
    if args.check_only:
        return
    paths = {"output": args.output.resolve()}
    if args.manifest is not None:
        paths["manifest"] = args.manifest.resolve()
    if args.save_results is not None:
        paths["save-results"] = args.save_results.resolve()
    if args.load_results is not None:
        paths["load-results"] = args.load_results.resolve()
    script_path = Path(__file__).resolve()
    if any(path == script_path for path in paths.values()):
        raise ValueError("artifact paths must not overwrite the experiment script")
    names = list(paths)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            if paths[left_name] == paths[right_name]:
                raise ValueError(
                    f"--{left_name} and --{right_name} must use distinct paths"
                )


def _validate_configuration(args: argparse.Namespace) -> None:
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie in (0,1)")
    if any(value <= 0.0 for value in args.kappa):
        raise ValueError("--kappa values must be positive")
    if args.reps < 2:
        raise ValueError("--reps must be at least 2")
    if args.n_fixed < 2 or args.n_validity < 2 or args.m_fixed < 1:
        raise ValueError(
            "--n-fixed and --n-validity must be at least 2; --m-fixed must be positive"
        )
    if any(value < 1 for value in args.m_grid) or any(value < 2 for value in args.n_grid):
        raise ValueError("sample-size grids contain invalid values")
    if args.hidden_units < 1:
        raise ValueError("--hidden-units must be positive")
    if args.mlp_steps < 1:
        raise ValueError("--mlp-steps must be positive")
    if args.check_only and (
        args.save_results is not None
        or args.load_results is not None
        or args.manifest is not None
    ):
        raise ValueError("--check-only cannot write or load run artifacts")
    if args.quick and args.load_results is not None:
        raise ValueError("--quick cannot be combined with --load-results")
    if args.load_results is not None:
        simulation_options = {
            "--alpha",
            "--kappa",
            "--seed",
            "--reps",
            "--n-fixed",
            "--n-validity",
            "--m-fixed",
            "--m-grid",
            "--n-grid",
            "--x-points",
            "--hidden-units",
            "--mlp-steps",
            "--quick",
        }
        explicit = {
            token.split("=", maxsplit=1)[0]
            for token in sys.argv[1:]
            if token.startswith("--")
        }
        ignored = sorted(explicit & simulation_options)
        if ignored:
            joined = ", ".join(ignored)
            raise ValueError(
                f"--load-results uses embedded simulation settings; remove {joined}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, nargs="+", default=list(DEFAULT_KAPPAS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--n-fixed", type=int, default=8192)
    parser.add_argument("--n-validity", type=int, default=2048)
    parser.add_argument("--m-fixed", type=int, default=4096)
    parser.add_argument("--m-grid", type=int, nargs="+", default=list(DEFAULT_M_GRID))
    parser.add_argument("--n-grid", type=int, nargs="+", default=list(DEFAULT_N_GRID))
    parser.add_argument("--x-points", type=int, default=1025)
    parser.add_argument("--hidden-units", type=int, default=20)
    parser.add_argument("--mlp-steps", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=ROOT / "figure" / "e3_cqr.pdf")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="write resolved configuration, RNG streams, hashes, and diagnostics as JSON",
    )
    results_group = parser.add_mutually_exclusive_group()
    results_group.add_argument(
        "--save-results",
        type=Path,
        help="save all per-replication numerical results as a compressed NPZ archive",
    )
    results_group.add_argument(
        "--load-results",
        type=Path,
        help="load a saved NPZ archive and redraw the PDF without simulation",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    _validate_configuration(args)
    _validate_artifact_paths(args)
    if args.check_only:
        checks = run_checks()
        print("E3 checks PASS")
        for name, value in checks.items():
            print(f"{name} = {value:.12g}")
        return
    if args.load_results is None:
        run_checks()

    saved_results = None
    saved_diagnostics = None
    if args.load_results is not None:
        (
            saved_metadata,
            coverage_samples,
            m_metric_samples,
            n_metric_samples,
        ) = load_results_archive(args.load_results)
        config = saved_metadata["resolved_config"]
        alpha = float(config["alpha"])
        kappas = tuple(float(value) for value in config["kappa"])
        seed = int(config["seed"])
        repetitions = int(config["repetitions"])
        m_grid = tuple(int(value) for value in config["m_grid"])
        n_grid = tuple(int(value) for value in config["n_grid"])
        n_fixed = int(config["n_fixed"])
        n_validity = int(config["n_validity"])
        m_fixed = int(config["m_fixed"])
        x_points = int(config["x_points"])
        hidden_units = int(config["hidden_units"])
        mlp_steps = int(config["mlp_steps"])
        quick = bool(config["quick"])
        saved_results = (
            coverage_samples,
            m_metric_samples,
            n_metric_samples,
        )
        saved_diagnostics = {
            name: float(value)
            for name, value in saved_metadata["diagnostics"].items()
        }
    else:
        alpha = args.alpha
        kappas = tuple(dict.fromkeys(float(value) for value in args.kappa))
        seed = args.seed
        m_grid = tuple(sorted(set(int(value) for value in args.m_grid)))
        n_grid = tuple(sorted(set(int(value) for value in args.n_grid)))
        repetitions = args.reps
        n_fixed = args.n_fixed
        n_validity = args.n_validity
        m_fixed = args.m_fixed
        x_points = args.x_points
        hidden_units = args.hidden_units
        mlp_steps = args.mlp_steps
        quick = args.quick
    if quick and args.load_results is None:
        repetitions = min(repetitions, 12)
        n_fixed = min(n_fixed, 512)
        n_validity = min(n_validity, 512)
        m_fixed = min(m_fixed, 1024)
        m_grid = tuple(value for value in m_grid if value <= 1024)
        n_grid = tuple(value for value in n_grid if value <= 512)
        x_points = min(x_points, 201)
        mlp_steps = min(mlp_steps, 400)
    if not m_grid or not n_grid:
        raise ValueError("quick-mode caps removed the complete m or n grid")
    if len(m_grid) < 2 or len(n_grid) < 2:
        raise ValueError("--m-grid and --n-grid must contain at least two unique values")
    if x_points % 2 == 0:
        x_points += 1

    started = time.perf_counter()
    (
        diagnostics,
        coverage_samples,
        m_metric_samples,
        n_metric_samples,
    ) = make_figure(
        alpha=alpha,
        kappas=kappas,
        m_grid=m_grid,
        n_grid=n_grid,
        n_fixed=n_fixed,
        n_validity=n_validity,
        m_fixed=m_fixed,
        repetitions=repetitions,
        x_points=x_points,
        hidden_units=hidden_units,
        mlp_steps=mlp_steps,
        seed=seed,
        output=args.output,
        saved_results=saved_results,
        saved_diagnostics=saved_diagnostics,
    )
    elapsed = time.perf_counter() - started
    config = _resolved_config(
        alpha=alpha,
        kappas=kappas,
        seed=seed,
        repetitions=repetitions,
        m_grid=m_grid,
        n_grid=n_grid,
        n_fixed=n_fixed,
        n_validity=n_validity,
        m_fixed=m_fixed,
        x_points=x_points,
        hidden_units=hidden_units,
        mlp_steps=mlp_steps,
        quick=quick,
    )
    if args.save_results is not None:
        write_results_archive(
            path=args.save_results,
            config=config,
            runtime_seconds=elapsed,
            diagnostics=diagnostics,
            coverage_samples=coverage_samples,
            m_metric_samples=m_metric_samples,
            n_metric_samples=n_metric_samples,
        )
    results_path = args.save_results or args.load_results
    if args.manifest is not None:
        write_run_manifest(
            path=args.manifest,
            output=args.output,
            alpha=alpha,
            kappas=kappas,
            seed=seed,
            repetitions=repetitions,
            m_grid=m_grid,
            n_grid=n_grid,
            n_fixed=n_fixed,
            n_validity=n_validity,
            m_fixed=m_fixed,
            x_points=x_points,
            hidden_units=hidden_units,
            mlp_steps=mlp_steps,
            quick=quick,
            runtime_seconds=elapsed,
            diagnostics=diagnostics,
            results=results_path,
            execution_mode="plot_only" if args.load_results is not None else "simulation",
        )
    print(f"wrote {args.output}")
    if args.save_results is not None:
        print(f"wrote {args.save_results}")
    if args.manifest is not None:
        print(f"wrote {args.manifest}")
    print(f"execution_mode = {'plot_only' if args.load_results is not None else 'simulation'}")
    print(f"repetitions = {repetitions}, seed = {seed}")
    print(f"runtime_seconds = {elapsed:.3f}")
    for name, value in diagnostics.items():
        print(f"{name} = {value:.12g}")


if __name__ == "__main__":
    main()
