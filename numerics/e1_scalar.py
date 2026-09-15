#!/usr/bin/env python3
"""Exact scalar numerical illustration for minimax calibration experiments.

The carrier-rank risk is evaluated as a Binomial mixture of Beta order
statistics.  No Monte Carlo or response data simulation is used.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc
from scipy.stats import beta as beta_distribution
from scipy.stats import binom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAPPAS = (1.0, 3.0, 9.0, 27.0)
PLATEAU_ALPHAS = (0.05, 0.10, 0.20)
BLUE = ("#9ecae1", "#4292c6", "#2171b5", "#084594")


def _validate(alpha: float, kappa: float, m: int) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if m < 1:
        raise ValueError("m must be positive")


def carrier_rank(n: np.ndarray | int, alpha: float) -> np.ndarray:
    """Return ``ceil((n+1)(1-alpha))`` without floating boundary errors."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    counts = np.asarray(n, dtype=np.int64)
    sample_size = counts + 1
    target = np.longdouble(1.0) - np.longdouble(alpha)
    product = sample_size.astype(np.longdouble) * target
    rank = np.ceil(product).astype(np.int64)

    # Correct the only numerically ambiguous products with exact float
    # arithmetic.  This preserves vectorized speed away from integer ranks.
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


def beta_absolute_deviation(
    n: np.ndarray | int,
    alpha: float,
) -> np.ndarray:
    """Compute ``r_n=E|B_{n,k_n}-(1-alpha)|`` in closed form."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    counts = np.asarray(n, dtype=np.int64)
    if np.any(counts < 0):
        raise ValueError("counts must be nonnegative")

    tau = 1.0 - alpha
    rank = carrier_rank(counts, alpha)
    result = np.full(counts.shape, alpha, dtype=float)
    interior = rank <= counts
    if np.any(interior):
        a = rank[interior]
        b = counts[interior] + 1 - a
        mean = a / (counts[interior] + 1.0)
        value = (
            2.0 * tau * betainc(a, b, tau)
            - 2.0 * mean * betainc(a + 1, b, tau)
            + mean
            - tau
        )
        result[interior] = np.maximum(value, 0.0)
    return result


def exact_risk(alpha: float, kappa: float, m: int) -> float:
    """Evaluate exact risk ``R(m,kappa)`` from the manuscript's Binomial-Beta
    identity (SM4.4; the reference tracks the compiled manuscript numbering).
    """
    _validate(alpha, kappa, m)
    one_plus_kappa = 1.0 + kappa
    n = np.arange(m + 1, dtype=np.int64)
    probabilities = binom.pmf(n, m, 1.0 / one_plus_kappa)
    mass = float(np.sum(probabilities))
    if abs(mass - 1.0) > 2e-10:
        raise ArithmeticError(f"Binomial mass error {mass - 1.0:+.3e} at m={m}")
    return float(np.dot(probabilities, beta_absolute_deviation(n, alpha)))


def finite_upper(kappa: float, m: np.ndarray | int) -> np.ndarray:
    """Finite achievability bound of Proposition 5.2, including its exact
    finite-m factors (the reference tracks the compiled manuscript numbering).
    """
    one_plus_kappa = 1.0 + kappa
    values = np.asarray(m, dtype=float)
    hit_probability = -np.expm1(
        (values + 1.0) * np.log1p(-1.0 / one_plus_kappa)
    )
    return 1.5 * np.sqrt(one_plus_kappa * hit_probability / (values + 1.0))


def le_cam_constant(alpha: float) -> float:
    return 0.125 * math.sqrt(math.log(2.0) * alpha * (1.0 - alpha))


def effective_validity_boundary(alpha: float) -> float:
    return (
        4.0
        * math.log(2.0)
        * alpha
        * (1.0 - alpha)
        / min(alpha, 1.0 - alpha) ** 2
    )


def asymptotic_constant(alpha: np.ndarray | float) -> np.ndarray:
    values = np.asarray(alpha, dtype=float)
    return np.sqrt(2.0 * values * (1.0 - values) / math.pi)


def _m_grid(kappa: float, maximum_effective_size: float, points: int) -> np.ndarray:
    one_plus_kappa = 1.0 + kappa
    maximum = max(1, int(round(one_plus_kappa * maximum_effective_size)))
    return np.unique(
        np.maximum(1, np.rint(np.geomspace(1.0, maximum, points)).astype(np.int64))
    )


def compute_curves(
    alpha: float,
    kappas: tuple[float, ...],
    maximum_effective_size: float,
    points: int,
) -> dict[float, dict[str, np.ndarray]]:
    curves: dict[float, dict[str, np.ndarray]] = {}
    for kappa in kappas:
        m = _m_grid(kappa, maximum_effective_size, points)
        risk = np.array([exact_risk(alpha, kappa, int(value)) for value in m])
        curves[kappa] = {
            "m": m,
            "x": m / (1.0 + kappa),
            "risk": risk,
            "upper": finite_upper(kappa, m),
        }
    return curves


def run_checks() -> dict[str, float]:
    """Focused deterministic checks for formula, bounds, collapse, and plateau."""
    alpha = 0.1
    kappas = DEFAULT_KAPPAS

    if int(carrier_rank(np.array([8]), alpha)[0]) != 9:
        raise AssertionError("rank failed below integral boundary")
    if int(carrier_rank(np.array([9]), alpha)[0]) != 9:
        raise AssertionError("rank failed at integral boundary")
    if int(carrier_rank(np.array([2]), 0.3333333333333333)[0]) != 3:
        raise AssertionError("rank failed below a floating integral boundary")
    if int(carrier_rank(np.array([9]), np.nextafter(0.1, 0.0))[0]) != 10:
        raise AssertionError("rank failed for alpha immediately below 0.1")
    if not math.isclose(float(beta_absolute_deviation(np.array([0]), alpha)[0]), alpha):
        raise AssertionError("empty-carrier loss is not alpha")

    n = 37
    rank = int(carrier_rank(np.array([n]), alpha)[0])
    b = n + 1 - rank
    numerical, _ = quad(
        lambda value: abs(value - (1.0 - alpha))
        * beta_distribution.pdf(value, rank, b),
        0.0,
        1.0,
        points=[1.0 - alpha],
        epsabs=2e-13,
    )
    closed = float(beta_absolute_deviation(np.array([n]), alpha)[0])
    if not math.isclose(closed, numerical, rel_tol=2e-11, abs_tol=2e-13):
        raise AssertionError("closed Beta absolute moment disagrees with quadrature")

    for kappa in kappas:
        for m in (1, 17, int((1.0 + kappa) * 200)):
            risk = exact_risk(alpha, kappa, m)
            if risk > float(finite_upper(kappa, m)) + 2e-13:
                raise AssertionError(f"finite upper bound violated at kappa={kappa}, m={m}")

    collapse_grid = np.unique(
        np.r_[
            np.arange(30, 101, dtype=np.int64),
            np.rint(np.geomspace(110.0, 10_000.0, 18)).astype(np.int64),
        ]
    )
    collapse_widths = []
    for effective_size in collapse_grid:
        collapsed = np.array(
            [
                exact_risk(
                    alpha,
                    kappa,
                    int(round((1.0 + kappa) * effective_size)),
                )
                for kappa in kappas
            ]
        )
        collapse_widths.append(float(np.ptp(collapsed) / np.mean(collapsed)))
    collapse_width = collapse_widths[0]
    maximum_collapse_width = max(collapse_widths)
    if maximum_collapse_width > 0.007:
        raise AssertionError(
            "E1 collapse too wide for effective sizes at least 30: "
            f"{maximum_collapse_width:.3%}"
        )

    plateau_x = 10_000.0
    plateau = np.array(
        [
            math.sqrt(plateau_x)
            * exact_risk(alpha, kappa, int((1.0 + kappa) * plateau_x))
            for kappa in kappas
        ]
    )
    limit = float(asymptotic_constant(alpha))
    plateau_error = float(np.max(np.abs(plateau - limit)) / limit)
    if plateau_error > 0.01:
        raise AssertionError(f"E1 plateau misses asymptotic constant: {plateau_error:.3%}")
    return {
        "collapse_relative_width_at_30": collapse_width,
        "collapse_max_relative_width_from_30": maximum_collapse_width,
        "plateau_relative_error_at_1e4": plateau_error,
        "plateau": float(np.mean(plateau)),
        "limit": limit,
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
            "font.size": 9.2,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.2,
            "legend.fontsize": 7.6,
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
    maximum_effective_size: float,
    points: int,
    output: Path,
    run_diagnostics: bool = True,
) -> dict[str, float]:
    """Render vector PDF and return numerical diagnostics."""
    curves = compute_curves(alpha, kappas, maximum_effective_size, points)
    plt = _setup_plotting()
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(7.25, 5.65))
    grid = fig.add_gridspec(2, 2)
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.09,
        top=0.80,
        wspace=0.38,
        hspace=0.58,
    )
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_normalized = fig.add_subplot(grid[0, 1])
    ax_alpha = fig.add_subplot(grid[1, :])
    colors = [BLUE[index * (len(BLUE) - 1) // max(1, len(kappas) - 1)] for index in range(len(kappas))]

    risk_handles = []
    for color, kappa in zip(colors, kappas):
        data = curves[kappa]
        label = fr"${kappa:g}$"
        risk_handle, = ax_raw.plot(
            data["x"], data["risk"], color=color, lw=1.8, label=label
        )
        risk_handles.append(risk_handle)
        ax_normalized.plot(
            data["x"], np.sqrt(data["x"]) * data["risk"], color=color, lw=1.8
        )
        # The finite achievability bound is almost collapsed, but retains
        # exact finite-m lattice terms.
        ax_raw.plot(data["x"], data["upper"], color="0.42", lw=0.75, ls="--", alpha=0.42)
        ax_normalized.plot(
            data["x"], np.sqrt(data["x"]) * data["upper"],
            color="0.42", lw=0.75, ls="--", alpha=0.42,
        )

    boundary = effective_validity_boundary(alpha)
    if maximum_effective_size >= boundary:
        lower_x = np.geomspace(boundary, maximum_effective_size, 300)
        lower = le_cam_constant(alpha) / np.sqrt(lower_x)
        lower_handle, = ax_raw.plot(
            lower_x,
            lower,
            color="#D55E00",
            lw=1.35,
            ls=(0, (5, 2)),
            label="Le Cam floor",
        )
        ax_normalized.hlines(
            le_cam_constant(alpha), boundary, maximum_effective_size,
            color="#D55E00", lw=1.35, ls=(0, (5, 2)),
        )
    else:
        lower_handle, = ax_raw.plot(
            [], [], color="#D55E00", lw=1.35, ls=(0, (5, 2)), label="Le Cam floor"
        )
    finite_handle, = ax_raw.plot([], [], color="0.42", lw=1.0, ls="--", label="finite achievability bound")
    limit = float(asymptotic_constant(alpha))
    limit_handle = ax_normalized.axhline(
        limit,
        color="0.15",
        lw=1.0,
        ls=":",
        label=fr"Gaussian limit ${limit:.4f}$",
    )

    for label, ax in zip(("(a)", "(b)"), (ax_raw, ax_normalized)):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="major", axis="both", color="0.90", lw=0.5)
        ax.set_xlabel(r"effective size $m/(1+\kappa)$")
        ax.text(-0.15, 1.04, label, transform=ax.transAxes, fontsize=9.5)
    ax_raw.set_ylabel(r"exact risk $R(m,\kappa)$")
    ax_normalized.set_ylabel(r"$\sqrt{m/(1+\kappa)}\,R(m,\kappa)$")
    ax_raw.set_title(fr"Exact risk and finite bounds, $\alpha={alpha:g}$")
    ax_normalized.set_title("Scaled risk")
    quantity_handles = (
        Line2D([], [], color=BLUE[-2], lw=1.8, label=r"exact $R(m,\kappa)$"),
        finite_handle,
        lower_handle,
        limit_handle,
    )
    fig.legend(
        handles=quantity_handles,
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.995),
        title="quantity",
        title_fontsize=7.4,
        handlelength=2.0,
        columnspacing=0.8,
    )
    fig.legend(
        handles=risk_handles,
        frameon=False,
        ncol=len(risk_handles),
        loc="upper right",
        bbox_to_anchor=(0.99, 0.995),
        title=r"$\kappa$ (color)",
        title_fontsize=7.4,
        handlelength=1.5,
        columnspacing=0.8,
    )

    alpha_grid = np.linspace(0.035, 0.22, 300)
    ax_alpha.plot(
        alpha_grid,
        asymptotic_constant(alpha_grid),
        color="0.15",
        lw=1.25,
        ls=":",
    )
    plateau_x = 10_000.0
    for index, alpha_value in enumerate(PLATEAU_ALPHAS):
        values = np.array(
            [
                math.sqrt(plateau_x)
                * exact_risk(
                    alpha_value,
                    kappa,
                    int(round((1.0 + kappa) * plateau_x)),
                )
                for kappa in kappas
            ]
        )
        ax_alpha.vlines(
            alpha_value,
            float(np.min(values)),
            float(np.max(values)),
            color=BLUE[-2],
            lw=1.0,
        )
        ax_alpha.plot(
            alpha_value,
            float(np.mean(values)),
            "o",
            color=BLUE[-1],
            ms=4.0,
            label=(
                r"exact at $x=10^4$"
                if index == 0
                else None
            ),
        )
    ax_alpha.grid(which="major", color="0.90", lw=0.5)
    ax_alpha.set_xlabel(r"miscoverage $\alpha$")
    ax_alpha.set_ylabel(r"scaled risk $\sqrt{x}\,R$")
    ax_alpha.set_title(
        r"Dependence on miscoverage, $x=m/(1+\kappa)$"
    )
    ax_alpha.text(-0.08, 1.04, "(c)", transform=ax_alpha.transAxes, fontsize=9.5)
    ax_alpha.legend(frameon=False, loc="lower right", handlelength=1.5)

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

    check = (
        run_checks()
        if run_diagnostics and math.isclose(alpha, 0.1) and kappas == DEFAULT_KAPPAS
        else {}
    )
    return check


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, nargs="+", default=list(DEFAULT_KAPPAS))
    parser.add_argument("--max-effective", type=float, default=10_000.0)
    parser.add_argument("--points", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "figure" / "e1_scalar.pdf")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kappas = tuple(dict.fromkeys(float(value) for value in args.kappa))
    if any(value <= 0.0 for value in kappas):
        raise ValueError("all kappa values must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if args.points < 3:
        raise ValueError("points must be at least three")
    if args.max_effective <= 0.0:
        raise ValueError("max-effective must be positive")

    started = time.perf_counter()
    if args.check_only:
        diagnostics = run_checks()
        print("E1 checks PASS:", ", ".join(f"{key}={value:.6g}" for key, value in diagnostics.items()))
        return 0
    maximum = min(args.max_effective, 500.0) if args.quick else args.max_effective
    points = min(args.points, 14) if args.quick else args.points
    diagnostics = make_figure(
        alpha=args.alpha,
        kappas=kappas,
        maximum_effective_size=maximum,
        points=points,
        output=args.output,
        run_diagnostics=not args.quick,
    )
    elapsed = time.perf_counter() - started
    detail = ""
    if diagnostics:
        detail = (
            f"; collapse width@30={diagnostics['collapse_relative_width_at_30']:.3%}"
            f"; plateau error@1e4={diagnostics['plateau_relative_error_at_1e4']:.3%}"
        )
    print(f"wrote {args.output.resolve()} in {elapsed:.2f}s{detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
