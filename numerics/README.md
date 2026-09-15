# Numerical experiments

`e1_scalar.py` and `e2_cellwise.py` generate the finite carrier experiments.
`e3_cqr_pipeline.py` runs learned CQR under a continuous covariate shift.
NumPy and SciPy perform the numerical work; Matplotlib writes vector PDFs.

## Reproduction

Run from the directory containing `numerics/` with Python 3.11, NumPy, SciPy, and
Matplotlib available:

```bash
python numerics/e1_scalar.py
python numerics/e2_cellwise.py
python numerics/e3_cqr_pipeline.py \
  --save-results numerics/e3_cqr_results.npz \
  --manifest numerics/e3_cqr_run.json
```

Outputs:

- `figure/e1_scalar.pdf`
- `figure/e2_cellwise.pdf`
- `figure/e3_cqr.pdf`
- `numerics/e3_cqr_results.npz` with every E3 per-replication result
- `numerics/e3_cqr_run.json` for the canonical E3 run manifest

Fast render and canonical deterministic checks:

```bash
python numerics/e1_scalar.py --quick
python numerics/e2_cellwise.py --quick
python numerics/e3_cqr_pipeline.py --quick
python numerics/e1_scalar.py --check-only
python numerics/e2_cellwise.py --check-only
python numerics/e3_cqr_pipeline.py --check-only
```

All scripts accept `--alpha` and a list-valued `--kappa`. E2 also accepts a
list-valued `--K`; E2 and E3 expose the seed and repetition/grid controls:

```bash
python numerics/e1_scalar.py --alpha 0.1 --kappa 1 3 9 27
python numerics/e2_cellwise.py \
  --alpha 0.1 --kappa 1 3 9 --K 23 64 256 --seed 20260826
python numerics/e3_cqr_pipeline.py \
  --alpha 0.1 --kappa 1 3 --reps 200 --seed 20260826 \
  --n-fixed 8192 --n-validity 2048 --m-fixed 4096 \
  --m-grid 128 192 256 384 512 \
  --n-grid 128 256 512 1024 2048 --x-points 1025 \
  --hidden-units 20 --mlp-steps 1800 \
  --output figure/e3_cqr.pdf \
  --save-results numerics/e3_cqr_results.npz \
  --manifest numerics/e3_cqr_run.json
```

Redraw E3 after changing labels, colors, or layout without rerunning any fit:

```bash
python numerics/e3_cqr_pipeline.py \
  --load-results numerics/e3_cqr_results.npz \
  --output figure/e3_cqr.pdf
```

The full E2 command evaluates all nine `(kappa, K)` pairs at 25 effective
sample sizes. It enforces at least `10000` replications per Monte Carlo point.
`--quick` caps the grid and uses at most `1000` replications for render smoke
testing. `--check-only` runs the canonical `alpha=0.1` diagnostics and writes
no figure.

The full E3 command uses `200` replications, `m in {128,192,256,384,512}`
with `n=2048` in panel (a), `n=8192` in panel (b), `m=4096` in panel (c), a
1025-point covariate grid, and seed `20260826`. The pinball MLP has one 20-unit
tanh layer and 1800 Adam steps with an optimizer stream fixed across
replications. `--quick` uses 12 replications, smaller sample grids, 201
quadrature points, and 400 optimizer steps.

The canonical command saves all 14 raw Monte Carlo matrices in
`numerics/e3_cqr_results.npz`: six panel-(a) coverage matrices, four panel-(b)
functional matrices, and four panel-(c) functional matrices. Each matrix has
one row per replication. Embedded metadata records the resolved configuration,
RNG streams, software versions, Git HEAD, script hash, runtime, and diagnostics.
`--load-results` reads these matrices and only renders the PDF.

`numerics/e3_cqr_run.json` records the command and SHA256 hashes of the script,
PDF, and results archive. Use separate output, results, and manifest paths for
quick or exploratory runs so they do not replace the canonical artifacts.

## Computation and checks

E1 evaluates

\[
R(m,\kappa)=
\mathbb E_{N\sim\mathrm{Bin}(m,(1+\kappa)^{-1})}
\mathbb E|B_{N,k_N}-(1-\alpha)|
\]

from the closed incomplete-beta formula. Its checks cover the degenerate
rank, the closed formula against quadrature, the finite bound, collapse over
effective sizes from `30` to `10000`, and the normalized plateau at `10000`.

E2 evaluates \(M_p=\{\mathbb E L_p^p\}^{1/p}\) from the one-cell
Binomial--Beta identity. For `p=2,8`, an exact Beta moment recurrence replaces
quadrature; the canonical check compares the recurrence with one-dimensional
SciPy quadrature. Monte Carlo samples the full multinomial count vector and
independent conditional Beta order statistics. Seeds are keyed by
`(seed, kappa, K, m, panel)`, so grid order does not change results.

The sample-level Jensen check
\(
\overline{L_p}\le(\overline{L_p^p})^{1/p}
\)
holds deterministically. The population inequality
\(
\mathbb E L_p\le M_p
\)
is checked with a `5 SE + 5e-5` numerical tolerance.  The absolute term covers
rare rank transitions for which a finite sample can have zero empirical
variance despite a nonzero population tail probability. A finite Monte Carlo point may
exceed the exact curve within sampling error; at `p=1`, the population
quantities are equal. Pointwise plot intervals use the normal approximation
`mean +/- 1.96 SE`; rare-event tails can make them anticonservative at small
effective sizes for `p=8`.

E3 calibrates the exponential tilt by solving

\[
\frac{\theta}{2}\coth(\theta/2)=1+\kappa,
\qquad
w_\theta(x)=\frac{\theta e^{\theta x}}{e^\theta-1}.
\]

The training fold fits a pinball MLP and affine quantile regression on source
data. The calibration fold constructs the exact weighted threshold, the
unweighted split threshold, and the historical population-normalized
diagnostic `Q(beta_m; Fhat_m^w)`. Gaussian conditional coverage is evaluated
through `Phi`; target marginal coverage and both `L2(Q_X)` functionals use
composite trapezoidal quadrature. No target responses are simulated.

Panel (a) reports Monte Carlo means and pointwise `mean +/- 1.96 SE`
intervals. Panels (b) and (c) report medians and pointwise empirical central
95% bands. The script stops if an exact weighted threshold is infinite on a
functional panel; all 200 canonical repetitions are finite. This is a
finite-run summary: the exact rule can have a rare `+infinity` threshold
outside the realized sample. The population-normalized rule attains
`+infinity` in some panel-(a) repetitions, where its coverage is one.

## Measured runtimes

Measured on the local Apple Silicon machine with Python 3.11 in `cqr_cs`:

| Command | Runtime |
|---|---:|
| `python numerics/e1_scalar.py` | 12.8 s |
| `python numerics/e2_cellwise.py` | 26.4 s |
| `python numerics/e1_scalar.py --quick` | 9.3 s |
| `python numerics/e2_cellwise.py --quick` | 1.7 s |
| canonical E3 command with saved results | 246.4 s |
| `python numerics/e3_cqr_pipeline.py --load-results numerics/e3_cqr_results.npz` | 0.5 s |
| `python numerics/e3_cqr_pipeline.py --quick` | 2.3 s |

## Figure notes

**E1.** Scalar carrier experiment with `alpha=0.1` and
`kappa in {1,3,9,27}`. Blue curves are exact Binomial--Beta mixtures. The
initial plateau comes from ranks with `k_N=N+1`, including the no-carrier
event; the selected threshold is `+infinity` and the error is `alpha`. The Le
Cam floor is restricted to its validity range. The exact finite bound retains
small finite-sample dependence on `kappa`. The plateau is the Gaussian
constant `sqrt(alpha(1-alpha)) E|Z| = sqrt(2 alpha(1-alpha)/pi)`, where
`E|Z|=sqrt(2/pi)` for a standard normal `Z`. Bound constants are not
optimized.

**E2.** Cellwise carrier experiment with `alpha=0.1`,
`kappa in {1,3,9}`, and `K in {23,64,256}`. Exact and Monte Carlo curves cover
the full Cartesian grid. Monte Carlo uses `10000` replications per point,
pointwise normal-approximation 95% intervals, and seed `20260826`. The normalized exact plateau is
`(kappa,K)`-independent asymptotically; the normalized mean risk retains a
finite-`K` Jensen gap for `p>1`. The `p=infinity` panel adds `K=1024` at
`kappa=3` and effective size `10000`, and fits `c sqrt(log K)` as an empirical
scaling probe. In panel (a), each lower dotted Fano curve is labeled by `K`;
horizontal dotted guides in panels (b1)--(b3) are exact Gaussian plateaus.
The `p=8` overshoot near effective size `10` is a finite-sample carrier-rank
transition before the Gaussian regime.

**E3.** The continuous experiment uses `alpha=0.1`, `kappa in {1,3}`, and
known exponential-tilt weights. Panel (a) compares exact weighted,
population-normalized, and unweighted calibration with misspecified affine
endpoints. Panel (b) shows the two target `L2` functionals for exact weighted
CQR with MLP endpoints and an `m^{-1/2}` reference. Panel (c) compares the MLP
and affine endpoint regimes at fixed large calibration size. Color, line
style, and marker meanings are listed below each panel.

Section 6 of the main article contains the experiment descriptions and the E3
figure. Section SM5 of the supplement contains the E1 and E2 figures and
additional computational details.

## Distributed artifact metadata

Local absolute paths in the E3 JSON manifest and embedded NPZ metadata were
replaced by relative paths. Numerical array payloads are unchanged. The manifest
retains the original results-archive hash and records the distributed archive hash.
The three Python scripts and three reference figure PDFs are unchanged.
