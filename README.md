# CQR under covariate shift

Code and numerical results for **Conformalized Quantile Regression and Minimax
Limits of Fixed-Score Calibration under Known Covariate Shift**, by Rustam Isaev,
Anton Conrad, Denis Belomestny, Eric Moulines, and Sergey Samsonov.

## Experiments

| Experiment | Script | Reference figure |
| --- | --- | --- |
| E1: scalar fixed-score calibration | [e1_scalar.py](numerics/e1_scalar.py) | [e1_scalar.pdf](figure/e1_scalar.pdf) |
| E2: atomwise fixed-score calibration | [e2_cellwise.py](numerics/e2_cellwise.py) | [e2_cellwise.pdf](figure/e2_cellwise.pdf) |
| E3: learned CQR under exponential covariate shift | [e3_cqr_pipeline.py](numerics/e3_cqr_pipeline.py) | [e3_cqr.pdf](figure/e3_cqr.pdf) |

E1 and E2 accompany the supplementary materials. E3 appears in Section 6 of the
main article. All experiments use synthetic distributions; no external dataset
is required. The scripts use NumPy, SciPy, and Matplotlib.

## Reproduce the figures

Use Python 3.11. Package versions recorded for the canonical E3 run are listed
in [requirements.txt](requirements.txt). Run these commands from this directory:

```sh
python -m pip install -r requirements.txt
python numerics/e1_scalar.py
python numerics/e2_cellwise.py
python numerics/e3_cqr_pipeline.py --load-results numerics/e3_cqr_results.npz --output figure/e3_cqr.pdf
```

The first two commands recompute the E1 and E2 experiments. The last command
redraws E3 from the included results without retraining. To rerun all 200 E3
replications and save fresh outputs separately:

```sh
python numerics/e3_cqr_pipeline.py --output figure/e3_cqr_rerun.pdf --save-results numerics/e3_cqr_rerun.npz --manifest numerics/e3_cqr_rerun.json
```

Canonical E3 settings: alpha 0.1; kappa 1 and 3; seed 20260826; 200 replications;
calibration sizes 128, 192, 256, 384, 512; training sizes 128, 256, 512, 1024,
2048; 1025 quadrature points; 20 tanh hidden units; 1800 Adam steps.

Detailed commands, panel-specific sample sizes, numerical checks, statistical
summaries, and measured runtimes are in [numerics/README.md](numerics/README.md).

The reference E1 and E2 PDFs record Matplotlib 3.11.1; the E3 PDF records
Matplotlib 3.8.0. Font and math-text layout can differ across Matplotlib
versions. The checked-in PDFs preserve the figures used in the article.

## Checks

```sh
python numerics/e1_scalar.py --check-only
python numerics/e2_cellwise.py --check-only
python numerics/e3_cqr_pipeline.py --check-only
```

The checks cover exact risk identities, moment calculations, rank conventions,
and calibration diagnostics. They do not replace the complete experiments.

## Saved results and provenance

[numerics/e3_cqr_results.npz](numerics/e3_cqr_results.npz) contains all 14 E3
Monte Carlo result matrices, each with 200 replications. Embedded JSON metadata
records the simulation configuration, random-number streams, software versions,
and diagnostics. [numerics/e3_cqr_run.json](numerics/e3_cqr_run.json) records the
canonical run and hashes of its script, figure, and result archive.

Absolute machine-local paths in the distributed metadata have been replaced by
relative paths. All 14 numerical array payloads, Python scripts, and reference
figures are unchanged. The manifest records both the original result-archive
hash and the hash of the distributed archive.

[INDEX.txt](INDEX.txt) lists the experiment files and their roles.
