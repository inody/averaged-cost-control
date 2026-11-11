# Averaged-cost control with shared feedback

This repository reproduces the numerical results and figures for the paper
*Averaged-Cost Control with Shared Feedback under Model Uncertainty*.

It implements two methods for the same finite-horizon shared-feedback design problem:

- **Nonlinear adjoint method:** time-dependent neural feedback optimized with forward state solves, backward adjoints, and a projected policy-parameter residual.
- **Linear moment method:** static linear feedback designed on a first-order linearization using conditional second moments, Lyapunov evaluation, and Anderson--Moore updates with Armijo backtracking.

Both controller classes are evaluated on the original nonlinear Van der Pol and Stuart--Landau oscillators. The experiments include uncertainty in the model parameter and initial state.

## Numerical laws

- Van der Pol: `omega ~ Uniform[-2, 4]` and `xi = (1, 1) + Uniform([-0.2, 0.2]^2)`.
- Stuart--Landau: `omega ~ Uniform[-2.1, 0.8]` and `xi = (0.5, 0.5) + Uniform([-0.2, 0.2]^2)`.
- The nominal design fixes both quantities at their means.

## Environment

The maintained environment uses Python 3.11, NumPy 2.4.2, SciPy 1.17.0, and Matplotlib 3.10.8. A LaTeX installation is required for publication-style figure text.

```bash
make setup
```

## Recompute the main paper results and figures

```bash
make paper-figures
```

This command starts from the fixed neural-policy warm starts in `data/warm_starts.json` and recomputes:

- averaged and nominal time-dependent neural feedback;
- averaged and nominal static linear feedback;
- independent final-law evaluations on the nonlinear oscillators;
- the four nonlinear trajectory figures;
- stationarity residuals and linear gradient checks.

It also renders the runtime-scaling figure from the recorded hardware timings and Figure 6 from the committed 34-law sensitivity summary. The six PDFs are collected under:

```text
outputs/paper-figures/figures/
```

The neural optimization uses a 400-iteration cap and can take several minutes. Existing output directories are not overwritten.

## Recompute Figure 6 from controller redesigns

After `make recompute`, run:

```bash
make sensitivity
```

For each plotted model-parameter or initial-state variance, this command redesigns the averaged neural policy and averaged static gain for that distribution, evaluates both averaged and nominal controllers on the same distribution, and renders `uncertainty-variance-cost-sensitivity.pdf`. The nominal controllers remain fixed because the mean parameter--initial-state pair is unchanged.

The sweep contains 34 unique uncertainty laws and may take tens of minutes. It is resumable: completed point files under `outputs/uncertainty-sensitivity/points/` are reused.

To render Figure 6 directly from the committed reference summary without rerunning the optimizations:

```bash
make sensitivity-figure
```

## Other entry points

Recompute the Table 1 laws and four trajectory figures without recorded-data figures:

```bash
make recompute
```

Run both the main recomputation and the full Figure 6 redesign sweep:

```bash
make full-recompute
```

Re-run the hardware-dependent scaling benchmark:

```bash
make benchmark
```

Absolute wall times are not expected to match the paper exactly. The reproducible claim is the contrast between particle-dependent nonlinear work and fixed-dimensional linear moment propagation.

## Verification

```bash
make verify
```

The checks cover model definitions, uncertainty laws, warm-start inventory, nonlinear adjoint gradients, linear moment gradients, sensitivity-data completeness, and release scope. After a numerical recomputation, `make verify-results` compares its costs, gains, residuals, inventories, and figure names with the release references.

## Repository scope

The repository contains only the code, fixed warm starts, recorded timing and sensitivity data, and tests needed for paper reproduction. Intermediate exploratory studies, presentation sources, manuscript sources, and development records are intentionally excluded.

## License

The reproduction code is distributed under the [MIT License](LICENSE). Third-party packages remain under their respective licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
