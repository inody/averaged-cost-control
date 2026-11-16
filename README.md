# Averaged-cost control with shared feedback

This repository reproduces the numerical evidence for *Averaged-Cost Control with Shared Feedback under Model Uncertainty*. It implements a nonlinear adjoint method for time-dependent neural feedback and a linear moment method for static feedback designed on oscillator linearizations. Both controller classes are evaluated on the original nonlinear Van der Pol and Stuart--Landau systems.

## Requirements

- Python 3.11 and [uv](https://docs.astral.sh/uv/)
- A LaTeX installation with Times-compatible fonts, `dvipng`, and Ghostscript (`gs`) for PDF/PNG figure rendering
- Poppler (`pdfinfo`) for PDF integrity checks

On Ubuntu, install the figure-system dependencies with:

```bash
sudo apt-get install -y cm-super dvipng ghostscript poppler-utils texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

The Python environment is locked to NumPy 2.4.2, SciPy 1.17.0, and Matplotlib 3.10.8.

```bash
make setup
```

## Fast workflow: replay all six figures

```bash
make figures
```

This regenerates all six manuscript PDFs under `outputs/frozen-figures/` from compact committed numerical inputs. It runs no optimization and no ODE integration. `data/frozen/manifest.json` maps each manuscript figure to its inputs and records SHA-256 checksums.

```bash
make verify
```

`make verify` runs the tests, validates the frozen-data manifest and arrays, replays all six figures in a temporary directory, checks the exact filename inventory, reads every PDF with Poppler, verifies the 34-law/36-record sensitivity structure, and regenerates the Table 1 display values. It does not start the long numerical campaigns.

## Full numerical workflow

```bash
make reproduce
```

This starts from the fixed distributed neural-policy warm starts in `data/warm_starts.json`, recomputes the main averaged and nominal controllers, runs the complete 34-law sensitivity redesign sweep, renders six figures, regenerates Table 1 rows, and performs final numerical and structural verification. Existing output is not overwritten. The sensitivity stage is resumable only when its source, lock file, warm starts, seeds, protocol, baseline summary, and continuation inputs match the saved campaign contract.

The full workflow can take tens of minutes. It deliberately excludes the machine-dependent timing benchmark. The fixed warm starts are part of the declared reproduction boundary: the earlier teacher-training process that produced them is not regenerated.

Useful staged aliases are:

```bash
make recompute       # main four-controller study and strict result checks
make sensitivity     # full sensitivity sweep; resumes compatible point files
make verify-results  # recheck a completed main study
make full-recompute  # alias for make reproduce
```

## Timing records and optional benchmark

Figure 2 uses `data/runtime_scaling.json`, which records measurements from the original hardware. Absolute wall times are not expected to transfer across machines; the relevant comparison is the observed scaling contrast between particle propagation and fixed-dimensional moment propagation.

To collect a new, machine-specific timing record after `make recompute`:

```bash
make benchmark
```

Timing is kept outside numerical acceptance checks.

## Paper correspondence

Generate the Table 1 rows from the committed reference values:

```bash
uv run python -m experiments.paper_correspondence
```

An optional local manuscript checkout can be checked without becoming a public dependency:

```bash
uv run python -m experiments.paper_correspondence --manuscript /path/to/body.tex
```

The six figure numbers, filenames, inputs, and replay command are listed in `data/frozen/manifest.json`.

## Numerical laws

- Van der Pol: `omega ~ Uniform[-2, 4]`, `xi = (1, 1) + Uniform([-0.2, 0.2]^2)`.
- Stuart--Landau: `omega ~ Uniform[-2.1, 0.8]`, `xi = (0.5, 0.5) + Uniform([-0.2, 0.2]^2)`.
- Nominal design fixes the model parameter and initial state at their means.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The reproduction code is distributed under the [MIT License](LICENSE); third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
