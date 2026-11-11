"""Render the paper runtime-scaling figure from recorded hardware timings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.figure_style import (
    AXES_LABEL_SIZE,
    AXES_TITLE_SIZE,
    COMPARATOR_COLOR,
    FONT_SIZE,
    LEGEND_SIZE,
    PRIMARY_METHOD_COLOR,
    TICK_LABEL_SIZE,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "runtime_scaling.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "runtime-scaling.pdf"


def configure_plotting() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}",
        "font.family": "serif",
        "font.size": FONT_SIZE + 2,
        "axes.labelsize": AXES_LABEL_SIZE + 2,
        "axes.titlesize": AXES_TITLE_SIZE + 2,
        "xtick.labelsize": TICK_LABEL_SIZE + 2,
        "ytick.labelsize": TICK_LABEL_SIZE + 2,
        "legend.fontsize": LEGEND_SIZE + 2,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "figure.dpi": 150,
        "savefig.dpi": 200,
    })


def render(data_path: Path, output_path: Path) -> None:
    data = json.loads(data_path.read_text())
    configure_plotting()
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 2.6), sharey=True, constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    labels = (("van-der-pol", "Van der Pol"), ("stuart-landau", "Stuart--Landau"))
    for axis, (key, title) in zip(axes, labels):
        rows = data["cases"][key]["rows"]
        sample_counts = np.asarray([row["M"] for row in rows])
        axis.loglog(
            sample_counts,
            [row["linear_moment_median_seconds"] for row in rows],
            "s-", color=PRIMARY_METHOD_COLOR, linewidth=3.6, markersize=5.0,
            label="Linear moment method",
        )
        axis.loglog(
            sample_counts,
            [row["nonlinear_adjoint_median_seconds"] for row in rows],
            "o-", color=COMPARATOR_COLOR, linewidth=3.6, markersize=5.0,
            label="Nonlinear adjoint method",
        )
        axis.set_title(title)
        axis.set_xlabel(r"Initial-state sample count $M$")
        axis.set_xticks(sample_counts, [str(value) for value in sample_counts])
    axes[0].set_ylabel(r"Median wall time $[\mathrm{s}]$")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncol=2, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.data, args.output)
