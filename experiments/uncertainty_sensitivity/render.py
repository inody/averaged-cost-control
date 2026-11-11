"""Render the Figure 6 uncertainty-sensitivity panel from a sweep summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "data" / "reference_sensitivity.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "paper-figures" / "uncertainty-variance-cost-sensitivity.pdf"

AVERAGED_COLOR = "#168ca6"
NOMINAL_COLOR = "#e36c09"
BASE_VARIANCES = {
    "van-der-pol": {"model variance": 3.0, "initial-state variance": 0.2**2 / 3.0},
    "stuart-landau": {"model variance": 1.45**2 / 3.0, "initial-state variance": 0.2**2 / 3.0},
}


def configure() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}",
        "font.family": "serif",
        "font.size": 13.5,
        "axes.labelsize": 14.5,
        "axes.titlesize": 15.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def axes_setup(axis, model: str, sweep: str) -> str:
    model_name = "Van der Pol" if model == "van-der-pol" else r"Stuart--Landau"
    uncertainty = "model-parameter variance" if sweep == "model variance" else "initial-state variance"
    axis.set_title(f"{model_name}: {uncertainty}")
    axis.set_xlabel(r"$\mathrm{Var}(\omega)$" if sweep == "model variance" else r"$\mathrm{Var}(\xi_k)$")
    axis.axvline(BASE_VARIANCES[model][sweep], color="0.35", linestyle=":", linewidth=1.25, zorder=0)
    axis.margins(x=0.05, y=0.08)
    return "omega_variance" if sweep == "model variance" else "initial_component_variance"


def render(summary_path: Path, output: Path) -> None:
    configure()
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "passed" or summary.get("unique_law_count") != 34:
        raise RuntimeError("sensitivity summary is incomplete")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.35), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.88), w_pad=0.035, h_pad=0.035)
    for row, model in enumerate(("van-der-pol", "stuart-landau")):
        for col, sweep in enumerate(("model variance", "initial-state variance")):
            axis = axes[row, col]
            xfield = axes_setup(axis, model, sweep)
            records = sorted(
                (record for record in summary["records"] if record["model"] == model and record["sweep"] == sweep),
                key=lambda record: record["law"][xfield],
            )
            if len(records) != 9:
                raise RuntimeError(f"expected nine points for {model}/{sweep}")
            xvalues = [record["law"][xfield] for record in records]
            for method, linestyle in (("neural", "-"), ("static", "--")):
                axis.plot(
                    xvalues,
                    [record[method]["averaged_cost"] for record in records],
                    color=AVERAGED_COLOR,
                    linestyle=linestyle,
                    linewidth=2.35,
                    marker="o",
                    markersize=4.5,
                )
                axis.plot(
                    xvalues,
                    [record[method]["nominal_cost"] for record in records],
                    color=NOMINAL_COLOR,
                    linestyle=linestyle,
                    linewidth=2.35,
                    marker="o",
                    markersize=4.5,
                )
            axis.set_ylabel("Finite-horizon cost")
    handles = [
        Line2D([0], [0], color=AVERAGED_COLOR, linewidth=2.7, marker="o", label="Averaged design"),
        Line2D([0], [0], color=NOMINAL_COLOR, linewidth=2.7, marker="o", label="Nominal design"),
        Line2D([0], [0], color="0.15", linewidth=2.4, linestyle="-", label="Time-dependent neural feedback"),
        Line2D([0], [0], color="0.15", linewidth=2.4, linestyle="--", label="Static linear feedback"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.7,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    render(arguments.summary, arguments.output)
