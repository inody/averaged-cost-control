"""Verify a full recomputation against the paper release inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference_results.json"
FIGURE_NAMES = {
    "nonlinear-vdp-direct-cost-neural-feedback-full-results.pdf",
    "nonlinear-stuart-landau-direct-cost-neural-feedback-full-results.pdf",
    "nonlinear-vdp-static-linear-feedback.pdf",
    "nonlinear-stuart-landau-static-linear-feedback.pdf",
}


def verify(summary_path: Path, figures_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    reference = json.loads(REFERENCE.read_text())
    failures: list[str] = []

    for case, methods in reference["neural"].items():
        for method, expected in methods.items():
            actual = summary["cases"][case]["policies"]["direct_neural"][method]
            cost = float(actual["final_evaluation"]["cost"])
            residual = float(actual["selected_tight_stationarity"]["projected_gradient_residual"])
            if not np.isclose(cost, expected["nonlinear_evaluation_cost"], rtol=2e-6, atol=2e-8):
                failures.append(f"neural cost mismatch: {case}/{method}: {cost}")
            if residual > 1e-4 or not actual["stationarity_certified"]:
                failures.append(f"neural stationarity failure: {case}/{method}: {residual}")
            if actual["final_evaluation"]["status"] != "finite full inventory":
                failures.append(f"incomplete neural inventory: {case}/{method}")

    for case, methods in reference["static_linear"].items():
        for method, expected in methods.items():
            actual = summary["linear_cases"][case]["methods"][method]
            linearized_cost = float(actual["evaluation_cost"])
            nonlinear_cost = float(actual["nonlinear_evaluation"]["cost"])
            gain = np.asarray(actual["gain"], dtype=float)
            residual = float(actual["stationarity_residual"])
            if not np.isclose(
                linearized_cost, expected["linearized_evaluation_cost"], rtol=2e-7, atol=2e-9
            ):
                failures.append(f"linearized cost mismatch: {case}/{method}: {linearized_cost}")
            if not np.isclose(
                nonlinear_cost, expected["nonlinear_evaluation_cost"], rtol=2e-6, atol=2e-8
            ):
                failures.append(f"nonlinear static cost mismatch: {case}/{method}: {nonlinear_cost}")
            if not np.allclose(gain, expected["gain"], rtol=2e-7, atol=2e-8):
                failures.append(f"linear gain mismatch: {case}/{method}: {gain.tolist()}")
            if residual > 1e-8 or not actual["stationarity_certified"]:
                failures.append(f"linear stationarity failure: {case}/{method}: {residual}")
            if actual["gradient_check"]["maximum_absolute_difference"] > 2e-7:
                failures.append(f"linear gradient check failure: {case}/{method}")
            if actual["nonlinear_evaluation"]["status"] != "finite full inventory":
                failures.append(f"incomplete nonlinear static inventory: {case}/{method}")

    existing = {path.name for path in figures_path.glob("*.pdf")}
    if existing != FIGURE_NAMES:
        failures.append(f"trajectory figure inventory mismatch: {sorted(existing)}")

    if failures:
        raise RuntimeError("\n".join(failures))
    print("Full recomputation verification passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    verify(args.summary, args.figures)
