"""Verify a full recomputation against the paper release inventory."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from experiments.neural_adjoint import JointPopulation, verify_rollout

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference_results.json"
FIGURE_NAMES = {
    "nonlinear-vdp-direct-cost-neural-feedback-full-results.pdf",
    "nonlinear-stuart-landau-direct-cost-neural-feedback-full-results.pdf",
    "nonlinear-vdp-static-linear-feedback.pdf",
    "nonlinear-stuart-landau-static-linear-feedback.pdf",
}
VERIFICATION_CONTRACT = {
    "schema_version": 1,
    "summary_schema_version": 2,
    "inventory_sizes": {
        "averaged_design": 256,
        "averaged_validation": 256,
        "nominal": 1,
        "final": 2048,
        "display": 32,
    },
    "neural_stationarity_tolerance": 1e-4,
    "linear_stationarity_tolerance": 1e-8,
    "linear_gradient_error_tolerance": 2e-7,
    "neural_cost_rtol": 2e-6,
    "neural_cost_atol": 2e-8,
    "linear_cost_rtol": 2e-7,
    "linear_cost_atol": 2e-9,
    "nonlinear_static_cost_rtol": 2e-6,
    "nonlinear_static_cost_atol": 2e-8,
    "gain_rtol": 2e-7,
    "gain_atol": 2e-8,
}


def _finite(value, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise RuntimeError(f"nonfinite value: {label}")
    return number


def _population(record: dict, label: str, expected_size: int) -> JointPopulation:
    if record.get("size") != expected_size:
        raise RuntimeError(f"particle inventory size mismatch: {label}")
    population = JointPopulation(
        omega=np.asarray(record.get("omega"), dtype=float),
        initial_states=np.asarray(record.get("initial_states"), dtype=float),
        weights=np.asarray(record.get("weights"), dtype=float),
    )
    if population.size != expected_size:
        raise RuntimeError(f"particle inventory shape mismatch: {label}")
    if not np.isclose(population.weights.sum(), 1.0, rtol=0.0, atol=2e-13):
        raise RuntimeError(f"particle weights are not normalized: {label}")
    return population


def _verify_evaluation(record: dict, population: JointPopulation, label: str) -> float:
    if record.get("status") != "finite full inventory":
        raise RuntimeError(f"incomplete evaluation: {label}")
    if record.get("successful_subset_averaged") is not False:
        raise RuntimeError(f"subset averaging was not explicitly disabled: {label}")
    cost = _finite(record.get("cost"), f"{label}/cost")
    per_particle = np.asarray(record.get("per_particle_cost"), dtype=float)
    verification = verify_rollout(
        {"cost": cost, "per_particle_cost": per_particle}, population, label
    )
    saved = record.get("verification")
    if not isinstance(saved, dict):
        raise RuntimeError(f"missing rollout verification record: {label}")
    if saved.get("particle_count") != population.size or saved.get("all_particles_retained") is not True or saved.get("all_costs_finite") is not True:
        raise RuntimeError(f"false or incomplete rollout verification: {label}")
    if not np.isclose(
        _finite(saved.get("weighted_cost_recomputed"), f"{label}/weighted cost"),
        verification["weighted_cost_recomputed"], rtol=2e-12, atol=2e-11,
    ):
        raise RuntimeError(f"saved weighted-cost reconstruction mismatch: {label}")
    return cost


def _verify_display(record: dict, population: JointPopulation, label: str) -> None:
    if record.get("status") != "finite full inventory" or record.get("successful_subset_averaged") is not False:
        raise RuntimeError(f"incomplete display evaluation: {label}")
    cost = _finite(record.get("cost"), f"{label}/cost")
    verification = record.get("verification")
    if not isinstance(verification, dict) or verification.get("particle_count") != population.size:
        raise RuntimeError(f"display inventory mismatch: {label}")
    if verification.get("all_particles_retained") is not True or verification.get("all_costs_finite") is not True:
        raise RuntimeError(f"display verification failed: {label}")
    weighted = _finite(verification.get("weighted_cost_recomputed"), f"{label}/weighted cost")
    if not np.isclose(cost, weighted, rtol=2e-12, atol=2e-11):
        raise RuntimeError(f"display weighted-cost mismatch: {label}")


def _verify_pdf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 100:
        raise RuntimeError(f"missing or empty trajectory PDF: {path.name}")
    data = path.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise RuntimeError(f"invalid trajectory PDF: {path.name}")
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)], capture_output=True, text=True, timeout=15, check=False
        )
    except FileNotFoundError as error:
        raise RuntimeError("pdfinfo is required for result verification") from error
    if result.returncode != 0 or not re.search(r"^Pages:\s+[1-9][0-9]*\s*$", result.stdout, re.M):
        raise RuntimeError(f"unreadable trajectory PDF: {path.name}")


def verify(summary_path: Path, figures_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    reference = json.loads(REFERENCE.read_text())
    contract = VERIFICATION_CONTRACT
    if summary.get("schema_version") != contract["summary_schema_version"] or summary.get("status") != "complete":
        raise RuntimeError("summary is not a completed schema-v2 recomputation")
    if summary.get("protocol", {}).get("all_or_nothing_full_inventory") is not True:
        raise RuntimeError("summary does not declare all-or-nothing full inventories")
    if set(summary.get("cases", {})) != set(reference["neural"]):
        raise RuntimeError("nonlinear case inventory mismatch")
    if set(summary.get("linear_cases", {})) != set(reference["static_linear"]):
        raise RuntimeError("linear case inventory mismatch")

    populations: dict[str, dict[str, JointPopulation]] = {}
    for case, case_record in summary["cases"].items():
        inventories = case_record.get("particle_inventories", {})
        if set(inventories) != set(contract["inventory_sizes"]):
            raise RuntimeError(f"particle inventory names mismatch: {case}")
        populations[case] = {
            name: _population(record, f"{case}/{name}", contract["inventory_sizes"][name])
            for name, record in inventories.items()
        }

    for case, methods in reference["neural"].items():
        actual_methods = summary["cases"][case]["policies"]["direct_neural"]
        if set(actual_methods) != set(methods):
            raise RuntimeError(f"neural method inventory mismatch: {case}")
        for method, expected in methods.items():
            actual = actual_methods[method]
            if actual.get("optimizer", {}).get("terminated") is not True:
                raise RuntimeError(f"neural optimizer did not complete: {case}/{method}")
            if actual.get("stationarity_certified") is not True:
                raise RuntimeError(f"neural stationarity is not certified: {case}/{method}")
            tight = actual.get("selected_tight_stationarity", {})
            if tight.get("status") != "finite":
                raise RuntimeError(f"neural tight stationarity did not complete: {case}/{method}")
            residual = _finite(tight.get("projected_gradient_residual"), f"{case}/{method}/neural residual")
            if not 0.0 <= residual <= contract["neural_stationarity_tolerance"]:
                raise RuntimeError(f"neural stationarity failure: {case}/{method}: {residual}")
            parameters = np.asarray(actual.get("selected_parameters"), dtype=float)
            if parameters.shape != (81,) or not np.isfinite(parameters).all():
                raise RuntimeError(f"invalid selected neural parameters: {case}/{method}")
            cost = _verify_evaluation(
                actual.get("final_evaluation", {}), populations[case]["final"], f"{case}/{method}/neural/final"
            )
            _verify_display(
                actual.get("display_evaluation", {}), populations[case]["display"], f"{case}/{method}/neural/display"
            )
            if not np.isclose(
                cost, expected["nonlinear_evaluation_cost"],
                rtol=contract["neural_cost_rtol"], atol=contract["neural_cost_atol"],
            ):
                raise RuntimeError(f"neural cost mismatch: {case}/{method}: {cost}")

    nonlinear_case = {"linear-vdp": "nonlinear-vdp", "linear-stuart-landau": "nonlinear-stuart-landau"}
    for case, methods in reference["static_linear"].items():
        actual_methods = summary["linear_cases"][case]["methods"]
        if set(actual_methods) != set(methods):
            raise RuntimeError(f"static method inventory mismatch: {case}")
        for method, expected in methods.items():
            actual = actual_methods[method]
            linearized_cost = _finite(actual.get("evaluation_cost"), f"{case}/{method}/linearized cost")
            gain = np.asarray(actual.get("gain"), dtype=float)
            if gain.shape != (2,) or not np.isfinite(gain).all():
                raise RuntimeError(f"invalid linear gain: {case}/{method}")
            residual = _finite(actual.get("stationarity_residual"), f"{case}/{method}/linear residual")
            gradient_error = _finite(
                actual.get("gradient_check", {}).get("maximum_absolute_difference"),
                f"{case}/{method}/gradient check",
            )
            if actual.get("stationarity_certified") is not True or not 0.0 <= residual <= contract["linear_stationarity_tolerance"]:
                raise RuntimeError(f"linear stationarity failure: {case}/{method}: {residual}")
            if not 0.0 <= gradient_error <= contract["linear_gradient_error_tolerance"]:
                raise RuntimeError(f"linear gradient check failure: {case}/{method}: {gradient_error}")
            ncase = nonlinear_case[case]
            nonlinear_cost = _verify_evaluation(
                actual.get("nonlinear_evaluation", {}), populations[ncase]["final"], f"{case}/{method}/nonlinear/final"
            )
            _verify_display(
                actual.get("nonlinear_display_evaluation", {}), populations[ncase]["display"], f"{case}/{method}/nonlinear/display"
            )
            if not np.isclose(
                linearized_cost, expected["linearized_evaluation_cost"],
                rtol=contract["linear_cost_rtol"], atol=contract["linear_cost_atol"],
            ):
                raise RuntimeError(f"linearized cost mismatch: {case}/{method}: {linearized_cost}")
            if not np.isclose(
                nonlinear_cost, expected["nonlinear_evaluation_cost"],
                rtol=contract["nonlinear_static_cost_rtol"], atol=contract["nonlinear_static_cost_atol"],
            ):
                raise RuntimeError(f"nonlinear static cost mismatch: {case}/{method}: {nonlinear_cost}")
            if not np.allclose(gain, expected["gain"], rtol=contract["gain_rtol"], atol=contract["gain_atol"]):
                raise RuntimeError(f"linear gain mismatch: {case}/{method}: {gain.tolist()}")

    existing = {path.name for path in figures_path.glob("*.pdf")}
    if existing != FIGURE_NAMES:
        raise RuntimeError(f"trajectory figure inventory mismatch: {sorted(existing)}")
    for name in FIGURE_NAMES:
        _verify_pdf(figures_path / name)
    print("Full recomputation verification passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    verify(args.summary, args.figures)
