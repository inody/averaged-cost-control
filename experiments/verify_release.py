"""Verify frozen replay figures, sensitivity records, and paper correspondence."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from experiments.frozen_figures import EXPECTED_FIGURES, load_manifest, load_trajectory_data
from experiments.paper_correspondence import verify as verify_table

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SENSITIVITY = ROOT / "data" / "reference_sensitivity.json"
REFERENCE_RESULTS = ROOT / "data" / "reference_results.json"
NEURAL_TOLERANCE = 1e-4
STATIC_TOLERANCE = 1e-8
SENSITIVITY_RTOL = 5e-3
SENSITIVITY_ATOL = 5e-5


def verify_pdf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 100:
        raise RuntimeError(f"missing or empty PDF: {path}")
    data = path.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise RuntimeError(f"invalid PDF container: {path}")
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)], capture_output=True, text=True, timeout=15, check=False
        )
    except FileNotFoundError as error:
        raise RuntimeError("pdfinfo is required to validate replayed PDFs") from error
    if result.returncode != 0:
        raise RuntimeError(f"pdfinfo could not read {path}: {result.stderr.strip()}")
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.M)
    if not match or int(match.group(1)) < 1:
        raise RuntimeError(f"PDF has no readable pages: {path}")


def _finite_number(value, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise RuntimeError(f"nonfinite sensitivity value: {label}")
    return number


def _record_key(record: dict) -> tuple[str, str, float]:
    return str(record["model"]), str(record["sweep"]), float(record["scale"])


def verify_sensitivity(summary_path: Path, reference_path: Path | None = None) -> None:
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "passed":
        raise RuntimeError("sensitivity summary did not pass")
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 36:
        raise RuntimeError("sensitivity summary must contain 36 plotted records")
    expected_models = {"van-der-pol", "stuart-landau"}
    expected_sweeps = {"model variance", "initial-state variance"}
    expected_scales = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}
    keys = [_record_key(record) for record in records]
    if len(set(keys)) != 36:
        raise RuntimeError("sensitivity plotted-record identities are not unique")
    if {key[0] for key in keys} != expected_models or {key[1] for key in keys} != expected_sweeps:
        raise RuntimeError("sensitivity model/sweep inventory mismatch")
    for model in expected_models:
        for sweep in expected_sweeps:
            if {key[2] for key in keys if key[:2] == (model, sweep)} != expected_scales:
                raise RuntimeError(f"sensitivity scale inventory mismatch: {model}/{sweep}")

    unique_laws: set[str] = set()
    baseline_by_model: dict[str, list[dict]] = {model: [] for model in expected_models}
    for record in records:
        model, sweep, scale = _record_key(record)
        law = record.get("law")
        if not isinstance(law, dict):
            raise RuntimeError(f"missing law: {model}/{sweep}/{scale}")
        law_numbers = {
            key: _finite_number(law[key], f"{model}/{sweep}/{scale}/law/{key}")
            for key in ("mean_omega", "omega_half_width", "omega_variance", "initial_radius", "initial_component_variance")
        }
        if law_numbers["omega_half_width"] < 0 or law_numbers["omega_variance"] < 0 or law_numbers["initial_radius"] < 0 or law_numbers["initial_component_variance"] < 0:
            raise RuntimeError(f"negative sensitivity variance/radius: {model}/{sweep}/{scale}")
        unique_laws.add(json.dumps(law, sort_keys=True, separators=(",", ":")))
        for method, tolerance in (("neural", NEURAL_TOLERANCE), ("static", STATIC_TOLERANCE)):
            block = record.get(method)
            if not isinstance(block, dict):
                raise RuntimeError(f"missing sensitivity method: {model}/{sweep}/{scale}/{method}")
            averaged = _finite_number(block["averaged_cost"], f"{model}/{sweep}/{scale}/{method}/averaged")
            nominal = _finite_number(block["nominal_cost"], f"{model}/{sweep}/{scale}/{method}/nominal")
            reduction = _finite_number(block["reduction_percent"], f"{model}/{sweep}/{scale}/{method}/reduction")
            residual = _finite_number(block["stationarity_residual"], f"{model}/{sweep}/{scale}/{method}/residual")
            if averaged <= 0 or nominal <= 0 or residual < 0 or residual > tolerance:
                raise RuntimeError(f"invalid sensitivity certification: {model}/{sweep}/{scale}/{method}")
            recomputed = 100.0 * (nominal - averaged) / nominal
            if not np.isclose(reduction, recomputed, rtol=2e-12, atol=2e-11):
                raise RuntimeError(f"sensitivity reduction mismatch: {model}/{sweep}/{scale}/{method}")
        if scale == 1.0:
            baseline_by_model[model].append(record)
    if len(unique_laws) != 34 or summary.get("unique_law_count") != 34:
        raise RuntimeError("sensitivity summary must contain 34 unique laws")
    for model, baseline in baseline_by_model.items():
        if len(baseline) != 2:
            raise RuntimeError(f"missing duplicated baseline records: {model}")
        first, second = baseline
        if first["law"] != second["law"] or first["neural"] != second["neural"] or first["static"] != second["static"]:
            raise RuntimeError(f"baseline duplication mismatch: {model}")

    if reference_path is not None:
        reference = json.loads(reference_path.read_text())
        reference_records = {_record_key(record): record for record in reference["records"]}
        if set(reference_records) != set(keys):
            raise RuntimeError("reference sensitivity identity mismatch")
        for record in records:
            key = _record_key(record)
            expected = reference_records[key]
            if record["law"] != expected["law"]:
                raise RuntimeError(f"sensitivity reference law mismatch: {key}")
            for method in ("neural", "static"):
                for field in ("averaged_cost", "nominal_cost", "reduction_percent"):
                    actual_value = float(record[method][field])
                    expected_value = float(expected[method][field])
                    if not np.isclose(actual_value, expected_value, rtol=SENSITIVITY_RTOL, atol=SENSITIVITY_ATOL):
                        raise RuntimeError(f"sensitivity reference mismatch: {key}/{method}/{field}")
    print(f"Sensitivity verification passed: {summary_path}")


def verify_sensitivity_points(points: Path, summary_path: Path | None = None, source_summary: Path | None = None) -> None:
    from experiments.uncertainty_sensitivity.run import build_campaign_contract, canonical_digest, validate_reusable_point

    campaign = json.loads((points.parent / "campaign_contract.json").read_text())
    campaign_core = {key: value for key, value in campaign.items() if key != "contract_sha256"}
    campaign_hash = canonical_digest(campaign_core)
    if campaign.get("contract_sha256") != campaign_hash:
        raise RuntimeError("sensitivity campaign contract checksum mismatch")
    if source_summary is not None and campaign != build_campaign_contract(source_summary):
        raise RuntimeError("sensitivity campaign does not match current source and baseline")
    summary_records = None
    if summary_path is not None:
        summary = json.loads(summary_path.read_text())
        summary_records = {_record_key(record): record for record in summary["records"]}
    paths = sorted(points.glob("*/*.json"))
    if len(paths) != 34:
        raise RuntimeError(f"expected 34 sensitivity point files, found {len(paths)}")
    contracts: set[str] = set()
    nominal_by_model: dict[str, tuple[str, np.ndarray]] = {}
    identities: set[tuple[str, str, float]] = set()
    expected_identities = {
        (model, "baseline", 1.0)
        for model in ("van-der-pol", "stuart-landau")
    } | {
        (model, sweep, scale)
        for model in ("van-der-pol", "stuart-landau")
        for sweep in ("model variance", "initial-state variance")
        for scale in (0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0)
    }
    for path in paths:
        point = json.loads(path.read_text())
        if point.get("status") != "passed":
            raise RuntimeError(f"sensitivity point did not pass: {path}")
        identity = (str(point.get("model")), str(point.get("sweep")), float(point.get("scale")))
        if identity in identities:
            raise RuntimeError(f"duplicate sensitivity point identity: {identity}")
        identities.add(identity)
        contract = point.get("campaign_contract_sha256")
        if not isinstance(contract, str) or len(contract) != 64:
            raise RuntimeError(f"missing campaign contract in {path}")
        contracts.add(contract)
        point_input = point.get("point_input_contract", {})
        if point_input.get("campaign_contract_sha256") != contract or (
            point_input.get("model"), point_input.get("sweep"), float(point_input.get("scale"))
        ) != identity:
            raise RuntimeError(f"point input contract does not match identity: {path}")
        if contract != campaign_hash or not validate_reusable_point(point, point_input):
            raise RuntimeError(f"invalid sensitivity point or campaign: {path}")
        if summary_records is not None:
            sweeps = ("model variance", "initial-state variance") if identity[1] == "baseline" else (identity[1],)
            for sweep in sweeps:
                record = summary_records[(identity[0], sweep, identity[2])]
                if record["law"] != point["law"]:
                    raise RuntimeError(f"summary/point law mismatch: {path}")
                for method in ("neural", "static"):
                    for field in ("averaged_cost", "nominal_cost", "reduction_percent"):
                        if record[method][field] != point[method][field]:
                            raise RuntimeError(f"summary/point result mismatch: {path}/{method}/{field}")
                    residual = point[method]["optimization"]["selected_tight_stationarity"]["projected_gradient_residual"] if method == "neural" else point[method]["averaged_design"]["stationarity_residual"]
                    if record[method]["stationarity_residual"] != residual:
                        raise RuntimeError(f"summary/point residual mismatch: {path}/{method}")
        controller = point.get("nominal_controller", {})
        neural_hash = controller.get("neural_parameters_sha256")
        gain = np.asarray(controller.get("static_gain"), dtype=float)
        if not isinstance(neural_hash, str) or len(neural_hash) != 64 or gain.shape != (2,) or not np.isfinite(gain).all():
            raise RuntimeError(f"invalid nominal-controller record: {path}")
        previous = nominal_by_model.setdefault(identity[0], (neural_hash, gain))
        if previous[0] != neural_hash or not np.array_equal(previous[1], gain):
            raise RuntimeError(f"nominal controller changed within sensitivity sweep: {identity[0]}")
        final_count = int(point["law"]["final_particle_count"])
        if final_count <= 0:
            raise RuntimeError(f"invalid final inventory count: {path}")
        for method, tolerance in (("neural", NEURAL_TOLERANCE), ("static", STATIC_TOLERANCE)):
            block = point[method]
            averaged = _finite_number(block["averaged_cost"], f"{path}/{method}/averaged")
            nominal = _finite_number(block["nominal_cost"], f"{path}/{method}/nominal")
            reduction = _finite_number(block["reduction_percent"], f"{path}/{method}/reduction")
            residual = (
                block["optimization"]["selected_tight_stationarity"]["projected_gradient_residual"]
                if method == "neural" else block["averaged_design"]["stationarity_residual"]
            )
            certified = (
                block["optimization"].get("stationarity_certified")
                if method == "neural" else block["averaged_design"].get("stationarity_certified")
            )
            if certified is not True or _finite_number(residual, f"{path}/{method}/residual") > tolerance:
                raise RuntimeError(f"uncertified sensitivity point: {path}/{method}")
            if not np.isclose(reduction, 100.0 * (nominal - averaged) / nominal, rtol=2e-12, atol=2e-11):
                raise RuntimeError(f"point reduction mismatch: {path}/{method}")
            for design, expected_cost in (("averaged", averaged), ("nominal", nominal)):
                final = block[f"{design}_final"]
                if method == "neural" and final.get("status") != "finite full inventory":
                    raise RuntimeError(f"incomplete final rollout: {path}/{method}/{design}")
                verification = final.get("verification", {})
                if verification.get("particle_count") != final_count or verification.get("all_particles_retained") is not True or verification.get("all_costs_finite") is not True:
                    raise RuntimeError(f"final inventory verification failed: {path}/{method}/{design}")
                if not np.isclose(_finite_number(final["cost"], f"{path}/{method}/{design}/cost"), expected_cost, rtol=2e-12, atol=2e-11):
                    raise RuntimeError(f"final cost record mismatch: {path}/{method}/{design}")
    if identities != expected_identities:
        raise RuntimeError("sensitivity point identity inventory mismatch")
    if len(contracts) != 1:
        raise RuntimeError("sensitivity points use inconsistent campaign contracts")
    print(f"Sensitivity point verification passed: {points}")


def verify_release(figures: Path, sensitivity: Path = REFERENCE_SENSITIVITY) -> None:
    load_manifest()
    load_trajectory_data()
    existing = {path.name for path in figures.glob("*.pdf")}
    if existing != set(EXPECTED_FIGURES):
        raise RuntimeError(f"six-figure inventory mismatch: {sorted(existing)}")
    for name in EXPECTED_FIGURES:
        verify_pdf(figures / name)
    verify_sensitivity(sensitivity, REFERENCE_SENSITIVITY if sensitivity != REFERENCE_SENSITIVITY else None)
    verify_table(REFERENCE_RESULTS)
    print("Frozen replay release verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--sensitivity", type=Path, default=REFERENCE_SENSITIVITY)
    parser.add_argument("--reference-sensitivity", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--results", type=Path, default=REFERENCE_RESULTS)
    parser.add_argument("--manuscript", type=Path)
    args = parser.parse_args()
    if args.figures is not None:
        load_manifest()
        load_trajectory_data()
        existing = {path.name for path in args.figures.glob("*.pdf")}
        if existing != set(EXPECTED_FIGURES):
            raise RuntimeError(f"six-figure inventory mismatch: {sorted(existing)}")
        for name in EXPECTED_FIGURES:
            verify_pdf(args.figures / name)
    verify_sensitivity(args.sensitivity, args.reference_sensitivity)
    if args.points is not None:
        baseline = args.results if args.results != REFERENCE_RESULTS else None
        verify_sensitivity_points(args.points, args.sensitivity, baseline)
    verify_table(args.results, args.manuscript)
    print("Release verification passed.")


if __name__ == "__main__":
    main()
