"""Redesign averaged controllers across the Figure 6 uncertainty laws."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "uncertainty-sensitivity"
POINTS = OUTPUT / "points"
SOURCE_SUMMARY = ROOT / "outputs" / "paper-figures" / "joint-study" / "summary.json"
SOURCE_SUMMARY_SHA256 = ""
WARM_STARTS = ROOT / "data" / "warm_starts.json"

from experiments.joint_uncertainty.run import (
    StaticLinearPolicy,
    digest,
    evaluate_full,
    legendre_uniform,
    initial_moment,
    linear_moment_evaluation,
    midpoint_uniform,
    optimize_neural,
    policy_from_record,
)
from experiments.models import case_data  # noqa: E402
from experiments.neural_adjoint import product_population, rollout, verify_rollout  # noqa: E402
from experiments.linear_moment import (  # noqa: E402
    STATIC_STATIONARITY_TOLERANCE,
    anderson_moore_iteration,
)

SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
LOW_SCALES = (0.75, 0.5, 0.25, 0.0)
HIGH_SCALES = (1.25, 1.5, 1.75, 2.0)
MODEL_SPECS = {
    "van-der-pol": {
        "case": "nonlinear-vdp",
        "linear_case": "linear-vdp",
        "mean_omega": 1.0,
        "base_half_width": 3.0,
        "base_radius": 0.2,
    },
    "stuart-landau": {
        "case": "nonlinear-stuart-landau",
        "linear_case": "linear-stuart-landau",
        "mean_omega": -0.65,
        "base_half_width": 1.45,
        "base_radius": 0.2,
    },
}
INITIAL_SEEDS = {
    "nonlinear-vdp": {"design": 6201, "validation": 6202},
    "nonlinear-stuart-landau": {"design": 7201, "validation": 7202},
}
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_SOURCE_FILES = (
    "experiments/uncertainty_sensitivity/run.py",
    "experiments/joint_uncertainty/run.py",
    "experiments/models.py",
    "experiments/neural_adjoint.py",
    "experiments/neural_policy.py",
    "experiments/linear_moment.py",
    "experiments/verify_results.py",
    "experiments/verify_release.py",
    "pyproject.toml",
    "uv.lock",
)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    temporary.replace(path)


def canonical_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_campaign_contract(
    source_summary: Path,
    *,
    source_files: tuple[Path, ...] | None = None,
    warm_starts: Path = WARM_STARTS,
) -> dict:
    files = source_files or tuple(ROOT / relative for relative in CONTRACT_SOURCE_FILES)
    source_hashes = {}
    for path in files:
        resolved = Path(path).resolve()
        try:
            label = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            label = resolved.name
        source_hashes[label] = digest(resolved)
    core = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "baseline_summary_sha256": digest(source_summary),
        "warm_starts_sha256": digest(warm_starts),
        "reference_results_sha256": digest(ROOT / "data" / "reference_results.json"),
        "reference_sensitivity_sha256": digest(ROOT / "data" / "reference_sensitivity.json"),
        "source_files": source_hashes,
        "protocol": {
            "scales": list(SCALES),
            "low_scales": list(LOW_SCALES),
            "high_scales": list(HIGH_SCALES),
            "model_specs": MODEL_SPECS,
            "initial_seeds": INITIAL_SEEDS,
            "design_rule": "32 midpoint model nodes by 8 antithetic initial states",
            "validation_rule": "32 Gauss--Legendre model nodes by 8 independent antithetic initial states",
            "final_rule": "128 Gauss--Legendre model nodes by 4-by-4 Gauss--Legendre initial states",
            "continuation": "baseline, then separate lower and higher variance branches",
            "neural_stationarity_tolerance": 1e-4,
            "static_stationarity_tolerance": STATIC_STATIONARITY_TOLERANCE,
            "all_or_nothing_final_evaluation": True,
        },
    }
    return {**core, "contract_sha256": canonical_digest(core)}


def ensure_campaign_contract(output: Path, contract: dict) -> Path:
    path = output / "campaign_contract.json"
    configuration = output / "configuration.json"
    if path.exists():
        try:
            saved = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid existing sensitivity contract: {path}") from error
        if saved != contract:
            raise RuntimeError(f"output directory has an incompatible sensitivity contract: {path}")
        return path
    if output.exists() and (configuration.exists() or any(output.iterdir())):
        raise RuntimeError(f"nonempty sensitivity output has no campaign contract: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, contract)
    return path


def point_input_contract(
    campaign_contract_sha256: str,
    model: str,
    sweep: str,
    scale: float,
    start_theta: np.ndarray,
    start_gain: np.ndarray,
) -> dict:
    core = {
        "campaign_contract_sha256": campaign_contract_sha256,
        "model": model,
        "sweep": sweep,
        "scale": float(scale),
        "start_theta_sha256": hashlib.sha256(np.asarray(start_theta, dtype="<f8").tobytes()).hexdigest(),
        "start_gain_sha256": hashlib.sha256(np.asarray(start_gain, dtype="<f8").tobytes()).hexdigest(),
    }
    return {**core, "point_input_sha256": canonical_digest(core)}


def validate_reusable_point(saved: dict, expected_input: dict) -> bool:
    if (
        saved.get("status") != "passed"
        or saved.get("campaign_contract_sha256") != expected_input["campaign_contract_sha256"]
        or saved.get("point_input_contract") != expected_input
    ):
        return False
    try:
        neural = saved["neural"]
        static = saved["static"]
        nominal = saved["nominal_controller"]
        if not isinstance(nominal["neural_parameters_sha256"], str) or len(nominal["neural_parameters_sha256"]) != 64:
            return False
        nominal_gain = np.asarray(nominal["static_gain"], dtype=float)
        if nominal_gain.shape != (2,) or not np.isfinite(nominal_gain).all():
            return False
        values = [
            neural["averaged_cost"], neural["nominal_cost"], neural["reduction_percent"],
            neural["optimization"]["selected_tight_stationarity"]["projected_gradient_residual"],
            static["averaged_cost"], static["nominal_cost"], static["reduction_percent"],
            static["averaged_design"]["stationarity_residual"],
        ]
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            return False
        if not (0.0 <= float(values[3]) <= 1e-4 and 0.0 <= float(values[7]) <= STATIC_STATIONARITY_TOLERANCE):
            return False
        optimization = neural["optimization"]
        if optimization.get("stationarity_certified") is not True or optimization.get("optimizer", {}).get("terminated") is not True:
            return False
        if optimization["selected_tight_stationarity"].get("status") != "finite" or static["averaged_design"].get("stationarity_certified") is not True:
            return False
        theta = np.asarray(optimization["selected_parameters"], dtype=float)
        gain = np.asarray(static["averaged_design"]["gain"], dtype=float)
        if theta.shape != (81,) or gain.shape != (2,) or not np.isfinite(theta).all() or not np.isfinite(gain).all():
            return False
        identity = (expected_input["model"], expected_input["sweep"], float(expected_input["scale"]))
        if (saved["model"], saved["sweep"], float(saved["scale"])) != identity:
            return False
        point_core = {key: value for key, value in expected_input.items() if key != "point_input_sha256"}
        if canonical_digest(point_core) != expected_input["point_input_sha256"]:
            return False
        data = case_data(MODEL_SPECS[identity[0]]["case"])
        rules = build_rules(*identity, data)
        law = saved["law"]
        expected_law = {
            "mean_omega": MODEL_SPECS[identity[0]]["mean_omega"],
            "omega_half_width": rules["omega_half_width"],
            "omega_variance": rules["omega_variance"],
            "initial_radius": rules["x0_radius"],
            "initial_component_variance": rules["initial_component_variance"],
            "design_particle_count": rules["averaged_design"].size,
            "validation_particle_count": rules["averaged_validation"].size,
            "final_particle_count": rules["final"].size,
        }
        for key, value in expected_law.items():
            if not np.isclose(float(law[key]), value, rtol=0.0, atol=2e-13):
                return False
        if not np.allclose(law["initial_center"], rules["x0_center"], rtol=0.0, atol=2e-13):
            return False
        from experiments.verify_results import _verify_evaluation
        for method, block in (("neural", neural), ("static", static)):
            if float(block["averaged_cost"]) <= 0 or float(block["nominal_cost"]) <= 0:
                return False
            recomputed = 100.0 * (float(block["nominal_cost"]) - float(block["averaged_cost"])) / float(block["nominal_cost"])
            if not np.isclose(recomputed, float(block["reduction_percent"]), rtol=2e-12, atol=2e-11):
                return False
            for design in ("averaged", "nominal"):
                cost = _verify_evaluation(block[f"{design}_final"], rules["final"], f"{identity}/{method}/{design}")
                if not np.isclose(cost, float(block[f"{design}_cost"]), rtol=2e-12, atol=2e-11):
                    return False
    except (KeyError, TypeError, ValueError, RuntimeError, IndexError, ZeroDivisionError):
        return False
    return True


def antithetic_initial_rule(center: np.ndarray, radius: float, seed: int):
    if radius == 0.0:
        return center[None, :], np.ones(1)
    draws = np.random.default_rng(seed).uniform(-1.0, 1.0, (4, 2))
    normalized = np.vstack((draws, -draws))
    return center[None, :] + radius * normalized, np.full(8, 1.0 / 8.0)


def legendre_initial_rule(center: np.ndarray, radius: float, order: int = 4):
    if radius == 0.0:
        return center[None, :], np.ones(1)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    weights = weights / 2.0
    first, second = np.meshgrid(nodes, nodes, indexing="ij")
    first_weights, second_weights = np.meshgrid(weights, weights, indexing="ij")
    states = center[None, :] + radius * np.column_stack((first.ravel(), second.ravel()))
    return states, (first_weights * second_weights).ravel()


def omega_rule(mean: float, half_width: float, count: int, family: str):
    if half_width == 0.0:
        return np.asarray([mean]), np.ones(1)
    lower, upper = mean - half_width, mean + half_width
    if family == "midpoint":
        return midpoint_uniform(lower, upper, count)
    if family == "legendre":
        return legendre_uniform(lower, upper, count)
    raise ValueError(family)


def build_rules(model: str, sweep: str, scale: float, data: dict) -> dict:
    spec = MODEL_SPECS[model]
    if sweep == "model variance":
        half_width = spec["base_half_width"] * scale
        radius = spec["base_radius"]
    elif sweep == "initial-state variance":
        half_width = spec["base_half_width"]
        radius = spec["base_radius"] * scale
    elif sweep == "baseline":
        half_width = spec["base_half_width"]
        radius = spec["base_radius"]
    else:
        raise ValueError(sweep)
    center = np.asarray(data["initial"], dtype=float)
    omega_design, omega_design_weights = omega_rule(spec["mean_omega"], half_width, 32, "midpoint")
    omega_validation, omega_validation_weights = omega_rule(spec["mean_omega"], half_width, 32, "legendre")
    omega_final, omega_final_weights = omega_rule(spec["mean_omega"], half_width, 128, "legendre")
    seeds = INITIAL_SEEDS[spec["case"]]
    x0_design, x0_design_weights = antithetic_initial_rule(center, radius, seeds["design"])
    x0_validation, x0_validation_weights = antithetic_initial_rule(center, radius, seeds["validation"])
    x0_final, x0_final_weights = legendre_initial_rule(center, radius, 4)
    return {
        "x0_center": center,
        "x0_radius": radius,
        "omega_half_width": half_width,
        "omega_variance": half_width**2 / 3.0,
        "initial_component_variance": radius**2 / 3.0,
        "omega_design": omega_design,
        "omega_design_weights": omega_design_weights,
        "omega_validation": omega_validation,
        "omega_validation_weights": omega_validation_weights,
        "omega_final": omega_final,
        "omega_final_weights": omega_final_weights,
        "x0_design": x0_design,
        "x0_design_weights": x0_design_weights,
        "x0_validation": x0_validation,
        "x0_validation_weights": x0_validation_weights,
        "x0_final": x0_final,
        "x0_final_weights": x0_final_weights,
        "averaged_design": product_population(
            omega_design, omega_design_weights, x0_design, x0_design_weights
        ),
        "averaged_validation": product_population(
            omega_validation, omega_validation_weights, x0_validation, x0_validation_weights
        ),
        "final": product_population(omega_final, omega_final_weights, x0_final, x0_final_weights),
    }


def point_name(sweep: str, scale: float) -> str:
    if sweep == "baseline":
        return "baseline"
    return f"{sweep.replace('-', '_').replace(' ', '_')}-scale-{scale:.2f}"


def point_path(model: str, sweep: str, scale: float) -> Path:
    return POINTS / model / f"{point_name(sweep, scale)}.json"


def evaluate_static(data, policy, gain, population, label):
    started = time.perf_counter()
    result = rollout(
        data, policy, np.asarray(gain, dtype=float), population, times=None,
        rtol=2e-10, atol=2e-12, max_step_divisor=400.0,
    )
    verification = verify_rollout(result, population, label)
    return {
        "status": "finite full inventory",
        "cost": float(result["cost"]),
        "per_particle_cost": result["per_particle_cost"],
        "successful_subset_averaged": False,
        "nfev": int(result["nfev"]),
        "elapsed_seconds": time.perf_counter() - started,
        "verification": verification,
    }


def optimize_neural_with_retry(data, case: str, rules: dict, warm_template: dict,
                               continuation_theta: np.ndarray, nominal_theta: np.ndarray):
    attempts = []

    def run(start_theta: np.ndarray, source: str):
        warm = copy.deepcopy(warm_template)
        warm["cases"][case]["averaged"]["parameters"] = np.asarray(start_theta, dtype=float).tolist()
        record, policy = optimize_neural(
            data, case, "averaged", rules["averaged_design"], rules["averaged_validation"], warm,
            maxiter=800, maxls=200, restart_stages=5, ftol=0.0, gtol=1e-12,
        )
        attempts.append({"source": source, "record": record})
        return record, policy

    record, policy = run(continuation_theta, "continuation")
    nominal_validation, _ = evaluate_full(
        data, policy_from_record({
            "architecture": record["architecture"]
        }), nominal_theta, rules["averaged_validation"], "redesign/nominal/validation"
    )
    selected_validation = float(record["selection"]["selected_validation_cost"])
    if nominal_validation["cost"] is not None and selected_validation > float(nominal_validation["cost"]):
        retry_record, retry_policy = run(nominal_theta, "nominal-controller restart")
        if float(retry_record["selection"]["selected_validation_cost"]) < selected_validation:
            record, policy = retry_record, retry_policy
    return record, policy, attempts, nominal_validation


def optimize_linear_from_start(data: dict, rules: dict, start_gain: np.ndarray) -> dict:
    parameters = rules["omega_design"]
    weights = rules["omega_design_weights"]
    moment0 = initial_moment(rules["x0_center"], rules["x0_radius"], True)
    started = time.perf_counter()
    iteration = anderson_moore_iteration(
        lambda gain: linear_moment_evaluation(data, gain, moment0, parameters, weights),
        start=np.asarray(start_gain, dtype=float),
    )
    gain = np.asarray(iteration["gain"], dtype=float)
    tight = linear_moment_evaluation(
        data, gain, moment0, parameters, weights, rtol=2e-12, atol=2e-14, divisor=480.0
    )
    if not (
        np.isfinite(tight["cost"])
        and np.all(np.isfinite(tight["gradient"]))
        and tight["stationarity_residual"] <= STATIC_STATIONARITY_TOLERANCE
    ):
        raise RuntimeError("linear residual check failed after continuation")
    evaluation = linear_moment_evaluation(
        data, gain, moment0, rules["omega_final"], rules["omega_final_weights"],
        rtol=2e-12, atol=2e-14, divisor=480.0,
    )
    return {
        "gain": gain,
        "start_gain": np.asarray(start_gain, dtype=float),
        "design_initial_second_moment": moment0,
        "design_cost": tight["cost"],
        "tight_gradient": tight["gradient"],
        "stationarity_residual": tight["stationarity_residual"],
        "stationarity_tolerance": STATIC_STATIONARITY_TOLERANCE,
        "stationarity_certified": True,
        "occupancy_condition_number": tight["occupancy_condition_number"],
        "linear_evaluation_cost": evaluation["cost"],
        "fixed_point_iteration": iteration,
        "elapsed_seconds": time.perf_counter() - started,
    }


def optimize_point(model: str, sweep: str, scale: float, start_theta: np.ndarray,
                   start_gain: np.ndarray, source: dict, warm_template: dict,
                   campaign_contract_sha256: str) -> dict:
    path = point_path(model, sweep, scale)
    input_contract = point_input_contract(
        campaign_contract_sha256, model, sweep, scale, start_theta, start_gain
    )
    if path.exists():
        try:
            saved = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid existing sensitivity point: {path}") from error
        if not validate_reusable_point(saved, input_contract):
            raise RuntimeError(f"incompatible or invalid existing sensitivity point: {path}")
        print(f"reuse {model} {sweep} scale={scale:g}", flush=True)
        return saved
    spec = MODEL_SPECS[model]
    case = spec["case"]
    data = case_data(case)
    rules = build_rules(model, sweep, scale, data)
    nominal_record = source["cases"][case]["policies"]["direct_neural"]["nominal"]
    nominal_theta = np.asarray(nominal_record["selected_parameters"], dtype=float)
    nominal_policy = policy_from_record(nominal_record)
    started = time.perf_counter()
    print(
        f"START {model} {sweep} scale={scale:g} "
        f"Var(omega)={rules['omega_variance']:.8g} Var(xi_k)={rules['initial_component_variance']:.8g}",
        flush=True,
    )

    averaged_record, averaged_policy, attempts, nominal_validation = optimize_neural_with_retry(
        data, case, rules, warm_template, np.asarray(start_theta, dtype=float), nominal_theta
    )
    averaged_theta = np.asarray(averaged_record["selected_parameters"], dtype=float)
    averaged_final, _ = evaluate_full(
        data, averaged_policy, averaged_theta, rules["final"],
        f"redesign/{model}/{sweep}/{scale}/neural/averaged/final",
    )
    nominal_final, _ = evaluate_full(
        data, nominal_policy, nominal_theta, rules["final"],
        f"redesign/{model}/{sweep}/{scale}/neural/nominal/final",
    )
    if averaged_final["cost"] is None or nominal_final["cost"] is None:
        raise RuntimeError(f"neural final rollout failed: {model}/{sweep}/{scale}")

    linear_data = dict(data)
    linear_data.update(case=spec["linear_case"], nonlinear=False)
    linear_averaged = optimize_linear_from_start(linear_data, rules, start_gain)
    nominal_gain = np.asarray(
        source["linear_cases"][spec["linear_case"]]["methods"]["nominal"]["gain"], dtype=float
    )
    static_policy = StaticLinearPolicy()
    static_averaged = evaluate_static(
        data, static_policy, linear_averaged["gain"], rules["final"],
        f"redesign/{model}/{sweep}/{scale}/static/averaged/final",
    )
    static_nominal = evaluate_static(
        data, static_policy, nominal_gain, rules["final"],
        f"redesign/{model}/{sweep}/{scale}/static/nominal/final",
    )
    neural_reduction = 100.0 * (nominal_final["cost"] - averaged_final["cost"]) / nominal_final["cost"]
    static_reduction = 100.0 * (static_nominal["cost"] - static_averaged["cost"]) / static_nominal["cost"]
    result = {
        "status": "passed",
        "source_summary_sha256": SOURCE_SUMMARY_SHA256,
        "campaign_contract_sha256": campaign_contract_sha256,
        "point_input_contract": input_contract,
        "model": model,
        "sweep": sweep,
        "scale": float(scale),
        "nominal_controller": {
            "neural_parameters_sha256": hashlib.sha256(np.asarray(nominal_theta, dtype="<f8").tobytes()).hexdigest(),
            "static_gain": nominal_gain,
        },
        "law": {
            "mean_omega": spec["mean_omega"],
            "omega_half_width": rules["omega_half_width"],
            "omega_variance": rules["omega_variance"],
            "initial_center": rules["x0_center"],
            "initial_radius": rules["x0_radius"],
            "initial_component_variance": rules["initial_component_variance"],
            "design_particle_count": rules["averaged_design"].size,
            "validation_particle_count": rules["averaged_validation"].size,
            "final_particle_count": rules["final"].size,
        },
        "neural": {
            "optimization": averaged_record,
            "attempts": attempts,
            "nominal_validation": nominal_validation,
            "averaged_final": averaged_final,
            "nominal_final": nominal_final,
            "averaged_cost": averaged_final["cost"],
            "nominal_cost": nominal_final["cost"],
            "reduction_percent": neural_reduction,
        },
        "static": {
            "averaged_design": linear_averaged,
            "nominal_gain": nominal_gain,
            "averaged_final": static_averaged,
            "nominal_final": static_nominal,
            "averaged_cost": static_averaged["cost"],
            "nominal_cost": static_nominal["cost"],
            "reduction_percent": static_reduction,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not validate_reusable_point(result, input_contract):
        raise RuntimeError(f"new sensitivity point failed verification: {path}")
    write_json(path, result)
    print(
        f"DONE {model} {sweep} scale={scale:g} elapsed={result['elapsed_seconds']:.1f}s "
        f"neural={averaged_final['cost']:.8g}/{nominal_final['cost']:.8g} "
        f"static={static_averaged['cost']:.8g}/{static_nominal['cost']:.8g}",
        flush=True,
    )
    return result


def existing_or_initial_theta(model: str, source: dict) -> np.ndarray:
    case = MODEL_SPECS[model]["case"]
    return np.asarray(
        source["cases"][case]["policies"]["direct_neural"]["averaged"]["selected_parameters"], dtype=float
    )


def selected_theta(result: dict) -> np.ndarray:
    return np.asarray(result["neural"]["optimization"]["selected_parameters"], dtype=float)


def selected_gain(result: dict) -> np.ndarray:
    return np.asarray(result["static"]["averaged_design"]["gain"], dtype=float)


def update_state(started: float) -> None:
    completed = list(POINTS.glob("*/*.json")) if POINTS.exists() else []
    write_json(OUTPUT / "state.json", {
        "status": "running",
        "completed_point_files": [str(path.relative_to(OUTPUT)) for path in sorted(completed)],
        "completed_count": len(completed),
        "target_unique_laws": 34,
        "elapsed_seconds": time.perf_counter() - started,
    })


def assemble_summary(started: float) -> dict:
    records = []
    for model in MODEL_SPECS:
        baseline = json.loads(point_path(model, "baseline", 1.0).read_text())
        for sweep in ("model variance", "initial-state variance"):
            for scale in SCALES:
                point = baseline if scale == 1.0 else json.loads(point_path(model, sweep, scale).read_text())
                records.append({
                    "model": model,
                    "sweep": sweep,
                    "scale": scale,
                    "law": point["law"],
                    "neural": {
                        "averaged_cost": point["neural"]["averaged_cost"],
                        "nominal_cost": point["neural"]["nominal_cost"],
                        "reduction_percent": point["neural"]["reduction_percent"],
                        "stationarity_residual": point["neural"]["optimization"]["selected_tight_stationarity"]["projected_gradient_residual"],
                    },
                    "static": {
                        "averaged_cost": point["static"]["averaged_cost"],
                        "nominal_cost": point["static"]["nominal_cost"],
                        "reduction_percent": point["static"]["reduction_percent"],
                        "stationarity_residual": point["static"]["averaged_design"]["stationarity_residual"],
                    },
                    "point_file": str(point_path(model, "baseline", 1.0).relative_to(OUTPUT) if scale == 1.0 else point_path(model, sweep, scale).relative_to(OUTPUT)),
                })
    return {
        "status": "passed",
        "description": "Each averaged controller is redesigned for its uncertainty law; nominal controllers remain fixed because the mean pair is unchanged.",
        "scales": list(SCALES),
        "unique_law_count": 34,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(output: Path = OUTPUT, source_summary: Path = SOURCE_SUMMARY) -> None:
    global OUTPUT, POINTS, SOURCE_SUMMARY, SOURCE_SUMMARY_SHA256
    OUTPUT = output.resolve()
    POINTS = OUTPUT / "points"
    SOURCE_SUMMARY = source_summary.resolve()
    if not SOURCE_SUMMARY.is_file():
        raise FileNotFoundError(f"baseline summary not found: {SOURCE_SUMMARY}; run make recompute first")
    started = time.perf_counter()
    SOURCE_SUMMARY_SHA256 = digest(SOURCE_SUMMARY)
    campaign_contract = build_campaign_contract(SOURCE_SUMMARY)
    ensure_campaign_contract(OUTPUT, campaign_contract)
    POINTS.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_SUMMARY.read_text())
    warm_template = json.loads(WARM_STARTS.read_text())
    try:
        source_summary_label = SOURCE_SUMMARY.relative_to(ROOT).as_posix()
    except ValueError:
        source_summary_label = SOURCE_SUMMARY.name
    configuration = {
        "source_summary": source_summary_label,
        "source_summary_sha256": SOURCE_SUMMARY_SHA256,
        "warm_starts": str(WARM_STARTS.relative_to(ROOT)),
        "warm_starts_sha256": digest(WARM_STARTS),
        "scales": list(SCALES),
        "model_specs": MODEL_SPECS,
        "initial_seeds": INITIAL_SEEDS,
        "design_rule": "32 midpoint model nodes by 8 antithetic initial states",
        "validation_rule": "32 Gauss--Legendre model nodes by 8 independent antithetic initial states",
        "final_rule": "128 Gauss--Legendre model nodes by 4-by-4 Gauss--Legendre initial states",
        "continuation": "optimize baseline, then continue separately toward lower and higher variance",
        "nominal_restart": "retry from nominal parameters when their validation cost is below the continuation result",
        "campaign_contract_sha256": campaign_contract["contract_sha256"],
    }
    configuration_path = OUTPUT / "configuration.json"
    if configuration_path.exists():
        if json.loads(configuration_path.read_text()) != configuration:
            raise RuntimeError(f"existing sensitivity configuration is incompatible: {configuration_path}")
    else:
        write_json(configuration_path, configuration)
    update_state(started)

    for model in MODEL_SPECS:
        initial = existing_or_initial_theta(model, source)
        spec = MODEL_SPECS[model]
        initial_gain = np.asarray(
            source["linear_cases"][spec["linear_case"]]["methods"]["averaged"]["gain"], dtype=float
        )
        baseline = optimize_point(
            model, "baseline", 1.0, initial, initial_gain, source, warm_template,
            campaign_contract["contract_sha256"],
        )
        root_theta = selected_theta(baseline)
        root_gain = selected_gain(baseline)
        update_state(started)
        for sweep in ("model variance", "initial-state variance"):
            theta = root_theta.copy()
            gain = root_gain.copy()
            for scale in LOW_SCALES:
                point = optimize_point(
                    model, sweep, scale, theta, gain, source, warm_template,
                    campaign_contract["contract_sha256"],
                )
                theta = selected_theta(point)
                gain = selected_gain(point)
                update_state(started)
            theta = root_theta.copy()
            gain = root_gain.copy()
            for scale in HIGH_SCALES:
                point = optimize_point(
                    model, sweep, scale, theta, gain, source, warm_template,
                    campaign_contract["contract_sha256"],
                )
                theta = selected_theta(point)
                gain = selected_gain(point)
                update_state(started)

    summary = assemble_summary(started)
    write_json(OUTPUT / "summary.json", summary)
    write_json(OUTPUT / "state.json", {
        "status": "passed",
        "completed_count": 34,
        "target_unique_laws": 34,
        "elapsed_seconds": summary["elapsed_seconds"],
        "summary": "summary.json",
    })
    lines = [
        "# Uncertainty-variance redesign sweep", "",
        "Each averaged controller is redesigned for its uncertainty law.  Nominal controllers remain fixed because the mean pair does not change.", "",
        f"- Status: `{summary['status']}`", f"- Unique laws: `{summary['unique_law_count']}`",
        f"- Elapsed seconds: `{summary['elapsed_seconds']:.6g}`", "",
    ]
    for model in MODEL_SPECS:
        lines.extend([f"## {model}", ""])
        for sweep in ("model variance", "initial-state variance"):
            lines.extend([f"### {sweep}", "", "| scale | neural reduction | static reduction |", "|---:|---:|---:|"])
            for record in summary["records"]:
                if record["model"] == model and record["sweep"] == sweep:
                    lines.append(f"| {record['scale']:.2f} | {record['neural']['reduction_percent']:.6g}% | {record['static']['reduction_percent']:.6g}% |")
            lines.append("")
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "passed", "output": str(OUTPUT), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY)
    arguments = parser.parse_args()
    main(arguments.output, arguments.source_summary)
