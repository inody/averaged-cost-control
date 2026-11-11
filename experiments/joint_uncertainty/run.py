"""Run the joint model/initial-state uncertainty study and render its figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from experiments.figure_style import (
    AXES_LABEL_SIZE,
    AXES_TITLE_SIZE,
    FONT_SIZE,
    LEGEND_SIZE,
    TICK_LABEL_SIZE,
)
from experiments.models import case_data
from experiments.neural_policy import DirectNeuralPolicy, NeuralFeedbackOperator, PolicyEvaluation
from experiments.neural_adjoint import (
    JointPopulation,
    objective_gradient,
    product_population,
    projected_gradient_norm,
    projected_gradient_residual,
    rollout,
    verify_rollout,
)
from experiments.linear_moment import (
    ARMIJO_CONSTANT,
    ARMIJO_CONTRACTION,
    INITIAL_GAIN,
    STATIC_STATIONARITY_TOLERANCE,
    anderson_moore_iteration,
    policy_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "joint-uncertainty-reproduction"
FIGURES = OUTPUT / "figures"
WARM_STARTS = ROOT / "data" / "warm_starts.json"
METHODS = ("averaged", "nominal")
CASES = ("nonlinear-vdp", "nonlinear-stuart-landau")
NEURAL_MAXITER = 400
CHECKPOINT_INTERVAL = 10
NEURAL_STATIONARITY_TOLERANCE = 1e-4
NEURAL_RELATIVE_OBJECTIVE_CHANGE_TOLERANCE = 1e-7
STUART_LANDAU_OMEGA_BOUNDS = (-2.1, 0.8)
STUART_LANDAU_NOMINAL_PARAMETER = sum(STUART_LANDAU_OMEGA_BOUNDS) / 2.0
SEEDS = {
    "nonlinear-vdp": {
        "x0_design": 20261601,
        "omega_validation": 20261602,
        "x0_validation": 20261603,
        "x0_final": 20261604,
        "omega_display": 20261605,
        "x0_display": 20261606,
    },
    "nonlinear-stuart-landau": {
        "x0_design": 20261611,
        "omega_validation": 20261612,
        "x0_validation": 20261613,
        "x0_final": 20261614,
        "omega_display": 20261615,
        "x0_display": 20261616,
    },
}
RADII = {"nonlinear-vdp": 0.2, "nonlinear-stuart-landau": 0.2}
NAMES = {
    ("nonlinear-vdp", "direct_neural"): "nonlinear-vdp-direct-cost-neural-feedback-full-results.pdf",
    ("nonlinear-stuart-landau", "direct_neural"): "nonlinear-stuart-landau-direct-cost-neural-feedback-full-results.pdf",
    ("nonlinear-vdp", "static_linear"): "nonlinear-vdp-static-linear-feedback.pdf",
    ("nonlinear-stuart-landau", "static_linear"): "nonlinear-stuart-landau-static-linear-feedback.pdf",
}


@dataclass(frozen=True)
class StaticLinearPolicy:
    """Static linear feedback evaluated on the original nonlinear oscillator."""

    parameter_count: int = 2
    parameter_bounds: tuple[tuple[float, float], ...] = ((-np.inf, np.inf), (-np.inf, np.inf))

    def evaluate(self, time: float, states: np.ndarray, parameters: np.ndarray) -> PolicyEvaluation:
        del time
        states = np.asarray(states, dtype=float)
        gain = np.asarray(parameters, dtype=float).reshape(2)
        controls = -(states @ gain)
        count = len(states)
        return PolicyEvaluation(
            gains=np.tile(gain, (count, 1)),
            controls=controls,
            control_state_jacobian=-np.tile(gain, (count, 1)),
            control_parameter_jacobian=-states.copy(),
        )


def policy_from_record(record: dict) -> DirectNeuralPolicy:
    """Reconstruct a neural policy from a saved optimization record."""

    architecture = record["architecture"]
    operator = NeuralFeedbackOperator(
        horizon=5.0,
        state_scale=np.asarray(architecture["state_scale"], dtype=float),
        control_scale=float(architecture["control_scale"]),
        hidden_width=int(architecture["hidden_width"]),
    )
    return DirectNeuralPolicy(operator)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(value):
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(native(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def midpoint_uniform(lower: float, upper: float, count: int):
    nodes = lower + (upper - lower) * (np.arange(count) + 0.5) / count
    return nodes, np.full(count, 1.0 / count)


def legendre_uniform(lower: float, upper: float, count: int):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return 0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes, weights / 2.0


def stratified_uniform(lower: float, upper: float, count: int, seed: int):
    rng = np.random.default_rng(seed)
    probabilities = (np.arange(count) + rng.uniform(size=count)) / count
    return lower + (upper - lower) * probabilities, np.full(count, 1.0 / count)


def sample_initial(center: np.ndarray, radius: float, count: int, seed: int):
    return center[None, :] + np.random.default_rng(seed).uniform(-radius, radius, (count, 2))


def rules(case: str, data: dict) -> dict:
    center = np.asarray(data["initial"], dtype=float)
    radius = RADII[case]
    seeds = SEEDS[case]
    if case == "nonlinear-vdp":
        omega_design, omega_design_weights = midpoint_uniform(-2.0, 4.0, 32)
        omega_final, omega_final_weights = legendre_uniform(-2.0, 4.0, 128)
        omega_validation, omega_validation_weights = stratified_uniform(-2.0, 4.0, 32, seeds["omega_validation"])
        omega_display, omega_display_weights = stratified_uniform(-2.0, 4.0, 32, seeds["omega_display"])
        omega_metadata = {"family": "Uniform", "lower": -2.0, "upper": 4.0, "mean": 1.0}
    else:
        lower, upper = STUART_LANDAU_OMEGA_BOUNDS
        omega_design, omega_design_weights = midpoint_uniform(lower, upper, 32)
        omega_final, omega_final_weights = legendre_uniform(lower, upper, 128)
        omega_validation, omega_validation_weights = stratified_uniform(
            lower, upper, 32, seeds["omega_validation"]
        )
        omega_display, omega_display_weights = stratified_uniform(
            lower, upper, 32, seeds["omega_display"]
        )
        omega_metadata = {
            "family": "Uniform",
            "lower": lower,
            "upper": upper,
            "mean": STUART_LANDAU_NOMINAL_PARAMETER,
        }
    x0_design = sample_initial(center, radius, 8, seeds["x0_design"])
    x0_validation = sample_initial(center, radius, 8, seeds["x0_validation"])
    x0_final = sample_initial(center, radius, 16, seeds["x0_final"])
    x0_display = sample_initial(center, radius, 32, seeds["x0_display"])
    avg_design = product_population(omega_design, omega_design_weights, x0_design, np.full(8, 1 / 8))
    avg_validation = product_population(omega_validation, omega_validation_weights, x0_validation, np.full(8, 1 / 8))
    nominal = product_population(np.asarray([omega_metadata["mean"]]), np.ones(1), center[None, :], np.ones(1))
    final = product_population(omega_final, omega_final_weights, x0_final, np.full(16, 1 / 16))
    display = JointPopulation(omega_display, x0_display, np.full(32, 1 / 32))
    return {
        "omega_metadata": omega_metadata,
        "omega_design": omega_design,
        "omega_design_weights": omega_design_weights,
        "omega_validation": omega_validation,
        "omega_validation_weights": omega_validation_weights,
        "omega_final": omega_final,
        "omega_final_weights": omega_final_weights,
        "omega_display": omega_display,
        "omega_display_weights": omega_display_weights,
        "x0_center": center,
        "x0_radius": radius,
        "x0_design": x0_design,
        "x0_validation": x0_validation,
        "x0_final": x0_final,
        "x0_display": x0_display,
        "averaged_design": avg_design,
        "averaged_validation": avg_validation,
        "nominal": nominal,
        "final": final,
        "display": display,
    }


def population_record(pop: JointPopulation) -> dict:
    return {"size": pop.size, "omega": pop.omega, "initial_states": pop.initial_states, "weights": pop.weights}


def evaluate_full(data, policy, parameters, population, label, times=None):
    started = time.perf_counter()
    try:
        result = rollout(data, policy, np.asarray(parameters), population, times=times, rtol=2e-10, atol=2e-12, max_step_divisor=400.0)
        verification = verify_rollout(result, population, label)
        record = {
            "status": "finite full inventory",
            "cost": result["cost"],
            "per_particle_cost": result["per_particle_cost"],
            "nfev": result["nfev"],
            "elapsed_seconds": time.perf_counter() - started,
            "verification": verification,
            "successful_subset_averaged": False,
        }
        return record, result
    except Exception as error:
        return {
            "status": "full-inventory rollout failed",
            "cost": None,
            "particle_count_attempted": population.size,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "successful_subset_averaged": False,
            "all_or_nothing_evaluation": True,
        }, None


def checkpoint(data, policy, theta, population, iteration, label, stationarity=None):
    evaluation, _ = evaluate_full(data, policy, theta, population, label)
    record = {"iteration": int(iteration), "parameters": np.asarray(theta),
              "admissible": evaluation["status"] == "finite full inventory", "validation": evaluation}
    if stationarity is not None:
        record["default_stationarity"] = stationarity
    return record


def optimize_neural(
    data, case, method, design_pop, validation_pop, warm_starts, *,
    maxiter=NEURAL_MAXITER, maxls=60, restart_stages=1, ftol=1e-13, gtol=2e-7,
):
    warm_start = warm_starts["cases"][case][method]
    settings = warm_start["operator"]
    operator = NeuralFeedbackOperator(
        float(settings["horizon"]),
        np.asarray(settings["state_scale"], dtype=float),
        float(settings["control_scale"]),
        hidden_width=int(settings["hidden_width"]),
    )
    policy = DirectNeuralPolicy(operator)
    start = np.asarray(warm_start["parameters"], dtype=float)
    failures = []
    objective_trace = []
    checkpoints = [checkpoint(data, policy, start, validation_pop, 0, f"{case}/{method}/neural/checkpoint-0")]
    accepted = 0
    previous_accepted_value = None
    stationarity_stop = None
    last_evaluation = {}
    started = time.perf_counter()

    def objective(theta):
        values = np.asarray(theta, dtype=float)
        try:
            value, gradient = objective_gradient(data, policy, values, design_pop)
        except Exception as error:
            failures.append({"evaluation_index": len(objective_trace) + len(failures), "parameters": values.tolist(),
                             "error_type": type(error).__name__, "error": str(error)})
            value = 1e12 + float(values @ values)
            gradient = 2.0 * values
        objective_trace.append(float(value))
        last_evaluation.update(parameters=values.copy(), value=float(value), gradient=np.asarray(gradient, dtype=float).copy())
        return value, gradient

    start_value, _ = objective(start)
    previous_accepted_value = float(start_value)

    def callback(theta):
        nonlocal accepted, previous_accepted_value, stationarity_stop
        values = np.asarray(theta, dtype=float)
        if not np.array_equal(values, last_evaluation.get("parameters")):
            value, gradient = objective(values)
        else:
            value = last_evaluation["value"]
            gradient = last_evaluation["gradient"]
        accepted += 1
        relative_change = (previous_accepted_value - value) / max(abs(previous_accepted_value), abs(value), 1.0)
        residual = projected_gradient_residual(values, gradient, policy.parameter_bounds)
        stationarity = {
            "design_cost": float(value),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "projected_gradient_residual": residual,
            "stationarity_tolerance": NEURAL_STATIONARITY_TOLERANCE,
            "relative_objective_change": float(relative_change),
            "relative_objective_change_tolerance": NEURAL_RELATIVE_OBJECTIVE_CHANGE_TOLERANCE,
        }
        should_stop = bool(
            residual <= NEURAL_STATIONARITY_TOLERANCE
            and abs(relative_change) <= NEURAL_RELATIVE_OBJECTIVE_CHANGE_TOLERANCE
        )
        if accepted % CHECKPOINT_INTERVAL == 0 or should_stop:
            cp = checkpoint(data, policy, values, validation_pop, accepted,
                            f"{case}/{method}/neural/checkpoint-{accepted}", stationarity)
            checkpoints.append(cp)
            print(case, method, "neural checkpoint", accepted, cp["validation"]["status"], cp["validation"]["cost"],
                  "pgrad", residual, flush=True)
        previous_accepted_value = float(value)
        if should_stop:
            stationarity_stop = {"iteration": accepted, **stationarity}
            raise StopIteration

    result = None
    restart = start
    for stage in range(restart_stages):
        result = minimize(
            objective, restart, method="L-BFGS-B", jac=True,
            bounds=policy.parameter_bounds, callback=callback,
            options={"ftol": ftol, "gtol": gtol, "maxiter": maxiter, "maxls": maxls,
                     "maxfun": max(30000, 50 * maxiter)},
        )
        print(
            case, method, "L-BFGS-B stage", stage + 1,
            "status", result.status, "nit", result.nit, str(result.message), flush=True,
        )
        if result.status == 99:
            break
        restart = np.asarray(result.x, dtype=float)
    assert result is not None
    terminal_iteration = accepted
    terminal_parameters = np.asarray(result.x, dtype=float)
    if checkpoints[-1]["iteration"] != terminal_iteration:
        if not np.array_equal(terminal_parameters, last_evaluation.get("parameters")):
            terminal_value, terminal_gradient = objective(terminal_parameters)
        else:
            terminal_value = last_evaluation["value"]
            terminal_gradient = last_evaluation["gradient"]
        terminal_residual = projected_gradient_residual(terminal_parameters, terminal_gradient, policy.parameter_bounds)
        checkpoints.append(checkpoint(
            data, policy, terminal_parameters, validation_pop, terminal_iteration,
            f"{case}/{method}/neural/checkpoint-{terminal_iteration}",
            {"design_cost": float(terminal_value), "gradient_norm": float(np.linalg.norm(terminal_gradient)),
             "projected_gradient_residual": terminal_residual,
             "stationarity_tolerance": NEURAL_STATIONARITY_TOLERANCE,
             "relative_objective_change": None,
             "relative_objective_change_tolerance": NEURAL_RELATIVE_OBJECTIVE_CHANGE_TOLERANCE},
        ))
    if stationarity_stop is None:
        raise RuntimeError(f"neural stationarity stop not reached: {case}/{method}")

    for cp in checkpoints:
        default = cp.get("default_stationarity", {})
        if default.get("projected_gradient_residual", np.inf) > NEURAL_STATIONARITY_TOLERANCE:
            cp["stationarity_certified"] = False
            continue
        theta = np.asarray(cp["parameters"], dtype=float)
        try:
            tight_cost, tight_gradient = objective_gradient(
                data, policy, theta, design_pop, rtol=2e-10, atol=2e-12, max_step_divisor=400.0
            )
            tight_residual = projected_gradient_residual(theta, tight_gradient, policy.parameter_bounds)
            cp["tight_stationarity"] = {
                "status": "finite", "design_cost": tight_cost, "gradient": tight_gradient,
                "gradient_norm": float(np.linalg.norm(tight_gradient)),
                "projected_gradient_norm": projected_gradient_norm(theta, tight_gradient, policy.parameter_bounds),
                "projected_gradient_residual": tight_residual,
                "stationarity_tolerance": NEURAL_STATIONARITY_TOLERANCE,
            }
            cp["stationarity_certified"] = bool(tight_residual <= NEURAL_STATIONARITY_TOLERANCE)
        except Exception as error:
            cp["tight_stationarity"] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            cp["stationarity_certified"] = False
    admissible = [cp for cp in checkpoints if cp["admissible"] and cp["validation"]["cost"] is not None
                  and cp.get("stationarity_certified", False)]
    if not admissible:
        raise RuntimeError(f"no stationarity-certified neural checkpoint: {case}/{method}")
    selected = min(admissible, key=lambda cp: (float(cp["validation"]["cost"]), cp["iteration"]))
    theta = np.asarray(selected["parameters"])
    tight = selected["tight_stationarity"]
    record = {
        "policy": "width-16 one-hidden-layer tanh state-time feedback",
        "architecture": {"hidden_width": operator.hidden_width, "parameter_count": policy.parameter_count,
                         "input": ["time", "q", "v"], "output": "scalar control", "coordinate_bounds": [-8.0, 8.0],
                         "state_scale": operator.state_scale, "control_scale": operator.control_scale},
        "warm_start": {"source": str(WARM_STARTS.relative_to(ROOT)),
                       "source_key": f"{case}/{method}", "parameters": start},
        "selection": {"metric": "stationarity-certified complete-inventory joint validation cost" if method == "averaged" else "stationarity-certified mean-pair nominal validation cost",
                      "baseline_included": True, "selected_iteration": selected["iteration"],
                      "selected_validation_cost": selected["validation"]["cost"],
                      "baseline_validation_cost": checkpoints[0]["validation"]["cost"],
                      "stationarity_tolerance": NEURAL_STATIONARITY_TOLERANCE,
                      "final_evaluation_used_for_selection": False, "checkpoint_interval": CHECKPOINT_INTERVAL},
        "optimizer": {"requested_accepted_iterations": maxiter,
                      "restart_stage_limit": restart_stages,
                      "terminated": bool(result.success or stationarity_stop is not None), "message": str(result.message),
                      "termination_reason": "projected_gradient_and_relative_objective_change",
                      "accepted_iterations": terminal_iteration, "function_evaluations": int(result.nfev),
                      "gradient_evaluations": int(result.njev), "reported_design_cost": float(result.fun),
                      "reported_gradient_norm": float(np.linalg.norm(result.jac)),
                      "stationarity_stop": stationarity_stop, "objective_trace": objective_trace,
                      "elapsed_seconds": time.perf_counter() - started},
        "checkpoints": checkpoints,
        "objective_failures": failures,
        "selected_parameters": theta,
        "selected_tight_stationarity": tight,
        "stationarity_certified": True,
    }
    return record, policy


def linear_data(case: str):
    nonlinear_case = "nonlinear-vdp" if case == "linear-vdp" else "nonlinear-stuart-landau"
    data = dict(case_data(nonlinear_case))
    data.update(case=case, nonlinear=False)
    return data


def initial_moment(center, radius, averaged):
    base = np.outer(center, center)
    return base + (radius ** 2 / 3.0) * np.eye(2) if averaged else base


def linear_moment_evaluation(data, gain, moment0, parameters, weights, *, rtol=2e-10, atol=2e-12, divisor=240.0):
    gain = np.asarray(gain, dtype=float)
    weights = np.asarray(weights, dtype=float)
    evaluation, closed = policy_evaluation(data, gain, parameters, rtol=rtol, atol=atol,
                                           max_step_divisor=divisor, dense_output=True)
    count = len(parameters)
    P0 = evaluation.y[:, -1].reshape(count, 2, 2)
    costs = 0.5 * np.einsum("nij,ji->n", P0, moment0)
    cost = float(weights @ costs)
    horizon = float(data["horizon"])
    def rhs(t, packed):
        moments = packed[:4 * count].reshape(count, 2, 2)
        derivative = np.einsum("nij,njk->nik", closed, moments) + np.einsum("nij,nkj->nik", moments, closed)
        P = evaluation.sol(t).reshape(count, 2, 2)
        occupancy_rate = np.einsum("n,nij->ij", weights, moments)
        cross_parts = np.einsum("ni,nij->nj", P[:, 1, :], moments)
        cross_rate = np.einsum("n,nj->j", weights, cross_parts)
        return np.r_[derivative.ravel(), occupancy_rate.ravel(), cross_rate]
    initial = np.r_[np.tile(moment0, (count, 1, 1)).ravel(), np.zeros(6)]
    forward = solve_ivp(rhs, (0.0, horizon), initial, method="DOP853", rtol=rtol, atol=atol, max_step=horizon/divisor)
    if not forward.success or forward.t[-1] != horizon:
        raise RuntimeError("linear moment solve failed: " + forward.message)
    occupancy = forward.y[-6:-2, -1].reshape(2, 2)
    cross = forward.y[-2:, -1]
    gradient = data["R"] * (gain @ occupancy) - cross
    target = np.linalg.solve(occupancy.T, cross) / data["R"]
    return {
        "cost": cost,
        "gradient": gradient,
        "stationarity_residual": float(np.linalg.norm(gradient, ord=np.inf)),
        "occupancy": occupancy,
        "cross": cross,
        "fixed_point_target": target,
        "occupancy_condition_number": float(np.linalg.cond(occupancy)),
        "per_parameter_cost": costs,
    }


def linear_objective_gradient(data, gain, moment0, parameters, weights, *, rtol=2e-10, atol=2e-12, divisor=240.0):
    result = linear_moment_evaluation(
        data, gain, moment0, parameters, weights, rtol=rtol, atol=atol, divisor=divisor
    )
    return result["cost"], result["gradient"], result["per_parameter_cost"]


def optimize_linear(data, case_rules, method):
    center = case_rules["x0_center"]
    radius = case_rules["x0_radius"]
    if method == "averaged":
        parameters = case_rules["omega_design"]
        weights = case_rules["omega_design_weights"]
        M0 = initial_moment(center, radius, True)
    else:
        parameters = np.asarray([data["nominal_parameter"]])
        weights = np.ones(1)
        M0 = initial_moment(center, radius, False)
    started = time.perf_counter()
    iteration = anderson_moore_iteration(
        lambda gain: linear_moment_evaluation(data, gain, M0, parameters, weights)
    )
    gain = np.asarray(iteration["gain"])
    tight = linear_moment_evaluation(
        data, gain, M0, parameters, weights, rtol=2e-12, atol=2e-14, divisor=480.0
    )
    qualified = bool(
        np.isfinite(tight["cost"])
        and np.all(np.isfinite(tight["gradient"]))
        and tight["stationarity_residual"] <= STATIC_STATIONARITY_TOLERANCE
    )
    if not qualified:
        raise RuntimeError(f"linear residual check failed: {data['case']}/{method}")
    iteration["elapsed_seconds"] = time.perf_counter() - started
    eval_M0 = initial_moment(center, radius, True)
    evaluation = linear_moment_evaluation(
        data, gain, eval_M0, case_rules["omega_final"], case_rules["omega_final_weights"],
        rtol=2e-12, atol=2e-14, divisor=480.0
    )
    point = INITIAL_GAIN.copy(); step = 1e-5
    check_cost, check_gradient, _ = linear_objective_gradient(data, point, M0, parameters, weights)
    finite_gradient = np.zeros(2)
    for i in range(2):
        direction = np.eye(2)[i] * step
        plus = linear_objective_gradient(data, point + direction, M0, parameters, weights)[0]
        minus = linear_objective_gradient(data, point - direction, M0, parameters, weights)[0]
        finite_gradient[i] = (plus - minus) / (2 * step)
    return {
        "gain": gain, "design_initial_second_moment": M0, "evaluation_initial_second_moment": eval_M0,
        "design_parameters": parameters, "design_weights": weights,
        "design_per_parameter_cost": tight["per_parameter_cost"],
        "design_cost": tight["cost"], "tight_gradient": tight["gradient"],
        "tight_gradient_norm": float(np.linalg.norm(tight["gradient"])),
        "stationarity_residual": tight["stationarity_residual"],
        "stationarity_tolerance": STATIC_STATIONARITY_TOLERANCE,
        "stationarity_certified": True,
        "occupancy_condition_number": tight["occupancy_condition_number"],
        "evaluation_cost": evaluation["cost"],
        "evaluation_per_parameter_cost": evaluation["per_parameter_cost"],
        "fixed_point_iteration": iteration,
        "gradient_check": {"point": point, "cost": check_cost, "lyapunov_moment_gradient": check_gradient,
                           "finite_difference_gradient": finite_gradient,
                           "maximum_absolute_difference": float(np.max(np.abs(check_gradient - finite_gradient)))},
    }


def configure_plot():
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}",
                         "font.family": "serif", "font.size": FONT_SIZE + 2,
                         "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
                         "axes.labelsize": AXES_LABEL_SIZE + 2, "axes.titlesize": AXES_TITLE_SIZE + 2,
                         "xtick.labelsize": TICK_LABEL_SIZE + 2, "ytick.labelsize": TICK_LABEL_SIZE + 2,
                         "legend.fontsize": LEGEND_SIZE + 2})


def plot_trajectory(path, population, trajectories, *, figure_size=(10.5, 6.4)):
    configure_plot()
    omega = population.omega
    norm = Normalize(float(omega.min()), float(omega.max()))
    cmap = plt.get_cmap("RdBu_r")
    fig, axes = plt.subplots(3, 2, figsize=figure_size, sharex=True, constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    titles = {"averaged": "Averaged design", "nominal": "Nominal design"}
    blocks = [[] for _ in range(3)]
    for column, method in enumerate(METHODS):
        result = trajectories[method]
        time_grid = result["time"]
        values_by_row = (result["states"][:, :, 0], result["states"][:, :, 1], result["controls"])
        for row, values in enumerate(values_by_row):
            for index, parameter in enumerate(omega):
                axes[row, column].plot(time_grid, values[index], color=cmap(norm(parameter)), alpha=.58, linewidth=.55)
            mean = np.einsum("n,nt->t", population.weights, values)
            axes[row, column].plot(time_grid, mean, color="#24343a", linewidth=2.5)
            axes[row, column].grid(alpha=.18)
            axes[row, column].set_xlim(time_grid[0], time_grid[-1])
            blocks[row].extend((values, mean[None, :]))
        axes[0, column].set_title(titles[method], color="black", pad=6)
        axes[2, column].set_xlabel("Time")
    for row, label in enumerate((r"$q$", r"$v$", r"$u$")):
        axes[row, 0].set_ylabel(label)
        low = min(float(np.min(x)) for x in blocks[row])
        high = max(float(np.max(x)) for x in blocks[row])
        margin = .045 * max(high - low, 1e-12)
        for column in range(2):
            axes[row, column].set_ylim(low - margin, high + margin)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal",
                      shrink=.56, aspect=40, pad=.055)
    cb.ax.text(1.03, .5, r"Model parameter $\omega$", transform=cb.ax.transAxes,
               ha="left", va="center", fontsize=FONT_SIZE + 2)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=190)
    plt.close(fig)


def contact_sheet(paths):
    images=[]
    for path in paths:
        image=plt.imread(path.with_suffix(".png")); images.append((path.stem,image))
    fig,axes=plt.subplots(2,2,figsize=(16,11),constrained_layout=True)
    for ax,(title,image) in zip(axes.ravel(),images):
        ax.imshow(image); ax.set_title(title,fontsize=12); ax.axis("off")
    for ax in axes.ravel()[len(images):]: ax.axis("off")
    fig.savefig(OUTPUT/"contact-sheet.png",dpi=140); plt.close(fig)


def main(output: Path = OUTPUT):
    global OUTPUT, FIGURES
    OUTPUT = output.resolve()
    FIGURES = OUTPUT / "figures"
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    FIGURES.mkdir(parents=True)
    warm_starts=json.loads(WARM_STARTS.read_text())
    script=Path(__file__)
    sources=[WARM_STARTS,
             ROOT/"experiments/models.py",
             ROOT/"experiments/neural_adjoint.py",
             ROOT/"experiments/neural_policy.py",
             ROOT/"experiments/linear_moment.py",script]
    summary={
        "schema_version":2,
        "status":"running", "role":"joint model/initial-state uncertainty experiment",
        "started_utc":now(),
        "protocol":{"static_feedback_initial_gain":INITIAL_GAIN,
                    "static_feedback_update":"moment-weighted Anderson--Moore direction with Armijo backtracking",
                    "static_feedback_armijo_constant":ARMIJO_CONSTANT,
                    "static_feedback_armijo_contraction":ARMIJO_CONTRACTION,
                    "static_feedback_stationarity_residual":"entrywise infinity norm of the shared-gain gradient",
                    "static_feedback_stationarity_tolerance":STATIC_STATIONARITY_TOLERANCE,
                    "neural_max_accepted_iterations":NEURAL_MAXITER,
                    "neural_checkpoint_interval":CHECKPOINT_INTERVAL,
                    "neural_stationarity_residual":"entrywise infinity norm of theta - projection(theta - gradient)",
                    "neural_stationarity_tolerance":NEURAL_STATIONARITY_TOLERANCE,
                    "neural_relative_objective_change_tolerance":NEURAL_RELATIVE_OBJECTIVE_CHANGE_TOLERANCE,
                    "neural_candidate_rule":"lowest validation cost among tightly stationarity-certified checkpoints",
                    "final_evaluation_used_for_selection":False,
                    "all_or_nothing_full_inventory":True,
                    "trajectory_display":"empirical trajectory-display law formed by 32 paired joint particles; colors encode omega, line variation includes the initial state, and the thick line is the equal-weight pointwise mean"},
        "seeds":SEEDS, "source_artifact_hashes":{str(path if path.is_absolute() and not str(path).startswith(str(ROOT)) else path.relative_to(ROOT)):digest(path) for path in sources},
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
        "cases":{}, "linear_cases":{}
    }
    write_json(OUTPUT/"summary.json",summary)
    trajectory_cache={}
    case_rules_cache={}
    for case in CASES:
        print("CASE",case,flush=True)
        data=case_data(case); cr=rules(case,data); case_rules_cache[case]=cr
        case_record={
            "system":data["system"],"horizon":data["horizon"],"Q":data["Q"],"R":data["R"],"gain_bounds":data["bounds"],
            "omega_distribution":cr["omega_metadata"],
            "initial_state_distribution":{"family":"independent coordinatewise Uniform", "mean":cr["x0_center"],
                                          "lower":cr["x0_center"]-cr["x0_radius"],"upper":cr["x0_center"]+cr["x0_radius"]},
            "rules":{"omega_design":cr["omega_design"],"omega_design_weights":cr["omega_design_weights"],
                     "omega_validation":cr["omega_validation"],"omega_validation_weights":cr["omega_validation_weights"],
                     "omega_final":cr["omega_final"],"omega_final_weights":cr["omega_final_weights"],
                     "omega_display":cr["omega_display"],"x0_design":cr["x0_design"],"x0_validation":cr["x0_validation"],
                     "x0_final":cr["x0_final"],"x0_display":cr["x0_display"]},
            "particle_inventories":{"averaged_design":population_record(cr["averaged_design"]),
                                    "averaged_validation":population_record(cr["averaged_validation"]),
                                    "nominal":population_record(cr["nominal"]),
                                    "final":population_record(cr["final"]),"display":population_record(cr["display"])},
            "policies":{"direct_neural":{}}
        }
        for method in METHODS:
            design_pop=cr["averaged_design"] if method=="averaged" else cr["nominal"]
            validation_pop=cr["averaged_validation"] if method=="averaged" else cr["nominal"]
            neural_record,neural_policy=optimize_neural(data,case,method,design_pop,validation_pop,warm_starts)
            final_record,_=evaluate_full(data,neural_policy,neural_record["selected_parameters"],cr["final"],f"{case}/{method}/neural/final")
            display_record,display_result=evaluate_full(data,neural_policy,neural_record["selected_parameters"],cr["display"],f"{case}/{method}/neural/display",times=np.linspace(0,data["horizon"],201))
            neural_record["final_evaluation"]=final_record; neural_record["display_evaluation"]={k:v for k,v in display_record.items() if k!="per_particle_cost"}
            case_record["policies"]["direct_neural"][method]=neural_record
            trajectory_cache[(case,"direct_neural",method)]=display_result
            summary["cases"][case]=case_record; write_json(OUTPUT/"summary.json",summary)
        summary["cases"][case]=case_record
        write_json(OUTPUT/"summary.json",summary)

    static_policy = StaticLinearPolicy()
    for lcase,ncase in (("linear-vdp","nonlinear-vdp"),("linear-stuart-landau","nonlinear-stuart-landau")):
        print("LINEAR DESIGN / NONLINEAR EVALUATION",lcase,flush=True)
        data=linear_data(lcase); nonlinear_data=case_data(ncase); cr=case_rules_cache[ncase]
        data["nominal_parameter"] = cr["omega_metadata"]["mean"]
        record={"system":data["system"],"horizon":data["horizon"],"Q":data["Q"],"R":data["R"],
                "design_model":"first-order linearization about the zero equilibrium",
                "evaluation_model":"original nonlinear oscillator",
                "initial_second_moment_formula":"M0=xi_bar*xi_bar^T+(r^2/3)I for averaged design/evaluation; nominal design uses xi_bar*xi_bar^T",
                "methods":{}}
        for method in METHODS:
            result=optimize_linear(data,cr,method)
            final_record,_=evaluate_full(
                nonlinear_data,static_policy,result["gain"],cr["final"],
                f"{ncase}/{method}/static-linear/final",
            )
            display_record,display_result=evaluate_full(
                nonlinear_data,static_policy,result["gain"],cr["display"],
                f"{ncase}/{method}/static-linear/display",times=np.linspace(0,data["horizon"],401),
            )
            result["nonlinear_evaluation"]=final_record
            result["nonlinear_display_evaluation"]={k:v for k,v in display_record.items() if k!="per_particle_cost"}
            record["methods"][method]=result
            trajectory_cache[(ncase,"static_linear",method)]=display_result
            summary["linear_cases"][lcase]=record; write_json(OUTPUT/"summary.json",summary)
        summary["linear_cases"][lcase]=record

    figure_paths=[]
    for case in CASES:
        cr=case_rules_cache[case]
        family="direct_neural"
        path=FIGURES/NAMES[(case,family)]
        figure_size = (10.5, 5.15)
        plot_trajectory(
            path, cr["display"],
            {method: trajectory_cache[(case, family, method)] for method in METHODS},
            figure_size=figure_size,
        )
        figure_paths.append(path)
    for lcase,ncase in (("linear-vdp","nonlinear-vdp"),("linear-stuart-landau","nonlinear-stuart-landau")):
        cr=case_rules_cache[ncase]; path=FIGURES/NAMES[(ncase,"static_linear")]
        plot_trajectory(
            path, cr["display"],
            {method: trajectory_cache[(ncase, "static_linear", method)] for method in METHODS},
            figure_size=(10.5, 5.15),
        )
        figure_paths.append(path)
    contact_sheet(figure_paths)

    comparisons={}
    for case in CASES:
        family="direct_neural"
        avg=summary["cases"][case]["policies"][family]["averaged"]["final_evaluation"]["cost"]
        nom=summary["cases"][case]["policies"][family]["nominal"]["final_evaluation"]["cost"]
        comparisons[case]={family:{"averaged_cost":avg,"nominal_cost":nom,
                                   "reduction_percent":None if avg is None or nom is None else 100*(nom-avg)/nom}}
    for lcase in ("linear-vdp","linear-stuart-landau"):
        avg=summary["linear_cases"][lcase]["methods"]["averaged"]["nonlinear_evaluation"]["cost"]
        nom=summary["linear_cases"][lcase]["methods"]["nominal"]["nonlinear_evaluation"]["cost"]
        comparisons[lcase]={"static_linear_feedback":{"averaged_cost":avg,"nominal_cost":nom,"reduction_percent":100*(nom-avg)/nom}}
    summary["comparisons"]=comparisons
    summary["figures"]={path.name:{"pdf":str(path.relative_to(OUTPUT)),"pdf_sha256":digest(path),
                                                  "png":str(path.with_suffix('.png').relative_to(OUTPUT)),"png_sha256":digest(path.with_suffix('.png'))}
                        for path in figure_paths}
    summary["status"]="complete"; summary["completed_utc"]=now(); write_json(OUTPUT/"summary.json",summary)

    verification={"status":"running","checks":{},"failures":[]}
    expected=[FIGURES/name for name in [
        "nonlinear-vdp-direct-cost-neural-feedback-full-results.pdf",
        "nonlinear-stuart-landau-direct-cost-neural-feedback-full-results.pdf",
        "nonlinear-vdp-static-linear-feedback.pdf","nonlinear-stuart-landau-static-linear-feedback.pdf"]]
    verification["checks"]["all_four_pdfs_exist"]=all(path.exists() and path.stat().st_size>0 for path in expected)
    verification["checks"]["all_four_pngs_exist"]=all(path.with_suffix('.png').exists() and path.with_suffix('.png').stat().st_size>0 for path in expected)
    inventory_checks={}; cost_checks={}; finite_checks={}
    for case in CASES:
        cr=case_rules_cache[case]
        inventory_checks[case]={"averaged_design":cr["averaged_design"].size==256,"averaged_validation":cr["averaged_validation"].size==256,
                                "nominal":cr["nominal"].size==1,"final":cr["final"].size==2048,"display":cr["display"].size==32}
        family="direct_neural"
        for method in METHODS:
            key=f"{case}/{family}/{method}"; rec=summary["cases"][case]["policies"][family][method]["final_evaluation"]
            finite_checks[key]=rec["status"]=="finite full inventory" and rec["cost"] is not None and np.isfinite(rec["cost"])
            if rec["cost"] is not None:
                recomputed=float(cr["final"].weights @ np.asarray(rec["per_particle_cost"]))
                cost_checks[key]={"reported":rec["cost"],"recomputed":recomputed,"matches":bool(np.isclose(rec["cost"],recomputed,rtol=2e-12,atol=2e-11)),
                                  "particle_count":len(rec["per_particle_cost"]),"full_inventory":len(rec["per_particle_cost"])==2048}
            else: cost_checks[key]={"matches":False,"full_inventory":False}
        lcase="linear-vdp" if case=="nonlinear-vdp" else "linear-stuart-landau"
        for method in METHODS:
            key=f"{case}/static_linear/{method}"
            rec=summary["linear_cases"][lcase]["methods"][method]["nonlinear_evaluation"]
            finite_checks[key]=rec["status"]=="finite full inventory" and rec["cost"] is not None and np.isfinite(rec["cost"])
            if rec["cost"] is not None:
                recomputed=float(cr["final"].weights @ np.asarray(rec["per_particle_cost"]))
                cost_checks[key]={"reported":rec["cost"],"recomputed":recomputed,"matches":bool(np.isclose(rec["cost"],recomputed,rtol=2e-12,atol=2e-11)),
                                  "particle_count":len(rec["per_particle_cost"]),"full_inventory":len(rec["per_particle_cost"])==2048}
            else: cost_checks[key]={"matches":False,"full_inventory":False}
    neural_stationarity={}
    for case in CASES:
        neural_stationarity[case]={method:summary["cases"][case]["policies"]["direct_neural"][method]["stationarity_certified"] for method in METHODS}
    linear_gradient={}; linear_stationarity={}
    for lcase in ("linear-vdp","linear-stuart-landau"):
        linear_gradient[lcase]={method:summary["linear_cases"][lcase]["methods"][method]["gradient_check"]["maximum_absolute_difference"] for method in METHODS}
        linear_stationarity[lcase]={method:summary["linear_cases"][lcase]["methods"][method]["stationarity_certified"] for method in METHODS}
    verification["checks"].update({"particle_inventories":inventory_checks,"finite_final_costs":finite_checks,
                                   "recomputed_weighted_costs":cost_checks,
                                   "neural_stationarity_certified":neural_stationarity,
                                   "all_neural_candidates_stationarity_certified":all(x for case in neural_stationarity.values() for x in case.values()),
                                   "linear_gradient_checks":linear_gradient,
                                   "linear_stationarity_certified":linear_stationarity,
                                   "all_linear_candidates_stationarity_certified":all(x for case in linear_stationarity.values() for x in case.values()),
                                   "all_final_inventories_complete":all(v["full_inventory"] and v["matches"] for v in cost_checks.values()),
                                   "all_final_costs_finite":all(finite_checks.values()),
                                   "maximum_linear_gradient_error":max(x for case in linear_gradient.values() for x in case.values())})
    passed=verification["checks"]["all_four_pdfs_exist"] and verification["checks"]["all_final_inventories_complete"] and verification["checks"]["all_final_costs_finite"] and verification["checks"]["all_neural_candidates_stationarity_certified"] and verification["checks"]["all_linear_candidates_stationarity_certified"] and verification["checks"]["maximum_linear_gradient_error"]<2e-7
    verification["status"]="passed" if passed else "failed"; verification["completed_utc"]=now()
    write_json(OUTPUT/"verification.json",verification)

    rows=[]
    labels=[("Van der Pol","Time-dependent neural feedback",comparisons["nonlinear-vdp"]["direct_neural"]),
            ("Van der Pol","Static linear feedback",comparisons["linear-vdp"]["static_linear_feedback"]),
            ("Stuart–Landau","Time-dependent neural feedback",comparisons["nonlinear-stuart-landau"]["direct_neural"]),
            ("Stuart–Landau","Static linear feedback",comparisons["linear-stuart-landau"]["static_linear_feedback"])]
    for model,policy,row in labels:
        reduction="undefined" if row["reduction_percent"] is None else f"{row['reduction_percent']:.2f}%"
        rows.append(f"| {model} | {policy} | {row['averaged_cost']:.6g} | {row['nominal_cost']:.6g} | {reduction} |")
    failures=[]
    for case in CASES:
        family="direct_neural"
        lcase="linear-vdp" if case=="nonlinear-vdp" else "linear-stuart-landau"
        for method in METHODS:
            rec=summary["cases"][case]["policies"][family][method]
            if rec["objective_failures"]: failures.append(f"{case}/{family}/{method}: {len(rec['objective_failures'])} guarded objective failures")
            failures.extend([f"{case}/{family}/{method}/checkpoint-{cp['iteration']}: {cp['validation']['error']}" for cp in rec["checkpoints"] if not cp["admissible"]])
            if rec["final_evaluation"]["status"] != "finite full inventory": failures.append(f"{case}/{family}/{method}/final: {rec['final_evaluation'].get('error')}")
            static=summary["linear_cases"][lcase]["methods"][method]["nonlinear_evaluation"]
            if static["status"] != "finite full inventory": failures.append(f"{case}/static_linear/{method}/final: {static.get('error')}")
    caveat=("Direct-neural refinement used the requested 400 accepted-iteration cap and checkpoint interval 10; no reduced pilot was used."
            if NEURAL_MAXITER==400 else f"Direct-neural refinement used a cap of {NEURAL_MAXITER} accepted iterations.")
    report=f"""# Joint model and initial-state uncertainty experiment

## Independent joint-evaluation costs

| Model | Policy | Averaged design | Nominal design | Reduction |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Protocol notes

- Van der Pol uses $\\omega\\sim\\mathrm{{Uniform}}[-2,4]$ and $x_0=(1,1)+\\mathrm{{Uniform}}([-0.2,0.2]^2)$.
- Stuart–Landau uses $\\omega\\sim\\mathrm{{Uniform}}[-2.1,0.8]$ and $x_0=(0.5,0.5)+\\mathrm{{Uniform}}([-0.2,0.2]^2)$.
- Averaged nonlinear design uses 32 parameter nodes times 8 initial-state points. Nominal design fixes both quantities at their means.
- Final nonlinear evaluation uses the common independent 128 times 16 inventory. It is performed only after controller/checkpoint selection.
- Linearized averaged initial-state uncertainty is represented exactly by $M_0=\\bar x_0\\bar x_0^T+(r^2/3)I$ and propagated by the forward second-moment equation; policy evaluation uses backward Lyapunov equations. The resulting static gains are evaluated and plotted on the original nonlinear oscillators.
- Trajectory panels use 32 held-out paired joint particles, common to both methods. Color encodes $\\omega$; curve-to-curve variation also contains initial-state variation. The thick curve is the equal-weight pointwise mean.
- {caveat}

## Failures and caveats

{('- None.' if not failures else chr(10).join('- '+item for item in failures))}

The numerical results are finite only on the explicitly recorded complete inventories; no successful-subset average is used. The finite optimization procedure does not establish global optimality.

## Readiness

**{'All four trajectory figures are ready for user inspection.' if passed else 'The four-figure set is not fully ready; see verification.json and failures above.'}**

Easy inspection: `contact-sheet.png`. Full records: `summary.json`; checks: `verification.json`.
"""
    (OUTPUT/"REPORT.md").write_text(report)
    # Refresh hashes after final report/verification creation.
    print(json.dumps({"status":verification["status"],"output":str(OUTPUT),"comparisons":comparisons},indent=2),flush=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    main(arguments.output)
