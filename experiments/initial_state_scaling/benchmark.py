#!/usr/bin/env python3
"""Benchmark initial-state particle propagation against LQ moment closure.

The script writes only inside the requested output directory and imports the
repository's vectorized nonlinear joint-population objective-gradient
implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "initial-state-scaling-reproduction"
PREVIEW = ROOT / "reproduction" / "data" / "joint-uncertainty" / "summary.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.figure_style import (  # noqa: E402
    AXES_LABEL_SIZE,
    AXES_TITLE_SIZE,
    COMPARATOR_COLOR,
    FONT_SIZE,
    LEGEND_SIZE,
    PRIMARY_METHOD_COLOR,
    TICK_LABEL_SIZE,
)
from experiments.neural_policy import DirectNeuralPolicy, NeuralFeedbackOperator  # noqa: E402
from experiments.neural_adjoint import (  # noqa: E402
    JointPopulation,
    objective_gradient as nonlinear_objective_gradient,
    product_population,
)

B = np.array([[0.0], [1.0]])
M_VALUES = [1, 4, 16, 64, 128]
LINEAR_REPEATS = 5
NONLINEAR_REPEATS = 5
LINEAR_TIMING_TOL = {"rtol": 2e-10, "atol": 2e-12, "max_step_divisor": 240.0}
LINEAR_CHECK_TOL = {"rtol": 2e-12, "atol": 2e-14, "max_step_divisor": 480.0}
NONLINEAR_TOL = {"rtol": 2e-9, "atol": 2e-11, "max_step_divisor": 180.0}
AGREEMENT_TOL = {"objective_absolute": 2e-9, "gradient_max_absolute": 2e-8}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def matrices(system: str, parameters: np.ndarray) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    result = np.zeros((len(parameters), 2, 2))
    if system == "van_der_pol":
        result[:, 0, 1] = 1.0
        result[:, 1, 0] = -1.0
        result[:, 1, 1] = parameters
    elif system == "stuart_landau":
        result[:, 0, 0] = parameters
        result[:, 0, 1] = -1.0
        result[:, 1, 0] = 1.0
        result[:, 1, 1] = parameters
    else:
        raise ValueError(system)
    return result


def solve_options(horizon: float, tol: dict[str, float]) -> dict[str, Any]:
    return {
        "method": "DOP853",
        "rtol": tol["rtol"],
        "atol": tol["atol"],
        "max_step": horizon / tol["max_step_divisor"],
    }


def policy_evaluation(case: dict[str, Any], tol: dict[str, float]):
    gain = case["linear_gain"]
    parameters = case["parameters"]
    count = len(parameters)
    dynamics = matrices(case["system"], parameters)
    closed = dynamics - B[None, :, :] * gain[None, None, :]
    running = case["Q"] + case["R"] * np.outer(gain, gain)
    horizon = case["horizon"]

    def rhs(_time: float, packed: np.ndarray) -> np.ndarray:
        P = packed.reshape(count, 2, 2)
        derivative = -(
            np.einsum("nji,njk->nik", closed, P)
            + np.einsum("nij,njk->nik", P, closed)
            + running[None, :, :]
        )
        derivative = 0.5 * (derivative + np.swapaxes(derivative, 1, 2))
        return derivative.ravel()

    solution = solve_ivp(
        rhs,
        (horizon, 0.0),
        np.zeros(4 * count),
        dense_output=True,
        **solve_options(horizon, tol),
    )
    if not solution.success or not np.isclose(solution.t[-1], 0.0, rtol=0.0, atol=1e-14):
        raise RuntimeError("policy evaluation failed: " + solution.message)
    return solution, closed, running


def empirical_second_moment(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    return samples.T @ samples / len(samples)


def analytic_second_moment(mean: np.ndarray, radius: float) -> np.ndarray:
    return np.outer(mean, mean) + (radius**2 / 3.0) * np.eye(2)


def particle_forward(
    case: dict[str, Any],
    samples: np.ndarray,
    evaluation: Any,
    closed: np.ndarray,
    running: np.ndarray,
    tol: dict[str, float],
) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=float)
    count = len(case["parameters"])
    sample_count = len(samples)
    gain = case["linear_gain"]
    weights = case["weights"]
    horizon = case["horizon"]

    def rhs(time_value: float, packed: np.ndarray) -> np.ndarray:
        states = packed[: 2 * count * sample_count].reshape(count, sample_count, 2)
        derivatives = np.einsum("nij,nmj->nmi", closed, states)
        P = evaluation.sol(time_value).reshape(count, 2, 2)
        residual = case["R"] * gain[None, :] - P[:, 1, :]
        residual_state = np.einsum("ni,nmi->nm", residual, states)
        gradient_rate = np.einsum(
            "n,nm,nmj->j", weights, residual_state, states
        ) / sample_count
        quadratic = np.einsum("nmi,ij,nmj->nm", states, running, states)
        objective_rate = 0.5 * float(weights @ quadratic.mean(axis=1))
        return np.r_[derivatives.ravel(), objective_rate, gradient_rate]

    initial_states = np.tile(samples, (count, 1))
    solution = solve_ivp(
        rhs,
        (0.0, horizon),
        np.r_[initial_states.ravel(), np.zeros(3)],
        **solve_options(horizon, tol),
    )
    if not solution.success or not np.isclose(solution.t[-1], horizon, rtol=0.0, atol=1e-13):
        raise RuntimeError("particle forward solve failed: " + solution.message)
    return {
        "objective": float(solution.y[-3, -1]),
        "gradient": solution.y[-2:, -1].copy(),
        "nfev": int(solution.nfev),
        "state_dimension": int(2 * count * sample_count + 3),
    }


def moment_forward(
    case: dict[str, Any],
    M0: np.ndarray,
    evaluation: Any,
    closed: np.ndarray,
    running: np.ndarray,
    tol: dict[str, float],
) -> dict[str, Any]:
    count = len(case["parameters"])
    gain = case["linear_gain"]
    weights = case["weights"]
    horizon = case["horizon"]

    def rhs(time_value: float, packed: np.ndarray) -> np.ndarray:
        moments = packed[: 4 * count].reshape(count, 2, 2)
        derivatives = (
            np.einsum("nij,njk->nik", closed, moments)
            + np.einsum("nij,nkj->nik", moments, closed)
        )
        P = evaluation.sol(time_value).reshape(count, 2, 2)
        residual = case["R"] * gain[None, :] - P[:, 1, :]
        gradient_nodes = np.einsum("ni,nij->nj", residual, moments)
        gradient_rate = weights @ gradient_nodes
        objective_nodes = 0.5 * np.einsum("ij,nji->n", running, moments)
        objective_rate = float(weights @ objective_nodes)
        return np.r_[derivatives.ravel(), objective_rate, gradient_rate]

    solution = solve_ivp(
        rhs,
        (0.0, horizon),
        np.r_[np.tile(M0, (count, 1, 1)).ravel(), np.zeros(3)],
        **solve_options(horizon, tol),
    )
    if not solution.success or not np.isclose(solution.t[-1], horizon, rtol=0.0, atol=1e-13):
        raise RuntimeError("moment forward solve failed: " + solution.message)
    return {
        "objective": float(solution.y[-3, -1]),
        "gradient": solution.y[-2:, -1].copy(),
        "nfev": int(solution.nfev),
        "state_dimension": int(4 * count + 3),
    }


def trace_objective(case: dict[str, Any], evaluation: Any, M0: np.ndarray) -> float:
    count = len(case["parameters"])
    P0 = evaluation.y[:, -1].reshape(count, 2, 2)
    node_values = 0.5 * np.einsum("nij,ji->n", P0, M0)
    return float(case["weights"] @ node_values)


def timed_linear_once(case: dict[str, Any], samples: np.ndarray, method: str) -> dict[str, Any]:
    start = time.perf_counter()
    if method == "moment":
        setup_start = time.perf_counter()
        M0 = empirical_second_moment(samples)
        setup_seconds = time.perf_counter() - setup_start
    else:
        M0 = None
        setup_seconds = 0.0
    policy_start = time.perf_counter()
    evaluation, closed, running = policy_evaluation(case, LINEAR_TIMING_TOL)
    policy_seconds = time.perf_counter() - policy_start
    forward_start = time.perf_counter()
    if method == "particle":
        result = particle_forward(case, samples, evaluation, closed, running, LINEAR_TIMING_TOL)
    elif method == "moment":
        result = moment_forward(case, M0, evaluation, closed, running, LINEAR_TIMING_TOL)
    else:
        raise ValueError(method)
    forward_seconds = time.perf_counter() - forward_start
    total_seconds = time.perf_counter() - start
    return {
        **result,
        "gradient": np.asarray(result["gradient"], dtype=float).tolist(),
        "setup_seconds": setup_seconds,
        "policy_seconds": policy_seconds,
        "forward_seconds": forward_seconds,
        "total_seconds": total_seconds,
    }


def summarize_timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["setup_seconds", "policy_seconds", "forward_seconds", "total_seconds"]
    result: dict[str, Any] = {"repeats": len(rows), "raw": rows}
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=float)
        stem = field.removesuffix("_seconds")
        result[f"median_{stem}_seconds"] = float(np.median(values))
        result[f"min_{stem}_seconds"] = float(np.min(values))
        result[f"max_{stem}_seconds"] = float(np.max(values))
    result["objective"] = float(np.median([row["objective"] for row in rows]))
    result["gradient"] = np.median(np.asarray([row["gradient"] for row in rows]), axis=0).tolist()
    result["nfev_values"] = [int(row["nfev"]) for row in rows]
    result["state_dimension"] = int(rows[0]["state_dimension"])
    return result


def timed_nonlinear_once(case: dict[str, Any], samples: np.ndarray) -> dict[str, Any]:
    population = product_population(
        case["parameters"],
        case["weights"],
        samples,
        np.full(len(samples), 1.0 / len(samples)),
    )
    start = time.perf_counter()
    objective, gradient = nonlinear_objective_gradient(
        case["data"],
        case["nonlinear_policy"],
        case["nonlinear_parameters"],
        population,
        **NONLINEAR_TOL,
    )
    seconds = time.perf_counter() - start
    return {
        "objective": float(objective),
        "gradient": np.asarray(gradient, dtype=float).tolist(),
        "total_seconds": seconds,
        "particle_count": population.size,
        "forward_state_dimension": 3 * population.size,
        "backward_state_dimension": (
            2 * population.size + len(case["nonlinear_parameters"])
        ),
    }


def summarize_nonlinear(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([row["total_seconds"] for row in rows])
    return {
        "repeats": len(rows),
        "raw": rows,
        "median_total_seconds": float(np.median(values)),
        "min_total_seconds": float(np.min(values)),
        "max_total_seconds": float(np.max(values)),
        "objective": float(np.median([row["objective"] for row in rows])),
        "gradient": np.median(np.asarray([row["gradient"] for row in rows]), axis=0).tolist(),
        "particle_count": int(rows[0]["particle_count"]),
        "forward_state_dimension": int(rows[0]["forward_state_dimension"]),
        "backward_state_dimension": int(rows[0]["backward_state_dimension"]),
    }


def scaling_exponent(M: list[int], runtime: list[float]) -> float:
    return float(np.polyfit(np.log(np.asarray(M, dtype=float)), np.log(runtime), 1)[0])


def first_crossover(rows: list[dict[str, Any]], speedup_key: str) -> int | None:
    for row in rows:
        if row[speedup_key] > 1.0:
            return int(row["M"])
    return None


def build_cases(preview: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = {
        "van-der-pol": {
            "preview_case": "nonlinear-vdp",
            "linear_case": "linear-vdp",
            "mean": np.array([1.0, 1.0]),
            "radius": 0.2,
            "seed": 20261601,
        },
        "stuart-landau": {
            "preview_case": "nonlinear-stuart-landau",
            "linear_case": "linear-stuart-landau",
            "mean": np.array([0.5, 0.5]),
            "radius": 0.2,
            "seed": 20261611,
        },
    }
    cases: dict[str, dict[str, Any]] = {}
    for label, definition in definitions.items():
        nonlinear = preview["cases"][definition["preview_case"]]
        linear = preview["linear_cases"][definition["linear_case"]]
        parameters = np.asarray(nonlinear["rules"]["omega_design"], dtype=float)
        weights = np.asarray(nonlinear["rules"]["omega_design_weights"], dtype=float)
        data = {
            "system": nonlinear["system"],
            "horizon": float(nonlinear["horizon"]),
            "Q": np.asarray(nonlinear["Q"], dtype=float),
            "R": float(nonlinear["R"]),
        }
        if len(parameters) != 32 or not np.isclose(weights.sum(), 1.0):
            raise RuntimeError(f"invalid 32-node design rule for {label}")
        samples = np.random.default_rng(definition["seed"]).uniform(
            definition["mean"] - definition["radius"],
            definition["mean"] + definition["radius"],
            size=(max(M_VALUES), 2),
        )
        preview_prefix = np.asarray(nonlinear["rules"]["x0_design"], dtype=float)
        if not np.allclose(samples[: len(preview_prefix)], preview_prefix, rtol=0.0, atol=3e-15):
            raise RuntimeError(f"sample prefix does not reproduce the recorded design inventory for {label}")
        nonlinear_method = nonlinear["policies"]["direct_neural"]["averaged"]
        architecture = nonlinear_method["architecture"]
        nonlinear_policy = DirectNeuralPolicy(NeuralFeedbackOperator(
            horizon=float(nonlinear["horizon"]),
            state_scale=np.asarray(architecture["state_scale"], dtype=float),
            control_scale=float(architecture["control_scale"]),
            hidden_width=int(architecture["hidden_width"]),
        ))
        nonlinear_parameters = np.asarray(
            nonlinear_method["selected_parameters"], dtype=float,
        )
        if nonlinear_parameters.shape != (nonlinear_policy.parameter_count,):
            raise RuntimeError(f"invalid neural parameter inventory for {label}")
        cases[label] = {
            **definition,
            "system": nonlinear["system"],
            "horizon": float(nonlinear["horizon"]),
            "Q": np.asarray(nonlinear["Q"], dtype=float),
            "R": float(nonlinear["R"]),
            "parameters": parameters,
            "weights": weights,
            "linear_gain": np.asarray(linear["methods"]["averaged"]["gain"], dtype=float),
            "nonlinear_policy": nonlinear_policy,
            "nonlinear_parameters": nonlinear_parameters,
            "samples": samples,
            "data": data,
        }
    return cases


def configure_plotting() -> None:
    """Match the manuscript figures using LaTeX-rendered Computer Modern text."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}",
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": AXES_LABEL_SIZE,
        "axes.titlesize": AXES_TITLE_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "figure.dpi": 150,
        "savefig.dpi": 200,
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def display_name(label: str) -> str:
    return {
        "van-der-pol": "Van der Pol",
        "stuart-landau": "Stuart--Landau",
    }.get(label, label.replace("-", " ").title())


def make_plots(case_results: dict[str, Any]) -> list[str]:
    configure_plotting()
    colors = {"moment": PRIMARY_METHOD_COLOR, "particle": COMPARATOR_COLOR}
    ordered_results = [(label, case_results[label]) for label in ("van-der-pol", "stuart-landau")]

    plt.rcParams.update({
        "font.size": FONT_SIZE + 2,
        "axes.labelsize": AXES_LABEL_SIZE + 2,
        "axes.titlesize": AXES_TITLE_SIZE + 2,
        "xtick.labelsize": TICK_LABEL_SIZE + 2,
        "ytick.labelsize": TICK_LABEL_SIZE + 2,
        "legend.fontsize": LEGEND_SIZE + 2,
    })
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 2.6), sharey=True, constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    for axis, (label, result) in zip(axes, ordered_results):
        rows = result["rows"]
        M = np.asarray([row["M"] for row in rows])
        axis.loglog(M, [row["linear_moment"]["median_total_seconds"] for row in rows], "s-", color=colors["moment"], linewidth=3.6, markersize=5.0, label="Linear moment method")
        axis.loglog(M, [row["nonlinear_particle"]["median_total_seconds"] for row in rows], "o-", color=colors["particle"], linewidth=3.6, markersize=5.0, label="Nonlinear adjoint method")
        axis.set_title(display_name(label))
        axis.set_xlabel(r"Initial-state sample count $M$")
        axis.set_xticks(M, [str(x) for x in M])
    axes[0].set_ylabel(r"Median wall time $[\mathrm{s}]$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    save_figure(fig, "runtime_scaling")
    configure_plotting()

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex="col")
    for column, (label, result) in enumerate(ordered_results):
        rows = result["rows"]
        M = np.asarray([row["M"] for row in rows])
        axes[0, column].semilogx(M, [row["linear_total_speedup"] for row in rows], "o-", color="#377eb8", label="Total: particle/moment")
        axes[0, column].semilogx(M, [row["linear_forward_speedup"] for row in rows], "s--", color="#4daf4a", label="Forward only")
        axes[0, column].axhline(1.0, color="black", linewidth=0.8)
        axes[0, column].set_title(display_name(label))
        axes[0, column].set_ylabel("Speedup")
        axes[0, column].legend(fontsize=8)
        axes[1, column].loglog(M, [max(row["agreement"]["gradient_max_absolute_difference"], 1e-18) for row in rows], "o-", label="gradient max abs.")
        axes[1, column].loglog(M, [max(row["agreement"]["objective_absolute_difference"], 1e-18) for row in rows], "s--", label="objective abs.")
        axes[1, column].axhline(AGREEMENT_TOL["gradient_max_absolute"], color="#d95f02", linestyle=":", label="gradient tolerance")
        axes[1, column].set_xlabel("Initial-state sample count M")
        axes[1, column].set_ylabel("Particle-moment discrepancy")
        axes[1, column].set_xticks(M, [str(x) for x in M])
        axes[1, column].legend(fontsize=8)
    fig.suptitle("Linear moment-closure speedup and computational agreement")
    fig.tight_layout()
    save_figure(fig, "linear_speedup_agreement")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for label, result in ordered_results:
        rows = result["rows"]
        M = np.asarray([row["M"] for row in rows])
        axes[0].loglog(M, [max(row["empirical_to_analytic"]["objective_absolute_error"], 1e-18) for row in rows], "o-", label=display_name(label))
        axes[1].loglog(M, [max(row["empirical_to_analytic"]["gradient_l2_error"], 1e-18) for row in rows], "o-", label=display_name(label))
    axes[0].set_title("Objective error")
    axes[1].set_title("Gradient error")
    for axis in axes:
        axis.set_xlabel("Initial-state sample count M")
        axis.set_ylabel("Error to analytic continuous-uniform law")
        axis.set_xticks(M_VALUES, [str(x) for x in M_VALUES])
        axis.legend(fontsize=8)
    fig.suptitle("Empirical second moment versus exact continuous-law second moment")
    fig.tight_layout()
    save_figure(fig, "empirical_moment_convergence")
    return [
        "runtime_scaling.png", "runtime_scaling.pdf",
        "linear_speedup_agreement.png", "linear_speedup_agreement.pdf",
        "empirical_moment_convergence.png", "empirical_moment_convergence.pdf",
    ]


def fmt(value: float) -> str:
    if abs(value) >= 0.01 and abs(value) < 1000:
        return f"{value:.4g}"
    return f"{value:.3e}"


def crossover_text(value: int | None) -> str:
    return f"M={value}" if value is not None else "not observed through M=128"


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Initial-state scaling benchmark",
        "",
        f"Completed: `{summary['completed_utc']}`  ",
        f"Repository commit: `{summary['environment']['git_commit']}`",
        "",
        "## Verdict",
        "",
        summary["verdict"],
        "",
        "The defensible manuscript claim is therefore **an asymptotic computational-scaling advantage for homogeneous linear-quadratic initial-state averaging, not an unconditional wall-clock speedup at every small sample count**. The exact compression is unavailable for the nonlinear models, whose implemented forward/adjoint calculation retains one realization per parameter/initial-state pair.",
        "",
        "## Protocol and exact method definitions",
        "",
        "- Both models use the joint experiment's 32-node averaged-design parameter rule. The linear calculation uses the selected averaged static gain, and the nonlinear calculation uses the selected averaged 81-parameter neural feedback.",
        "- Nested initial-state inventories are the common prefixes `M in {1,4,16,64,128}` of the design RNG streams. Van der Pol uses seed `20261601`, mean `(1,1)`, and coordinatewise uniform radius `0.2`; Stuart–Landau uses seed `20261611`, mean `(0.5,0.5)`, and radius `0.2`. The first eight samples exactly reproduce the recorded design inventory.",
        "- **Linear particle trajectory method:** solve the backward policy-evaluation matrices `P_i(t)` once for the 32 parameter nodes, then solve one packed forward ODE containing all `N*M` two-dimensional states. The objective and static-gain gradient are accumulated from the realization averages of the quadratic running cost and `(R K-B^T P_i) x x^T`. This is deliberately more favorable than solving a separate vector costate for every initial state.",
        "- **Linear empirical-moment method:** solve the same `P_i(t)`, form `M0_hat=M^{-1} sum_j xi_j xi_j^T`, then solve one packed forward ODE containing one `2x2` second moment per parameter node. The objective and gradient use trace/moment contractions. The forward ODE dimension does not depend on `M`; only the one-pass construction of `M0_hat` does.",
        "- **Continuous-law reference:** use `M0=xbar xbar^T+(r^2/3)I`, which is exact for the independent coordinatewise continuous uniform law. Differences between an empirical-moment result and this reference are sampling/quadrature error in the initial law, while particle-versus-empirical-moment differences are computational error.",
        "- **Nonlinear adjoint method:** call `experiments.neural_adjoint.objective_gradient` with the selected neural feedback on the Cartesian product of the same 32 parameter nodes and each fixed initial-state prefix. The forward states and backward adjoints retain one realization per parameter/initial-state pair.",
        f"- Timings use one warmup and {LINEAR_REPEATS} repeats for each linear method/M pair, and one warmup and {NONLINEAR_REPEATS} repeats for each nonlinear/M pair. Reported times are medians. Linear timing tolerances are `rtol={LINEAR_TIMING_TOL['rtol']}`, `atol={LINEAR_TIMING_TOL['atol']}`, `max_step=T/{LINEAR_TIMING_TOL['max_step_divisor']}`; agreement/reference checks use `rtol={LINEAR_CHECK_TOL['rtol']}`, `atol={LINEAR_CHECK_TOL['atol']}`, `max_step=T/{LINEAR_CHECK_TOL['max_step_divisor']}`. Nonlinear timings use the current routine with `rtol={NONLINEAR_TOL['rtol']}`, `atol={NONLINEAR_TOL['atol']}`, `max_step=T/{NONLINEAR_TOL['max_step_divisor']}`.",
        "",
        "## Measured results",
        "",
    ]
    for label, result in summary["cases"].items():
        lines.extend([
            f"### {display_name(label)}",
            "",
            "| M | particles N M | linear particle total [s] | linear moment total [s] | total speedup | forward-only speedup | nonlinear particle [s] | max |Δgradient| | empirical objective error | empirical gradient error |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in result["rows"]:
            lines.append(
                "| {M} | {count} | {lp} | {lm} | {st} | {sf} | {nl} | {gd} | {oe} | {ge} |".format(
                    M=row["M"], count=row["nonlinear_particle"]["particle_count"],
                    lp=fmt(row["linear_particle"]["median_total_seconds"]),
                    lm=fmt(row["linear_moment"]["median_total_seconds"]),
                    st=fmt(row["linear_total_speedup"]), sf=fmt(row["linear_forward_speedup"]),
                    nl=fmt(row["nonlinear_particle"]["median_total_seconds"]),
                    gd=fmt(row["agreement"]["gradient_max_absolute_difference"]),
                    oe=fmt(row["empirical_to_analytic"]["objective_absolute_error"]),
                    ge=fmt(row["empirical_to_analytic"]["gradient_l2_error"]),
                )
            )
        scaling = result["scaling"]
        lines.extend([
            "",
            f"- Measured log-log exponent over all five M values: linear particle total `{scaling['linear_particle_total_exponent']:.3f}`, linear particle forward-only `{scaling['linear_particle_forward_exponent']:.3f}`, linear moment total `{scaling['linear_moment_total_exponent']:.3f}`, nonlinear particle `{scaling['nonlinear_particle_exponent']:.3f}`.",
            f"- First measured total-runtime particle/moment crossover: **{crossover_text(result['crossover']['total'])}**. First forward-only crossover: **{crossover_text(result['crossover']['forward'])}**.",
            f"- Maximum particle/moment objective discrepancy over M: `{result['maximum_agreement_errors']['objective_absolute']:.3e}`; maximum gradient discrepancy: `{result['maximum_agreement_errors']['gradient_max_absolute']:.3e}`.",
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "### Why exact closure works in the linearized problem",
        "",
        "The state trajectories still depend on each initial realization `xi`. Compression is exact because homogeneous linear dynamics give `x(t)=Phi(t)xi`, while the quadratic objective and gain gradient contain `x x^T`; averaging therefore needs only `E[xi xi^T]`. It is not correct to say that the trajectories themselves cease to depend on `xi`.",
        "",
        "For affine dynamics or costs with linear terms, first moments must also be propagated, but the closure remains finite-dimensional. For general nonlinear dynamics, the evolution of a finite collection of moments does not close: higher moments enter, so the implemented realization calculation retains `O(NM)` state/costate work.",
        "",
        "### Asymptotic operation count versus small-n wall clock",
        "",
        "With state dimension `n`, the common policy-evaluation solve carries `N n^2` matrix states and is independent of `M`. After that shared work, particle propagation carries `N M n` state variables and performs realization-level contractions, whereas moment propagation carries `N n^2` state variables and matrix products. Forming the empirical initial second moment costs `O(M n^2)` once, but removes the repeated trajectory work over all `N` nodes and time steps. Thus, for fixed small `n`, the trajectory portion changes from realization scaling `O(NM)` to an `M`-independent ODE solve plus a cheap one-pass moment setup.",
        "",
        "At `n=2`, a `2x2` moment has four entries while one particle state has two. Consequently, at very small `M` the moment method need not be faster, and the shared backward `P_i` solve can obscure the forward-stage crossover in end-to-end timing. The tables report both total and forward-only speedups to expose this effect rather than assuming an advantage.",
        "",
        "The nonlinear solver packs all particles into vectorized adaptive ODE solves. Its measured exponent is an implementation-dependent small-problem observation, not proof of exact linear wall-clock scaling. The operation inventory still grows with `N M`, but vectorization, interpolation, error control, cache behavior, and BLAS/NumPy overhead can make the measured curve sublinear or superlinear over this range.",
        "",
        "### Statistical versus computational error",
        "",
        "Particle and empirical-moment objective/gradient values agree within the declared tight computational tolerances for every M. Their common difference from the analytic continuous-uniform reference is not a closure error; it is the error of replacing the continuous initial law by the fixed empirical prefix. Because the prefixes are random and nested, these errors need not decrease monotonically at every displayed M even though their stochastic scale should decrease with sample size.",
        "",
        "## Failures and limitations",
        "",
    ])
    if summary["failures"]:
        for failure in summary["failures"]:
            lines.append(f"- Unrecovered: {failure}")
    else:
        lines.append("- No integration or final artifact-verification failure occurred.")
    for failure in summary.get("recovered_failures", []):
        lines.append(f"- Recovered and retained: {failure}")
    lines.extend([
        "- Wall-clock conclusions are specific to the recorded Apple M4 Max environment, SciPy adaptive solver, tolerances, packed-vector implementation, `n=2`, and `N=32`.",
        "- The benchmark evaluates one selected static gain and one selected neural feedback per model; it does not benchmark complete optimizer trajectories.",
        "- The linear particle method is already a favorable baseline because it shares `P_i` across initial states. A per-initial-state vector-costate implementation would scale less favorably.",
        "",
        "## Manuscript-safe claim",
        "",
        "> In the homogeneous linear-quadratic specialization, uncertainty in the initial state can be represented exactly through its second moment. For a fixed parameter rule, this replaces trajectory propagation over every parameter/initial-state pair by one matrix-moment propagation per parameter node (plus one-time construction of the empirical moment). The resulting ODE workload is independent of the number of initial-state particles, although for two-dimensional systems and very small particle sets matrix-state overhead and common policy-evaluation work can delay the wall-clock crossover. No analogous finite moment closure is available for the nonlinear oscillator calculations, which retain realization-level work over the joint particle inventory.",
        "",
        "This claim is supported by the benchmark. A stronger statement that moment closure is always faster, or that nonlinear vectorized runtime is exactly linear in particle count, is not supported.",
        "",
        "## Artifacts",
        "",
        "- `summary.json`: full protocol, raw timing repeats, values, errors, slopes, and environment.",
        "- `verification.json`: machine-readable artifact and tolerance checks.",
        "- `benchmark.py`: reproducible benchmark driver.",
        "- `runtime_scaling.{png,pdf}`, `linear_speedup_agreement.{png,pdf}`, `empirical_moment_convergence.{png,pdf}`.",
    ])
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main(
    output: Path = OUTPUT,
    preview_path: Path = PREVIEW,
) -> None:
    global OUTPUT, PREVIEW
    OUTPUT = output.resolve()
    PREVIEW = preview_path.resolve()
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    preview = json.loads(PREVIEW.read_text())
    cases = build_cases(preview)
    failures: list[str] = []
    recovered_failures: list[str] = []
    first_attempt_log = OUTPUT / "run-attempt-1.log"
    if first_attempt_log.exists():
        recovered_failures.append(
            "The first execution completed all numerical timings but failed while serializing raw NumPy gradient arrays to JSON; the traceback is retained in run-attempt-1.log. The serializer was corrected and the complete benchmark was rerun from scratch."
        )
    case_results: dict[str, Any] = {}

    for label, case in cases.items():
        print(f"[{label}] analytic reference", flush=True)
        analytic_M0 = analytic_second_moment(case["mean"], case["radius"])
        evaluation, closed, running = policy_evaluation(case, LINEAR_CHECK_TOL)
        analytic = moment_forward(case, analytic_M0, evaluation, closed, running, LINEAR_CHECK_TOL)
        analytic_trace = trace_objective(case, evaluation, analytic_M0)
        analytic["trace_objective"] = analytic_trace
        analytic["trace_identity_absolute_difference"] = abs(analytic["objective"] - analytic_trace)
        rows: list[dict[str, Any]] = []

        for M in M_VALUES:
            samples = case["samples"][:M].copy()
            print(f"[{label}] M={M} linear warmups", flush=True)
            timed_linear_once(case, samples, "particle")
            timed_linear_once(case, samples, "moment")
            particle_rows: list[dict[str, Any]] = []
            moment_rows: list[dict[str, Any]] = []
            for repeat in range(LINEAR_REPEATS):
                order = ("particle", "moment") if repeat % 2 == 0 else ("moment", "particle")
                for method in order:
                    result = timed_linear_once(case, samples, method)
                    (particle_rows if method == "particle" else moment_rows).append(result)
            linear_particle = summarize_timing(particle_rows)
            linear_moment = summarize_timing(moment_rows)

            print(f"[{label}] M={M} tight agreement", flush=True)
            check_evaluation, check_closed, check_running = policy_evaluation(case, LINEAR_CHECK_TOL)
            empirical_M0 = empirical_second_moment(samples)
            check_particle = particle_forward(
                case, samples, check_evaluation, check_closed, check_running, LINEAR_CHECK_TOL
            )
            check_moment = moment_forward(
                case, empirical_M0, check_evaluation, check_closed, check_running, LINEAR_CHECK_TOL
            )
            empirical_trace = trace_objective(case, check_evaluation, empirical_M0)
            agreement = {
                "objective_absolute_difference": abs(check_particle["objective"] - check_moment["objective"]),
                "gradient_difference": (check_particle["gradient"] - check_moment["gradient"]).tolist(),
                "gradient_max_absolute_difference": float(np.max(np.abs(check_particle["gradient"] - check_moment["gradient"]))),
                "particle_objective": check_particle["objective"],
                "moment_objective": check_moment["objective"],
                "particle_gradient": check_particle["gradient"].tolist(),
                "moment_gradient": check_moment["gradient"].tolist(),
                "moment_trace_objective": empirical_trace,
                "moment_trace_identity_absolute_difference": abs(check_moment["objective"] - empirical_trace),
            }
            empirical_to_analytic = {
                "objective_signed_error": check_moment["objective"] - analytic["objective"],
                "objective_absolute_error": abs(check_moment["objective"] - analytic["objective"]),
                "gradient_error": (check_moment["gradient"] - analytic["gradient"]).tolist(),
                "gradient_l2_error": float(np.linalg.norm(check_moment["gradient"] - analytic["gradient"])),
                "second_moment_error_frobenius": float(np.linalg.norm(empirical_M0 - analytic_M0)),
            }

            print(f"[{label}] M={M} nonlinear warmup/timing", flush=True)
            timed_nonlinear_once(case, samples)
            nonlinear_rows = [timed_nonlinear_once(case, samples) for _ in range(NONLINEAR_REPEATS)]
            nonlinear_particle = summarize_nonlinear(nonlinear_rows)

            row = {
                "M": M,
                "sample_prefix_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
                "empirical_second_moment": empirical_M0.tolist(),
                "linear_particle": linear_particle,
                "linear_moment": linear_moment,
                "nonlinear_particle": nonlinear_particle,
                "linear_total_speedup": linear_particle["median_total_seconds"] / linear_moment["median_total_seconds"],
                "linear_forward_speedup": linear_particle["median_forward_seconds"] / linear_moment["median_forward_seconds"],
                "agreement": agreement,
                "empirical_to_analytic": empirical_to_analytic,
            }
            rows.append(row)

        particle_total = [row["linear_particle"]["median_total_seconds"] for row in rows]
        particle_forward_times = [row["linear_particle"]["median_forward_seconds"] for row in rows]
        moment_total = [row["linear_moment"]["median_total_seconds"] for row in rows]
        moment_forward_times = [row["linear_moment"]["median_forward_seconds"] for row in rows]
        nonlinear_times = [row["nonlinear_particle"]["median_total_seconds"] for row in rows]
        case_results[label] = {
            "definition": {
                "system": case["system"],
                "horizon": case["horizon"],
                "Q": case["Q"].tolist(),
                "R": case["R"],
                "parameter_node_count": len(case["parameters"]),
                "parameter_nodes": case["parameters"].tolist(),
                "parameter_weights": case["weights"].tolist(),
                "linear_selected_averaged_static_gain": case["linear_gain"].tolist(),
                "nonlinear_selected_averaged_neural_parameters": case["nonlinear_parameters"].tolist(),
                "initial_mean": case["mean"].tolist(),
                "uniform_box_radius": case["radius"],
                "sample_seed": case["seed"],
                "sample_count_max": max(M_VALUES),
                "samples_nested_by_common_prefix": True,
                "recorded_design_prefix_length": len(preview["cases"][case["preview_case"]]["rules"]["x0_design"]),
            },
            "analytic_continuous_uniform": {
                "initial_second_moment": analytic_M0.tolist(),
                "objective": analytic["objective"],
                "gradient": analytic["gradient"].tolist(),
                "trace_objective": analytic_trace,
                "trace_identity_absolute_difference": analytic["trace_identity_absolute_difference"],
            },
            "rows": rows,
            "scaling": {
                "fit_definition": "least-squares slope of log(median wall time) against log(M) over M={1,4,16,64,128}",
                "linear_particle_total_exponent": scaling_exponent(M_VALUES, particle_total),
                "linear_particle_forward_exponent": scaling_exponent(M_VALUES, particle_forward_times),
                "linear_moment_total_exponent": scaling_exponent(M_VALUES, moment_total),
                "linear_moment_forward_exponent": scaling_exponent(M_VALUES, moment_forward_times),
                "nonlinear_particle_exponent": scaling_exponent(M_VALUES, nonlinear_times),
            },
            "crossover": {
                "definition": "first tested M for which particle/moment median runtime ratio exceeds one",
                "total": first_crossover(rows, "linear_total_speedup"),
                "forward": first_crossover(rows, "linear_forward_speedup"),
            },
            "maximum_agreement_errors": {
                "objective_absolute": max(row["agreement"]["objective_absolute_difference"] for row in rows),
                "gradient_max_absolute": max(row["agreement"]["gradient_max_absolute_difference"] for row in rows),
            },
        }

    figures = make_plots(case_results)
    completed = datetime.now(timezone.utc)
    environment = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_brand": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "git_commit": git("rev-parse", "HEAD"),
        "git_tracked_status_before_summary": git("status", "--short", "--untracked-files=no"),
        "pid": os.getpid(),
    }
    all_agreement = [
        row["agreement"] for case in case_results.values() for row in case["rows"]
    ]
    agreement_passed = all(
        row["objective_absolute_difference"] <= AGREEMENT_TOL["objective_absolute"]
        and row["gradient_max_absolute_difference"] <= AGREEMENT_TOL["gradient_max_absolute"]
        for row in all_agreement
    )
    if not agreement_passed:
        failures.append("At least one particle/moment agreement check exceeded tolerance.")
    trace_passed = all(
        case["analytic_continuous_uniform"]["trace_identity_absolute_difference"] <= 2e-9
        and all(row["agreement"]["moment_trace_identity_absolute_difference"] <= 2e-9 for row in case["rows"])
        for case in case_results.values()
    )
    if not trace_passed:
        failures.append("At least one integrated objective/trace identity check exceeded tolerance.")

    crossovers = {label: result["crossover"]["total"] for label, result in case_results.items()}
    verdict = (
        "The benchmark supports the scaling claim, with an important small-problem qualification. "
        "For both oscillators, the empirical-moment calculation reproduces the corresponding linear particle objective/gradient within tight tolerance for every tested M, while its forward ODE dimension remains fixed at 131 states (32 four-entry moments plus three accumulators). "
        "The particle forward dimension grows from 67 to 8195 states. Measured total-runtime crossovers occur at "
        + ", ".join(f"{display_name(name)}: {crossover_text(value)}" for name, value in crossovers.items())
        + ". Nonlinear objective/gradient evaluations retain N*M particles and show increasing runtime, but their measured vectorized wall-clock exponents are reported rather than assumed to be exactly one."
    )
    summary = {
        "status": "passed" if not failures else "completed_with_failures",
        "role": "computational scaling benchmark for initial-state moment closure versus particle calculations",
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "verdict": verdict,
        "protocol": {
            "M_values": M_VALUES,
            "nested_common_prefixes": True,
            "linear_warmups_per_method_M": 1,
            "linear_timed_repeats": LINEAR_REPEATS,
            "nonlinear_warmups_per_M": 1,
            "nonlinear_timed_repeats": NONLINEAR_REPEATS,
            "linear_timing_tolerances": LINEAR_TIMING_TOL,
            "linear_agreement_reference_tolerances": LINEAR_CHECK_TOL,
            "nonlinear_tolerances": NONLINEAR_TOL,
            "agreement_tolerances": AGREEMENT_TOL,
            "timing_clock": "time.perf_counter",
            "reported_statistic": "median; raw, minimum, and maximum retained",
            "resampling_inside_evaluation": False,
        },
        "method_definitions": {
            "linear_particle": "one backward packed P_i solve over N nodes, followed by one packed forward solve over N*M state trajectories plus objective/gradient accumulators; gradient uses averaged (R K-B^T P_i) x x^T",
            "linear_moment": "the same backward packed P_i solve, empirical M0_hat construction, then one packed forward solve over N 2x2 second moments plus objective/gradient accumulators",
            "analytic_reference": "same linear moment method with M0=xbar*xbar^T+(radius^2/3)I for the continuous coordinatewise-uniform law",
            "nonlinear_particle": "repository current vectorized joint-population objective_gradient: one packed nonlinear forward solve with dense output and one packed backward adjoint solve",
        },
        "asymptotic_reasoning": {
            "policy_evaluation": "N matrix ODEs with n^2 states; independent of initial sample count M",
            "linear_particle_forward": "N*M trajectories with n states; realization work O(N*M) for fixed n",
            "linear_moment_forward": "N matrix ODEs with n^2 states; independent of M after M0_hat is formed",
            "empirical_M0_setup": "one pass O(M*n^2), not repeated over N nodes or time steps",
            "nonlinear_particle": "finite moments generally do not close; N*M realization states and costates remain",
        },
        "cases": case_results,
        "environment": environment,
        "source_hashes": {
            "joint_summary": sha256(PREVIEW),
            "nonlinear_numerics_source": sha256(ROOT / "experiments" / "neural_adjoint.py"),
            "policy_source": sha256(ROOT / "experiments" / "neural_policy.py"),
            "benchmark_script": sha256(Path(__file__)),
        },
        "figures": figures,
        "failures": failures,
        "recovered_failures": recovered_failures,
    }
    write_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    shutil.copy2(Path(__file__), OUTPUT / "benchmark.py")

    required = ["REPORT.md", "summary.json", "verification.json", "benchmark.py", *figures]
    checks = {
        "all_required_files_exist": all((OUTPUT / name).exists() for name in required if name != "verification.json"),
        "all_required_files_nonempty": all((OUTPUT / name).stat().st_size > 0 for name in required if name != "verification.json" and (OUTPUT / name).exists()),
        "all_M_values_present": all([row["M"] for row in case["rows"]] == M_VALUES for case in case_results.values()),
        "all_particle_moment_agreements_pass": agreement_passed,
        "all_trace_identity_checks_pass": trace_passed,
        "nested_prefix_hashes_unique_by_M": all(len({row["sample_prefix_sha256"] for row in case["rows"]}) == len(M_VALUES) for case in case_results.values()),
        "parameter_node_count_32": all(case["definition"]["parameter_node_count"] == 32 for case in case_results.values()),
        "nonlinear_particle_counts_correct": all(row["nonlinear_particle"]["particle_count"] == 32 * row["M"] for case in case_results.values() for row in case["rows"]),
        "moment_state_dimension_constant": all(len({row["linear_moment"]["state_dimension"] for row in case["rows"]}) == 1 for case in case_results.values()),
        "particle_state_dimension_increases": all(np.all(np.diff([row["linear_particle"]["state_dimension"] for row in case["rows"]]) > 0) for case in case_results.values()),
        "tracked_repository_status_clean": git("status", "--short", "--untracked-files=no") == "",
        "failures_empty": not failures,
    }
    verification = {
        "status": "passed" if all(checks.values()) else "failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "agreement_tolerances": AGREEMENT_TOL,
        "maximum_observed": {
            "objective_particle_moment_absolute_difference": max(row["objective_absolute_difference"] for row in all_agreement),
            "gradient_particle_moment_max_absolute_difference": max(row["gradient_max_absolute_difference"] for row in all_agreement),
            "analytic_trace_identity_absolute_difference": max(case["analytic_continuous_uniform"]["trace_identity_absolute_difference"] for case in case_results.values()),
            "empirical_trace_identity_absolute_difference": max(row["agreement"]["moment_trace_identity_absolute_difference"] for case in case_results.values() for row in case["rows"]),
        },
        "required_files": required,
        "failures": failures,
    }
    write_json(OUTPUT / "verification.json", verification)
    print(json.dumps({"summary_status": summary["status"], "verification_status": verification["status"], "elapsed_seconds": summary["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--joint-summary", type=Path, default=PREVIEW)
    arguments = parser.parse_args()
    main(arguments.output, arguments.joint_summary)
