"""Linear moment/Lyapunov method and Anderson--Moore iteration."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

B = np.array([[0.0], [1.0]])
INITIAL_GAIN = np.array([1.0, 4.0])
STATIC_STATIONARITY_TOLERANCE = 1e-8
ARMIJO_CONSTANT = 1e-4
ARMIJO_CONTRACTION = 0.5
MINIMUM_STEP_SIZE = 2.0 ** -20
MAXIMUM_FIXED_POINT_ITERATIONS = 400


def matrices(data: dict, parameters: np.ndarray) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    result = np.zeros((len(parameters), 2, 2))
    if data["system"] == "van_der_pol":
        result[:, 0, 1] = 1.0
        result[:, 1, 0] = -1.0
        result[:, 1, 1] = parameters
    elif data["system"] == "stuart_landau":
        result[:, 0, 0] = parameters
        result[:, 0, 1] = -1.0
        result[:, 1, 0] = 1.0
        result[:, 1, 1] = parameters
    else:
        raise ValueError(data["system"])
    return result


def policy_evaluation(
    data: dict,
    gain: np.ndarray,
    parameters: np.ndarray,
    *,
    rtol: float,
    atol: float,
    max_step_divisor: float,
    dense_output: bool,
):
    gain = np.asarray(gain, dtype=float).reshape(2)
    parameters = np.asarray(parameters, dtype=float)
    count = len(parameters)
    dynamics = matrices(data, parameters)
    closed = dynamics - B[None, :, :] * gain[None, None, :]
    running = data["Q"] + data["R"] * np.outer(gain, gain)
    horizon = float(data["horizon"])

    def rhs(_time: float, packed: np.ndarray) -> np.ndarray:
        P = packed.reshape(count, 2, 2)
        derivative = -(
            np.einsum("nji,njk->nik", closed, P)
            + np.einsum("nij,njk->nik", P, closed)
            + running[None, :, :]
        )
        return (0.5 * (derivative + np.swapaxes(derivative, 1, 2))).ravel()

    solution = solve_ivp(
        rhs,
        (horizon, 0.0),
        np.zeros(4 * count),
        method="DOP853",
        dense_output=dense_output,
        rtol=rtol,
        atol=atol,
        max_step=horizon / max_step_divisor,
    )
    if not solution.success or solution.t[-1] != 0.0:
        raise RuntimeError("policy-evaluation solve failed: " + solution.message)
    return solution, closed


def anderson_moore_iteration(evaluate, start: np.ndarray = INITIAL_GAIN) -> dict:
    gain = np.asarray(start, dtype=float).copy()
    current = evaluate(gain)
    evaluations = 1
    trace = []
    for iteration in range(MAXIMUM_FIXED_POINT_ITERATIONS + 1):
        trace.append({
            "iteration": iteration,
            "gain": gain.copy(),
            "design_cost": current["cost"],
            "gradient": current["gradient"],
            "stationarity_residual": current["stationarity_residual"],
            "occupancy_condition_number": current["occupancy_condition_number"],
        })
        if current["stationarity_residual"] <= STATIC_STATIONARITY_TOLERANCE:
            return {
                "start": np.asarray(start, dtype=float),
                "gain": gain,
                "iterations": iteration,
                "function_evaluations": evaluations,
                "trace": trace,
                "terminal": current,
            }
        direction = current["fixed_point_target"] - gain
        slope = float(current["gradient"] @ direction)
        if not np.isfinite(slope) or slope >= 0.0:
            raise RuntimeError("Anderson--Moore direction is not a descent direction")
        step_size = 1.0
        while step_size >= MINIMUM_STEP_SIZE:
            trial_gain = gain + step_size * direction
            try:
                trial = evaluate(trial_gain)
                evaluations += 1
            except (FloatingPointError, RuntimeError, ValueError):
                trial = None
            if trial is not None and trial["cost"] <= (
                current["cost"] + ARMIJO_CONSTANT * step_size * slope
            ):
                trace[-1]["step_size"] = step_size
                gain = trial_gain
                current = trial
                break
            step_size *= ARMIJO_CONTRACTION
        else:
            raise RuntimeError("Anderson--Moore Armijo search failed")
    raise RuntimeError("Anderson--Moore iteration limit reached")
