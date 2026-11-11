"""Oscillator data used by the paper reproduction."""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
GAIN_BOUNDS = ((-20.0, 20.0), (-5.0, 20.0))
CASES = ("nonlinear-vdp", "nonlinear-stuart-landau")


def uniform_midpoint_rule(lower: float, upper: float, size: int):
    nodes = lower + (upper - lower) * (np.arange(size) + 0.5) / size
    return nodes, np.full(size, 1.0 / size)


def uniform_legendre_rule(lower: float, upper: float, order: int):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes, weights / 2.0


def gaussian_midpoint_quantiles(mean: float, sigma: float, size: int):
    normal = NormalDist(mean, sigma)
    probabilities = [(index + 0.5) / size for index in range(size)]
    parameters = np.asarray([normal.inv_cdf(value) for value in probabilities])
    return parameters, np.full(size, 1.0 / size)


def gaussian_hermite_rule(mean: float, sigma: float, order: int):
    nodes, weights = hermgauss(order)
    return mean + np.sqrt(2.0) * sigma * nodes, weights / np.sqrt(np.pi)


def case_data(case: str) -> dict:
    """Return one of the two model definitions used in the paper."""
    if case not in CASES:
        raise ValueError(f"unknown active case: {case}")
    if case == "nonlinear-vdp":
        lower, upper = -2.0, 4.0
        initial = np.array([1.0, 1.0])
        Q = np.eye(2)
        R = 0.2
        system = "van_der_pol"
    else:
        lower, upper = -2.1, 0.8
        initial = np.array([0.5, 0.5])
        Q = 2.0 * np.eye(2)
        R = 1.0
        system = "stuart_landau"
    design_parameters, design_weights = uniform_midpoint_rule(lower, upper, 32)
    evaluation_parameters, evaluation_weights = uniform_legendre_rule(lower, upper, 128)
    return {
        "case": case,
        "system": system,
        "nonlinear": True,
        "nominal_parameter": 0.5 * (lower + upper),
        "design_parameters": design_parameters,
        "design_weights": design_weights,
        "evaluation_parameters": evaluation_parameters,
        "evaluation_weights": evaluation_weights,
        "horizon": 5.0,
        "initial": initial,
        "Q": Q,
        "R": R,
        "bounds": GAIN_BOUNDS,
        "times": np.linspace(0.0, 5.0, 401),
    }


def rollout(
    data: dict,
    gain: np.ndarray,
    parameters: np.ndarray,
    weights: np.ndarray,
    *,
    times: np.ndarray | None,
    rtol: float,
    atol: float,
    max_step_divisor: float,
) -> dict:
    """Integrate every parameter node under one static gain."""
    gain = np.asarray(gain, dtype=float).reshape(2)
    parameters = np.asarray(parameters, dtype=float)
    weights = np.asarray(weights, dtype=float)
    count = len(parameters)
    initial = np.r_[np.tile(data["initial"], count), np.zeros(count)]

    def rhs(_time: float, packed: np.ndarray) -> np.ndarray:
        states = packed[: 2 * count].reshape(count, 2)
        inputs = -(states @ gain)
        q, v = states[:, 0], states[:, 1]
        if data["system"] == "van_der_pol":
            derivatives = np.column_stack(
                (v, -q + parameters * (1.0 - q**2) * v + inputs)
            )
        else:
            radius_squared = q**2 + v**2
            derivatives = np.column_stack(
                (
                    parameters * q - v - radius_squared * q,
                    q + parameters * v - radius_squared * v + inputs,
                )
            )
        running = 0.5 * (
            np.einsum("ni,ij,nj->n", states, data["Q"], states)
            + data["R"] * inputs**2
        )
        result = np.r_[derivatives.ravel(), running]
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite static-gain rollout")
        return result

    solution = solve_ivp(
        rhs,
        (0.0, data["horizon"]),
        initial,
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
        max_step=data["horizon"] / max_step_divisor,
    )
    if not solution.success or solution.t[-1] != data["horizon"]:
        raise RuntimeError("static-gain rollout failed: " + solution.message)
    costs = solution.y[2 * count :]
    result = {
        "cost": float(weights @ costs[:, -1]),
        "per_particle_cost": costs[:, -1].copy(),
        "nfev": int(solution.nfev),
    }
    if times is not None:
        states = solution.y[: 2 * count].reshape(
            count, 2, len(solution.t)
        ).transpose(0, 2, 1)
        result.update(
            time=solution.t.copy(),
            states=states,
            inputs=-np.einsum("nti,i->nt", states, gain),
            cumulative_cost=costs.copy(),
        )
    return result
