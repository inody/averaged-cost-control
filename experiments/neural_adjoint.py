"""Forward, adjoint, optimization, and rollout routines for neural feedback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from experiments.neural_policy import PolicyEvaluation


class Policy(Protocol):
    parameter_count: int
    parameter_bounds: list[tuple[float, float]]

    def evaluate(
        self, time: float, states: np.ndarray, parameters: np.ndarray,
    ) -> PolicyEvaluation: ...


@dataclass(frozen=True)
class JointPopulation:
    """Weighted joint particles ``(omega, x0)`` without subset filtering."""

    omega: np.ndarray
    initial_states: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        omega = np.asarray(self.omega, dtype=float)
        initial = np.asarray(self.initial_states, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if omega.ndim != 1 or initial.shape != (len(omega), 2):
            raise ValueError("joint particles have incompatible shapes")
        if weights.shape != omega.shape or len(omega) == 0:
            raise ValueError("joint weights have the wrong shape")
        if not all(np.isfinite(value).all() for value in (omega, initial, weights)):
            raise ValueError("joint population must be finite")
        if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=2e-13):
            raise ValueError("joint weights must be positive and sum to one")

    @property
    def size(self) -> int:
        return len(self.omega)


def product_population(
    omega: np.ndarray,
    omega_weights: np.ndarray,
    initial_states: np.ndarray,
    initial_weights: np.ndarray,
) -> JointPopulation:
    omega = np.asarray(omega, dtype=float)
    omega_weights = np.asarray(omega_weights, dtype=float)
    initial_states = np.asarray(initial_states, dtype=float)
    initial_weights = np.asarray(initial_weights, dtype=float)
    if omega_weights.shape != omega.shape:
        raise ValueError("omega weights mismatch")
    if initial_states.ndim != 2 or initial_states.shape[1] != 2:
        raise ValueError("initial states must have shape (M,2)")
    if initial_weights.shape != (len(initial_states),):
        raise ValueError("initial weights mismatch")
    return JointPopulation(
        omega=np.repeat(omega, len(initial_states)),
        initial_states=np.tile(initial_states, (len(omega), 1)),
        weights=np.repeat(omega_weights, len(initial_states))
        * np.tile(initial_weights, len(omega)),
    )


def dynamics(
    data: dict, omega: np.ndarray, states: np.ndarray, controls: np.ndarray,
) -> np.ndarray:
    q, v = states[:, 0], states[:, 1]
    if data["system"] == "van_der_pol":
        return np.column_stack((v, -q + omega * (1.0 - q**2) * v + controls))
    if data["system"] == "stuart_landau":
        radius_squared = q**2 + v**2
        return np.column_stack((
            omega * q - v - radius_squared * q,
            q + omega * v - radius_squared * v + controls,
        ))
    raise ValueError(data["system"])


def open_loop_jacobian(
    data: dict, omega: np.ndarray, states: np.ndarray,
) -> np.ndarray:
    q, v = states[:, 0], states[:, 1]
    jacobian = np.empty((len(states), 2, 2))
    if data["system"] == "van_der_pol":
        jacobian[:, 0, 0] = 0.0
        jacobian[:, 0, 1] = 1.0
        jacobian[:, 1, 0] = -1.0 - 2.0 * omega * q * v
        jacobian[:, 1, 1] = omega * (1.0 - q**2)
    elif data["system"] == "stuart_landau":
        jacobian[:, 0, 0] = omega - (3.0 * q**2 + v**2)
        jacobian[:, 0, 1] = -1.0 - 2.0 * q * v
        jacobian[:, 1, 0] = 1.0 - 2.0 * q * v
        jacobian[:, 1, 1] = omega - (q**2 + 3.0 * v**2)
    else:
        raise ValueError(data["system"])
    return jacobian


def _forward_solve(
    data: dict,
    policy: Policy,
    theta: np.ndarray,
    population: JointPopulation,
    *,
    times: np.ndarray | None,
    dense_output: bool,
    rtol: float,
    atol: float,
    max_step_divisor: float,
):
    count = population.size
    horizon = float(data["horizon"])
    Q = np.asarray(data["Q"], dtype=float)
    R = float(data["R"])

    def rhs(time: float, packed: np.ndarray) -> np.ndarray:
        states = packed[:2 * count].reshape(count, 2)
        evaluation = policy.evaluate(time, states, theta)
        derivatives = dynamics(data, population.omega, states, evaluation.controls)
        running = 0.5 * (
            np.einsum("ni,ij,nj->n", states, Q, states)
            + R * evaluation.controls**2
        )
        result = np.r_[derivatives.ravel(), running]
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite joint rollout")
        return result

    initial = np.r_[population.initial_states.ravel(), np.zeros(count)]
    solution = solve_ivp(
        rhs,
        (0.0, horizon),
        initial,
        method="DOP853",
        t_eval=times,
        dense_output=dense_output,
        rtol=rtol,
        atol=atol,
        max_step=horizon / max_step_divisor,
    )
    if not solution.success or solution.t[-1] != horizon:
        raise RuntimeError("joint forward solve failed: " + solution.message)
    return solution


def objective_cost(
    data: dict,
    policy: Policy,
    theta: np.ndarray,
    population: JointPopulation,
    *,
    rtol: float = 2e-9,
    atol: float = 2e-11,
    max_step_divisor: float = 180.0,
) -> float:
    solution = _forward_solve(
        data, policy, np.asarray(theta, dtype=float), population,
        times=None, dense_output=False, rtol=rtol, atol=atol,
        max_step_divisor=max_step_divisor,
    )
    costs = solution.y[2 * population.size:, -1]
    return float(population.weights @ costs)


def objective_gradient(
    data: dict,
    policy: Policy,
    theta: np.ndarray,
    population: JointPopulation,
    *,
    rtol: float = 2e-9,
    atol: float = 2e-11,
    max_step_divisor: float = 180.0,
) -> tuple[float, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (policy.parameter_count,) or not np.isfinite(theta).all():
        raise ValueError("invalid policy parameters")
    count = population.size
    horizon = float(data["horizon"])
    Q = np.asarray(data["Q"], dtype=float)
    R = float(data["R"])
    forward = _forward_solve(
        data, policy, theta, population,
        times=None, dense_output=True, rtol=rtol, atol=atol,
        max_step_divisor=max_step_divisor,
    )
    cost = float(population.weights @ forward.y[2 * count:, -1])

    def backward_rhs(time: float, packed: np.ndarray) -> np.ndarray:
        costates = packed[:2 * count].reshape(count, 2)
        states = forward.sol(time)[:2 * count].reshape(count, 2)
        evaluation = policy.evaluate(time, states, theta)
        jacobian = open_loop_jacobian(data, population.omega, states)
        jacobian[:, 1, :] += evaluation.control_state_jacobian
        running_gradient = (
            states @ Q.T
            + R * evaluation.controls[:, None]
            * evaluation.control_state_jacobian
        )
        costate_derivative = -(
            np.einsum("nji,nj->ni", jacobian, costates) + running_gradient
        )
        hamiltonian_u = R * evaluation.controls + costates[:, 1]
        parameter_integrand = (
            hamiltonian_u[:, None] * evaluation.control_parameter_jacobian
        )
        accumulator_derivative = -(
            population.weights[:, None] * parameter_integrand
        ).sum(axis=0)
        result = np.r_[costate_derivative.ravel(), accumulator_derivative]
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite joint adjoint")
        return result

    backward = solve_ivp(
        backward_rhs,
        (horizon, 0.0),
        np.zeros(2 * count + policy.parameter_count),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=horizon / max_step_divisor,
    )
    if not backward.success or backward.t[-1] != 0.0:
        raise RuntimeError("joint adjoint solve failed: " + backward.message)
    return cost, backward.y[-policy.parameter_count:, -1].copy()


def guarded_objective_gradient(
    data: dict, policy: Policy, theta: np.ndarray, population: JointPopulation,
) -> tuple[float, np.ndarray]:
    values = np.asarray(theta, dtype=float)
    try:
        return objective_gradient(data, policy, values, population)
    except (FloatingPointError, RuntimeError, ValueError):
        return 1e12 + float(values @ values), 2.0 * values


def projected_gradient_mapping(
    parameters: np.ndarray,
    gradient: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    grad = np.asarray(gradient, dtype=float)
    lower = np.asarray([bound[0] for bound in bounds])
    upper = np.asarray([bound[1] for bound in bounds])
    return values - np.clip(values - grad, lower, upper)


def projected_gradient_norm(
    parameters: np.ndarray,
    gradient: np.ndarray,
    bounds: list[tuple[float, float]],
) -> float:
    return float(np.linalg.norm(projected_gradient_mapping(parameters, gradient, bounds)))


def projected_gradient_residual(
    parameters: np.ndarray,
    gradient: np.ndarray,
    bounds: list[tuple[float, float]],
) -> float:
    """Entrywise infinity norm of the projected stationarity mapping."""
    mapping = projected_gradient_mapping(parameters, gradient, bounds)
    return float(np.linalg.norm(mapping, ord=np.inf))


def optimize_multistart(
    data: dict,
    policy: Policy,
    population: JointPopulation,
    starts: list[np.ndarray],
    *,
    maxiter: int,
    label: str,
) -> tuple[np.ndarray, list[dict]]:
    records: list[dict] = []
    for index, start in enumerate(starts, start=1):
        print(f"[{label}] start {index}/{len(starts)}", flush=True)
        result = minimize(
            lambda theta: guarded_objective_gradient(
                data, policy, theta, population
            ),
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            jac=True,
            bounds=policy.parameter_bounds,
            options={
                "ftol": 1e-13,
                "gtol": 2e-7,
                "maxiter": maxiter,
                "maxls": 60,
            },
        )
        finite = bool(
            np.isfinite(result.fun)
            and result.fun < 1e11
            and np.isfinite(result.x).all()
            and np.isfinite(result.jac).all()
        )
        record = {
            "start": np.asarray(start, dtype=float).tolist(),
            "finite": finite,
            "optimizer_terminated": bool(result.success),
            "message": str(result.message),
            "parameters": np.asarray(result.x, dtype=float).tolist(),
            "design_cost": float(result.fun),
            "gradient_norm": float(np.linalg.norm(result.jac)),
            "projected_gradient_norm": projected_gradient_norm(
                result.x, result.jac, policy.parameter_bounds
            ),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "gradient_evaluations": int(result.njev),
        }
        records.append(record)
        print(
            f"[{label}] cost={record['design_cost']:.10g} "
            f"pgrad={record['projected_gradient_norm']:.3g} "
            f"nit={record['iterations']} finite={finite}",
            flush=True,
        )
    if not all(row["finite"] for row in records):
        raise RuntimeError(f"at least one declared optimization start failed: {label}")
    selected = min(records, key=lambda row: row["design_cost"])
    return np.asarray(selected["parameters"], dtype=float), records


def finite_difference_check_all(
    data: dict,
    policy: Policy,
    theta: np.ndarray,
    population: JointPopulation,
    *,
    step: float = 1e-5,
) -> dict:
    cost, gradient = objective_gradient(
        data, policy, theta, population,
        rtol=2e-11, atol=2e-13, max_step_divisor=500.0,
    )
    rows = []
    for index in range(policy.parameter_count):
        direction = np.zeros(policy.parameter_count)
        direction[index] = step
        plus = objective_cost(
            data, policy, theta + direction, population,
            rtol=2e-11, atol=2e-13, max_step_divisor=500.0,
        )
        minus = objective_cost(
            data, policy, theta - direction, population,
            rtol=2e-11, atol=2e-13, max_step_divisor=500.0,
        )
        numeric = (plus - minus) / (2.0 * step)
        absolute = float(abs(gradient[index] - numeric))
        rows.append({
            "index": index,
            "adjoint": float(gradient[index]),
            "finite_difference": float(numeric),
            "absolute_difference": absolute,
            "scaled_difference": float(absolute / max(1.0, abs(numeric))),
        })
    return {
        "cost": cost,
        "step": step,
        "particle_count": population.size,
        "checked_parameter_count": len(rows),
        "rows": rows,
        "maximum_absolute_difference": max(
            row["absolute_difference"] for row in rows
        ),
        "maximum_scaled_difference": max(
            row["scaled_difference"] for row in rows
        ),
    }


def rollout(
    data: dict,
    policy: Policy,
    theta: np.ndarray,
    population: JointPopulation,
    *,
    times: np.ndarray | None,
    rtol: float = 2e-10,
    atol: float = 2e-12,
    max_step_divisor: float = 350.0,
) -> dict:
    solution = _forward_solve(
        data, policy, np.asarray(theta, dtype=float), population,
        times=times, dense_output=False, rtol=rtol, atol=atol,
        max_step_divisor=max_step_divisor,
    )
    count = population.size
    costs = solution.y[2 * count:]
    result = {
        "cost": float(population.weights @ costs[:, -1]),
        "per_particle_cost": costs[:, -1].copy(),
        "nfev": int(solution.nfev),
    }
    if times is not None:
        states = solution.y[:2 * count].reshape(
            count, 2, len(solution.t)
        ).transpose(0, 2, 1)
        controls = np.empty((count, len(solution.t)))
        gains = np.empty((count, len(solution.t), 2))
        for index, time in enumerate(solution.t):
            evaluation = policy.evaluate(time, states[:, index], theta)
            controls[:, index] = evaluation.controls
            gains[:, index] = evaluation.gains
        result.update(
            time=solution.t.copy(),
            states=states,
            controls=controls,
            gains=gains,
            cumulative_cost=costs.copy(),
        )
    return result


def verify_rollout(result: dict, population: JointPopulation, label: str) -> dict:
    per_particle = np.asarray(result["per_particle_cost"], dtype=float)
    if per_particle.shape != (population.size,):
        raise RuntimeError(f"particle inventory mismatch: {label}")
    if not np.isfinite(per_particle).all() or np.any(per_particle < -1e-11):
        raise RuntimeError(f"invalid particle costs: {label}")
    recomputed = float(population.weights @ per_particle)
    if not np.isclose(recomputed, result["cost"], rtol=2e-12, atol=2e-11):
        raise RuntimeError(f"weighted cost mismatch: {label}")
    return {
        "particle_count": population.size,
        "all_particles_retained": True,
        "all_costs_finite": True,
        "weighted_cost_recomputed": recomputed,
    }
