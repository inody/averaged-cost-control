"""Neural shared-feedback policy used in the nonlinear adjoint method."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyEvaluation:
    gains: np.ndarray
    controls: np.ndarray
    control_state_jacobian: np.ndarray
    control_parameter_jacobian: np.ndarray


@dataclass(frozen=True)
class NeuralFeedbackOperator:
    """One-hidden-layer tanh map from normalized ``(t,x)`` to control."""

    horizon: float
    state_scale: np.ndarray
    control_scale: float
    hidden_width: int = 16

    def __post_init__(self) -> None:
        scale = np.asarray(self.state_scale, dtype=float)
        if scale.shape != (2,) or np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError("state_scale must contain two positive values")
        if self.horizon <= 0 or self.control_scale <= 0 or self.hidden_width < 1:
            raise ValueError("invalid neural-policy settings")
        scale = scale.copy()
        scale.setflags(write=False)
        object.__setattr__(self, "state_scale", scale)

    @property
    def parameter_count(self) -> int:
        return 5 * self.hidden_width + 1

    def unpack(self, parameters: np.ndarray):
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.parameter_count,):
            raise ValueError("wrong neural parameter count")
        cursor = 0
        W1 = values[cursor:cursor + 3 * self.hidden_width].reshape(self.hidden_width, 3)
        cursor += 3 * self.hidden_width
        b1 = values[cursor:cursor + self.hidden_width]
        cursor += self.hidden_width
        w2 = values[cursor:cursor + self.hidden_width]
        return W1, b1, w2, float(values[-1])

    def initial_parameters(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        W1 = rng.normal(scale=0.5, size=(self.hidden_width, 3))
        b1 = rng.normal(scale=0.2, size=self.hidden_width)
        w2 = rng.normal(scale=0.05 / np.sqrt(self.hidden_width), size=self.hidden_width)
        return np.r_[W1.ravel(), b1, w2, 0.0]

    def normalized_inputs(self, time: np.ndarray, states: np.ndarray) -> np.ndarray:
        return np.column_stack((
            2.0 * np.asarray(time, dtype=float) / self.horizon - 1.0,
            np.asarray(states, dtype=float)[:, 0] / self.state_scale[0],
            np.asarray(states, dtype=float)[:, 1] / self.state_scale[1],
        ))


@dataclass(frozen=True)
class DirectNeuralPolicy:
    operator: NeuralFeedbackOperator
    parameter_bound: float = 8.0

    @property
    def parameter_count(self) -> int:
        return self.operator.parameter_count

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        return [(-self.parameter_bound, self.parameter_bound)] * self.parameter_count

    def evaluate(
        self, time: float, states: np.ndarray, parameters: np.ndarray,
    ) -> PolicyEvaluation:
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError("states must have shape (N,2)")
        inputs = self.operator.normalized_inputs(np.full(len(states), float(time)), states)
        W1, b1, w2, b2 = self.operator.unpack(parameters)
        hidden = np.tanh(inputs @ W1.T + b1)
        hidden_slope = 1.0 - hidden**2
        scale = float(self.operator.control_scale)
        controls = scale * (hidden @ w2 + b2)

        state_weights = W1[:, 1:] / self.operator.state_scale[None, :]
        control_state_jacobian = scale * (
            (hidden_slope * w2[None, :]) @ state_weights
        )

        count = len(states)
        width = self.operator.hidden_width
        parameter_jacobian = np.empty((count, self.parameter_count))
        hidden_factor = scale * hidden_slope * w2[None, :]
        cursor = 0
        parameter_jacobian[:, cursor:cursor + 3 * width] = (
            hidden_factor[:, :, None] * inputs[:, None, :]
        ).reshape(count, 3 * width)
        cursor += 3 * width
        parameter_jacobian[:, cursor:cursor + width] = hidden_factor
        cursor += width
        parameter_jacobian[:, cursor:cursor + width] = scale * hidden
        parameter_jacobian[:, -1] = scale
        return PolicyEvaluation(
            gains=-control_state_jacobian,
            controls=controls,
            control_state_jacobian=control_state_jacobian,
            control_parameter_jacobian=parameter_jacobian,
        )
