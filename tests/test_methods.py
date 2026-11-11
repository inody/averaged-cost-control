import json
from pathlib import Path

import numpy as np
import pytest

from experiments.joint_uncertainty.run import (
    StaticLinearPolicy,
    initial_moment,
    linear_data,
    linear_moment_evaluation,
    linear_objective_gradient,
    rules,
)
from experiments.uncertainty_sensitivity.run import build_rules
from experiments.linear_moment import INITIAL_GAIN
from experiments.models import case_data
from experiments.neural_policy import DirectNeuralPolicy, NeuralFeedbackOperator
from experiments.neural_adjoint import (
    JointPopulation,
    objective_cost,
    objective_gradient,
    projected_gradient_residual,
)

ROOT = Path(__file__).resolve().parents[1]


def test_joint_laws_match_the_paper():
    vdp = rules("nonlinear-vdp", case_data("nonlinear-vdp"))
    assert vdp["averaged_design"].size == 32 * 8
    assert np.isclose(vdp["averaged_design"].weights.sum(), 1.0)
    assert vdp["omega_metadata"] == {
        "family": "Uniform", "lower": -2.0, "upper": 4.0, "mean": 1.0,
    }
    assert np.all((-2.0 <= vdp["omega_design"]) & (vdp["omega_design"] <= 4.0))
    assert np.all(np.abs(vdp["x0_design"] - np.array([1.0, 1.0])) <= 0.2)

    stuart = rules("nonlinear-stuart-landau", case_data("nonlinear-stuart-landau"))
    assert stuart["omega_metadata"] == {
        "family": "Uniform", "lower": -2.1, "upper": 0.8, "mean": -0.65,
    }
    np.testing.assert_allclose(stuart["nominal"].omega, [-0.65])
    assert np.all(np.abs(stuart["x0_design"] - np.array([0.5, 0.5])) <= 0.2)


def test_sensitivity_rules_keep_the_mean_pair_fixed():
    for model, case, mean in (
        ("van-der-pol", "nonlinear-vdp", 1.0),
        ("stuart-landau", "nonlinear-stuart-landau", -0.65),
    ):
        data = case_data(case)
        for sweep in ("model variance", "initial-state variance"):
            for scale in (0.0, 2.0):
                study = build_rules(model, sweep, scale, data)
                assert np.isclose(study["omega_design_weights"] @ study["omega_design"], mean)
                np.testing.assert_allclose(
                    study["x0_design_weights"] @ study["x0_design"], data["initial"], atol=2e-16
                )
                assert np.isclose(study["averaged_design"].weights.sum(), 1.0)
                assert np.isclose(study["final"].weights.sum(), 1.0)


def test_static_linear_policy_derivatives():
    policy = StaticLinearPolicy()
    states = np.array([[1.0, -2.0], [0.5, 0.25]])
    gain = np.array([1.5, 3.0])
    evaluation = policy.evaluate(0.3, states, gain)
    np.testing.assert_allclose(evaluation.controls, -(states @ gain))
    np.testing.assert_allclose(evaluation.control_parameter_jacobian, -states)
    np.testing.assert_allclose(evaluation.control_state_jacobian, -np.tile(gain, (2, 1)))


def test_uniform_initial_second_moment():
    center = np.array([0.5, 0.5])
    expected = np.outer(center, center) + (0.2**2 / 3.0) * np.eye(2)
    np.testing.assert_allclose(initial_moment(center, 0.2, True), expected)
    np.testing.assert_allclose(initial_moment(center, 0.2, False), np.outer(center, center))


def test_linear_moment_gradient_matches_finite_difference():
    data = linear_data("linear-vdp")
    gain = np.array([1.0, 4.0])
    moment0 = initial_moment(np.array([1.0, 1.0]), 0.1, True)
    parameters = np.array([-0.5, 1.5])
    weights = np.array([0.4, 0.6])
    _, gradient, _ = linear_objective_gradient(
        data, gain, moment0, parameters, weights, divisor=160.0
    )
    step = 1e-5
    finite = np.zeros(2)
    for index in range(2):
        direction = np.eye(2)[index] * step
        plus = linear_objective_gradient(
            data, gain + direction, moment0, parameters, weights, divisor=160.0
        )[0]
        minus = linear_objective_gradient(
            data, gain - direction, moment0, parameters, weights, divisor=160.0
        )[0]
        finite[index] = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite, rtol=3e-7, atol=3e-7)


def test_anderson_moore_direction_is_descent():
    data = linear_data("linear-vdp")
    result = linear_moment_evaluation(
        data,
        INITIAL_GAIN,
        np.outer(data["initial"], data["initial"]),
        np.array([data["nominal_parameter"]]),
        np.ones(1),
    )
    direction = result["fixed_point_target"] - INITIAL_GAIN
    assert result["gradient"] @ direction < 0.0


def test_policy_derivatives_and_adjoint():
    operator = NeuralFeedbackOperator(
        horizon=5.0, state_scale=np.array([1.0, 1.0]),
        control_scale=1.0, hidden_width=2,
    )
    policy = DirectNeuralPolicy(operator)
    parameters = operator.initial_parameters(22)
    states = np.array([[0.4, -0.2], [1.1, 0.7]])
    evaluation = policy.evaluate(1.7, states, parameters)
    step = 1e-6
    shifted = states.copy(); shifted[:, 0] += step
    plus = policy.evaluate(1.7, shifted, parameters).controls
    shifted[:, 0] -= 2.0 * step
    minus = policy.evaluate(1.7, shifted, parameters).controls
    np.testing.assert_allclose(
        evaluation.control_state_jacobian[:, 0], (plus - minus) / (2.0 * step),
        rtol=0.0, atol=2e-9,
    )

    data = case_data("nonlinear-stuart-landau")
    population = JointPopulation(
        omega=np.array([-0.2, 0.15]),
        initial_states=np.array([[0.45, 0.55], [0.6, 0.4]]),
        weights=np.array([0.4, 0.6]),
    )
    _, gradient = objective_gradient(
        data, policy, parameters, population,
        rtol=2e-11, atol=2e-13, max_step_divisor=500.0,
    )
    for index in (0, 3, policy.parameter_count - 1):
        direction = np.zeros(policy.parameter_count); direction[index] = 1e-5
        plus = objective_cost(
            data, policy, parameters + direction, population,
            rtol=2e-11, atol=2e-13, max_step_divisor=500.0,
        )
        minus = objective_cost(
            data, policy, parameters - direction, population,
            rtol=2e-11, atol=2e-13, max_step_divisor=500.0,
        )
        assert gradient[index] == pytest.approx((plus - minus) / 2e-5, abs=2e-7)


def test_projected_residual_and_warm_start_inventory():
    assert projected_gradient_residual(
        np.array([0.0, -8.0, 8.0]), np.array([4e-5, 1.0, -1.0]),
        [(-8.0, 8.0)] * 3,
    ) == pytest.approx(4e-5)
    warm = json.loads((ROOT / "data" / "warm_starts.json").read_text())
    assert set(warm["cases"]) == {"nonlinear-vdp", "nonlinear-stuart-landau"}
    for methods in warm["cases"].values():
        assert set(methods) == {"averaged", "nominal"}
        for record in methods.values():
            assert len(record["parameters"]) == 81
            assert record["operator"]["hidden_width"] == 16
