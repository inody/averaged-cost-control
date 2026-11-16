import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.verify_results import FIGURE_NAMES, REFERENCE, verify


def population(size: int) -> dict:
    return {
        "size": size,
        "omega": np.zeros(size).tolist(),
        "initial_states": np.zeros((size, 2)).tolist(),
        "weights": np.full(size, 1.0 / size).tolist(),
    }


def evaluation(cost: float, size: int, *, include_costs: bool = True) -> dict:
    record = {
        "status": "finite full inventory",
        "cost": cost,
        "nfev": 1,
        "successful_subset_averaged": False,
        "verification": {
            "particle_count": size,
            "all_particles_retained": True,
            "all_costs_finite": True,
            "weighted_cost_recomputed": cost,
        },
    }
    if include_costs:
        record["per_particle_cost"] = np.full(size, cost).tolist()
    return record


def valid_summary() -> dict:
    reference = json.loads(REFERENCE.read_text())
    summary = {
        "schema_version": 2,
        "status": "complete",
        "protocol": {"all_or_nothing_full_inventory": True},
        "cases": {},
        "linear_cases": {},
    }
    for case, methods in reference["neural"].items():
        inventories = {
            "averaged_design": population(256),
            "averaged_validation": population(256),
            "nominal": population(1),
            "final": population(2048),
            "display": population(32),
        }
        policies = {}
        for method, expected in methods.items():
            cost = expected["nonlinear_evaluation_cost"]
            policies[method] = {
                "optimizer": {"terminated": True},
                "stationarity_certified": True,
                "selected_tight_stationarity": {
                    "status": "finite",
                    "projected_gradient_residual": expected["stationarity_residual"],
                },
                "selected_parameters": np.zeros(81).tolist(),
                "final_evaluation": evaluation(cost, 2048),
                "display_evaluation": evaluation(cost, 32, include_costs=False),
            }
        summary["cases"][case] = {
            "particle_inventories": inventories,
            "policies": {"direct_neural": policies},
        }
    nonlinear_case = {"linear-vdp": "nonlinear-vdp", "linear-stuart-landau": "nonlinear-stuart-landau"}
    for case, methods in reference["static_linear"].items():
        actual = {}
        for method, expected in methods.items():
            cost = expected["nonlinear_evaluation_cost"]
            actual[method] = {
                "evaluation_cost": expected["linearized_evaluation_cost"],
                "gain": expected["gain"],
                "stationarity_residual": expected["stationarity_residual"],
                "stationarity_certified": True,
                "gradient_check": {"maximum_absolute_difference": expected["gradient_check_error"]},
                "nonlinear_evaluation": evaluation(cost, 2048),
                "nonlinear_display_evaluation": evaluation(cost, 32, include_costs=False),
            }
        summary["linear_cases"][case] = {"methods": actual, "nonlinear_case": nonlinear_case[case]}
    return summary


def valid_figures(path: Path) -> None:
    path.mkdir()
    for name in FIGURE_NAMES:
        fig, axis = plt.subplots(figsize=(1, 1))
        axis.plot([0, 1], [0, 1])
        fig.savefig(path / name)
        plt.close(fig)


def run_verify(tmp_path: Path, summary: dict) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    figures = tmp_path / "figures"
    valid_figures(figures)
    verify(summary_path, figures)


def test_fail_closed_verifier_accepts_complete_record(tmp_path):
    run_verify(tmp_path, valid_summary())


@pytest.mark.parametrize(
    "mutation",
    (
        lambda summary: summary["cases"]["nonlinear-vdp"]["policies"]["direct_neural"]["averaged"]["selected_tight_stationarity"].__setitem__("projected_gradient_residual", float("nan")),
        lambda summary: summary["cases"]["nonlinear-vdp"]["policies"]["direct_neural"]["averaged"]["final_evaluation"].pop("per_particle_cost"),
        lambda summary: summary["cases"]["nonlinear-vdp"]["particle_inventories"]["final"]["weights"].__setitem__(0, 0.0),
        lambda summary: summary["linear_cases"]["linear-vdp"]["methods"]["averaged"]["gradient_check"].__setitem__("maximum_absolute_difference", float("nan")),
        lambda summary: summary["linear_cases"]["linear-vdp"]["methods"]["averaged"].__setitem__("stationarity_certified", False),
        lambda summary: summary["linear_cases"]["linear-vdp"]["methods"]["averaged"].__setitem__("stationarity_residual", -1.0),
        lambda summary: summary["linear_cases"]["linear-vdp"]["methods"]["averaged"]["gradient_check"].__setitem__("maximum_absolute_difference", -1.0),
        lambda summary: summary["cases"]["nonlinear-vdp"]["policies"]["direct_neural"]["averaged"]["display_evaluation"]["verification"].__setitem__("weighted_cost_recomputed", 9.0),
    ),
)
def test_fail_closed_verifier_rejects_adversarial_records(tmp_path, mutation):
    summary = valid_summary()
    mutation(summary)
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError)):
        run_verify(tmp_path, summary)


def test_fail_closed_verifier_rejects_zero_byte_pdf(tmp_path):
    summary = valid_summary()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    figures = tmp_path / "figures"
    valid_figures(figures)
    (figures / next(iter(FIGURE_NAMES))).write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty trajectory PDF"):
        verify(summary_path, figures)
