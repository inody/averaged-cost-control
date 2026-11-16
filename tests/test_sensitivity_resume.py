import json
from pathlib import Path

import numpy as np
import pytest

from experiments.uncertainty_sensitivity import run


def contract_inputs(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"complete"}\n')
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    warm = tmp_path / "warm.json"
    warm.write_text('{"cases":{}}\n')
    return summary, source, warm


def test_campaign_contract_changes_with_code_config_and_inputs(tmp_path, monkeypatch):
    summary, source, warm = contract_inputs(tmp_path)
    base = run.build_campaign_contract(summary, source_files=(source,), warm_starts=warm)

    source.write_text("VALUE = 2\n")
    changed_code = run.build_campaign_contract(summary, source_files=(source,), warm_starts=warm)
    assert changed_code["contract_sha256"] != base["contract_sha256"]

    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(run, "SCALES", run.SCALES + (2.25,))
    changed_config = run.build_campaign_contract(summary, source_files=(source,), warm_starts=warm)
    assert changed_config["contract_sha256"] != base["contract_sha256"]

    monkeypatch.setattr(run, "SCALES", tuple(base["protocol"]["scales"]))
    summary.write_text('{"status":"changed"}\n')
    changed_input = run.build_campaign_contract(summary, source_files=(source,), warm_starts=warm)
    assert changed_input["contract_sha256"] != base["contract_sha256"]

    summary.write_text('{"status":"complete"}\n')
    warm.write_text('{"cases":{"changed":true}}\n')
    changed_warm_start = run.build_campaign_contract(summary, source_files=(source,), warm_starts=warm)
    assert changed_warm_start["contract_sha256"] != base["contract_sha256"]


def test_incompatible_contract_fails_before_configuration_overwrite(tmp_path):
    summary, source, warm = contract_inputs(tmp_path)
    contract = run.build_campaign_contract(summary, source_files=(source,), warm_starts=warm)
    output = tmp_path / "campaign"
    run.ensure_campaign_contract(output, contract)
    configuration = output / "configuration.json"
    configuration.write_text('{"sentinel":true}\n')

    source.write_text("VALUE = 2\n")
    incompatible = run.build_campaign_contract(summary, source_files=(source,), warm_starts=warm)
    with pytest.raises(RuntimeError, match="incompatible sensitivity contract"):
        run.ensure_campaign_contract(output, incompatible)
    assert configuration.read_text() == '{"sentinel":true}\n'


def reusable_record(input_contract: dict) -> dict:
    model = input_contract["model"]
    rules = run.build_rules(model, input_contract["sweep"], input_contract["scale"], run.case_data(run.MODEL_SPECS[model]["case"]))

    def final(cost):
        population = rules["final"]
        return {
            "status": "finite full inventory", "cost": cost,
            "per_particle_cost": np.full(population.size, cost).tolist(),
            "successful_subset_averaged": False,
            "verification": {
                "particle_count": population.size, "all_particles_retained": True,
                "all_costs_finite": True, "weighted_cost_recomputed": cost,
            },
        }

    return {
        "model": model, "sweep": input_contract["sweep"], "scale": input_contract["scale"],
        "law": {
            "mean_omega": run.MODEL_SPECS[model]["mean_omega"],
            "omega_half_width": rules["omega_half_width"], "omega_variance": rules["omega_variance"],
            "initial_radius": rules["x0_radius"], "initial_component_variance": rules["initial_component_variance"],
            "initial_center": rules["x0_center"].tolist(),
            "design_particle_count": rules["averaged_design"].size,
            "validation_particle_count": rules["averaged_validation"].size,
            "final_particle_count": rules["final"].size,
        },
        "status": "passed",
        "campaign_contract_sha256": input_contract["campaign_contract_sha256"],
        "point_input_contract": input_contract,
        "nominal_controller": {
            "neural_parameters_sha256": "c" * 64,
            "static_gain": [1.0, 2.0],
        },
        "neural": {
            "averaged_cost": 1.0,
            "nominal_cost": 2.0,
            "reduction_percent": 50.0,
            "optimization": {
                "selected_tight_stationarity": {"status": "finite", "projected_gradient_residual": 5e-5},
                "selected_parameters": [0.0] * 81,
                "stationarity_certified": True, "optimizer": {"terminated": True},
            },
            "averaged_final": final(1.0), "nominal_final": final(2.0),
        },
        "static": {
            "averaged_cost": 1.5,
            "nominal_cost": 2.0,
            "reduction_percent": 25.0,
            "averaged_design": {"stationarity_residual": 5e-9, "gain": [1.0, 2.0], "stationarity_certified": True},
            "averaged_final": final(1.5), "nominal_final": final(2.0),
        },
    }


def test_compatible_point_resumption_and_invalid_point_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "POINTS", tmp_path / "points")
    theta = np.zeros(81)
    gain = np.array([3.0, 4.0])
    campaign_sha = "a" * 64
    expected = run.point_input_contract(campaign_sha, "van-der-pol", "baseline", 1.0, theta, gain)
    path = run.point_path("van-der-pol", "baseline", 1.0)
    path.parent.mkdir(parents=True)
    saved = reusable_record(expected)
    path.write_text(json.dumps(saved))

    resumed = run.optimize_point(
        "van-der-pol", "baseline", 1.0, theta, gain, {}, {}, campaign_sha
    )
    assert resumed == saved

    saved["neural"]["optimization"]["selected_tight_stationarity"]["projected_gradient_residual"] = float("nan")
    path.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="invalid existing sensitivity point"):
        run.optimize_point(
            "van-der-pol", "baseline", 1.0, theta, gain, {}, {}, campaign_sha
        )


@pytest.mark.parametrize("mutation", (
    lambda record: record["neural"]["optimization"]["selected_parameters"].__setitem__(0, float("nan")),
    lambda record: record["static"]["averaged_design"].__setitem__("stationarity_certified", False),
    lambda record: record["neural"]["optimization"].__setitem__("selected_parameters", [0.0]),
    lambda record: record["static"]["averaged_final"].pop("per_particle_cost"),
    lambda record: record["neural"]["averaged_final"]["per_particle_cost"].__setitem__(0, 10.0),
    lambda record: record["law"].__setitem__("final_particle_count", 1),
    lambda record: record["law"].__setitem__("omega_variance", 0.0),
    lambda record: record["static"]["averaged_design"].__setitem__("stationarity_residual", -1.0),
))
def test_reusable_points_reject_corrupt_controllers_inventories_and_laws(mutation):
    expected = run.point_input_contract("a" * 64, "van-der-pol", "baseline", 1.0, np.zeros(81), np.ones(2))
    saved = reusable_record(expected)
    assert run.validate_reusable_point(saved, expected)
    mutation(saved)
    assert not run.validate_reusable_point(saved, expected)


def test_final_point_verification_checks_complete_records(tmp_path):
    from experiments.verify_release import verify_sensitivity_points

    campaign = {"schema_version": 1}
    campaign_hash = run.canonical_digest(campaign)
    run.write_json(tmp_path / "campaign_contract.json", {**campaign, "contract_sha256": campaign_hash})
    points = tmp_path / "points"
    for model in run.MODEL_SPECS:
        identities = [("baseline", 1.0)] + [
            (sweep, scale) for sweep in ("model variance", "initial-state variance")
            for scale in run.SCALES if scale != 1.0
        ]
        for sweep, scale in identities:
            expected = run.point_input_contract(campaign_hash, model, sweep, scale, np.zeros(81), np.ones(2))
            run.write_json(points / model / f"{run.point_name(sweep, scale)}.json", reusable_record(expected))
    verify_sensitivity_points(points)
    first = points / "van-der-pol" / "baseline.json"
    record = json.loads(first.read_text())
    record["neural"]["averaged_final"].pop("per_particle_cost")
    run.write_json(first, record)
    with pytest.raises(RuntimeError, match="invalid sensitivity point"):
        verify_sensitivity_points(points)


def test_point_input_contract_binds_continuation_state():
    first = run.point_input_contract("b" * 64, "van-der-pol", "model variance", 0.5, np.array([1.0]), np.array([2.0]))
    changed_theta = run.point_input_contract("b" * 64, "van-der-pol", "model variance", 0.5, np.array([1.1]), np.array([2.0]))
    changed_gain = run.point_input_contract("b" * 64, "van-der-pol", "model variance", 0.5, np.array([1.0]), np.array([2.1]))
    assert first["point_input_sha256"] != changed_theta["point_input_sha256"]
    assert first["point_input_sha256"] != changed_gain["point_input_sha256"]
