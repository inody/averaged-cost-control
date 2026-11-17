import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_has_fast_and_full_reproduction_entry_points():
    readme = (ROOT / "README.md").read_text()
    makefile = (ROOT / "Makefile").read_text()
    for command in ("make figures", "make verify", "make reproduce", "make recompute", "make sensitivity", "make benchmark"):
        assert command in readme
    for target in ("figures:", "reproduce:", "recompute:", "sensitivity:", "benchmark:", "verify:"):
        assert target in makefile
    assert "experiments.frozen_figures" in makefile
    assert "joint_uncertainty.run" in makefile
    assert "uncertainty_sensitivity.run" in makefile
    assert "verify_release" in makefile


def test_no_development_directories_are_present():
    for relative in ("docs", "slidev", "runs", "artifacts", "private", "latex"):
        assert not (ROOT / relative).exists()


def test_expected_release_data_are_present():
    for relative in (
        "data/warm_starts.json",
        "data/runtime_scaling.json",
        "data/reference_results.json",
        "data/reference_sensitivity.json",
        "data/frozen/trajectory_figures.npz",
        "data/frozen/manifest.json",
        "CITATION.cff",
        ".github/workflows/tests.yml",
    ):
        assert (ROOT / relative).is_file()


def test_recorded_runtime_scaling_matches_the_current_laws():
    timing = json.loads((ROOT / "data" / "runtime_scaling.json").read_text())
    expected = {
        "van-der-pol": (-0.000644102427335143, 0.6734120680629152),
        "stuart-landau": (-0.0015781332285288588, 0.664533703482434),
    }
    for case, (linear_exponent, nonlinear_exponent) in expected.items():
        assert [row["M"] for row in timing["cases"][case]["rows"]] == [1, 4, 16, 64, 128]
        scaling = timing["cases"][case]["scaling"]
        assert scaling["linear_moment_total_exponent"] == linear_exponent
        assert scaling["nonlinear_particle_exponent"] == nonlinear_exponent


def test_reference_sensitivity_is_complete_and_certified():
    summary = json.loads((ROOT / "data" / "reference_sensitivity.json").read_text())
    assert summary["status"] == "passed"
    assert summary["unique_law_count"] == 34
    assert len(summary["records"]) == 36
    assert {record["model"] for record in summary["records"]} == {
        "van-der-pol", "stuart-landau",
    }
    assert {record["sweep"] for record in summary["records"]} == {
        "model variance", "initial-state variance",
    }
    assert max(record["neural"]["stationarity_residual"] for record in summary["records"]) <= 1e-4
    assert max(record["static"]["stationarity_residual"] for record in summary["records"]) <= 1e-8


def test_published_paper_links_and_preferred_citation_are_present():
    readme = (ROOT / "README.md").read_text()
    citation = (ROOT / "CITATION.cff").read_text()
    paper = citation.split("\npreferred-citation:\n", 1)[1]
    assert "https://arxiv.org/abs/2611.19019" in readme
    assert "https://doi.org/10.48550/arXiv.2611.19019" in readme
    assert '  type: article\n' in paper
    assert '  title: "Averaged-Cost Control with Shared Feedback under Model Uncertainty"' in paper
    assert "  year: 2026\n" in paper
    assert "  journal: arXiv\n" in paper
    assert '  doi: "10.48550/arXiv.2611.19019"' in paper
    assert '  url: "https://arxiv.org/abs/2611.19019"' in paper
    for family, given in (
        ("Bettiol", "Piernicola"),
        ("Inoue", "Daisuke"),
        ("Gómez-Aedo", "Matías"),
        ("Kalise", "Dante"),
        ("Vinter", "Richard B."),
    ):
        assert f"    - family-names: {family}\n      given-names: {given}\n" in paper


def test_release_license_is_declared():
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'license = "MIT"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
