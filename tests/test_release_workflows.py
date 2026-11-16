import json
from pathlib import Path

import pytest

from experiments.frozen_figures import EXPECTED_FIGURES, load_manifest, load_trajectory_data
from experiments.paper_correspondence import table_rows
from experiments.verify_release import verify_sensitivity

ROOT = Path(__file__).resolve().parents[1]


def test_ci_installs_pdf_and_tex_png_rendering_dependencies():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    install = next(line for line in workflow.splitlines() if "sudo apt-get install" in line)
    assert {"dvipng", "ghostscript", "poppler-utils", "texlive-latex-base", "texlive-latex-extra", "texlive-fonts-recommended", "cm-super"} <= set(install.split())
    assert 'for tool in latex dvipng gs pdfinfo; do command -v "$tool"; done' in workflow


def test_frozen_manifest_covers_exactly_six_manuscript_figures():
    manifest = load_manifest()
    assert tuple(record["filename"] for record in manifest["figures"]) == EXPECTED_FIGURES
    assert [record["manuscript_number"] for record in manifest["figures"]] == list(range(1, 7))
    assert set(load_trajectory_data()) == {"vdp_neural", "stuart_neural", "vdp_static", "stuart_static"}


def test_reference_sensitivity_passes_mechanical_checks():
    verify_sensitivity(ROOT / "data" / "reference_sensitivity.json")


def test_sensitivity_checks_reject_inconsistent_reduction(tmp_path):
    payload = json.loads((ROOT / "data" / "reference_sensitivity.json").read_text())
    payload["records"][0]["neural"]["reduction_percent"] += 1.0
    path = tmp_path / "sensitivity.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="reduction mismatch"):
        verify_sensitivity(path)


def test_table1_display_precision_matches_declared_values():
    rows = table_rows(ROOT / "data" / "reference_results.json")
    assert [row.split(" & ")[-3:] for row in rows] == [
        ["1.379", "1.765", "21.9\\%\\\\"],
        ["1.563", "1.656", "5.62\\%\\\\"],
        ["0.2845", "0.3953", "28.0\\%\\\\"],
        ["0.2952", "0.3878", "23.9\\%\\\\"],
    ]
