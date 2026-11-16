"""Generate Table 1 values and optionally compare them with a manuscript source."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data" / "reference_results.json"

ROW_SPECS = (
    ("Nonlinear Van der Pol", "Nonlinear", "Time-dependent neural", "nonlinear-vdp", "neural", ".4g", ".4g", ".1f"),
    ("Nonlinear Van der Pol", "Linearized", "Static linear", "linear-vdp", "static", ".4g", ".4g", ".2f"),
    ("Nonlinear Stuart--Landau", "Nonlinear", "Time-dependent neural", "nonlinear-stuart-landau", "neural", ".4g", ".4g", ".1f"),
    ("Nonlinear Stuart--Landau", "Linearized", "Static linear", "linear-stuart-landau", "static", ".4g", ".4g", ".1f"),
)


def _costs(payload: dict, key: str, family: str) -> tuple[float, float]:
    if "cases" in payload:
        if family == "neural":
            methods = payload["cases"][key]["policies"]["direct_neural"]
            return tuple(float(methods[name]["final_evaluation"]["cost"]) for name in ("averaged", "nominal"))
        methods = payload["linear_cases"][key]["methods"]
        return tuple(float(methods[name]["nonlinear_evaluation"]["cost"]) for name in ("averaged", "nominal"))
    section = "neural" if family == "neural" else "static_linear"
    methods = payload[section][key]
    return tuple(float(methods[name]["nonlinear_evaluation_cost"]) for name in ("averaged", "nominal"))


def table_rows(results_path: Path) -> list[str]:
    payload = json.loads(results_path.read_text())
    rows = []
    for system, dynamics, feedback, key, family, avg_fmt, nom_fmt, reduction_fmt in ROW_SPECS:
        averaged, nominal = _costs(payload, key, family)
        if not np.isfinite([averaged, nominal]).all() or nominal <= 0:
            raise RuntimeError(f"invalid Table 1 costs: {key}/{family}")
        reduction = 100.0 * (nominal - averaged) / nominal
        rows.append(
            f"{system} & {dynamics} & {feedback} & "
            f"{averaged:{avg_fmt}} & {nominal:{nom_fmt}} & {reduction:{reduction_fmt}}\\%\\\\"
        )
    return rows


def normalize(row: str) -> str:
    return " ".join(row.replace("\\multirow{2}{*}{", "").replace("}", "").split())


def manuscript_rows(path: Path) -> list[str]:
    text = path.read_text()
    match = re.search(r"\\label\{tab:evaluation-costs\}.*?\\begin\{tabular\}.*?\\midrule(.*?)\\bottomrule", text, re.S)
    if not match:
        raise RuntimeError("Table 1 was not found in the manuscript")
    rows = []
    current_system = None
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("\\multirow"):
            system_match = re.match(r"\\multirow\{2\}\{\*\}\{([^}]*)\}", line)
            if not system_match:
                raise RuntimeError(f"cannot parse manuscript Table 1 row: {line}")
            current_system = system_match.group(1)
            line = line[system_match.end():].strip()
        if line.startswith("&"):
            if current_system is None:
                raise RuntimeError("Table 1 continuation row has no system")
            line = current_system + " " + line
        if "&" in line and line.endswith("\\\\"):
            rows.append(line)
    if len(rows) != 4:
        raise RuntimeError(f"expected four Table 1 rows, found {len(rows)}")
    return rows


def verify(results_path: Path, manuscript: Path | None = None) -> list[str]:
    generated = table_rows(results_path)
    print("\n".join(generated))
    if manuscript is not None:
        observed = manuscript_rows(manuscript)
        mismatches = [
            (expected, actual)
            for expected, actual in zip(generated, observed)
            if normalize(expected) != normalize(actual)
        ]
        if mismatches:
            details = "\n".join(f"generated: {a}\nmanuscript: {b}" for a, b in mismatches)
            raise RuntimeError("Table 1 mismatch:\n" + details)
        print(f"Table 1 matches {manuscript}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manuscript", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = verify(args.results, args.manuscript)
    if args.output:
        if args.output.exists():
            raise RuntimeError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
