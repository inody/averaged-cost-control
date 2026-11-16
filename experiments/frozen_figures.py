"""Replay all six manuscript figures from committed frozen numerical inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.initial_state_scaling.render_recorded import render as render_runtime
from experiments.joint_uncertainty.run import JointPopulation, plot_trajectory
from experiments.uncertainty_sensitivity.render import render as render_sensitivity

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "frozen"
MANIFEST = FROZEN / "manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "frozen-figures"

EXPECTED_FIGURES = (
    "nonlinear-vdp-direct-cost-neural-feedback-full-results.pdf",
    "initial-state-runtime-scaling.pdf",
    "nonlinear-stuart-landau-direct-cost-neural-feedback-full-results.pdf",
    "nonlinear-vdp-static-linear-feedback.pdf",
    "nonlinear-stuart-landau-static-linear-feedback.pdf",
    "uncertainty-variance-cost-sensitivity.pdf",
)
TRAJECTORY_SPECS = {
    "nonlinear-vdp-direct-cost-neural-feedback-full-results.pdf": "vdp_neural",
    "nonlinear-stuart-landau-direct-cost-neural-feedback-full-results.pdf": "stuart_neural",
    "nonlinear-vdp-static-linear-feedback.pdf": "vdp_static",
    "nonlinear-stuart-landau-static-linear-feedback.pdf": "stuart_static",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported frozen-data manifest schema")
    figures = manifest.get("figures")
    if not isinstance(figures, list):
        raise RuntimeError("frozen-data manifest has no figure inventory")
    names = tuple(item.get("filename") for item in figures)
    if names != EXPECTED_FIGURES:
        raise RuntimeError(f"frozen figure inventory mismatch: {names}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("frozen-data manifest has no input inventory")
    required_inputs = {
        "data/frozen/trajectory_figures.npz", "data/runtime_scaling.json",
        "data/reference_sensitivity.json", "data/reference_results.json",
    }
    paths = [record.get("path") for record in files]
    if len(paths) != len(required_inputs) or set(paths) != required_inputs:
        raise RuntimeError("frozen-data input inventory mismatch")
    if [item.get("manuscript_number") for item in figures] != list(range(1, 7)):
        raise RuntimeError("manuscript figure numbering mismatch")
    for record in files:
        relative = record.get("path")
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"invalid frozen input path: {relative!r}")
        source = ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"missing frozen input: {relative}")
        if source.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"frozen input size mismatch: {relative}")
        if sha256(source) != record.get("sha256"):
            raise RuntimeError(f"frozen input checksum mismatch: {relative}")
    return manifest


def _finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise RuntimeError(f"nonfinite frozen trajectory array: {name}")
    return array


def load_trajectory_data(path: Path = FROZEN / "trajectory_figures.npz") -> dict[str, dict]:
    result: dict[str, dict] = {}
    with np.load(path, allow_pickle=False) as archive:
        expected_keys: set[str] = set()
        for prefix in TRAJECTORY_SPECS.values():
            keys = {
                "time", "omega", "initial_states", "weights",
                "averaged_states", "averaged_controls",
                "nominal_states", "nominal_controls",
            }
            expected_keys.update(f"{prefix}_{key}" for key in keys)
            arrays = {key: _finite(f"{prefix}_{key}", archive[f"{prefix}_{key}"]) for key in keys}
            count = 32
            steps = 201 if prefix.endswith("neural") else 401
            if arrays["time"].shape != (steps,):
                raise RuntimeError(f"invalid time-grid shape: {prefix}")
            if not np.all(np.diff(arrays["time"]) > 0) or arrays["time"][0] != 0.0 or arrays["time"][-1] != 5.0:
                raise RuntimeError(f"invalid time grid: {prefix}")
            if arrays["omega"].shape != (count,) or arrays["initial_states"].shape != (count, 2):
                raise RuntimeError(f"invalid display population shape: {prefix}")
            weights = arrays["weights"]
            if weights.shape != (count,) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=2e-13):
                raise RuntimeError(f"invalid display weights: {prefix}")
            for method in ("averaged", "nominal"):
                if arrays[f"{method}_states"].shape != (count, steps, 2):
                    raise RuntimeError(f"invalid state-array shape: {prefix}/{method}")
                if arrays[f"{method}_controls"].shape != (count, steps):
                    raise RuntimeError(f"invalid control-array shape: {prefix}/{method}")
                if not np.allclose(arrays[f"{method}_states"][:, 0], arrays["initial_states"], rtol=0.0, atol=2e-13):
                    raise RuntimeError(f"initial states do not match trajectories: {prefix}/{method}")
            result[prefix] = arrays
        if set(archive.files) != expected_keys:
            raise RuntimeError("trajectory archive key inventory mismatch")
    return result


def render_all(output: Path = DEFAULT_OUTPUT) -> tuple[Path, ...]:
    load_manifest()
    trajectories = load_trajectory_data()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)

    for filename, prefix in TRAJECTORY_SPECS.items():
        arrays = trajectories[prefix]
        population = JointPopulation(
            omega=arrays["omega"],
            initial_states=arrays["initial_states"],
            weights=arrays["weights"],
        )
        payload = {
            method: {
                "time": arrays["time"],
                "states": arrays[f"{method}_states"],
                "controls": arrays[f"{method}_controls"],
            }
            for method in ("averaged", "nominal")
        }
        plot_trajectory(output / filename, population, payload, figure_size=(10.5, 5.15))

    render_runtime(ROOT / "data" / "runtime_scaling.json", output / "initial-state-runtime-scaling.pdf")
    render_sensitivity(
        ROOT / "data" / "reference_sensitivity.json",
        output / "uncertainty-variance-cost-sensitivity.pdf",
    )
    paths = tuple(output / name for name in EXPECTED_FIGURES)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"figure replay did not create: {missing}")
    print(f"Replayed six manuscript figures in {output}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args()
    if args.verify_manifest:
        load_manifest()
        load_trajectory_data()
        print("Frozen-data manifest and trajectory arrays passed validation.")
        return
    render_all(args.output)


if __name__ == "__main__":
    main()
