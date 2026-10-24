"""Validate saved prompt-gradient files and print a compact report for a GPU smoke run."""

import argparse
import json
import math
from pathlib import Path

import torch


def check_step(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError(f"Unsupported gradient manifest version in {directory}")
    if directory.name != f"step_{manifest['step']}":
        raise ValueError(f"Step mismatch in {directory}")
    if not manifest["groups"]:
        raise ValueError(f"No prompt gradients captured in {directory}")
    groups = []
    seen = set()
    for entry in manifest["groups"]:
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("Gradient file names must not contain directories")
        payload = torch.load(directory / filename, map_location="cpu", weights_only=True)
        if payload["schema_version"] != 1 or payload["step"] != manifest["step"]:
            raise ValueError(f"Gradient metadata mismatch in {filename}")
        group_id = payload["group_id"]
        if group_id != entry["group_id"] or group_id in seen:
            raise ValueError(f"Duplicate or mismatched group ID in {filename}")
        seen.add(group_id)
        if len(payload["sequences"]) != entry["sequence_count"]:
            raise ValueError(f"Sequence count mismatch in {filename}")
        if payload["normalizers"] != manifest["normalizers"]:
            raise ValueError(f"Normalization mismatch in {filename}")
        if payload["gradients"].keys() != payload["parameter_metadata"].keys():
            raise ValueError(f"Gradient parameter names mismatch in {filename}")
        squared_norm = 0.0
        unused = 0
        for name, gradient in payload["gradients"].items():
            if gradient is None:
                unused += 1
                continue
            if gradient.dtype != torch.float32 or list(gradient.shape) != payload["parameter_metadata"][name]["shape"]:
                raise ValueError(f"Gradient shape/dtype mismatch: {filename}: {name}")
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError(f"Non-finite gradient: {filename}: {name}")
            squared_norm += torch.linalg.vector_norm(gradient, dtype=torch.float64).item() ** 2
        groups.append(
            {
                "group_id": group_id,
                "sequence_count": entry["sequence_count"],
                "gradient_norm": math.sqrt(squared_norm),
                "unused_parameters": unused,
            }
        )
    return {
        "step": manifest["step"],
        "microbatches": manifest["microbatches"],
        "batch_gradient_verified": manifest["verification"]["checked"],
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Run directory, prompt_gradients directory, or step directory")
    parser.add_argument("steps", type=int, nargs="*", help="Expected capture steps, e.g. 1 2")
    args = parser.parse_args()
    root = args.directory
    if (root / "prompt_gradients").is_dir():
        root = root / "prompt_gradients"
    directories = [root] if root.name.startswith("step_") else sorted(root.glob("step_*"))
    if not directories:
        raise FileNotFoundError(f"No captured steps found in {root}")
    reports = [check_step(directory) for directory in directories]
    actual_steps = {report["step"] for report in reports}
    if args.steps and actual_steps != set(args.steps):
        raise ValueError(f"Expected steps {sorted(set(args.steps))}, found {sorted(actual_steps)}")
    print(json.dumps({"directory": str(root), "steps": reports}, indent=2))


if __name__ == "__main__":
    main()
