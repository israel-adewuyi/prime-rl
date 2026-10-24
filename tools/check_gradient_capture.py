"""Validate rollout-gradient metrics and optional checkpoint gradient/update tensors."""

import argparse
import json
import math
from pathlib import Path

import torch


def check_statistics(path: Path, expected_steps: list[int]) -> list[dict]:
    reports = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("producer") != "trainer" or "snr/count" not in row:
                continue
            step = row["step"]
            if not isinstance(step, int) or isinstance(step, bool) or step < 1:
                raise ValueError(f"Invalid statistics step: {step!r}")
            if step in reports:
                raise ValueError(f"Multiple statistics records for step {step}")
            count = row["snr/count"]
            if (
                not isinstance(count, (int, float))
                or isinstance(count, bool)
                or not math.isfinite(count)
                or count < 1
                or int(count) != count
            ):
                raise ValueError(f"Invalid rollout count at step {step}: {count!r}")
            mean_norm = row["snr/mean_norm"]
            if not math.isfinite(mean_norm) or mean_norm < 0:
                raise ValueError(f"Invalid mean gradient norm at step {step}")
            variance = row.get("snr/variance")
            value = row.get("snr/value")
            if count < 2:
                if "snr/variance" in row or "snr/value" in row:
                    raise ValueError(f"Variance and SNR must be omitted for count < 2 at step {step}")
            else:
                if variance is None or not math.isfinite(variance) or variance < 0:
                    raise ValueError(f"Invalid sample variance at step {step}")
                if variance == 0:
                    if "snr/value" in row:
                        raise ValueError(f"SNR must be omitted for zero variance at step {step}")
                elif (
                    value is None
                    or not math.isfinite(value)
                    or value < 0
                    or not math.isclose(value, mean_norm**2 / variance, rel_tol=1e-6, abs_tol=1e-12)
                ):
                    raise ValueError(f"Power SNR does not match mean_norm^2 / variance at step {step}")
            reports[step] = {
                "step": step,
                **{key: value for key, value in row.items() if key.startswith("snr/")},
            }
    if expected_steps and set(reports) != set(expected_steps):
        raise ValueError(f"Expected statistics steps {sorted(set(expected_steps))}, found {sorted(reports)}")
    return [reports[step] for step in sorted(reports)]


def check_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported gradient/update schema in {path}")
    if path.parent.parent.name != f"step_{payload['step']}":
        raise ValueError(f"Checkpoint step mismatch in {path}")
    if payload["gradients_before_clipping"] is not True or payload["update_definition"] != "after_minus_before":
        raise ValueError(f"Unexpected gradient/update semantics in {path}")
    metadata = payload["parameter_metadata"]
    if not metadata:
        raise ValueError(f"No parameter metadata in {path}")
    norms = {}
    promoted_dtypes = {"torch.float16": torch.float32, "torch.bfloat16": torch.float32}
    promoted_dtypes.update({"torch.float32": torch.float32, "torch.float64": torch.float64})
    for collection in ("batch_gradients", "applied_updates"):
        tensors = payload[collection]
        if tensors.keys() != metadata.keys():
            raise ValueError(f"Parameter names differ between {collection} and metadata in {path}")
        squared_norm = 0.0
        for name, tensor in tensors.items():
            dtype_key = "gradient_dtype" if collection == "batch_gradients" else "parameter_dtype"
            source_dtype = metadata[name][dtype_key]
            if tensor is None and collection == "batch_gradients" and source_dtype is None:
                continue
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != promoted_dtypes.get(source_dtype):
                raise ValueError(f"Invalid tensor dtype: {path}: {collection}: {name}")
            if list(tensor.shape) != metadata[name]["shape"] or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid tensor shape or non-finite values: {path}: {collection}: {name}")
            squared_norm += torch.linalg.vector_norm(tensor, dtype=torch.float64).item() ** 2
        norms[collection] = math.sqrt(squared_norm)
    return {"step": payload["step"], "path": str(path), "parameter_count": len(metadata), "norms": norms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run directory")
    parser.add_argument("steps", type=int, nargs="*", help="Expected statistics steps, e.g. 1 2")
    args = parser.parse_args()
    config_path = args.run_dir / "configs" / "latest" / "resolved" / "trainer.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    monitor = config.get("monitors", {}).get("file")
    metric_path = Path(monitor.get("path", "metrics.jsonl")) if monitor is not None else Path("metrics.jsonl")
    metric_path = args.run_dir / "monitors" / "file" / metric_path
    statistics = check_statistics(metric_path, args.steps) if metric_path.is_file() else []
    if not statistics and (args.steps or config.get("gradient_capture") is not None):
        raise ValueError(f"No trainer rollout-gradient statistics in {metric_path}")

    checkpoint_config = config.get("ckpt") or {}
    checkpoint_root = Path(checkpoint_config.get("output_dir") or args.run_dir) / "checkpoints"
    checkpoint_dirs = sorted(checkpoint_root.glob("step_*/trainer"), key=lambda path: int(path.parent.name[5:]))
    required = checkpoint_config.get("save_gradients_and_updates", False)
    checkpoints = []
    for directory in checkpoint_dirs:
        path = directory / "gradient_update.pt"
        if required or path.exists():
            checkpoints.append(check_checkpoint(path))
    if required and not checkpoints:
        raise FileNotFoundError(f"No trainer checkpoint gradient/update tensors found in {checkpoint_root}")
    if not statistics and not checkpoints:
        raise FileNotFoundError(f"No rollout statistics or checkpoint gradient/update tensors in {args.run_dir}")
    print(json.dumps({"run_dir": str(args.run_dir), "statistics": statistics, "checkpoints": checkpoints}, indent=2))


if __name__ == "__main__":
    main()
