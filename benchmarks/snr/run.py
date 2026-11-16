"""Verify and time SNR accumulation from a bounded pool of GPU gradients."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prime_rl.trainer.rl.gradient_statistics import CUDAGradientStatistics, GradientStatistics  # noqa: E402

IMPLEMENTATIONS = {"cpu": GradientStatistics, "cuda": CUDAGradientStatistics}


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def shape(value: str) -> tuple[int, ...]:
    return tuple(positive_int(dimension) for dimension in value.split("x"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Result JSON file")
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--parameters", type=positive_int, help="Total elements; default: 1,000,000")
    layout.add_argument("--shapes", nargs="+", type=shape, help="Explicit tensor shapes, e.g. 1024x1024 1024")
    parser.add_argument("--tensor-numel", type=positive_int, help="Split --parameters into tensors; default: 1,000,000")
    parser.add_argument("--rollouts", type=positive_int, default=64)
    parser.add_argument("--pool-size", type=positive_int, default=4, help="Precomputed gradients to cycle through")
    parser.add_argument("--dtype", choices=["bfloat16", "float32", "float64"], default="float32")
    parser.add_argument("--device", default="cuda:0", help="GPU containing the input gradients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, help="PyTorch CPU threads; defaults to the environment setting")
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--repeats", type=positive_int, default=10)
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS))
    args = parser.parse_args()
    if args.shapes and args.tensor_numel is not None:
        parser.error("--tensor-numel applies only to --parameters")
    if len(set(args.implementations)) != len(args.implementations):
        parser.error("--implementations must not contain duplicates")
    if not args.shapes:
        parameters = args.parameters or 1_000_000
        chunk = args.tensor_numel or 1_000_000
        args.shapes = [(min(chunk, parameters - offset),) for offset in range(0, parameters, chunk)]
    args.pool_size = min(args.pool_size, args.rollouts)
    return args


def accumulate(factory, pool, rollouts: int) -> GradientStatistics:
    accumulator = factory()
    for index in range(rollouts):
        accumulator.update(pool[index % len(pool)])
    return accumulator


def verify(pool, rollouts: int, implementations: list[str]) -> dict:
    """Compare S, Q, n, and the metrics before collecting any timings."""
    rtol, atol = 1e-8, 1e-12
    reference = accumulate(GradientStatistics, pool, rollouts)
    expected_sum = dict(reference.gradient_sum)
    expected_q = float(reference.squared_norm_sum)
    expected_metrics = reference.finalize()
    results = {}
    for name in implementations:
        candidate = accumulate(IMPLEMENTATIONS[name], pool, rollouts)
        if candidate.count != rollouts or candidate.gradient_sum.keys() != expected_sum.keys():
            raise AssertionError(f"{name}: rollout count or parameter names differ from the CPU reference")
        max_sum_error = 0.0
        for key, expected in expected_sum.items():
            actual = candidate.gradient_sum[key]
            if expected is None:
                if actual is not None:
                    raise AssertionError(f"{name}: expected unused parameter {key}")
                continue
            if actual is None:
                raise AssertionError(f"{name}: missing gradient sum for {key}")
            actual = actual.cpu()
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=f"{name}: gradient sum {key}")
            max_sum_error = max(max_sum_error, (actual - expected).abs().max().item())
        actual_q = float(candidate.squared_norm_sum)
        if not math.isclose(actual_q, expected_q, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"{name}: squared norm sum {actual_q} != {expected_q}")
        metrics = candidate.finalize()
        if metrics.keys() != expected_metrics.keys():
            raise AssertionError(f"{name}: metric keys differ from the CPU reference")
        for key, expected in expected_metrics.items():
            if not math.isclose(metrics[key], expected, rel_tol=rtol, abs_tol=atol):
                raise AssertionError(f"{name}: {key} {metrics[key]} != {expected}")
        results[name] = {
            "passed": True,
            "rtol": rtol,
            "atol": atol,
            "max_abs_gradient_sum_error": max_sum_error,
            "abs_squared_norm_sum_error": abs(actual_q - expected_q),
            "abs_metric_errors": {key: abs(metrics[key] - value) for key, value in expected_metrics.items()},
            "metrics": metrics,
        }
    return results


def measure(factory, pool, rollouts: int, device: torch.device) -> dict:
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    metrics = accumulate(factory, pool, rollouts).finalize()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    return {
        "seconds": seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "extra_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline_allocated,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "metrics": metrics,
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA/ROCm inputs, including when measuring the CPU accumulator")
    torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    # CPU generation makes the input stream independent of the GPU RNG implementation.
    generator = torch.Generator().manual_seed(args.seed)
    pool = [
        {
            f"parameter_{index}": torch.randn(tensor_shape, generator=generator, dtype=dtype).to(device)
            for index, tensor_shape in enumerate(args.shapes)
        }
        for _ in range(args.pool_size)
    ]
    print(f"Verifying {args.implementations} on {args.rollouts} rollouts...", flush=True)
    verification = verify(pool, args.rollouts, args.implementations)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    for name in args.implementations:
        for _ in range(args.warmup):
            measure(IMPLEMENTATIONS[name], pool, args.rollouts, device)
    samples = {name: [] for name in args.implementations}
    for repeat in range(args.repeats):
        # Alternate ordering so one implementation does not always run first.
        names = args.implementations if repeat % 2 == 0 else list(reversed(args.implementations))
        for name in names:
            result = measure(IMPLEMENTATIONS[name], pool, args.rollouts, device)
            expected = verification[name]["metrics"]
            if result["metrics"].keys() != expected.keys() or any(
                not math.isclose(result["metrics"][key], value, rel_tol=1e-8, abs_tol=1e-12)
                for key, value in expected.items()
            ):
                raise AssertionError(f"{name}: repeated accumulation changed its metrics")
            samples[name].append(result)
            print(f"{name}: repeat {repeat + 1}/{args.repeats}, {result['seconds']:.6f} s", flush=True)
    summary = {}
    for name, rows in samples.items():
        timings = [row["seconds"] for row in rows]
        summary[name] = {
            "median_seconds": statistics.median(timings),
            "mean_seconds": statistics.mean(timings),
            "stdev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "min_seconds": min(timings),
            "max_seconds": max(timings),
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
            "extra_peak_allocated_bytes": max(row["extra_peak_allocated_bytes"] for row in rows),
        }
    properties = torch.cuda.get_device_properties(device)
    source_paths = [Path(__file__), REPO_ROOT / "src/prime_rl/trainer/rl/gradient_statistics.py"]
    result = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": git_output("rev-parse", "HEAD"), "dirty": bool(git_output("status", "--porcelain"))},
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "device": str(device),
        },
        "workload": {
            "shapes": [list(tensor_shape) for tensor_shape in args.shapes],
            "parameters": sum(math.prod(tensor_shape) for tensor_shape in args.shapes),
            "input_dtype": args.dtype,
            "accumulation_dtype": "float64",
            "rollouts": args.rollouts,
            "pool_size": args.pool_size,
            "seed": args.seed,
            "pool_reused": True,
        },
        "timing": {"method": "synchronized_wall_clock", "warmup": args.warmup, "repeats": args.repeats},
        "verification": verification,
        "samples": samples,
        "summary": summary,
    }
    if "cpu" in summary and "cuda" in summary:
        result["cuda_speedup_vs_cpu"] = summary["cpu"]["median_seconds"] / summary["cuda"]["median_seconds"]
        print(f"CUDA speedup vs CPU: {result['cuda_speedup_vs_cpu']:.3f}x")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
