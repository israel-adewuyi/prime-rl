"""Compare two SNR benchmark result files with matching workloads and environments."""

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-implementation", default="cuda")
    parser.add_argument("--candidate-implementation", default="cuda")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    for field in ("schema_version", "workload", "environment"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Cannot compare runs with different {field}")
    if baseline["schema_version"] != 1:
        raise ValueError("Unsupported benchmark schema")
    if baseline["timing"]["method"] != candidate["timing"]["method"]:
        raise ValueError("Cannot compare different timing methods")
    for data, name in ((baseline, args.baseline_implementation), (candidate, args.candidate_implementation)):
        if not data["verification"][name]["passed"]:
            raise ValueError(f"Verification failed for {name}")
    before_verification = baseline["verification"][args.baseline_implementation]
    after_verification = candidate["verification"][args.candidate_implementation]
    for field in ("rtol", "atol"):
        if before_verification[field] != after_verification[field]:
            raise ValueError(f"Cannot compare different verification {field}")
    before_metrics = before_verification["metrics"]
    after_metrics = after_verification["metrics"]
    if before_metrics.keys() != after_metrics.keys() or before_metrics["snr/count"] != after_metrics["snr/count"]:
        raise ValueError("Cannot compare different metric keys or rollout counts")
    for key, value in before_metrics.items():
        if not (math.isfinite(value) and math.isfinite(after_metrics[key])) or not math.isclose(
            value,
            after_metrics[key],
            rel_tol=before_verification["rtol"],
            abs_tol=before_verification["atol"],
        ):
            raise ValueError(f"Cannot compare numerically different {key} values")
    before = baseline["summary"][args.baseline_implementation]
    after = candidate["summary"][args.candidate_implementation]
    print(f"Baseline:  {before['median_seconds']:.6f} s (std {before['stdev_seconds']:.6f})")
    print(f"Candidate: {after['median_seconds']:.6f} s (std {after['stdev_seconds']:.6f})")
    print(f"Speedup:   {before['median_seconds'] / after['median_seconds']:.3f}x")
    for label, row in (("Baseline", before), ("Candidate", after)):
        print(f"{label} extra peak GPU allocation: {row['extra_peak_allocated_bytes'] / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()
