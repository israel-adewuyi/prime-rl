"""Launch one member of the exp3 sweep from a shared base configuration."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

MODEL = "LiquidAI/LFM2.5-230M"
REVISION = "40cb2ad3b3044d5a41eee083a6103c8b523afa45"
RATES = [f"{mantissa}e-{exponent:02d}" for exponent in range(7, 2, -1) for mantissa in (1, 5)]
RUNS = [(rate, 200, False) for rate in RATES] + [("1e-06", 500, False), ("1e-06", 1000, False), ("1e-06", 5, True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--index", type=int, choices=range(len(RUNS)))
    parser.add_argument("--gpus", default="0,1", help="Inference GPU,trainer GPU within the allocation")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp3-20261024"))
    parser.add_argument("--steps", type=int, help="Explicit override for a short timing probe")
    parser.add_argument("--name", help="Distinct run name for a timing probe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.list:
        for index, (lr, steps, capture) in enumerate(RUNS):
            print(json.dumps(dict(index=index, lr=lr, steps=steps, capture=capture)))
        return
    if args.index is None:
        parser.error("--index is required unless --list is used")
    devices = args.gpus.split(",")
    if len(devices) != 2 or not all(device.isdigit() for device in devices) or devices[0] == devices[1]:
        parser.error("--gpus requires two distinct nonnegative GPU ordinals")
    lr, steps, capture = RUNS[args.index]
    steps = args.steps if args.steps is not None else steps
    if steps < 1 or (capture and steps != 5):
        parser.error("steps must be positive; the capture experiment requires five updates")
    name = args.name or ("lfm-gradient-capture-5" if capture else f"lfm-lr-{lr}-{steps}")
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip():
        parser.error("Commit tracked source changes before launching an experiment")
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    port = 4000 + args.index * 20
    command = [
        "uv", "--no-config", "run", "--no-project", "--active", "--no-sync", "rl",
        "@", str(Path(__file__).with_name("base.toml")),
        "--model.name", model_path,
        "--run.name", name, "--output-dir", str(args.output_dir.resolve()),
        "--max-steps", str(steps), "--trainer.optim.lr", lr,
        "--inference.server.port", str(port), "--inference.backend-port", str(port + 1),
        "--inference.vllm.data-parallel-rpc-port", str(27000 + args.index * 20),
        "--orchestrator.model.client.base-url", f"http://127.0.0.1:{port}/v1",
        "--orchestrator.model.client.admin-base-url", json.dumps([f"http://127.0.0.1:{port + 1}/v1"]),
        "--rollout-transport.port", str(26000 + args.index * 20), "--no-dashboard",
    ]
    if capture:
        command += ["--trainer.gradient-capture.steps", "[1,2,3,4,5]", "--trainer.gradient-capture.verify"]
    if args.dry_run:
        command += ["--dry-run"]
    print(json.dumps(dict(commit=revision, model=MODEL, model_revision=REVISION, index=args.index,
                          name=name, lr=lr, steps=steps, capture=capture, gpus=args.gpus,
                          command=command)), flush=True)
    print(shlex.join(command), flush=True)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=args.gpus, CUDA_VISIBLE_DEVICES=args.gpus)
    raise SystemExit(subprocess.call(command, cwd=root, env=env))


if __name__ == "__main__":
    main()
