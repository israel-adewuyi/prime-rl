"""Exp5: launch one explicitly selected Qwen2.5 placement trial."""

import argparse
import copy
import json
import os
import shlex
import subprocess
import tomllib
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
# task, physical GPU(s), listener slot, inference seed, checkpoint interval
CONDITIONS = {
    "alphabet-solo": ("alphabet-sort", "0", 0, 42, 10),
    "alphabet-pair": ("alphabet-sort", "1,2", 1, 42, None),
    "shared-alphabet-0": ("alphabet-sort", "3", 2, 42, None),
    "shared-alphabet-1": ("alphabet-sort", "3", 3, 43, None),
    "shared-reverse-0": ("reverse-text", "3", 4, 42, None),
    "shared-reverse-1": ("reverse-text", "3", 5, 43, None),
}


def build_raw(condition: str, output_dir: Path, model_path: str, gpus: str | None = None):
    task, default_gpus, slot, seed, checkpoint_interval = CONDITIONS[condition]
    devices = (gpus or default_gpus).split(",")
    expected_devices = 2 if condition == "alphabet-pair" else 1
    if len(devices) != expected_devices or len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        raise ValueError(f"{condition} requires {expected_devices} distinct physical GPU ordinal(s)")
    raw = tomllib.loads(Path(__file__).with_name("base.toml").read_text())
    raw["model"]["name"] = model_path
    raw["run"] = {"name": condition}
    raw["output_dir"] = str(output_dir.resolve())
    raw["trainer"]["trace_path"] = str(output_dir.resolve() / condition / "torch-profile")
    if checkpoint_interval is not None:
        raw["ckpt"] = {"interval": checkpoint_interval}
    for split in ("train", "eval"):
        raw["orchestrator"][split]["source"] = [
            dict(
                name=task + ("-heldout" if split == "eval" else ""),
                env=dict(
                    taskset=dict(id=task),
                    agent=dict(harness=dict(id="null"), runtime=dict(type="subprocess")),
                ),
            )
        ]
    port = 4000 + slot * 50
    ports = dict(
        router=port,
        backend=port + 1,
        router_metrics=port + 21000,
        engine_rpc=18000 + slot * 50,
        rollout=21000 + slot * 50,
    )
    raw["inference"]["server"] = dict(host="127.0.0.1", port=ports["router"])
    raw["inference"]["backend_port"] = ports["backend"]
    raw["inference"]["vllm"].update(seed=seed, data_parallel_rpc_port=ports["engine_rpc"])
    raw["orchestrator"]["model"] = dict(
        client=dict(
            base_url=f"http://127.0.0.1:{ports['router']}/v1",
            admin_base_url=[f"http://127.0.0.1:{ports['backend']}/v1"],
        )
    )
    raw["rollout_transport"] = dict(type="zmq", port=ports["rollout"])
    return raw, devices, ports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        help="Override the physical GPU assignment; first inference, last trainer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configs without starting engines or rollouts",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip():
        parser.error("Commit tracked source changes before launching")
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "FALSE"
    from huggingface_hub import snapshot_download

    from prime_rl.configs.rl import RLConfig
    from prime_rl.utils.config import dump_resolved_config
    from prime_rl.utils.process import (
        DEFAULT_COMMON_ENV_VARS,
        DEFAULT_INFERENCE_ENV_VARS,
        ProcessGroup,
    )

    model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    raw, devices, ports = build_raw(args.condition, args.output_dir, model_path, args.gpus)
    raw["dry_run"] = args.dry_run
    full_config = RLConfig.model_validate(copy.deepcopy(raw))
    launch_dir = args.output_dir.resolve() / ".launch" / args.condition
    launch_dir.mkdir(parents=True, exist_ok=False)
    inference = dump_resolved_config(full_config.inference)
    inference["output_dir"] = str(launch_dir / "inference")
    inference["dry_run"] = args.dry_run
    inference_path = launch_dir / "inference.json"
    inference_path.write_text(json.dumps(inference, indent=2))
    # Both placements use the same external-inference path; only GPU masks differ.
    raw["inference"] = None
    raw["deployment"]["num_infer_gpus"] = 0
    config = RLConfig.model_validate(copy.deepcopy(raw))
    config_path = launch_dir / "rl.json"
    config_path.write_text(json.dumps(dump_resolved_config(config), indent=2))
    prefix = ["uv", "--no-config", "run", "--no-project", "--active"]
    infer_cmd = prefix + ["inference", "@", str(inference_path)]
    train_cmd = prefix + ["rl", "@", str(config_path)]
    record = dict(
        commit=commit,
        model=MODEL,
        model_revision=REVISION,
        condition=args.condition,
        dry_run=args.dry_run,
        inference_command=infer_cmd,
        rl_command=train_cmd,
        inference_gpu=devices[0],
        trainer_gpu=devices[-1],
        ports=ports,
        trace_step=config.trainer.trace_step,
        checkpoint_interval=config.trainer.ckpt.interval if config.trainer.ckpt else None,
        flash_attention_backend="ck",
    )
    (launch_dir / "launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    print(shlex.join(infer_cmd), flush=True)
    print(shlex.join(train_cmd), flush=True)
    if args.dry_run:
        subprocess.run(infer_cmd, cwd=repo, check=True)
        subprocess.run(train_cmd, cwd=repo, check=True)
        return
    common_env = dict(os.environ, **DEFAULT_COMMON_ENV_VARS)
    with ProcessGroup() as processes:
        processes.start(
            "inference",
            infer_cmd,
            env={
                **common_env,
                **DEFAULT_INFERENCE_ENV_VARS,
                "HIP_VISIBLE_DEVICES": devices[0],
                "CUDA_VISIBLE_DEVICES": devices[0],
            },
            log_path=launch_dir / "inference.log",
        )
        processes.start(
            "rl",
            train_cmd,
            env={
                **common_env,
                "HIP_VISIBLE_DEVICES": devices[-1],
                "CUDA_VISIBLE_DEVICES": devices[-1],
            },
            log_path=launch_dir / "rl.log",
        )
        processes.wait("rl")


if __name__ == "__main__":
    main()
