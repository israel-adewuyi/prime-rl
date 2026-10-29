"""Exp4 Qwen2.5: built-in tasks, matched gradient capture, GPU placement/sharing."""

import argparse
import json
import os
import shlex
import subprocess
import tomllib
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TRACKS = ("alphabet", "alphabet-capture", "reverse")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", choices=TRACKS, required=True)
    p.add_argument("--gpus", default="0,1", help="One ordinal colocates inference/trainer; two separate them")
    p.add_argument("--slot", type=int, required=True, choices=range(20))
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--requests", type=int, choices=(128, 256, 512), default=128)
    p.add_argument("--head-chunk", type=int, default=8192)
    p.add_argument("--flash-attention-backend", choices=("ck", "triton"), default="ck")
    p.add_argument("--profile-step", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE" if args.flash_attention_backend == "triton" else "FALSE"
    gpus = args.gpus.split(",")
    if len(gpus) not in (1, 2) or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        p.error("--gpus needs one ordinal or two distinct ordinals")
    if args.steps < 1 or args.batch_size < 16 or args.batch_size % 16 or args.head_chunk < 1:
        p.error("positive steps/chunk and batch divisible by group_size=16 required")
    if args.profile_step is not None and not 1 <= args.profile_step <= args.steps:
        p.error("profile step must fall within the run")
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip():
        p.error("Commit tracked source changes before launching")
    from huggingface_hub import snapshot_download

    from prime_rl.configs.rl import RLConfig
    from prime_rl.utils.config import dump_resolved_config
    from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, ProcessGroup

    model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    raw = tomllib.loads(Path(__file__).with_name("base.toml").read_text())
    raw["model"]["name"] = model_path
    raw["run"] = dict(name=args.name)
    raw["output_dir"] = str(args.output_dir.resolve())
    raw["max_steps"] = args.steps
    raw["dashboard"] = False
    raw["dry_run"] = args.dry_run
    raw["trainer"]["optim"]["lr"] = args.lr
    raw["trainer"]["model"]["fused_lm_head_token_chunk_size"] = args.head_chunk
    raw["orchestrator"]["batch_size"] = args.batch_size
    raw["orchestrator"]["concurrency"] = dict.fromkeys(
        ("initial_inflight", "min_inflight", "max_inflight"), args.requests
    )
    task = "reverse-text" if args.track == "reverse" else "alphabet-sort"
    for split in ("train", "eval"):
        raw["orchestrator"][split]["source"] = [
            dict(
                name=task + ("-heldout" if split == "eval" else ""),
                env=dict(taskset=dict(id=task), agent=dict(harness=dict(id="null"), runtime=dict(type="subprocess"))),
            )
        ]
    if args.track == "alphabet-capture":
        raw["trainer"]["gradient_capture"] = dict(steps=list(range(1, args.steps + 1)), verify=True)
    if args.profile_step:
        raw["trainer"]["trace_path"] = str(args.output_dir.resolve() / args.name / "torch-profile")
        raw["trainer"]["trace_step"] = args.profile_step
    port = 4000 + args.slot * 50
    raw["inference"]["server"] = dict(host="127.0.0.1", port=port)
    raw["inference"]["backend_port"] = port + 1
    raw["inference"]["vllm"].update(
        seed=args.seed, max_num_seqs=args.requests, data_parallel_rpc_port=18000 + args.slot * 50
    )
    raw["orchestrator"]["model"] = dict(
        client=dict(base_url=f"http://127.0.0.1:{port}/v1", admin_base_url=[f"http://127.0.0.1:{port + 1}/v1"])
    )
    raw["rollout_transport"] = dict(type="zmq", port=21000 + args.slot * 50)
    config = RLConfig.model_validate(raw)
    launch_dir = args.output_dir.resolve() / ".launch" / args.name
    launch_dir.mkdir(parents=True, exist_ok=False)
    inference = dump_resolved_config(config.inference)
    inference["output_dir"] = str(launch_dir / "inference")
    inference["dry_run"] = args.dry_run
    inference_path = launch_dir / "inference.json"
    inference_path.write_text(json.dumps(inference, indent=2))
    # Both placements use the stock external-inference path; only physical GPU masks differ.
    raw["inference"] = None
    raw["deployment"]["num_infer_gpus"] = 0
    config = RLConfig.model_validate(raw)
    config_path = launch_dir / "rl.json"
    config_path.write_text(json.dumps(dump_resolved_config(config), indent=2))
    prefix = ["uv", "--no-config", "run", "--no-project", "--active"]
    infer_cmd = prefix + ["inference", "@", str(inference_path)]
    train_cmd = prefix + ["rl", "@", str(config_path)]
    record = dict(
        commit=commit,
        model=MODEL,
        model_revision=REVISION,
        args=vars(args) | {"output_dir": str(args.output_dir)},
        inference_command=infer_cmd,
        rl_command=train_cmd,
        inference_gpu=gpus[0],
        trainer_gpu=gpus[-1],
        router_metrics_port=port + 21000,
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
                "HIP_VISIBLE_DEVICES": gpus[0],
                "CUDA_VISIBLE_DEVICES": gpus[0],
            },
            log_path=launch_dir / "inference.log",
        )
        processes.start(
            "rl",
            train_cmd,
            env={**common_env, "HIP_VISIBLE_DEVICES": gpus[-1], "CUDA_VISIBLE_DEVICES": gpus[-1]},
            log_path=launch_dir / "rl.log",
        )
        processes.wait("rl")


if __name__ == "__main__":
    main()
