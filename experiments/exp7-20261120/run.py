"""Launch an exp7 sweep condition with a colocated trainer and inference engine."""

import argparse
import json
import os
import shlex
import subprocess
import tomllib
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    trials = tomllib.loads((here / "sweep.toml").read_text())["trials"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=trials)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Physical GPU shared by inference and trainer")
    parser.add_argument("--retry", action="store_true", help="Use a distinct run directory for a restarted condition")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error("--gpu must be one physical GPU ordinal")
    repo = here.parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    rl_raw = tomllib.loads((here / "rl.toml").read_text())
    inference_raw = tomllib.loads((here / "inference.toml").read_text())
    for raw in (rl_raw, inference_raw):
        os.environ.update(raw.get("env_vars", {}))
    from prime_rl.configs.inference import InferenceConfig
    from prime_rl.configs.rl import RLConfig
    from prime_rl.utils.config import dump_resolved_config
    from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, ProcessGroup

    condition = args.condition
    trial = trials[condition]
    cap = trial["episode_cap"]
    # Port offsets separate all concurrent listeners, including router metrics.
    slot = int(args.gpu)
    port = 4600 + 50 * slot
    run_name = f"{condition}-retry" if args.retry else condition
    rl_raw["run"]["name"] = run_name
    orch = rl_raw["orchestrator"]
    orch["batch_size"] = trial["batch_size"]
    orch["train"]["group_size"] = trial["rollouts_per_prompt"]
    orch["concurrency"].update(initial_inflight=cap, min_inflight=cap, max_inflight=cap,
                               admission_burst=trial["admission_burst"])
    orch["model"]["client"].update(base_url=f"http://127.0.0.1:{port}/v1",
                                  admin_base_url=[f"http://127.0.0.1:{port + 1}/v1"])
    rl_raw["rollout_transport"]["port"] = 21600 + 50 * slot
    inference_raw["server"]["port"] = port
    inference_raw["backend_port"] = port + 1
    inference_raw["vllm"].update(max_num_seqs=cap, data_parallel_rpc_port=18600 + 50 * slot)
    os.environ["NCCL_RAS_ADDR"] = f"127.0.0.1:{27500 + slot}"
    rl_raw.update(output_dir=str(args.output_dir.resolve()), dry_run=args.dry_run)
    launch_dir = args.output_dir.resolve() / ".launch" / run_name
    launch_dir.mkdir(parents=True, exist_ok=False)
    inference_raw.update(output_dir=str(launch_dir / "inference"), dry_run=args.dry_run)
    config = RLConfig.model_validate(rl_raw)
    inference = InferenceConfig.model_validate(inference_raw)
    if config.model.name != inference.vllm.model:
        raise ValueError("The two TOMLs must select the same model snapshot")
    if not Path(config.model.name).is_dir():
        raise FileNotFoundError(config.model.name)
    prefix = ["uv", "--no-config", "run", "--no-project", "--active"]
    commands = {}
    for name, resolved in (("rl", config), ("inference", inference)):
        path = launch_dir / f"{name}.json"
        path.write_text(json.dumps(dump_resolved_config(resolved), indent=2))
        commands[name] = prefix + [name, "@", str(path)]
    record = dict(
        commit=commit,
        source_tomls=[str(here / "rl.toml"), str(here / "inference.toml"), str(here / "sweep.toml")],
        trial=trial,
        model=config.model.name,
        condition=condition,
        dry_run=args.dry_run,
        inference_command=commands["inference"],
        rl_command=commands["rl"],
        inference_gpu=args.gpu,
        trainer_gpu=args.gpu,
        batch_size=config.orchestrator.batch_size,
        max_steps=config.trainer.max_steps,
        capture_steps=config.trainer.gradient_capture.steps,
        statistics_device=config.trainer.gradient_capture.statistics_device,
        checkpoint_interval=config.trainer.ckpt.interval if config.trainer.ckpt else None,
    )
    (launch_dir / "launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    for command in commands.values():
        print(shlex.join(command), flush=True)
    if args.dry_run:
        for name in ("inference", "rl"):
            subprocess.run(commands[name], cwd=repo, check=True)
        return
    common_env = dict(os.environ, **DEFAULT_COMMON_ENV_VARS)
    common_env.update(HIP_VISIBLE_DEVICES=args.gpu, CUDA_VISIBLE_DEVICES=args.gpu)
    os.chdir(repo)
    with ProcessGroup() as processes:
        processes.start(
            "inference", commands["inference"],
            env={**common_env, **DEFAULT_INFERENCE_ENV_VARS, **inference.env_vars},
            log_path=launch_dir / "inference.log",
        )
        processes.start("rl", commands["rl"], env=common_env, log_path=launch_dir / "rl.log")
        processes.wait("rl")


if __name__ == "__main__":
    main()
