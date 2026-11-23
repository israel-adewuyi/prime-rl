"""Run one checkpoint and repeat on a dedicated inference/trainer GPU pair."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from prepare import HERE, fingerprint, load_toml, resolve_checkpoint
from sweep import captures


def resolved_configs(model, tokenizer, capture, port, state):
    from prime_rl.configs.inference import InferenceConfig
    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.configs.trainer import TrainerConfig

    raw = load_toml("rl.toml")
    trainer = raw["trainer"]
    trainer["model"]["name"] = model
    trainer["tokenizer"] = dict(name=tokenizer)
    trainer.update(output_dir=state, max_steps=1)
    orch = raw["orchestrator"]
    orch["tokenizer"] = dict(name=tokenizer)
    orch["model"] = dict(
        name=model,
        client=dict(
            base_url=f"http://127.0.0.1:{port}/v1",
            admin_base_url=[f"http://127.0.0.1:{port + 1}/v1"],
        ),
    )
    orch["batch_size"] = capture.batch_size
    orch["train"]["select"].update(seed=capture.prompt_seed, limit=capture.prompt_count)
    orch["train"]["group_size"] = capture.group_size
    orch["train"]["sampling"]["temperature"] = capture.temperature
    inference = load_toml("inference.toml")
    inference.update(output_dir=state, backend_port=port + 1)
    inference["server"]["port"] = port
    inference["vllm"].update(
        model=model, tokenizer=tokenizer, seed=capture.engine_seed, data_parallel_rpc_port=port + 14000
    )
    return (
        TrainerConfig.model_validate(trainer),
        OrchestratorConfig.model_validate(orch),
        InferenceConfig.model_validate(inference),
    )


def write_json(path, obj):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=path.parent, encoding="utf-8") as temp:
        json.dump(obj, temp, allow_nan=False, ensure_ascii=False, indent=2)
    Path(temp.name).replace(path)


async def record_run(run_dir, metrics):
    from prime_rl.configs.monitors import FileMonitorConfig
    from prime_rl.monitors.file.monitor import FileMonitor

    monitor = FileMonitor(FileMonitorConfig(path=Path("metrics.pending.jsonl")))
    pending = run_dir / "monitors/file/metrics.pending.jsonl"
    pending.unlink(missing_ok=True)
    await monitor.init(run_dir, producer="trainer")
    try:
        await monitor.log(metrics, step=1)
    finally:
        await monitor.finalize()
        monitor.file.close()
    pending.replace(pending.with_name("metrics.jsonl"))


def selected_tasks(source):
    import verifiers.v1 as vf

    taskset = vf.load_taskset(source.env.taskset).select(source.select)
    tasks = {task.data.idx: task for task in taskset}
    if len(tasks) != source.select.limit:
        raise ValueError(f"Expected {source.select.limit} distinct AlphabetSort tasks, got {len(tasks)}")
    return tasks


def worker(args, conditions, port, state):
    from cohort import SeededAlphabetSort, collect, score_and_pack

    from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
    from prime_rl.orchestrator.batch import BatchPacker
    from prime_rl.orchestrator.clients import InferenceClient, check_inference_ready
    from prime_rl.trainer.rl.data import FixedPolicyBatch
    from prime_rl.trainer.rl.train import train
    from prime_rl.utils.config import dump_resolved_config
    from prime_rl.utils.pathing import create_attempt_dirs

    phase = json.loads((state / "launch.json").read_text())
    model = phase["checkpoint"]
    tokenizer_model = phase["tokenizer"]
    tokenizer = tokenizer_model["path"]
    trainer, orch, inference = resolved_configs(model["path"], tokenizer, conditions[0], port, state)
    source = orch.train.source[0]
    env = SeededAlphabetSort(source.env)
    clients = InferenceClient(
        orch.model.client,
        model["path"],
        train_client_type="renderer",
        renderer_config=orch.renderer,
        renderer_model_name=tokenizer,
    )
    packer = BatchPacker(orch)
    algo = GRPOAlgorithm(source.algo, clients)
    max_inflight = load_toml("sweep.toml")["max_inflight_episodes"]
    with ExitStack() as stack:
        loop = stack.enter_context(asyncio.Runner())
        stack.callback(lambda: loop.run(clients.aclose()))
        loop.run(check_inference_ready(orch.model.client, model["path"]))

        def batches():
            for capture in conditions:
                run_dir = args.output / capture.name
                run_dir.mkdir(parents=True, exist_ok=True)
                _, resolved, _ = resolved_configs(model["path"], tokenizer, capture, port, state)
                source = resolved.train.source[0]
                tasks = selected_tasks(source)
                prompt_ids = list(tasks)
                spec = dict(
                    condition=asdict(capture),
                    checkpoint=model,
                    tokenizer=tokenizer_model,
                    prompt_ids=prompt_ids,
                    source_sha=phase["source_sha"],
                    max_inflight_episodes=max_inflight,
                )
                identity = fingerprint(dict(spec=spec, rl=load_toml("rl.toml"), inference=load_toml("inference.toml")))
                config_path = run_dir / "config.json"
                metrics_path = run_dir / "monitors/file/metrics.jsonl"
                previous = json.loads(config_path.read_text()) if config_path.exists() else {}
                if previous and previous.get("identity") != identity:
                    raise ValueError(f"{capture.name}: output directory belongs to a different configuration")
                if metrics_path.exists() and metrics_path.stat().st_size:
                    (row,) = (json.loads(line) for line in metrics_path.read_text().splitlines())
                    if row.get("snr/all_scored/count") != capture.batch_size or (
                        "nonzero_count" not in previous
                        or row.get("snr/nonzero_advantage/count") != previous["nonzero_count"]
                    ):
                        raise ValueError(f"{capture.name}: incomplete result")
                    print(f"Already captured {capture.name}", flush=True)
                    continue

                configs, logs = create_attempt_dirs(run_dir)
                for role, config in (("trainer", trainer), ("orchestrator", resolved), ("inference", inference)):
                    write_json(configs / f"{role}.json", dump_resolved_config(config))
                for role in ("trainer", "inference"):
                    (logs / f"{role}.log").symlink_to(os.path.relpath(state / f"{role}.log", logs))
                metadata = dict(
                    **spec,
                    identity=identity,
                    optimizer_step=False,
                    launch=phase,
                    versions={p: version(p) for p in ("torch", "vllm", "transformers")},
                )
                write_json(config_path, metadata)
                print(f"Generating {capture.name}", flush=True)
                try:
                    episodes = loop.run(collect(capture, tasks, env, clients, source, max_inflight))
                    micro, scored, active = loop.run(score_and_pack(capture, prompt_ids, episodes, algo, packer))
                    del episodes
                    metadata["scored_per_prompt"] = dict.fromkeys(prompt_ids, capture.group_size)
                    metadata["nonzero_count"] = active
                    write_json(config_path, metadata)

                    def record(metrics, directory=run_dir):
                        # Linux reports the lifetime peak of this trainer/orchestrator process in KiB.
                        import resource

                        metrics["memory/worker_peak_rss_gib"] = (
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
                        )
                        loop.run(record_run(directory, metrics))

                    yield FixedPolicyBatch(capture.name, scored, active, micro, record)
                except Exception:
                    print(f"Failed {capture.name}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                    raise

        train(trainer, fixed_batches=batches())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", choices=load_toml("sweep.toml")["checkpoints"])
    parser.add_argument("--repeat", type=int, required=True, choices=range(1, 5))
    parser.add_argument("--export-root", type=Path, help="Directory containing the original Exp5 model exports")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", help="One exact capture name for a smoke check")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-state", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.only:
        args.output = args.output / ".smoke"
    conditions = [
        c
        for c in captures()
        if c.checkpoint == args.checkpoint and c.repeat == args.repeat and (args.only is None or args.only == c.name)
    ]
    if not conditions:
        parser.error("--only must name a capture in the selected checkpoint and repeat")
    slot = args.repeat - 1
    port = 4600 + 50 * slot
    sweep = load_toml("sweep.toml")
    checkpoints = sweep["checkpoints"]
    tokenizer_label = sweep["tokenizer_checkpoint"]
    if args.dry_run:
        for capture in conditions:
            resolved_configs(
                checkpoints[args.checkpoint]["model"], checkpoints[tokenizer_label]["model"], capture, port, args.output
            )
            print(capture.name)
        return
    if args.worker_state:
        worker(args, conditions, port, args.worker_state)
        return

    from filelock import FileLock

    from prime_rl.utils.config import dump_resolved_config
    from prime_rl.utils.process import ProcessGroup

    models = {
        label: resolve_checkpoint(checkpoints[label], offline=True, export_root=args.export_root)
        for label in {tokenizer_label, args.checkpoint}
    }
    model_path = models[args.checkpoint]["path"]
    tokenizer = models[tokenizer_label]["path"]
    for capture in conditions:
        resolved_configs(model_path, tokenizer, capture, port, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    phase_dir = args.output / ".workers"
    phase_dir.mkdir(exist_ok=True)
    state = Path(tempfile.mkdtemp(prefix=f"{args.checkpoint}-rep{args.repeat:02d}-", dir=phase_dir))
    _, _, infer = resolved_configs(model_path, tokenizer, conditions[0], port, state)
    infer_path = state / "inference.json"
    write_json(infer_path, dump_resolved_config(infer))
    engine = [str(Path(sys.executable).with_name("inference")), "@", str(infer_path)]
    command = [sys.executable, str(HERE / "run.py"), *sys.argv[1:], "--worker-state", str(state)]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    launch = dict(
        source_sha=revision,
        checkpoint=models[args.checkpoint],
        tokenizer=models[tokenizer_label],
        inference_command=engine,
        trainer_command=command,
        inference_gpu=2 * slot,
        trainer_gpu=2 * slot + 1,
        log_dir=str(state),
    )
    write_json(state / "launch.json", launch)
    print(json.dumps(launch), flush=True)
    common = {
        **os.environ,
        **load_toml("rl.toml")["env_vars"],
        "PYTHONUNBUFFERED": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(19500 + slot),
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
    }
    lock_dir = args.output.parent if args.only else args.output
    with FileLock(str(lock_dir / f".pair-{slot}.lock"), timeout=0), ProcessGroup() as group:
        for role, cmd, gpu in (("inference", engine, 2 * slot), ("trainer", command, 2 * slot + 1)):
            process_env = dict(
                common,
                HIP_VISIBLE_DEVICES=str(gpu),
                CUDA_VISIBLE_DEVICES=str(gpu),
                NCCL_RAS_ADDR=f"127.0.0.1:{27500 + gpu}",
            )
            group.start(role, cmd, process_env, state / f"{role}.log")
        group.wait("trainer")


if __name__ == "__main__":
    main()
