import subprocess
import time
from pathlib import Path


LEARNING_RATES = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
NUM_RUNS = 3

BASE_TRAIN = Path("configs/alphabet_sort/rl/train.toml")
BASE_ORCH = Path("configs/alphabet_sort/rl/orch.toml")
INFER_CONFIG = Path("configs/alphabet_sort/rl/infer.toml")

OUTPUT_ROOT = Path("outputs/dppo_sweep")
TRAINER_GPU_IDS = "2"
INFERENCE_GPU_IDS = "2"
GPU_MEMORY_UTILIZATION = "0.4"
TOKEN_METADATA_TOP_LOGPROBS = "5"
TOKEN_METADATA_MAX_EXAMPLES = "16"
TOKEN_METADATA_SEED = "2001"


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("-", "")


for lr in LEARNING_RATES:
    lr_str = format_lr(lr)
    for run_idx in range(1, NUM_RUNS + 1):
        run_stem = f"dppo_sweep_lr={lr_str}_run{run_idx}"
        train_run_name = f"train_{run_stem}"
        orch_run_name = f"orch_{run_stem}"

        cmd = [
            "uv",
            "run",
            "rl",
            "--trainer",
            "@",
            BASE_TRAIN.as_posix(),
            "--orchestrator",
            "@",
            BASE_ORCH.as_posix(),
            "--inference",
            "@",
            INFER_CONFIG.as_posix(),
            "--trainer.optim.lr",
            str(lr),
            "--trainer.wandb.name",
            train_run_name,
            "--orchestrator.wandb.name",
            orch_run_name,
            "--output-dir",
            OUTPUT_ROOT.as_posix(),
            "--trainer-gpu-ids",
            TRAINER_GPU_IDS,
            "--inference-gpu-ids",
            INFERENCE_GPU_IDS,
            "--inference.gpu-memory-utilization",
            GPU_MEMORY_UTILIZATION,
        ]
        if run_idx == 1:
            monitor_path = OUTPUT_ROOT / "rollout_monitor" / run_stem
            cmd.extend(
                [
                    "--orchestrator.eval.save.disk",
                    "--orchestrator.eval.save.disk.path",
                    monitor_path.as_posix(),
                    "--orchestrator.eval.save.token-metadata.enabled",
                    "true",
                    "--orchestrator.eval.save.token-metadata.path",
                    monitor_path.as_posix(),
                    "--orchestrator.eval.save.token-metadata.top-logprobs",
                    TOKEN_METADATA_TOP_LOGPROBS,
                    "--orchestrator.eval.save.token-metadata.max-examples",
                    TOKEN_METADATA_MAX_EXAMPLES,
                    "--orchestrator.eval.save.token-metadata.seed",
                    TOKEN_METADATA_SEED,
                ]
            )

        print(f"\n{'=' * 80}")
        print(f"Starting {run_stem} ({lr=}, output_dir={OUTPUT_ROOT})")
        print(f"{'=' * 80}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nWARNING: {run_stem} failed with code {result.returncode}")
        else:
            print(f"\nFinished {run_stem}")

        time.sleep(10)

print("\nAll sweep runs completed.")
