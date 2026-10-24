# Rollout gradient statistics with LiquidAI LFM2.5-230M

The two-update demo trains [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)
on reverse-text with GRPO advantages and the default IPO policy surrogate.
Updates 1 and 2 report rollout-gradient mean norm, sample variance, and power
SNR through the normal monitors. Both checkpoints include the raw batch
gradient and actual parameter update.

The demo uses reverse-text's built-in reward unchanged. It scores similarity
to the reversed target inside the required `<reversed_text>` tags. No reward
or batching change is needed for gradient statistics.

Use a Linux ROCm environment provisioned according to the
[AMD setup instructions](../amd/qwen3/README.md#installation). The full RL
demo needs **two GPUs: one trainer GPU and one inference GPU**. The focused
FSDP acceptance test needs one GPU. Run these commands from the repository
root with the provisioned environment activated:

```bash
# Configuration validation without launching training.
uv --no-config run --no-project --active --no-sync rl \
    @ examples/prompt-gradient-capture/rl.toml --dry-run

# Statistics and checkpoint tensor logic.
uv --no-config run --no-project --active --no-sync python -m pytest -q \
    tests/unit/train/rl/test_gradient_statistics.py \
    tests/unit/train/test_update_capture.py

# Loss and rollout-metadata checks (loss tests require a GPU).
uv --no-config run --no-project --active --no-sync python -m pytest -q \
    tests/unit/train/rl/test_loss.py tests/unit/orchestrator/test_batch.py

# Configuration checks.
uv --no-config run --no-project --active --no-sync python -m pytest -q \
    tests/unit/test_configs.py -k 'gradient_capture or checkpoint_gradient or prompt-gradient-capture'

# Packed-model equivalence, boundary isolation, repeated backward, and FSDP.
uv --no-config run --no-project --active --no-sync python -m pytest -q \
    tests/unit/train/models/test_lfm2_portability.py

# Captured updates followed by an uncaptured update against ordinary training.
uv --no-config run --no-project --active --no-sync python -m pytest -q \
    tests/unit/train/rl/test_gradient_capture_fsdp.py

# Full rollout -> GRPO -> training -> weight-refresh pipeline.
bash scripts/run_prompt_gradient_demo.sh

# Validate statistics and checkpoint tensor files after the run.
uv --no-config run --no-project --active --no-sync python \
    tools/check_gradient_capture.py outputs/lfm25-gradient-stats 1 2
```

The script accepts additional TOML overlays and ordinary CLI overrides:

```bash
bash scripts/run_prompt_gradient_demo.sh --run.name lfm25-stats-repeat
```

The two controls are independent. `trainer.gradient_capture.steps` selects
the updates where rollout statistics are measured. `[ckpt] interval` selects
normal checkpoints, and `save_gradients_and_updates = true` adds tensors to
those checkpoints, including the final checkpoint. Neither selection enables
the other.

For this demo, verify trainer rows at steps 1 and 2 in
`outputs/lfm25-gradient-stats/monitors/file/metrics.jsonl`. They should contain
`snr/count` and `snr/mean_norm`, plus `snr/variance` for at least two rollouts
and `snr/value` for positive variance. The checkpoint files are:

```text
outputs/lfm25-gradient-stats/checkpoints/step_1/trainer/gradient_update.pt
outputs/lfm25-gradient-stats/checkpoints/step_2/trainer/gradient_update.pt
```

Each file contains named CPU `batch_gradients`, captured before clipping,
and `applied_updates`, calculated as parameters after minus before that
optimizer update. Unused gradients are represented by `None`. No individual
rollout gradient files are saved. For 230M trainable parameters in FP32,
the two tensor collections total roughly 1.84 GB per checkpoint, in addition
to the normal model and optimizer checkpoint. The statistics accumulator
also holds one CPU FP64 gradient sum, approximately 1.84 GB at this size.

GRPO sequences with zero advantages are filtered before reaching the trainer.
Groups with equal rollout rewards provide no training data; generation must
continue until the batch fills. Stochastic sampling cannot guarantee reward
variation. The controlled FSDP tests use nonzero objectives independently of
rollout quality. This demo checks the pipeline rather than reward improvement.

Keep `tests/fixtures/gradient_capture/fsdp.toml` with the test results. The
run's `configs/` directory records the resolved configuration and command.
GPU tests and the RL demo must pass on the target runtime before treating
the path as validated. They are not launched during Windows development.
See [the feature documentation](../../docs/prompt_gradient_capture.md) for
statistic definitions, tensor semantics, and execution constraints.
