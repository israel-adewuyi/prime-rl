# Prompt gradient capture with LiquidAI LFM2.5-230M

The two-update demo trains [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)
on reverse-text with GRPO advantages, saving each prompt's total batch-normalized
gradient on updates 1 and 2. The trainer retains its configured policy loss;
the example uses the repository's default IPO surrogate.

The demo uses reverse-text's built-in reward unchanged. It scores similarity
to the reversed target inside the task's required `<reversed_text>` tags.
Gradient capture does not require a custom reward or a different policy loss.

Use a Linux ROCm environment provisioned according to the
[AMD setup instructions](../amd/qwen3/README.md#installation). The complete RL
demo uses **two GPUs: one trainer GPU and one inference GPU**. The focused
FSDP acceptance test needs only one GPU.

From the repository root, with the environment activated:

```bash
# Configuration validation; no training or GPU allocation.
uv --no-config run --no-project --active --no-sync rl \
    @ examples/prompt-gradient-capture/rl.toml --dry-run

# Loss and prompt-metadata checks (the loss tests require a GPU).
uv --no-config run --no-project --active --no-sync python -m pytest -q \
    tests/unit/train/rl/test_loss.py tests/unit/orchestrator/test_batch.py
# Configuration checks.
uv --no-config run --no-project --active --no-sync python -m pytest -q \
    tests/unit/test_configs.py -k 'gradient_capture or prompt-gradient-capture'

# Real packed-model equivalence, boundary isolation, repeated backward, and FSDP.
uv --no-config run --no-project --active --no-sync python -m pytest -q \
    tests/unit/train/models/test_lfm2_portability.py

# Two captures followed by an uncaptured update; compare with ordinary training.
uv --no-config run --no-project --active --no-sync python -m pytest -q \
    tests/unit/train/rl/test_gradient_capture_fsdp.py

# Full rollout -> GRPO -> training -> weight-refresh pipeline.
bash scripts/run_prompt_gradient_demo.sh

# Inspect artifacts after the RL demo.
uv --no-config run --no-project --active --no-sync python \
    tools/check_prompt_gradients.py outputs/lfm25-prompt-gradients 1 2
```

The demo script accepts ordinary CLI overrides and additional TOML overlays:

```bash
bash scripts/run_prompt_gradient_demo.sh --run.name lfm25-capture-repeat
```

Keep the FSDP test fixture at `tests/fixtures/gradient_capture/fsdp.toml` with
the test results. The configuration and full run command are also saved in the
RL run's `configs/` directory. Tests and the RL run must complete on the target
GPU runtime before treating the capture path as validated. They are not run
as part of Windows development.

Capture output is under `outputs/lfm25-prompt-gradients/prompt_gradients/`.
Both `step_1/manifest.json` and `step_2/manifest.json` should exist, verification
should have passed, and every referenced prompt file should contain finite
CPU FP32 gradients (or explicit `None` entries for unused parameters).
GRPO sequences with zero advantages are filtered before reaching the trainer.
If all rollouts in a group receive the same reward, that group supplies no
training data and rollout generation must continue. Stochastic sampling cannot guarantee different rewards. The controlled FSDP test provides nonzero objectives
independently of rollout quality. The demo checks the pipeline and is not a
reward-improvement benchmark.

Full FP32 gradients for a 230M-parameter model take approximately 0.92 GB per
prompt per captured update, before metadata. Two prompt groups per update
over two updates need roughly 3.7 GB of gradient files; actual group counts
depend on how samples are assembled into training batches.

See [the feature documentation](../../docs/prompt_gradient_capture.md) for
normalization, artifact semantics, and supported settings.
