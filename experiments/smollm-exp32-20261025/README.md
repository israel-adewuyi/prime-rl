# Exp3.2: SmolLM2 Instruct learning-rate and batch-size sweep

One base config and `run.py` define twelve 200-update runs: batch sizes 256/512
crossed with learning rates 1e-7, 5e-7, 1e-6, 5e-6, 1e-5 and 5e-5.
Indices 12/13 repeat batch256, LR1e-6 for 500/1,000 updates. Index14 captures
rollout-gradient statistics and verifies five real RL updates; its final
checkpoint also contains pre-clipping batch gradients and actual AdamW updates.

The model is `HuggingFaceTB/SmolLM2-135M-Instruct`, pinned to
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, with its native chat template.
The baseline source was `9f891e305588434459e76ad17bf2bd306bbf039b`.
Both unchanged ROCm admission functions reject its `llama` architecture before
model execution. The accompanying narrow repair admits PrimeRL's existing
Llama implementation; it does not establish numerical equivalence with HF.

Both training and evaluation cap completion length at128 tokens. The per-episode
inference context limit is512. Packing defaults to16,384 and can be overridden
for timing probes or capture. `--pack` sets both the orchestrator's actual packing
length and the trainer's accepted length. Group size16 means16/32 admitted prompt groups
per batch256/512; discarded/no-signal groups can increase sampled prompts.
BF16 computation/inference and FP32 optimization/reduction defaults are unchanged.
The built-in reverse-text task and reward are unchanged.

Held-out task indices0:32 are excluded from training and evaluated at initialization
and every20 updates, one deterministic completion per task. Report mean reward
separately from full-credit accuracy (`solved_all` at eval group size1): partial
SequenceMatcher credit is not exact reversal accuracy. Count evaluation errors
and incomplete evaluations explicitly.

Inference uses max128 sequences and an explicit1GiB KV cache, with
`gpu_memory_utilization=0.10`. The explicit cache size overrides percentage-based
cache sizing; neither setting limits compute. Filesystem broadcasts are temporary.
Ordinary sweep jobs do not save checkpoints. Capture checkpoint tensors are
validated with `tools/check_gradient_capture.py` before artifact cleanup.

```bash
uv --no-config run --no-project --active python experiments/smollm-exp32-20261025/run.py --list
uv --no-config run --no-project --active python experiments/smollm-exp32-20261025/run.py \
  --index 2 --gpus 0,1 --output-dir /mnt/shared/.research-session/exp3.2-20261025/runs
```

The first GPU serves inference; the second trains. Each run needs a unique slot,
output name and device assignment. `--slot` reserves disjoint router, backend,
router-metrics, rollout and data-parallel RPC ports below the ephemeral range.
Timing probes use separate slots and `--steps`, `--pack`, `--batch-size`, `--name`.
`--dry-run` resolves configs only; it does not exercise model loading or rollouts.
The launcher records the source commit, pinned model and exact arguments; PrimeRL
writes resolved component configs. Target steps are not evidence of completion.

The session budget is eight MI355X GPUs for120 minutes from node join, plus up to
10 minutes to release capacity. Runtime image:
`vllm/vllm-openai-rocm@sha256:596db6504233a5d8575bdc0f6c4c8b245bd7f496bb4024705208081923afb9b3`.
The experiment report records actual packing/concurrency, timings, failures,
learning curves, memory/activity samples and completed update counts.
