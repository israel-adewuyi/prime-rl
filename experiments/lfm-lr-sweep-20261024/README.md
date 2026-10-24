# Exp3 LFM learning-rate sweep

One base config and `run.py` define this experiment before execution. The grid is
`{1, 5} x {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}`, with 200 optimizer updates per rate.
Indices 10 and 11 repeat 1e-6 for 500 and 1,000 updates. Index 12 runs five ordinary
RL updates with prompt-gradient capture and verification enabled.

Every run uses the instruction-tuned LFM2.5-230M snapshot
`40cb2ad3b3044d5a41eee083a6103c8b523afa45`, batch size 512 and group size 16
(32 prompt groups), BF16 computation/inference and unchanged FP32 optimization
and reduction defaults. The sequence limit is 256; training responses are capped
at 64 tokens and held-out evaluation responses at 128 tokens.
Training and held-out evaluation both use the built-in reverse-text reward.
Held-out task indices 0:32 are excluded from training and evaluated every 50
updates. This reward differs from exp2's historical overrides; evaluation scores
are not directly comparable with exp2's custom exact-match metric.

Inference admits 128 concurrent sequences with an explicit 1 GiB KV cache.
`gpu_memory_utilization=0.10` remains a startup memory check; the explicit cache
size overrides percentage-based cache sizing. Neither setting is a compute quota.
No checkpoint saving is configured. Filesystem broadcasts are temporary weights.

From the repository root, in the prepared ROCm environment:

```bash
uv --no-config run --no-project --active python experiments/lfm-lr-sweep-20261024/run.py --list
uv --no-config run --no-project --active python experiments/lfm-lr-sweep-20261024/run.py \
  --index 2 --gpus 0,1 --output-dir /mnt/shared/.research-session/exp3-20261024/runs
```

The first GPU serves inference and the second trains. Each index reserves its own
router, engine, router metrics, data-parallel RPC and rollout transport ports.
Do not launch the same index twice concurrently. For a separate timing probe,
use `--steps 3 --name lfm-timing-3` before launching the corresponding sweep run.
`--dry-run` resolves the actual PrimeRL config without launching GPU processes.

The launcher records the clean source commit, model revision and full command.
PrimeRL also records resolved component configs and command.txt with each run.
Logs, run names, timing and achieved update counts belong in the experiment
report, not in separate hand-written per-trial TOMLs. Requested update counts
are targets; any rental cutoff or failure must be reported as partial execution.
