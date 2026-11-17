# Exp6.1

The committed TOMLs define the experiment. Launchers read them without changing
task, model, sampling, batch size, optimizer, capture schedule or benchmark values.
They supply output paths, dry-run mode and physical GPU placement only.

- `rl.toml`: AlphabetSort's built-in reward, pinned Qwen2.5-0.5B-Instruct,
  batch512/group16, LR1e-6, 150 optimizer updates, SNR at1,10,20,...,150.
  Captures occur before clipping and the corresponding optimizer update; capture1
  therefore measures gradients at the initial model. FP32 optimization/reduction
  defaults remain unchanged. Checkpoints are disabled.
- `inference.toml`: the same pinned model, BF16, CK attention, TP1, shared GPU1.
- `benchmark.toml`: 512 and1024 precomputed gradient samples at1M and494M elements,
  CPU/GPU verification and synchronized timings. This measures accumulation,
  excluding rollout generation and per-sequence backward extraction. The full-size
  CPU reference at these sample counts can take many minutes; the150-update RL run
  remains the priority under the rental cutoff.
- `synth.toml` / `retail.toml`: serving comparisons and10 retail tasks ×3 trials,
  with Qwen for agent, user simulator and semantic judge.

Use the prepared ROCm environment and shared model cache. Run CPU resolution first
with a distinct output directory:

```bash
uv --no-config run --no-project --active python experiments/exp61-20261117/run.py --output-dir /mnt/shared/.research-session/exp61-20261117/preflight/rl-dry --dry-run
uv --no-config run --no-project --active python experiments/exp61-20261117/benchmark.py /mnt/shared/.research-session/exp61-20261117/benchmarks/snr --plan
```

On an authorized GPU allocation, omit `--dry-run` and use `--output-dir` pointing
to the fresh run's `runs/` directory. GPU ownership:0 accumulation benchmark,
1 RL inference/training,2–3 nanoTitan EP2,4–7 Qwen serving and retail.
Keep terminal transcripts and component logs on persistent shared storage.
