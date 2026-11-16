# Exp6 PrimeRL

Qwen2.5-0.5B-Instruct, pinned revision 7ae557604adf67be50417f59c2c2f167def9a775.
Built-in AlphabetSort reward; 150 updates; LR 1e-6; B32/group16; CK attention;
BF16 compute/inference and unchanged FP32 optimization/reduction defaults.
GPU1 holds inference and trainer. GPU0 is reserved for accumulation benchmarks.
Statistics are computed on GPU at updates 10,20,...,150. Capture verification is
off for the main measurement; run the existing focused GPU tests before training.
There are no model checkpoints, and no per-rollout gradient dumps.

Activate the prepared PrimeRL environment, set HF_HOME=/mnt/shared/.cache/huggingface
and FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE. From this repository:

```bash
uv --no-config run --no-project --active python experiments/exp6-20261116/run.py --condition alphabet-snr150 --output-dir /mnt/shared/.research-session/exp6-20261116/runs --dry-run
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 bash experiments/exp6-20261116/benchmark.sh /mnt/shared/.research-session/exp6-20261116/benchmarks/snr
uv --no-config run --no-project --active python experiments/exp6-20261116/run.py --condition alphabet-snr150 --output-dir /mnt/shared/.research-session/exp6-20261116/runs
```

Use a different output root for the dry run. Benchmark JSON contains verification,
raw synchronized timings, memory use, source hashes, and environment details.
These benchmarks measure accumulation of existing gradients, not full capture
or training throughput. The synthetic large layout matches Qwen's parameter count
but uses equal-size chunks rather than its actual tensor shapes.

## Qwen3.8 synthesis serving

`synthdata.py` uses the pinned Qwen3.8-27B revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0` on GPUs4/5/6/7.
It compares TP4/DP1, TP2/DP2, and TP1/DP4 at concurrency1/16/64,
with 128 requests per condition, nominal 512-token inputs and 128-token outputs.
Thinking is disabled for this text workload; all layouts use the same seed,
8K context limit and BF16 weights. Vision is disabled. Random prompt token counts
can differ after chat formatting; use each result's measured token totals.
The installed vLLM runtime is the prepared image environment, not the historical
editable vLLM checkout. Record its version in every GPU session report.

```bash
uv --no-config run --no-project --active python experiments/exp6-20261116/synthdata.py plan --output-dir /mnt/shared/.research-session/exp6-20261116/benchmarks/synth
uv --no-config run --no-project --active python experiments/exp6-20261116/synthdata.py benchmark --output-dir /mnt/shared/.research-session/exp6-20261116/benchmarks/synth
uv --no-config run --no-project --active python experiments/exp6-20261116/synthdata.py serve --layout dp4 --output-dir /mnt/shared/.research-session/exp6-20261116/benchmarks/synth
```

Benchmark output remains visible in the terminal. Open a second live pane for
`tail -F .../benchmarks/synth/<layout>/server.log` while a layout starts.
Result JSON includes throughput and TTFT/TPOT/ITL/end-to-end latency percentiles.
Choose the final layout from GPU measurements; the default final server uses DP4.
The final service binds container localhost:8300; use a localhost-only tunnel for
user access. CPU checks validate flags/imports, not GPU engine startup or serving.
