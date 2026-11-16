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
