# SNR accumulation benchmark

This benchmark measures the statistics of **already-computed GPU gradients**.
It runs no model, backward pass, optimizer, or rollout generator. The runner
imports the production accumulators from `gradient_statistics.py`, which
requires only PyTorch. Run from a checkout with Git and a CUDA/ROCm-enabled
PyTorch environment; no additional dependencies are needed.

## Run

From the repository root, using the installed environment:

```bash
uv run --no-sync python benchmarks/snr/run.py outputs/snr/initial.json
```

The defaults are 1,000,000 elements in one tensor, FP32 input, FP64 accumulation,
64 rollouts, a pool of four gradients, two warmups, and ten measured repetitions
for each of the CPU and CUDA implementations. Both implementations receive the
same inputs on GPU, so the CPU measurements include device-to-host transfers.

A larger workload with 230 million elements split into 230 parameter tensors:

```bash
uv run --no-sync python benchmarks/snr/run.py outputs/snr/230m.json \
    --parameters 230000000 --tensor-numel 1000000 --rollouts 2048 \
    --pool-size 2 --dtype bfloat16 --warmup 2 --repeats 10
```

Each repetition processes all 2,048 rollout gradients. This can be slow for the
CPU reference. Use smaller workloads while developing, and `--repeats 3` for an
initial large-workload measurement. The workload size describes gradient
elements; a model checkpoint is not needed.

To control parameter shapes directly:

```bash
uv run --no-sync python benchmarks/snr/run.py outputs/snr/shapes.json \
    --shapes 4096x4096 4096x1024 4096 --rollouts 64 --dtype float32
```

`--parameters` and `--shapes` are mutually exclusive. `--tensor-numel` applies
only to the total-element layout. Other controls include `--device cuda:1`,
`--threads 8`, `--seed 42`, and `--implementations cuda`. Even when timing only
CUDA, verification still runs the CPU reference first. There is no CPU-input
fallback: absence of a GPU is an error.

## Verification and timing

Before any warmups or timings, the runner compares each implementation with the
CPU reference on the configured workload: the full gradient sum `S`, squared
norm sum `Q`, count, and every final metric. Failures abort the run. Correctness
results and absolute differences are saved with the timings. This verification
is independent of the trainer's `gradient_capture.verify`; normal training can
keep that setting false.

Input construction and verification are outside the measured region. Each
measurement includes accumulator construction, all updates, finalization, and
release of the retained gradient sum. Wall-clock timing is synchronized at its
boundaries to include CPU work, GPU work, transfers, and completion of queued
operations. There is no synchronization between individual updates in the
CUDA accumulator. Implementation order alternates between repetitions.

Results contain raw timings, median/mean/standard deviation, peak allocated
GPU memory, additional peak allocation above the input-pool baseline, and
per-repetition peak reserved memory. Reserved memory includes PyTorch's cache
from earlier verification/warmups/implementations; additional peak **allocated**
memory is the useful measure of accumulator overhead. The retained FP64 sum
costs eight bytes per parameter, plus per-parameter conversion temporaries.

The input pool is generated once on CPU with a fixed seed, transferred before
verification, and cycled through without modification. Its memory is bounded
by `min(pool_size, rollouts) * parameters * input_element_size`. The repeated
inputs can be more cache-friendly than unique rollout gradients. Keep pool
size fixed across comparisons; these are accumulation timings, not estimates
of full training throughput or the SNR of actual rollouts.

Nonfinite gradients and norm overflow are rejected. The CUDA implementation
detects these during finalization rather than synchronizing on every parameter.
CPU and CUDA reductions can differ slightly in floating-point rounding.

## Compare implementations and revisions

The runner prints the median CUDA speedup over CPU when both are measured.
To compare revisions using the same workload and environment:

```bash
uv run --no-sync python benchmarks/snr/compare.py \
    outputs/snr/before.json outputs/snr/after.json
```

Or compare CPU with CUDA within one result file:

```bash
uv run --no-sync python benchmarks/snr/compare.py \
    outputs/snr/initial.json outputs/snr/initial.json \
    --baseline-implementation cpu --candidate-implementation cuda
```

The comparison requires matching workload, environment, timing method, and
verification tolerances. Both runs must pass verification, and their selected
implementations' final metrics must agree across runs within those tolerances.
Warmup/repetition counts are recorded but need not match. Git revision, dirty
status, and source hashes identify the code used.
Save results from before and after each change rather than overwriting them.
Keep other GPU workloads off the benchmark device when measuring.

To try another accumulator, add a factory to `IMPLEMENTATIONS` in `run.py`.
It must implement `update(gradients)` and `finalize()`, and expose `gradient_sum`,
`squared_norm_sum`, and `count` for untimed verification. Each update must consume
its inputs without mutating them or retaining their autograd graph. Finalization
must return the same metric keys and clear the state. No benchmark framework
or plugin dependency is needed.

The focused correctness tests, including CUDA cases, are:

```bash
uv run --no-sync pytest tests/unit/train/rl/test_gradient_statistics.py
```

## Enable in training

```toml
[trainer.gradient_capture]
steps = [1, 10, 50]
statistics_device = "cuda"
verify = false
```

`statistics_device` defaults to `"cpu"`. CUDA accumulation preserves the FP64
statistics and per-rollout gradient extraction; it changes where statistics
are computed. The benchmark does not measure the cost of gradient extraction.
