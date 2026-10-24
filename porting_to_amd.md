# PrimeRL on AMD ROCm

The experimental ROCm profile supports dense Qwen3 and LFM2 full-parameter
training with FlashAttention 2 and vLLM inference. Other architectures and
CUDA-specific kernels are not implied to work by a successful import.

## Setup

Use Linux x86_64, Python 3.12, a compatible ROCm driver, and a dedicated activated
environment containing matching ROCm PyTorch and vLLM builds. Preserve the
provisioned GPU binaries; do not install CUDA extras into this environment.

```bash
git submodule update --init --recursive
bash scripts/install_rocm.sh rl  # or sft when inference is not needed
uv --no-config pip check --python "$VIRTUAL_ENV/bin/python"
```

The installer resolves application dependencies separately from the compiled
GPU stack and records its manifest under `$VIRTUAL_ENV/prime-rl-runtime/`.
Keep code fixes in the repository and dependency choices in the ROCm profile;
do not patch site-packages. Root `uv sync --all-extras` selects CUDA sources and
is not the ROCm installation procedure.

The Transformers constraint permits provisioned 5.x versions, but this is not
an assertion that every such version is compatible. Record and validate the
exact PyTorch, HIP, vLLM, Transformers, FA2 and dependency revisions together.

## Supported settings

- Select `attn = "flash_attention_2"`; set
  `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` before importing FlashAttention.
- Use `cp=1`, `ep=1` (or `auto` resolving to 1), full-parameter training, and
  filesystem weight broadcasts. Quantization and LoRA are outside this profile.
- Keep `optimization_dtype` and `reduce_dtype` unchanged. BF16 finite gradients,
  batch-gradient reconstruction and cross-backend numerical equivalence are
  different checks.
- LFM2 uses independent packed-document forwards and stateless short convolution;
  it resets history at document boundaries and preserves native checkpoint names.
  This adapter is a correctness-oriented path, not an optimized throughput claim.
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn`. ROCm still uses `torch.cuda`,
  `device="cuda"` and the distributed `nccl` backend name (implemented by RCCL).

See [Qwen3 configurations](examples/amd/qwen3/README.md) and the
[LFM2 gradient-capture example](examples/prompt-gradient-capture/README.md).

## Checks and unresolved limits

From the repository root with the ROCm environment activated:

```bash
uv --no-config run --no-project --active python -m pytest -q tests/unit/test_configs.py -k amd
uv --no-config run --no-project --active python -m pytest -q tests/unit/train/models/test_qwen3_portability.py
uv --no-config run --no-project --active python -m pytest -q tests/unit/train/models/test_lfm2_portability.py
uv --no-config run --no-project --active python -m pytest -q tests/unit/train/rl/test_gradient_capture_fsdp.py
VLLM_WORKER_MULTIPROC_METHOD=spawn uv --no-config run --no-project --active python -m pytest -q tests/unit/inference/test_qwen3_weight_refresh.py
```

The BF16 packed Qwen3 comparison against HF SDPA can fail gradients even when
logits and loss pass. Backend choice and accumulation order require isolation;
do not treat a finite-gradient smoke run as numerical equivalence. The diagnostic
retains original gates and reports per-parameter differences:

```bash
uv --no-config run --no-project --active python tools/diagnose_qwen3_rocm.py gradients
uv --no-config run --no-project --active python tools/diagnose_qwen3_rocm.py memory
```

A HIP invalid-argument error in vLLM memory discovery occurs before weight
refresh is tested. Check memory APIs in a fresh spawned process; increasing
cache size is not a fix for a failed memory-query API. Successful weight reload
also does not prove probability agreement across HF and vLLM.

Full repository GPU suites include architectures and dependencies outside the
dense Qwen3/LFM2 profile and currently have unresolved failures. Distributed
multi-rank correctness, all optimizer/resume paths and NVIDIA regression
coverage require separate checks.

## Experiment provenance

Commit source changes and the experiment's TOMLs before launching. Push to the
user's origin, check out that exact commit on the server, and record the commit,
submodule revisions and runtime manifest in the results. A server checkout is an
execution copy of this Git history, not a separate source of changes.

Keep logs, checkpoints, capture tensors and generated reports outside tracked
source. [The exp2 definition](experiments/lfm-lr-sweep-20261023/README.md) is a
post-run record of historical settings; it is not a retroactive execution SHA.
