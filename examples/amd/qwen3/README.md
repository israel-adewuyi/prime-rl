# Dense Qwen3 on AMD MI355X

This is an experimental ROCm profile for full-parameter Qwen3-0.6B, 1.7B, 4B,
and 8B training. Hardware acceptance tests must pass before using it for research
runs. It uses FA2, AdamW, activation checkpointing, and filesystem policy updates.
Compilation, parameter fusions, activation/optimizer offload, quantization, LoRA,
and context/expert parallelism are disabled in the supplied configs. The model's
optimization and reduction dtype defaults remain in effect.

## Validation scope

Use the acceptance checks below on the actual ROCm runtime. CPU import and
configuration checks do not establish kernel correctness or training stability.
The strict BF16 cross-backend gradient comparison remains an unresolved gate;
do not relax its tolerances to make a run pass. See
[the AMD guide](../../../porting_to_amd.md) for the supported scope and diagnostics.

## Installation

Use Linux x86_64 with Python 3.12 and an MI355X-compatible ROCm driver/runtime.
Use a dedicated environment (for example `.venv-rocm`), not the CUDA workspace
environment. Activate it before running the following commands.

Provision **ROCm PyTorch 2.13.0** first. For RL, also provision a **ROCm build of
vLLM 0.30.0 compiled against that PyTorch runtime**. A matching provider image or
source build is preferable to mixing wheels from incompatible runtimes. Refer to
[PyTorch's ROCm packages](https://download.pytorch.org/whl/rocm7.2/torch/) and the
[vLLM 0.30.0 ROCm installation instructions](https://docs.vllm.ai/en/v0.30.0/getting_started/installation/gpu/).
These are candidate runtime versions matching this checkout, not a claim of a
hardware-tested image. Do not substitute a newer vLLM without checking this
repository's worker and server extensions.

Initialize the pinned submodules, then install the profile:

```bash
git submodule update --init --recursive
bash scripts/install_rocm.sh sft
# For a runtime that also contains ROCm vLLM:
bash scripts/install_rocm.sh rl
```

The installer checks the runtime, constrains its installed package versions, and
builds FlashAttention 2.8.3 with its Triton AMD backend. It installs the local
Python packages with `uv pip`, bypassing the root workspace's CUDA indexes and
wheel overrides. It fails on dependency conflicts instead of replacing the
runtime. The installed package list and initial runtime manifest are saved under
`$VIRTUAL_ENV/prime-rl-runtime/`; keep these together with the container digest,
driver version, repository commit, and submodule revisions for reproducibility.

Use `uv run --no-project --active --no-sync` for this environment. A root
`uv sync` uses the CUDA sources in the workspace lock and is
**not** the ROCm installation procedure. The separate dependency profile is
`environments/rocm/pyproject.toml`. Do not run root `uv sync --all-extras`
inside the ROCm environment.

## Acceptance tests

From the repository root, with the ROCm environment activated:

```bash
# Validate example schemas on Linux before allocating GPUs.
# Includes RL configs: requires reverse-text, installed by the rl setup mode.
uv run --no-project --active --no-sync python -m pytest -q tests/unit/test_configs.py -k amd

# CPU-capable boundary checks, followed by real FA2 forward/backward, sequence
# isolation, and optimizer-state resume checks on the GPU. No checkpoint download.
uv run --no-project --active --no-sync python -m pytest -q tests/unit/train/models/test_qwen3_portability.py
uv run --no-project --active --no-sync python -m pytest -q tests/unit/utils/test_gpu_visibility.py

# Requires vLLM. Compares prompt logprobs, reloads changed weights, then restores
# the original policy and verifies the served logprobs again. One GPU.
uv run --no-project --active --no-sync python -m pytest -q tests/unit/inference/test_qwen3_weight_refresh.py
```

Both GPU test files also run on NVIDIA with FA2, providing a regression gate for
the shared code. Missing FlashAttention/vLLM is a failure on a GPU machine;
GPU cases skip only when no GPU is available.

The optimizer-resume unit test checks model/optimizer state directly. The following
SFT runs separately exercise the real trainer, FSDP, and distributed checkpoint
manager with deterministic synthetic packed data:

```bash
uv run --no-project --active --no-sync sft @ examples/amd/qwen3/sft.toml --output-dir outputs/amd --run.name sft-06b
uv run --no-project --active --no-sync sft @ examples/amd/qwen3/sft.toml --output-dir outputs/amd --run.name sft-06b --resume --max-steps 15
```

Check finite losses/gradient norms, an increasing optimizer step, checkpoint
creation at steps 5/10, and continuation from step 10 on resume. The GPU unit
test checks resumed versus uninterrupted parameter equality; a successful trainer
resume alone does not establish that equality for the entire data pipeline.

Repeat the initial ten-step run using one model overlay at a time:

```bash
uv run --no-project --active --no-sync sft @ examples/amd/qwen3/sft.toml @ examples/amd/qwen3/1.7b.toml --output-dir outputs/amd --run.name sft-17b
uv run --no-project --active --no-sync sft @ examples/amd/qwen3/sft.toml @ examples/amd/qwen3/4b.toml --output-dir outputs/amd --run.name sft-4b
uv run --no-project --active --no-sync sft @ examples/amd/qwen3/sft.toml @ examples/amd/qwen3/8b.toml --output-dir outputs/amd --run.name sft-8b
```

## End-to-end RL

The RL example requires two GPUs: one for training and one for inference. It uses
the reverse-text environment as a pipeline smoke test; reward improvement is not
an acceptance criterion for this short run. Select the two allocated GPUs using
`ROCR_VISIBLE_DEVICES` before launching. The launcher partitions HIP-visible GPU
ordinals and sets both HIP and CUDA visibility consistently for each component.

```bash
uv run --no-project --active --no-sync rl @ examples/amd/qwen3/rl.toml --output-dir outputs/amd --run.name rl-06b
```

Require multiple rollout/training/weight-refresh cycles without hangs, and verify
inference acknowledges successive policy versions. Repeat with the model overlays
after 0.6B passes. For distributed trainer validation, increase
`--deployment.num-train-gpus 2` (RL then needs a third GPU for inference).

Record peak allocated/reserved GPU memory, training tokens/s, rollout tokens/s,
weight-refresh time, sequence length, and batch size for each model. The examples
use 512-token SFT and 1024-token RL contexts; they do not establish long-context
capacity. MI355X performance and maximum batch sizes must be measured on hardware.
MFU uses the dense BF16 peak from the
[AMD MI355X product brief](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf).

## Scope

Other architectures, GPU-direct weight transfer, disaggregated inference, and
quantization are outside this initial ROCm profile. Selecting an unsupported
architecture or backend fails explicitly. Optional dependencies are loaded when
their feature is selected; missing dependencies are not silently ignored.

ROCm retains the `torch.cuda` API and the distributed `nccl` backend name. NVML
NUMA pinning and NVIDIA-only NVLink transport heuristics are skipped on ROCm;
the process keeps the affinity assigned by its launcher or scheduler.
