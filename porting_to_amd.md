# Porting PrimeRL to AMD ROCm

## Scope and evidence

This checkout's initial ROCm target is dense Qwen3 training with FlashAttention 2 and vLLM inference with filesystem weight updates. Importing PrimeRL successfully does not establish that every architecture, CUDA extension, distributed strategy, or weight transport works on AMD.

On the National Compute MI355X workspace, the installer completed, all 297 installed packages passed `uv pip check`, the trainer and optimizer imported, and a tiny custom PrimeRL Qwen3 model completed a BF16 FA2 forward/backward pass with finite gradients. The standard lock passed `uv lock --check --offline` after updating its project dependency metadata.

Two stronger user-run gates remain failing: BF16 packed-gradient agreement against Hugging Face SDPA, and vLLM filesystem weight refresh initialization. The tiny smoke pass does **not** prove numerical equivalence or successful end-to-end RL. The follow-up changes below were prepared locally after the authorized GPU session expired; they have not been verified on a GPU.

## Runtime and repositories

The PrimeRL environment used Python 3.12.14, PyTorch `2.13.0+rocm7.2`, HIP `7.2.53211`, Transformers `5.19.0`, FlashAttention `2.8.3`, and an editable ROCm vLLM reporting `0.30.0+rocm723`. Its source was `/mnt/shared/vllm-rocm`, revision `ced6857`. Package version strings alone do not identify the actual source or binary compatibility; record the Git revision, wheel provenance, compiler/toolchain, and runtime manifest together.

The earlier DeepSeek serving experiments used a different vLLM source/runtime: revision `1a001d584244bd21829a5e5bd4a376ea141c0ce3`, version `0.30.1rc1.dev616+g1a001d584.rocm723`, and image digest `sha256:596db6504233a5d8575bdc0f6c4c8b245bd7f496bb4024705208081923afb9b3`. Do not conflate that runtime with the PrimeRL environment.

Shared paths:

- `/mnt/shared/prime-rl`: copied local PrimeRL checkout, initially branch `feat/amd-qwen3`, HEAD `ced491fe8960c4437b45a668a4858e08eb60bec8`.
- `/mnt/shared/nc-vllm`: DeepSeek setup, source, weights, profiles, and results.
- `/mnt/shared/nanotitan-dev`: sibling workspace.
- `/mnt/shared/prime-rl/.venv-rocm`: dedicated PrimeRL ROCm environment.

Use `/mnt/shared` for durable work. Container-local `/tmp`, processes, and terminal connections are disposable. A restart preserves shared files but interrupts running work. A localhost code-server port-forward provides browser access only while its tunnel and container are running.

## Dependency resolution

The original installer froze **every installed package** into constraints. That forced Transformers `5.19.0` alongside PrimeRL's exact `5.6.2` requirement and froze OpenAI `3.26.1` alongside verifiers' `<3` requirement. An existing container environment is not a compatible application lock.

The corrected `tools/check_rocm_runtime.py` records the complete runtime in `runtime.json` but constrains the compiled GPU stack and provisioned Transformers version. Python application dependencies can resolve together. In RL mode, `scripts/install_rocm.sh` explicitly includes vLLM in the solve so its transitive requirements participate too. This also resolved Prime CLI's `typer<0.26` versus FastAPI Cloud CLI's `typer>=0.26.1`, selecting FastAPI Cloud CLI `0.24.0`.

PrimeRL's declared Transformers requirement is now `>=5.6.2,<6`. The normal uv profile retains its `5.6.2` override and existing locked version; ROCm installation preserves provisioned `5.19.0`. This is not a claim that every Transformers 5.x release has been validated. The provisioned vLLM requires `transformers>=5.10.4`, so forcing `5.6.2` into this environment is incompatible. See [vLLM 0.30.0 requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/requirements/common.txt).

Do not use the CUDA `[gpu]`/`[all]` extras to construct this ROCm environment: the standard project sources and extras include CUDA wheels and NVIDIA-specific libraries. Provision matching ROCm PyTorch and ROCm vLLM first, then:

```bash
cd /mnt/shared/prime-rl
source .venv-rocm/bin/activate
bash scripts/install_rocm.sh rl
uv --no-config pip check --python "$VIRTUAL_ENV/bin/python"
```

Do not patch files inside site-packages. Keep fixes in source, dependency profiles, or a pinned upstream fork. The runtime manifest, installed package list, and constraints live under `$VIRTUAL_ENV/prime-rl-runtime`. The successful install log was saved as `/mnt/shared/prime-rl-rocm-install.log` and copied locally into the vLLM setup's ignored `.local` directory.

## AMD backend selection

PyTorch ROCm deliberately uses `torch.cuda`, `device="cuda"`, and many CUDA-labelled messages. Those names do not mean an NVIDIA GPU is present. Check `torch.version.hip`, actual device properties, and the vLLM platform backend. Distributed PyTorch's `nccl` backend name also appears on ROCm, where communication uses RCCL.

For this training profile, select `flash_attention_2` and enable `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` before importing FlashAttention. The installer builds FA2 for the AMD Triton implementation. FA3/FA4 and CUDA-specific Quack kernels are not interchangeable AMD backends. Optional libraries should load lazily so dense Qwen3 imports do not require unrelated MoE, NVIDIA, FP8, or transport kernels. See the [FlashAttention AMD instructions](https://github.com/Dao-AILab/flash-attention#amd-rocm-support).

Do not change `optimization_dtype` or `reduce_dtype` to make a test pass. BF16 forward correctness, finite gradients, gradient equivalence, optimizer behavior, and checkpoint resume are separate gates.

## Failure: packed Qwen3 gradients

The reported run passed logits and cross-entropy comparisons, then failed the gradient comparison at `model.embed_tokens.weight` for both tied and untied embeddings. This is the first parameter examined, not proof that only embeddings differ. The failing test never reached its document-isolation check. Optimizer/checkpoint resume passed separately.

The reference uses two separate HF SDPA document forwards; PrimeRL uses one packed FA2 forward. Different attention kernels, BF16 rounding, GEMM shapes, and gradient accumulation orders can produce differences. A varlen/GQA backward bug is another possibility. Both tie settings failing means weight tying alone is insufficient as an explanation. The current traceback hides discrepancy details because its assertion message is only the parameter name.

The test now retains the same tolerances but includes PyTorch's full mismatch report. `tools/diagnose_qwen3_rocm.py` reports per-parameter maximum error, relative L2 error, cosine similarity, failing-element counts, finite status, packed versus separate logits, and document isolation. Run:

```bash
uv --no-config run --no-project --active python tools/diagnose_qwen3_rocm.py gradients > gradients-default.jsonl
uv --no-config run --no-project --active python tools/diagnose_qwen3_rocm.py gradients --reference-backend math > gradients-math.jsonl
uv --no-config run --no-project --active pytest -q tests/unit/train/models/test_qwen3_portability.py
```

Next steps depend on the evidence:

1. Compare SDPA default versus explicitly selected math reference; a materially changed result implicates reference-backend numerics.
2. Compare packed FA2 against separate FA2 forwards and backwards with identical weights/tokens; this isolates packing from HF differences. The diagnostic currently compares their forward logits; a backward comparison remains to be run.
3. Reduce to attention-only Q/K/V gradients for lengths `[9,15]`, four query heads, two KV heads, head dimension 128, causal masking, and zero dropout. Compare against an FP32 mathematical reference.
4. If backward is incorrect, validate a pinned AMD FlashAttention/AITER fix or use a correctness-first attention backend with explicit packed document boundaries. An ordinary causal mask over all 24 tokens would leak across documents.
5. Only adjust a BF16-specific tolerance after measuring reproducible numerical error across parameters and seeds against a stronger reference. Do not broadly loosen tolerances or skip the failing gate.

No gradient correctness fix is claimed yet. HF's current Qwen3 RMSNorm formula matches the inspected PrimeRL fallback; there is no source evidence here for changing RMSNorm merely to address the failure. [HF Qwen3 5.19.0 source](https://raw.githubusercontent.com/huggingface/transformers/v5.19.0/src/transformers/models/qwen3/modeling_qwen3.py).

## Failure: vLLM memory snapshot initialization

The important exception is `hipErrorInvalidValue` from `torch.accelerator.get_memory_info(device)` in vLLM's initial `MemorySnapshot`. It happens before model loading and before any weight-refresh RPC. This run therefore says nothing yet about filesystem reload correctness. Increasing `gpu_memory_utilization`, disabling prefix caching, or changing checkpoints does not address this specific API failure.

The test also reports Python using `fork` from a multithreaded parent. GPU runtime state should not be inherited that way. The prepared test change sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` before GPU discovery. Explicit `spawn` is the first fix to verify:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn uv --no-config run --no-project --active pytest -q tests/unit/inference/test_qwen3_weight_refresh.py
uv --no-config run --no-project --active python tools/diagnose_qwen3_rocm.py memory
```

The memory probe compares `torch.cuda.mem_get_info` and `torch.accelerator.get_memory_info` in a fresh process after device initialization. If both fail, investigate device visibility and the PyTorch/HIP/driver combination. If only the accelerator API fails, a scoped ROCm fallback to `torch.cuda.mem_get_info` in the vLLM source, or an upstream PyTorch fix, is a candidate. Probe first: that fallback is not yet implemented or validated. Apply any fallback consistently to other memory probes too, and preserve device selection and `(free,total)` units. Do not swallow arbitrary accelerator errors.

The test must eventually verify original prompt log-probabilities, reload the zeroed LM head, observe uniform log-probabilities, reload original weights, and recover the original result. Startup alone is insufficient.

## Warnings and verification workflow

`--no-sync` has no effect with `--no-project`; omit it. `--no-config` avoids accidentally discovering the normal CUDA uv profile. `--active` uses the activated ROCm environment.

The dangling `~orch-...dist-info` directory warning did not block the successful installer or GPU smoke test. Inspect it as an installation remnant before any cleanup; do not remove the real torch distribution. Triton autotuner and `torch.jit.script_method` deprecation warnings are separate from the failing assertions and memory API exception.

Record exact runtime/source versions, seed, test command, environment flags, backend, full exception, and numerical diagnostics. Proceed from imports and dependency checks to isolated kernels, packed model equivalence/isolation, optimizer resume, inference log-probabilities and reloads, then a small RL job. Multi-GPU correctness and performance need their own gates.

## DeepSeek inference, parallelism, benchmarks, and profiling

The earlier DeepSeek V4 Flash serving runs used `deepseek-ai/DeepSeek-V4-Flash-0731`, model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`. That work does not validate PrimeRL's DeepSeek training path on AMD.

TP8 used tensor parallelism across eight GPUs, with DP1, PP1, and expert parallelism disabled. All 43 layers ran on every rank. Attention's 64 heads were divided into eight heads per rank; each rank retained the 256 experts with tensor-sharded intermediate projections (2048 to 256 per rank). Some projections were replicated. RCCL/AITER collectives and replicated components mean memory and latency do not simply scale as one eighth.

Standalone TP1 and TP8 eager profiles were collected for 1024-token prefill/one output token and prefill plus 16 output tokens. TP1 is a whole-model single-GPU run; rank 0 of TP8 is one shard of an eight-GPU execution. Those are different comparisons. GPU-only merged traces help inspect concurrent work; raw per-rank CPU/GPU traces preserve launch and synchronization context. A single finite-gradient smoke test is a much weaker result than numerical parity.

Benchmark requests covered TTFT, inter-token latency, throughput, concurrency, and context-length behavior; all 12 tested points completed without request failures (2064 requests). These are measurements for those exact workloads, not blanket performance guarantees. Keep warmup/compilation separate, report workload/input/output lengths and latency percentiles, and distinguish aggregate throughput from per-request latency. Eager traces do not represent compiled production throughput. No performance optimization was validated in those runs.

Local artifacts are under `vllm-national-compute/national-compute/results/20261003T220926Z` and `results/tp-comparison-20261003T224711Z` relative to the sibling local checkout. Large `kubectl cp` transfers were unreliable; 2 MiB archive chunks followed by SHA-256 verification worked. Copies must preserve Git/submodule state and relevant untracked files without carrying credentials or virtual environments.

The cluster/token used here exposes AMD MI355X capacity; NVIDIA requires separately granted access. Allocation was for a full eight-GPU node. A workspace stop ends its processes; provider release/billing follows cluster policy. Do not extend a timed rental merely to finish a test without spend authorization.

## October 13, 2026: verified source fixes and training runs

This section supersedes the unresolved October 9 status above; that evidence is retained as history. The transferred branch remained `feat/amd-qwen3`, HEAD `ced491fe8960c4437b45a668a4858e08eb60bec8`, including the user's six initially modified tracked files and the untracked diagnostic/handoff. No commit/reset or whole-tree synchronization was performed. Initial manifests, original patch, transfer SHA-256, submodule revisions, exact commands, failed attempts and final patches are in `/mnt/shared/.research-session/20261013/primerl`. The local research-manager notebook is `../research_manager/prime-rl.md`, with downloaded evidence under `../research_manager/.nc-session/primerl/artifacts/primerl` and a continuously mirrored `prime-rl-terminal.txt`.

### Runtime distinction

Dedicated SFT environment `/mnt/shared/prime-rl/.venv-rocm`: Python 3.12.12, official Torch `2.13.0+rocm7.2`, HIP `7.2.53211`, Triton-ROCm 3.7.1, Transformers 5.19.0, FlashAttention 2.8.3. AMD Triton FA2 was enabled with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`.

Inference and actual RL used an explicitly separate, coherent image-derived environment `/mnt/shared/prime-rl/.venv-image-rl`, inheriting compiled packages from pinned image `vllm/vllm-openai-rocm@sha256:596db6504233a5d8575bdc0f6c4c8b245bd7f496bb4024705208081923afb9b3`: Python 3.12.14, glibc 2.35, Torch `2.13.0+git733fca1`, HIP `7.2.53211`, Triton `3.8.0+git669b31ac`, Transformers 5.17.0, FA2 2.8.3, vLLM `0.30.1rc1.dev616+g1a001d584.rocm723`, AMD AITER 0.1.23. The application overlay's actual combined environment passed `python -m pip check`; installed application versions and exact installation commands are preserved. `PYTHON_EXEC=$VIRTUAL_ENV/bin/python` was necessary for inherited torchrun to launch the overlay interpreter.

The strict vLLM 0.30.0 target remains **blocked**. Source `ced6857afa0ea7b2e3f0846a62e1394e90f15607` was restored at `/mnt/shared/vllm-rocm`; the official wheel requires glibc 2.39 and a different Torch build. It was not installed into glibc 2.35 or relabeled as compatible. The strict runtime checker remains unchanged and rejects the newer image. Successful image-derived experiments do not imply the strict installer profile is reproducible yet.

### Demonstrated causes and fixes

1. **Packed FA2 backward depended on Q/K layout.** Original unchanged portability tests reproduced two failures and three passes. Packed forward logits/loss and document isolation agreed; default and math references both showed large Q/K model-gradient errors. Contiguous attention-only inputs matched packed/separate FA2 gradients exactly. Actual post-RoPE Q/K strides `[128,3072,1]`, with identical values, produced large attention-backward errors (packed V versus FP32 relative L2 1.27758, cosine 0.18392). Seeds 123, 42 and 321 were recorded. The source integration fix normalizes Q/K/V with `.contiguous()` at the existing ROCm FA2 call boundary. No kernels were written or changed.
2. **Random variable-length fake data had inconsistent field lengths.** The first SFT attempt reaching loss had 512 targets versus a 345-element mask. Slicing already-drawn random IDs to `seq_len + 1` fixes alignment while preserving RNG consumption and checkpoint replay. The existing replay test now checks all field lengths and causal shifts.
3. **Application dependencies had moved.** The ROCm profile now pins W&B 0.27.0 and wandb-workspaces 0.4.3, matching the main lock and avoiding the removed `wandb_gql` import. The RL installer declares the repository's exact router 0.2.2 wheel. Actual installation and combined-environment dependency checking passed; this does not clear the strict vLLM binary blocker.
4. **Newer vLLM moved a serving flag.** Real `/generate` requests initially failed HTTP 500 because `ModelConfig.enable_return_routed_experts` moved to `VllmConfig.aux_output_config`. The handler selects the new config when present and otherwise the legacy config, then reads the flag directly. The async handler regression checks old/new layouts with the flag true/false, parent delegation and response preservation. Before the fix: two failures and five passes; after: seven passes. NIXL/disaggregated-only compatibility remains outside this ROCm scope.
5. **The short rollout fixture exhausted its token budget thinking.** All 3,616 persisted episodes in `rl-06b-image-r3` contained reasoning only, no answer tags/content, `finish_reason=length`, exactly 128 completion tokens and zero LCS reward. The AMD RL example now selects the existing Qwen3 renderer with `enable_thinking=false`. The real GRPO algorithm, LCS reward, token budget and numerical settings are unchanged; no debug algorithm was substituted.

### Numerical gates and SFT

The unchanged portability test passed **5/5** after the FA2 layout fix. Across all 49 recorded parameters there were zero elements outside the unchanged combined tolerances: maximum relative L2 was 0.016674 against default HF, 0.015401 against math HF, and 0.003617 for packed/separate FA2. Both document-isolation checks were exactly zero. Random-data/replay and SFT-loss tests passed **17/17**. Source lint/format, Bash syntax and Git whitespace checks passed.

Fresh-process `torch.cuda.mem_get_info` and `torch.accelerator.get_memory_info` both passed in both environments; the earlier HIP invalid-argument failure was not reproduced, and no memory-API fallback was added. Explicit-spawn inference verified original, changed and restored log-probabilities. Five additional changed/restored cycles completed ten reload RPCs, recovering original log-probabilities exactly each time. In the coherent image environment, the unchanged five training gates plus inference refresh gate passed **6/6**.

| Model | SFT evidence | Final loss | Peak allocated GiB |
| --- | --- | ---: | ---: |
| Qwen3-0.6B | 10 steps, checkpoint resume from 10 through 15 | 12.0351 at step 15 | 11.7 on resume |
| Qwen3-1.7B | 10 steps; checkpoints 5/10 verified | 12.119416 | 29.3066 |
| Qwen3-4B | 10 steps; checkpoints 5/10 verified | 12.006597 | 65.2051 |
| Qwen3-8B | 10 steps; checkpoints 5/10 verified | 12.032191 | 129.09375 |

Every actual optimizer scalar in the DCP checkpoints matched its checkpoint number (310 scalars for 0.6B/1.7B, 398 for 4B, 399 for 8B). All training losses/gradient norms were finite. Larger models were not resumed; only 0.6B has resume-to-15 evidence. These synthetic SFT runs establish mechanics/correctness under these workloads, not task quality or convergence. The GPU device name was blank/generic, making the trainer's A100-derived MFU estimate invalid; do not report its MFU as MI355X performance.

### Real RL and evidence limits

Actual reverse-text GRPO runs `rl-06b-image-r4`, `rl-1.7b-image` and `rl-4b-image` each completed ten optimizer steps, with finite losses and positive finite gradient norms, successful trainer/orchestrator completion, and **11 successful weight-update RPCs** (startup plus ten updates). Actual DCP optimizer scalars at steps 5/10 were verified, and sampled first-layer Q-projection weights changed between checkpoints. Final losses/norms were 0.0034/4.6911, 0.0275/2.1335 and -0.0146/10.1047 respectively. Root-run `rl-8b-image-root` also completed ten real steps on GPUs 6/7 at 14:34 UTC, with 11 successful update RPCs, all 399 optimizer scalars correct at checkpoints 5/10, and 16,771,096/16,777,216 sampled Q-weight elements changed (max 5.04963e-6). Its final loss/norm were 0.0696/2.8536, peak allocated memory 129.7 GiB. All four model sizes therefore passed this short real RL pipeline; 8B retained the documented shutdown-only EngineDeadError.

Final policy v10 application is supported by successful RPCs plus source/control-flow evidence: final-batch draining awaits `wait_for_version(10)`; the watcher advances only after the awaited update succeeds. Logs show normal draining/completion within seconds and no 1200-second timeout warning. Do not invent an explicit v10 INFO acknowledgement where one is absent.

All configs used FA2, AdamW, no compilation/quantization/LoRA, cp=1, ep=1, filesystem weight updates and unchanged optimization/reduction dtypes. Each RL job had one trainer GPU and one inference GPU; this is not a distributed multi-rank trainer validation. A larger run's copied base `rl.toml` still names 0.6B: its additional size overlay, resolved trainer/orchestrator/inference JSON and actual model-loading log establish the effective model.

Preserved caveats: FSDP output-view warnings; the existing singleton unbiased `loss/std` statistic is non-finite and dropped while actual loss/gradient norms are finite; some successful runs emit `EngineDeadError` during engine teardown after training/orchestrator completion. No long-run convergence, production throughput, all-feature support, NVIDIA regression, or strict vLLM 0.30.0 compatibility is claimed.

Reproduction uses the recorded scripts `run-image-rl.sh`, `run-rl-scaling.sh`, `collect-rl-evidence.py`, `verify-layout-fix.sh`, and SFT runners in the artifact directory. Always activate the matching environment and invoke through `uv --no-config run --no-project --active`; avoid root CUDA uv synchronization. GPU authorization ended at the supplied October 13 session deadline; re-running requires new authorized compute.

### Bounded two-GPU inference follow-up

At the user's additional request, `rl-8b-dp2-image` used two independent 8B inference replicas (physical GPUs 4/5, DP=2, TP=1) and one trainer GPU (6). Both EngineCore_DP0 and EngineCore_DP1 completed model/KV-cache initialization and pause/resume cycles. The router logged 272 finished sessions on replica `@0` and 275 on `@1`; these are observations of traffic to both replicas, not a performance or load-balancing benchmark.

The run completed two real optimizer steps (loss/norm -0.0054/6.7762 and 0.0129/3.1158), both finite with positive gradients. Startup policy and policy v1 each received a successful update RPC, and the dispatcher resumed v1. The authorized time bound sent SIGTERM at 14:43:50 UTC, before the unchanged 14:44:22 work deadline. The run did not reach checkpoint interval 5 or ten-step completion, so **DP=2 remains partially validated**, with no checkpoint/resume claim. `rl-8b-dp2-image-partial-evidence.json` and full component logs/configs/metrics preserve this result, including intentional shutdown traces. Collection afterward ran only on the existing CPU staging pod; no GPU extension was requested or made.
