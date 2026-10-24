# Prompt gradient capture

Gradient capture saves each prompt group's contribution to the complete
training objective on selected optimizer updates. Enable it in the RL TOML:

```toml
[trainer.gradient_capture]
steps = [1, 10, 50]
verify = false
```

The section is optional. Update numbers are one-based and must be unique,
positive, and no greater than `max_steps` when a maximum is configured.
Gradients are evaluated before clipping and the optimizer update. A capture
at update 1 describes the weights entering the first update.

Each sequence contributes a differentiable scalar containing all active loss
terms with exactly the training batch's normalization. Capture sums those
scalars for each `group_id` in a microbatch before calling
`torch.autograd.grad`. If a group spans microbatches, capture accumulates its
detached gradients on CPU while normal training completes each microbatch's
backward pass. Model weights stay fixed throughout that training update.
There is no extra division by rollout count or prompt count.

Only rollouts present in that update contribute. A group split across updates
produces separate gradient files evaluated at different parameter versions.
Prompt identity follows rollout group IDs; repeated text does not merge groups.
The environment is retained as provenance, and one group ID spanning different
environments is rejected.

The loss interface is independent of the chosen algorithm or policy-loss
formula: sequence contributions add up to the scalar used by training.
GRPO is selected explicitly in the demo with
`[orchestrator.train.algo] type = "grpo"`; `trainer.loss` still controls the
policy surrogate and may be changed independently.

## Supported execution

Capture uses the existing FSDP wrapper on one trainer GPU. Set `cp = 1`,
`ep = 1`, and `dp_replicate = 1`. Compilation, activation checkpointing,
activation offload, FSDP CPU offload, optimizer-in-backward offload,
quantization, and multimodal training are unsupported with capture enabled.
The supplied demo disables them explicitly where required. The trainer's
optimization and reduction dtype settings retain their usual defaults.

Saved tensors are CPU FP32. Conversion to FP32 does not recover precision
lost in BF16 compute. Expect small differences between the sum of separately
computed prompt gradients and a single backward through their summed loss.

The LFM2 adapter uses the Hugging Face backbone with a stateless PyTorch
short-convolution implementation. Each packed sequence is forwarded
independently, resetting attention and convolution history at its boundary.
The adapter disables inference caches, keeps native checkpoint names, and
feeds concatenated hidden states to the PrimeRL LM head. This favors a clear
correctness path for the small model; it does not claim optimized LFM2
throughput. The demo uses FA2 attention and the chunked LM head.

## Verification

Set `verify = true` for acceptance runs. Verification checks that gradient
capture leaves accumulated training gradients unchanged and that prompt
contributions reconstruct the ordinary pre-clipping batch gradient. Failures
raise errors rather than silently omitting a capture. `verify_rtol` defaults
to `0.05` and `verify_atol` to `1e-6` for mixed-precision comparisons.
The check is per parameter tensor, using the L2 norm:

```text
||sum(prompt_gradients) - batch_gradient||₂
    <= verify_atol * sqrt(numel) + verify_rtol * ||batch_gradient||₂
```

It is a norm comparison, not an elementwise tolerance check.

The focused tests cover compute-parameter identification, repeated prompt
derivatives followed by ordinary backward, multiple microbatches, two captured
optimizer updates and a subsequent uncaptured update against a baseline with
capture disabled. The LFM2 tests compare native-checkpoint logits and gradients
with Hugging Face, verify isolation between packed sequences, and exercise
tied embeddings and the chunked head under production FSDP wrapping.

Run commands and the two-update RL example are in
[examples/prompt-gradient-capture](../examples/prompt-gradient-capture/README.md).
The FSDP test is TOML-driven through
`tests/fixtures/gradient_capture/fsdp.toml`. Configuration validation and static
checks do not establish ROCm numerical correctness; run the GPU tests and
the real RL demo in the provisioned Linux environment.

## Artifacts

By default captures are written beneath the run directory:

```text
prompt_gradients/
  step_1/
    manifest.json
    prompt_000000.pt
    prompt_000001.pt
```

`trainer.gradient_capture.output_dir` overrides the capture directory. A
relative path is resolved under the trainer run directory. Each prompt file
contains named gradient tensors, the group and environment identity, trace
and branch provenance, update number, and normalization metadata. Unused
parameters have `None` entries. The manifest records the files and verification
results. Inspect a completed run with:

```bash
uv --no-config run --no-project --active --no-sync python \
    tools/check_prompt_gradients.py outputs/lfm25-prompt-gradients 1 2
```
