# Training And Eval Notes

## Checkpoints And HF

Use `--ckpt` to write local checkpoints. Add `--ckpt.interval N` to save during training instead of only at the end. HF-compatible model weights are written under `outputs/weights/step_<N>`.

Upload after training with:

```bash
uv run hf upload <user>/<repo> outputs/weights/step_<N>
```

For RL-only artifact uploads during training, use:

```bash
--trainer.hf-artifacts.repo-id <user>/<repo>
--trainer.hf-artifacts.interval N
```

This uploads weights, grads, and deltas under `rl_steps/step_<N>`; it is separate from normal resumable checkpoints.

## Eval Top Logprobs

Eval already supports per-token top-logprobs. Configure:

```toml
[save.token_metadata]
enabled = true
top_logprobs = 5
max_examples = 16
seed = 2001
```

This saves locally as a HuggingFace Dataset, not to W&B or TensorBoard. `max_examples` keeps token metadata for a deterministic prompt subset while eval metrics still use all completions. If `path` is unset, output is step-scoped:

```text
<output_dir>/evals/step_<ckpt_step>/<env_name_or_id>/token_metadata
```

Do not set a fixed `path` for multi-step evals unless the code is changed to include the step; a custom path is reused for every evaluated checkpoint.

## RL Sample Logging

RL trajectory/sample logging happens in the orchestrator from training rollouts. It logs min-length, max-length, and random problem groups every `wandb.log_extras.interval`.

Configure in the orchestrator config:

```toml
[wandb.log_extras]
samples = true
distributions = true
interval = 50
```

Training rollouts currently log sampled-token logprobs, but not top-5 candidate logprobs. Adding RL top-logprobs is feasible, but should be opt-in because `top_logprobs` must be requested for all generated training rollouts, even though only a few samples are logged.
