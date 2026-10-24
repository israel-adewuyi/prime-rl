# Rollout gradient statistics and checkpoint updates

The trainer can measure variation between rollout gradients on selected
updates and save the batch gradient and actual parameter update with normal
checkpoints. The two schedules are independent:

```toml
[trainer.gradient_capture]
steps = [1, 10, 50]
verify = false

[ckpt]
interval = 10
save_gradients_and_updates = true
```

Omit `trainer.gradient_capture` to disable rollout statistics.
`save_gradients_and_updates` defaults to `false`; enabling it adds tensors
to checkpoints of completed optimizer updates, including the final checkpoint. Statistics
steps are one-based, unique, positive, and no greater than `max_steps` when
a maximum is configured. A statistic at update 1 describes the weights
entering the first update.

## Statistics

Each prepared rollout sequence contributes a differentiable scalar containing
the complete training objective with exactly the training batch's
normalization. `torch.autograd.grad` computes its parameter gradient before
normal backward, clipping, and the optimizer update. The existing microbatch
schedule is retained. Sequences are not regrouped by prompt: in the single-turn
workflow used here, each sequence is one rollout. There is no extra division
of each rollout loss by rollout count.

At each selected update, the trainer accumulates a CPU FP64 gradient sum `S`,
the sum of squared gradient norms `Q`, and the rollout count `n`. It reports:

```text
S = sum(g_i)
Q = sum(||g_i||^2)
mu = S / n
V = (Q - ||S||^2 / n) / (n - 1)
```

| Metric | Meaning |
| --- | --- |
| `snr/count` | Number of rollout gradients in this update, `n`. |
| `snr/mean_norm` | Norm of the mean rollout gradient, `||mu||`. |
| `snr/variance` | Sample gradient variance `V`, the trace of the sample covariance. |
| `snr/value` | Power SNR, `||mu||^2 / V`. |

Variance is undefined for fewer than two rollouts and is omitted. Values
within the FP64 roundoff bound are treated as zero. The ratio
is omitted when variance is undefined or zero. These statistics reset for
each selected update; they are not averages across training updates. Real
sequences with trace identity count even when their gradient is zero.
Synthetic padding sequences with no trace identity and no active loss do not.

Scalars go to the ordinary trainer monitors. With the default file monitor,
read `monitors/file/metrics.jsonl` beneath the run directory: trainer rows
have `"producer": "trainer"`, an integer `step`, and flat `snr/...` keys.
An enabled W&B monitor receives the same metrics. Individual rollout gradient
files are not written; the running accumulation uses memory proportional to
one model gradient, rather than the number of rollouts. The FP64 sum for
230M parameters takes approximately 1.84 GB of CPU memory.

The loss interface is independent of the algorithm or policy surrogate.
Sequence contributions add up to the scalar used by training. The demo sets
`[orchestrator.train.algo] type = "grpo"` explicitly and keeps the repository's
default IPO policy surrogate. `trainer.loss` can be changed independently.

## Checkpoint tensors

With `save_gradients_and_updates = true`, each normal trainer checkpoint has:

```text
checkpoints/step_10/trainer/gradient_update.pt
```

The file contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | Artifact schema version, currently `1`. |
| `step` | Completed optimizer update represented by the checkpoint. |
| `batch_gradients` | Named raw accumulated batch gradients, saved before clipping; unused parameters have `None`. |
| `applied_updates` | Named parameter differences, `parameter_after - parameter_before`. |
| `parameter_metadata` | Each parameter's shape, original `parameter_dtype`, and `gradient_dtype`. |
| `gradients_before_clipping` | `true`. |
| `update_definition` | `"after_minus_before"`. |

Tensors are stored on CPU in FP32, preserving FP64 when the source is FP64.
An applied update includes the effects of the optimizer, its state, weight
decay, and gradient clipping; it is not merely the negative learning rate
times the raw gradient. These files follow the normal checkpoint directory
and retention policy. Checkpoint tensor saving does not require a statistics
capture at the same step.
Checkpoint tensor saving applies to RL with one trainer GPU and requires
optimizer-in-backward offload to be disabled.

## Verification and supported execution

Set `verify = true` for statistics acceptance runs. Verification checks that
capture leaves accumulated optimizer gradients unchanged and that the sum
of rollout contributions reconstructs the pre-clipping batch gradient.
Failures raise errors. `verify_rtol` defaults to `0.05` and `verify_atol` to
`1e-6`; the check uses each parameter tensor's L2 norm:

```text
||S - batch_gradient||
    <= verify_atol * sqrt(numel) + verify_rtol * ||batch_gradient||
```

Capture uses the existing FSDP wrapper on one trainer GPU with `cp = 1`,
`ep = 1`, and `dp_replicate = 1`. Compilation, activation checkpointing,
activation offload, FSDP CPU offload, optimizer-in-backward offload,
quantization, and multimodal training are unsupported with capture enabled.
The optimization and reduction dtype settings retain their usual defaults.
FP64 accumulation does not recover precision lost in BF16 computation.
Separate backward traversals can introduce small floating-point differences.

The LFM2 adapter resets attention and convolution history at packed-sequence
boundaries, uses stateless PyTorch convolution, disables inference caches, and
feeds concatenated hidden states to the PrimeRL head. The demo uses FA2 and
the chunked head. It is a correctness-oriented path, not a throughput claim.

The tests cover explicit-gradient versus streaming statistics, FSDP compute
parameter identification, repeated extraction followed by ordinary backward,
captured and uncaptured updates against ordinary training, and checkpoint
gradient/update tensor values. The LFM2 tests also cover HF equivalence,
packed-sequence isolation, tied embeddings, and the chunked head.

See the [demo and GPU commands](../examples/prompt-gradient-capture/README.md).
Inspect its completed run with:

```bash
uv --no-config run --no-project --active --no-sync python \
    tools/check_gradient_capture.py outputs/lfm25-gradient-stats 1 2
```

The checker validates the expected statistics steps in normal metric logs
and any saved checkpoint gradient/update tensors. For a run that only enables
checkpoint tensors, omit the expected statistics steps. GPU numerical correctness
requires running the tests and demo in the provisioned Linux environment.
