# Non-Default RL Losses

Loss config lives in `train.toml`; advantage config lives in `orch.toml`.

## GRPO

```toml
# train.toml
[loss]
type = "grpo"
ratio_type = "token"
clip_eps = 0.2
beta = 0.0
```

```toml
# orch.toml
[advantage]
type = "grpo"
eps = 1e-6
```

Advantage: `(reward - group_mean) / (group_std + eps)`.

## MaxRL

```toml
# train.toml
[loss]
type = "max_rl"
ratio_type = "token"
clip_eps = 0.2
beta = 0.0
```

```toml
# orch.toml
[advantage]
type = "max_rl"
eps = 1e-6
```

Advantage: `(reward - group_mean) / (group_mean + eps)`.

## DPPO

```toml
# train.toml
[loss]
type = "dppo"
ratio_type = "token"
dppo_delta = 0.2
```

```toml
# orch.toml
[advantage]
type = "default"
```

DPPO filters token updates with the `dppo_delta` threshold. It does not have a separate DPPO advantage type in this repo.

## Notes

`grpo`, `max_rl`, and `dppo` require `ratio_type = "token"`. `beta > 0` is not supported on this branch because reference logprobs are not wired.
