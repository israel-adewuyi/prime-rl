from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from prime_rl.trainer.rl.config import LossConfig


@dataclass(frozen=True)
class SequenceLossState:
    trainer_logprobs: Tensor
    log_importance_ratio: Tensor
    token_importance_ratio: Tensor
    sequence_importance_ratio: Tensor
    token_mismatch_kl: Tensor
    geo_seq_ratio: Tensor
    token_mask_low: Tensor
    token_mask_high: Tensor
    geo_mask_low: Tensor
    geo_mask_high: Tensor
    seq_mask_low: Tensor
    seq_mask_high: Tensor
    is_masked: Tensor
    loss_mask: Tensor
    scaled_advantages: Tensor


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def _as_metric_tensor(value: Tensor | float | bool) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().float().reshape(-1)
    return torch.tensor([float(value)], dtype=torch.float32)


def _aggregate_metrics(metric_lists: dict[str, list[Tensor]]) -> dict[str, Tensor]:
    return {key: torch.cat(values) for key, values in metric_lists.items() if values}


def build_sequence_loss_state(
    trainer_logprobs: Tensor,
    inference_logprobs: Tensor,
    advantages: Tensor,
    loss_mask: Tensor,
    loss_config: LossConfig,
) -> SequenceLossState:
    log_importance_ratio = trainer_logprobs - inference_logprobs
    token_importance_ratio = torch.exp(log_importance_ratio)
    geo_seq_ratio = torch.exp(_safe_mean(log_importance_ratio, loss_mask))
    token_mismatch_kl = token_importance_ratio - log_importance_ratio - 1

    seq_log_importance_ratio = torch.clamp(log_importance_ratio[loss_mask].sum().detach(), max=10.0)
    sequence_importance_ratio = torch.clamp(torch.exp(seq_log_importance_ratio), max=loss_config.sequence_clip_high)

    seq_min_ratio = torch.where(loss_mask, token_importance_ratio, torch.inf).min()
    seq_max_ratio = torch.where(loss_mask, token_importance_ratio, -torch.inf).max()
    seq_mask_low = seq_min_ratio < loss_config.sequence_mask_low
    seq_mask_high = seq_max_ratio > loss_config.sequence_mask_high

    token_mask_low = token_importance_ratio < loss_config.token_mask_low
    token_mask_high = token_importance_ratio > loss_config.token_mask_high

    geo_mask_low = geo_seq_ratio < loss_config.geo_mask_low
    geo_mask_high = geo_seq_ratio > loss_config.geo_mask_high

    is_masked = token_mask_low | token_mask_high | geo_mask_low | geo_mask_high | seq_mask_low | seq_mask_high

    return SequenceLossState(
        trainer_logprobs=trainer_logprobs,
        log_importance_ratio=log_importance_ratio,
        token_importance_ratio=token_importance_ratio,
        sequence_importance_ratio=sequence_importance_ratio,
        token_mismatch_kl=token_mismatch_kl,
        geo_seq_ratio=geo_seq_ratio,
        token_mask_low=token_mask_low,
        token_mask_high=token_mask_high,
        geo_mask_low=geo_mask_low,
        geo_mask_high=geo_mask_high,
        seq_mask_low=seq_mask_low,
        seq_mask_high=seq_mask_high,
        is_masked=is_masked,
        loss_mask=loss_mask,
        scaled_advantages=loss_config.adv_tau * advantages,
    )


def _get_active_importance_ratio(state: SequenceLossState, loss_config: LossConfig) -> Tensor:
    if loss_config.ratio_type == "sequence":
        return state.sequence_importance_ratio
    return state.token_importance_ratio


def _compute_loss_from_ratio(
    state: SequenceLossState,
    importance_ratio: Tensor,
    keep_mask: Tensor,
    loss_config: LossConfig,
) -> Tensor:
    coeff = importance_ratio * (state.scaled_advantages - loss_config.kl_tau * state.log_importance_ratio)
    loss = -(coeff.detach() * state.trainer_logprobs)[keep_mask].sum()
    if loss_config.ratio_type == "sequence":
        loss = loss / torch.clamp_min(state.loss_mask.sum(), 1)
    return loss


def _reduce_masked_sequence(
    state: SequenceLossState,
    loss_config: LossConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    keep_mask = state.loss_mask & ~state.is_masked
    loss = _compute_loss_from_ratio(state, _get_active_importance_ratio(state, loss_config), keep_mask, loss_config)
    metrics = {
        "loss_masked_keep_frac": keep_mask[state.loss_mask].float(),
        "loss_masked_drop_frac": state.is_masked[state.loss_mask].float(),
        "loss_masked_drop_token_low_frac": state.token_mask_low[state.loss_mask].float(),
        "loss_masked_drop_token_high_frac": state.token_mask_high[state.loss_mask].float(),
        "loss_masked_drop_sequence_low_frac": _as_metric_tensor(state.seq_mask_low),
        "loss_masked_drop_sequence_high_frac": _as_metric_tensor(state.seq_mask_high),
        "loss_masked_drop_geo_low_frac": _as_metric_tensor(state.geo_mask_low),
        "loss_masked_drop_geo_high_frac": _as_metric_tensor(state.geo_mask_high),
        "loss_masked_kept_mismatch_kl_mean": _as_metric_tensor(_safe_mean(state.token_mismatch_kl, keep_mask)),
        "loss_masked_dropped_mismatch_kl_mean": _as_metric_tensor(
            _safe_mean(state.token_mismatch_kl, state.loss_mask & state.is_masked)
        ),
    }
    return loss, metrics


def _reduce_vanilla_sequence(
    state: SequenceLossState,
    loss_config: LossConfig,
) -> Tensor:
    return _compute_loss_from_ratio(state, _get_active_importance_ratio(state, loss_config), state.loss_mask, loss_config)


def _clip_diagnostic_tensors(
    active_ratio: Tensor,
    clipped_ratio: Tensor,
    loss_mask: Tensor,
    ratio_type: str,
    low: float,
    high: float,
) -> dict[str, Tensor]:
    clip_low = active_ratio < low
    clip_high = active_ratio > high
    if ratio_type == "sequence":
        return {
            "loss_clipped_clip_frac": _as_metric_tensor(clip_low | clip_high),
            "loss_clipped_clip_low_frac": _as_metric_tensor(clip_low),
            "loss_clipped_clip_high_frac": _as_metric_tensor(clip_high),
            "loss_clipped_ratio_preclip_mean": _as_metric_tensor(active_ratio),
            "loss_clipped_ratio_postclip_mean": _as_metric_tensor(clipped_ratio),
        }
    return {
        "loss_clipped_clip_frac": (clip_low | clip_high)[loss_mask].float(),
        "loss_clipped_clip_low_frac": clip_low[loss_mask].float(),
        "loss_clipped_clip_high_frac": clip_high[loss_mask].float(),
        "loss_clipped_ratio_preclip_mean": _as_metric_tensor(_safe_mean(active_ratio, loss_mask)),
        "loss_clipped_ratio_postclip_mean": _as_metric_tensor(_safe_mean(clipped_ratio, loss_mask)),
    }


def _reduce_clipped_sequence(
    state: SequenceLossState,
    loss_config: LossConfig,
    clip_epsilon: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    active_ratio = _get_active_importance_ratio(state, loss_config)
    clip_low = 1.0 - clip_epsilon
    clip_high = 1.0 + clip_epsilon
    clipped_ratio = torch.clamp(active_ratio, min=clip_low, max=clip_high)
    loss = _compute_loss_from_ratio(state, clipped_ratio, state.loss_mask, loss_config)
    metrics = _clip_diagnostic_tensors(active_ratio, clipped_ratio, state.loss_mask, loss_config.ratio_type, clip_low, clip_high)
    return loss, metrics


def compute_landscape_loss_metrics(
    trainer_logprobs: list[Tensor],
    inference_logprobs: list[Tensor],
    advantages: list[Tensor],
    loss_mask: list[Tensor],
    loss_config: LossConfig,
    loss_scale: int,
    clip_epsilon: float,
) -> dict[str, Tensor]:
    totals = {
        "loss_masked": trainer_logprobs[0].new_tensor(0.0),
        "loss_vanilla": trainer_logprobs[0].new_tensor(0.0),
        "loss_clipped": trainer_logprobs[0].new_tensor(0.0),
    }
    metric_lists: dict[str, list[Tensor]] = defaultdict(list)

    for seq_trainer_logprobs, seq_inference_logprobs, seq_advantages, seq_loss_mask in zip(
        trainer_logprobs, inference_logprobs, advantages, loss_mask
    ):
        state = build_sequence_loss_state(
            trainer_logprobs=seq_trainer_logprobs,
            inference_logprobs=seq_inference_logprobs,
            advantages=seq_advantages,
            loss_mask=seq_loss_mask,
            loss_config=loss_config,
        )
        masked_loss, masked_metrics = _reduce_masked_sequence(state, loss_config)
        vanilla_loss = _reduce_vanilla_sequence(state, loss_config)
        clipped_loss, clipped_metrics = _reduce_clipped_sequence(state, loss_config, clip_epsilon)

        totals["loss_masked"] = totals["loss_masked"] + masked_loss
        totals["loss_vanilla"] = totals["loss_vanilla"] + vanilla_loss
        totals["loss_clipped"] = totals["loss_clipped"] + clipped_loss

        shared_metrics = {
            "loss_shared_mismatch_kl_mean": _as_metric_tensor(_safe_mean(state.token_mismatch_kl, state.loss_mask)),
            "loss_shared_geo_seq_ratio_mean": _as_metric_tensor(state.geo_seq_ratio),
        }
        for metrics in (shared_metrics, masked_metrics, clipped_metrics):
            for key, value in metrics.items():
                metric_lists[key].append(value.detach().float().reshape(-1).cpu())

    aggregated = _aggregate_metrics(metric_lists)
    for key, value in totals.items():
        aggregated[key] = (value / loss_scale).detach().float().cpu().reshape(())
    return aggregated
