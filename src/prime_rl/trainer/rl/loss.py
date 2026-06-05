from typing import Any

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from prime_rl.trainer.rl.config import LossConfig


def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Apply top-k mask to the logits."""
    no_top_k_mask = k == logits.shape[1]
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    k_index = k.sub(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def _apply_top_k_top_p_2d(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    """Mirrors vLLM 0.10.2 top-k/top-p filtering semantics."""
    if p is None:
        if k is None:
            return logits
        return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Apply vLLM-style top-k/top-p masking to logits with shape [batch, seq, vocab]."""
    batch, seq, vocab = logits.shape
    flat_logits = logits.reshape(batch * seq, vocab)
    device = flat_logits.device

    k = None
    if top_k is not None and 0 < top_k < vocab:
        k = torch.full((flat_logits.shape[0],), top_k, dtype=torch.long, device=device)

    p = None
    if top_p is not None and top_p < 1.0:
        p = torch.full((flat_logits.shape[0],), top_p, dtype=flat_logits.dtype, device=device)

    if k is None and p is None:
        return logits

    with torch.no_grad():
        filtered_logits = _apply_top_k_top_p_2d(flat_logits.detach().clone(), k, p)
        support_mask = torch.isfinite(filtered_logits).reshape(batch, seq, vocab)
    return logits.masked_fill(~support_mask, -float("inf"))


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a zero logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    zeros = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([zeros, logits], dim=1)  # (batch, seq, vocab)
    return logits


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def _masked_values_or_zero(values: Tensor, mask: Tensor) -> Tensor:
    masked_values = values[mask]
    if masked_values.numel() == 0:
        return torch.zeros(1, dtype=values.dtype, device=values.device)
    return masked_values


def compute_loss(
    trainer_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    inference_logprobs: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    advantages: Any,  # list of Float[Tensor, "seq_i"] with potentially different seq_i lengths
    loss_mask: Any,  # list of Bool[Tensor, "seq_i"] with potentially different seq_i lengths
    loss_config: LossConfig,
    loss_scale: int,
    selected_in_support: Any | None = None,  # list of Bool[Tensor, "seq_i"] matching trainer_logprobs
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Args:
        trainer_logprobs: Log probabilities tensor for packed sequences
        inference_logprobs: Old log probabilities tensor for packed sequences
        advantages: Advantages tensor for packed sequences
        loss_mask: Loss mask tensor for packed sequences
        loss_config: Loss configuration object
        loss_scale: Scale factor to normalize the loss

    Returns:
        Tuple of (scaled_loss, aggregated_loss_tensors)
    """

    total_loss = 0
    total_mismatch_kl = []
    total_masked_mismatch_kl = []
    total_unmasked_mismatch_kl = []
    total_is_masked = []
    total_is_masked_low = []
    total_is_masked_high = []
    total_sequence_masked_low = []
    total_clip_frac = []
    total_importance_ratio = []
    total_clipped_ratio = []
    total_masked_importance_ratio = []
    total_masked_clipped_ratio = []
    total_log_importance_ratio = []
    total_surrogate_unclipped = []
    total_surrogate_clipped = []
    total_ratio_oob_frac = []
    total_active_clip_frac = []
    total_adv_abs_mean = []
    total_adv_std = []
    total_valid_tokens = []
    total_support_mismatch_frac = []
    dppo_loss_scale = 0
    total_dppo_rejected_mismatch_kl = []
    total_dppo_kept_mismatch_kl = []
    total_dppo_rejected = []
    total_dppo_eligible_importance_ratio = []
    total_dppo_kept_importance_ratio = []

    if selected_in_support is None:
        selected_in_support = [torch.ones_like(mask, dtype=torch.bool) for mask in loss_mask]

    for trainer_logprobs, inference_logprobs, advantages, loss_mask, selected_in_support in zip(
        trainer_logprobs, inference_logprobs, advantages, loss_mask, selected_in_support
    ):
        assert trainer_logprobs.shape == selected_in_support.shape
        log_importance_ratio = trainer_logprobs - inference_logprobs
        metric_mask = loss_mask & selected_in_support
        support_mismatch = loss_mask & ~selected_in_support
        total_support_mismatch_frac.append(_safe_mean(support_mismatch.float(), loss_mask).unsqueeze(0))

        if loss_config.type == "grpo" or loss_config.type == "max_rl":
            importance_ratio = torch.exp(log_importance_ratio)
            clipped_ratio = torch.clamp(importance_ratio, 1 - loss_config.clip_eps, 1 + loss_config.clip_eps)
            surrogate_unclipped = importance_ratio * advantages
            surrogate_clipped = clipped_ratio * advantages
            surrogate = torch.minimum(surrogate_unclipped, surrogate_clipped)
            token_mismatch_kl = torch.exp(log_importance_ratio) - log_importance_ratio - 1
            total_loss = total_loss - surrogate[loss_mask].sum()
            mismatch_kl = _safe_mean(token_mismatch_kl, metric_mask)
            is_clipped = (importance_ratio < 1 - loss_config.clip_eps) | (importance_ratio > 1 + loss_config.clip_eps)
            clip_mask = is_clipped[loss_mask]
            active_clip_mask = (surrogate_clipped < surrogate_unclipped)[loss_mask]
            masked_advantages = advantages[loss_mask]
            total_mismatch_kl.append(mismatch_kl)
            total_masked_mismatch_kl.append(_safe_mean(token_mismatch_kl, metric_mask & is_clipped))
            total_unmasked_mismatch_kl.append(_safe_mean(token_mismatch_kl, metric_mask & ~is_clipped))
            total_is_masked.append(clip_mask.float())
            total_is_masked_low.append((importance_ratio[loss_mask] < 1 - loss_config.clip_eps).float())
            total_is_masked_high.append((importance_ratio[loss_mask] > 1 + loss_config.clip_eps).float())
            total_sequence_masked_low.append(torch.tensor(0.0, device=trainer_logprobs.device))
            total_clip_frac.append(clip_mask.float())
            total_importance_ratio.append(importance_ratio)
            total_clipped_ratio.append(clipped_ratio)
            total_masked_importance_ratio.append(importance_ratio[loss_mask])
            total_masked_clipped_ratio.append(clipped_ratio[loss_mask])
            total_log_importance_ratio.append(_masked_values_or_zero(log_importance_ratio, metric_mask))
            total_surrogate_unclipped.append(surrogate_unclipped[loss_mask])
            total_surrogate_clipped.append(surrogate_clipped[loss_mask])
            total_ratio_oob_frac.append(clip_mask.float().mean().unsqueeze(0))
            total_active_clip_frac.append(active_clip_mask.float().mean().unsqueeze(0))
            total_adv_abs_mean.append(masked_advantages.abs().mean().unsqueeze(0))
            total_adv_std.append(masked_advantages.std(unbiased=False).unsqueeze(0))
            total_valid_tokens.append(loss_mask.sum().unsqueeze(0).float())
            continue
        
        elif loss_config.type == "dppo":
            threshold = loss_config.dppo_delta
            importance_ratio = torch.exp(log_importance_ratio)
            token_mismatch_kl = torch.exp(log_importance_ratio) - log_importance_ratio - 1
            D = torch.abs(torch.exp(trainer_logprobs) - torch.exp(inference_logprobs))
            dppo_bad_mask = ((advantages > 0) & (importance_ratio > 1) & (D > threshold)) | (
                (advantages < 0) & (importance_ratio < 1) & (D > threshold)
            )
            token_mask = loss_mask & ~dppo_bad_mask
            dppo_rejected_mask = loss_mask & dppo_bad_mask
            masked_advantages = advantages[token_mask]
            loss = (importance_ratio * advantages)[token_mask].sum()
            total_loss = total_loss - loss
            dppo_loss_scale += int(token_mask.sum().item())

            total_mismatch_kl.append(_safe_mean(token_mismatch_kl, metric_mask))
            total_dppo_rejected_mismatch_kl.append(_safe_mean(token_mismatch_kl, dppo_rejected_mask & selected_in_support))
            total_dppo_kept_mismatch_kl.append(_safe_mean(token_mismatch_kl, token_mask & selected_in_support))
            total_dppo_rejected.append(dppo_bad_mask[loss_mask].float())
            total_sequence_masked_low.append(torch.tensor(0.0, device=trainer_logprobs.device))
            total_dppo_eligible_importance_ratio.append(importance_ratio[loss_mask])
            total_dppo_kept_importance_ratio.append(importance_ratio[token_mask])
            total_log_importance_ratio.append(_masked_values_or_zero(log_importance_ratio, metric_mask))
            total_adv_abs_mean.append(_safe_mean(advantages.abs(), token_mask).unsqueeze(0))
            total_valid_tokens.append(token_mask.sum().unsqueeze(0).float())
            continue

        # Compute trainer-inference mismatch KL
        token_mismatch_kl = torch.exp(log_importance_ratio) - log_importance_ratio - 1

        if loss_config.ratio_type == "sequence":
            if support_mismatch.any():
                raise ValueError("ratio_type='sequence' is not supported with outside-support sampled tokens.")
            seq_log_importance_ratio = (log_importance_ratio[loss_mask]).sum()
            log_importance_ratio = trainer_logprobs - trainer_logprobs.detach() + seq_log_importance_ratio.detach()
            log_importance_ratio = torch.clamp(log_importance_ratio, max=10.0)

        importance_ratio = torch.exp(log_importance_ratio)
        is_masked_low = importance_ratio < loss_config.mask_ratio_low
        is_masked_high = importance_ratio > loss_config.mask_ratio_high
        is_masked = is_masked_low | is_masked_high
        seq_min_ratio = importance_ratio.masked_fill(~loss_mask, torch.inf).min()
        seq_should_mask = seq_min_ratio < loss_config.sequence_mask_ratio_low
        is_masked = is_masked | seq_should_mask
        keep_mask = loss_mask & ~is_masked
        loss = (-importance_ratio * advantages)[keep_mask].sum()
        if loss_config.kl_mask_type == "masked":
            kl_mask = loss_mask & is_masked
        elif loss_config.kl_mask_type == "unmasked":
            kl_mask = keep_mask
        elif loss_config.kl_mask_type == "all":
            kl_mask = loss_mask
        else:
            raise ValueError(f"Invalid KL mask type: {loss_config.kl_mask_type}")
        if loss_config.kl_tau > 0:
            loss = loss + loss_config.kl_tau * (log_importance_ratio[kl_mask & selected_in_support]).sum()

        # Apply sequence-level normalization if configured
        if loss_config.ratio_type == "sequence":
            loss = loss / torch.clamp_min(loss_mask.sum(), 1)

        total_loss = total_loss + loss

        mismatch_kl = token_mismatch_kl[metric_mask].sum() / torch.clamp_min(metric_mask.sum(), 1)
        masked_mismatch_kl = token_mismatch_kl[metric_mask & is_masked].sum() / torch.clamp_min(
            (metric_mask & is_masked).sum(), 1
        )
        unmasked_mismatch_kl = token_mismatch_kl[keep_mask & selected_in_support].sum() / torch.clamp_min(
            (keep_mask & selected_in_support).sum(), 1
        )

        # Aggregate loss tensors
        total_mismatch_kl.append(mismatch_kl)
        total_masked_mismatch_kl.append(masked_mismatch_kl)
        total_unmasked_mismatch_kl.append(unmasked_mismatch_kl)
        total_is_masked.append(is_masked[loss_mask].float())
        total_is_masked_low.append(is_masked_low[loss_mask].float())
        total_is_masked_high.append(is_masked_high[loss_mask].float())
        total_sequence_masked_low.append(seq_should_mask.float())

    # Apply loss scaling
    if loss_config.type == "dppo":
        loss_scale = max(dppo_loss_scale, 1)
    scaled_loss = total_loss / loss_scale

    if loss_config.type == "dppo":
        return scaled_loss, {
            "mismatch_kl": torch.stack(total_mismatch_kl),
            "dppo_rejected_mismatch_kl": torch.stack(total_dppo_rejected_mismatch_kl),
            "dppo_kept_mismatch_kl": torch.stack(total_dppo_kept_mismatch_kl),
            "dppo_rejected": torch.cat(total_dppo_rejected),
            "sequence_masked_low": torch.stack(total_sequence_masked_low),
            "dppo_eligible_importance_ratio": torch.cat(total_dppo_eligible_importance_ratio),
            "dppo_kept_importance_ratio": torch.cat(total_dppo_kept_importance_ratio),
            "log_importance_ratio": torch.cat(total_log_importance_ratio),
            "adv_abs_mean": torch.cat(total_adv_abs_mean),
            "valid_tokens": torch.cat(total_valid_tokens),
            "support_mismatch_frac": torch.cat(total_support_mismatch_frac),
        }

    loss_tensors = {
        "mismatch_kl": torch.stack(total_mismatch_kl),
        "masked_mismatch_kl": torch.stack(total_masked_mismatch_kl),
        "unmasked_mismatch_kl": torch.stack(total_unmasked_mismatch_kl),
        "is_masked": torch.cat(total_is_masked),
        "is_masked_low": torch.cat(total_is_masked_low),
        "is_masked_high": torch.cat(total_is_masked_high),
        "sequence_masked_low": torch.stack(total_sequence_masked_low),
        "support_mismatch_frac": torch.cat(total_support_mismatch_frac),
    }
    if total_clip_frac:
        loss_tensors["clip_frac"] = torch.cat(total_clip_frac)
        loss_tensors["importance_ratio"] = torch.cat(total_importance_ratio)
        loss_tensors["clipped_ratio"] = torch.cat(total_clipped_ratio)
        loss_tensors["masked_importance_ratio"] = torch.cat(total_masked_importance_ratio)
        loss_tensors["masked_clipped_ratio"] = torch.cat(total_masked_clipped_ratio)
        loss_tensors["log_importance_ratio"] = torch.cat(total_log_importance_ratio)
        loss_tensors["surrogate_unclipped"] = torch.cat(total_surrogate_unclipped)
        loss_tensors["surrogate_clipped"] = torch.cat(total_surrogate_clipped)
        loss_tensors["ratio_oob_frac"] = torch.cat(total_ratio_oob_frac)
        loss_tensors["active_clip_frac"] = torch.cat(total_active_clip_frac)
        loss_tensors["adv_abs_mean"] = torch.cat(total_adv_abs_mean)
        loss_tensors["adv_std"] = torch.cat(total_adv_std)
        loss_tensors["valid_tokens"] = torch.cat(total_valid_tokens)
    return scaled_loss, loss_tensors
