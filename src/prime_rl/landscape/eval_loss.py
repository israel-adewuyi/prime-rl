import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from loguru import logger

from prime_rl.trainer.model import forward
from prime_rl.trainer.rl.loss import (
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.trainer.utils import get_response_lengths

LOSS_EVAL_COLUMNS = (
    "loss_grpo",
    "loss_grpo_unclipped",
    "loss_kl_valid_mean",
    "loss_kl_valid_clipped_mean",
    "loss_clip_frac",
    "loss_clip_low_frac",
    "loss_clip_high_frac",
    "loss_ratio_mean",
    "loss_clipped_ratio_mean",
    "loss_log_ratio_mean",
    "loss_surrogate_mean",
    "loss_surrogate_clipped_mean",
    "loss_batch_adv_mean",
    "loss_batch_adv_std",
    "loss_batch_adv_abs_mean",
    "loss_valid_tokens",
)


def micro_batch_to_tensor(micro_batch) -> dict:
    return {
        "input_ids": torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
        "position_ids": torch.tensor(micro_batch.position_ids, dtype=torch.long).unsqueeze(0),
        "advantages": torch.tensor(micro_batch.advantages, dtype=torch.float).unsqueeze(0),
        "inference_logprobs": torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
        "loss_mask": torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
        "temperatures": torch.tensor(micro_batch.temperatures, dtype=torch.float).unsqueeze(0),
        "pixel_values": torch.tensor(micro_batch.pixel_values, dtype=torch.float)
        if micro_batch.pixel_values is not None
        else None,
        "image_grid_thw": torch.tensor(micro_batch.image_grid_thw, dtype=torch.long)
        if micro_batch.image_grid_thw is not None
        else None,
    }


def _mean_or_zero(tensors: list[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    values = torch.cat(tensors)
    if values.numel() == 0:
        return 0.0
    return float(values.mean().item())


def _compute_grpo_loss_metrics(
    trainer_logprobs: list[torch.Tensor],
    inference_logprobs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    valid_token_mask: list[torch.Tensor],
    loss_scale: int,
    clip_epsilon: float,
) -> dict[str, torch.Tensor]:
    totals = {
        "loss_grpo": trainer_logprobs[0].new_tensor(0.0),
        "loss_grpo_unclipped": trainer_logprobs[0].new_tensor(0.0),
    }
    metric_lists: dict[str, list[torch.Tensor]] = {
        "loss_kl_valid_mean": [],
        "loss_kl_valid_clipped_mean": [],
        "loss_clip_frac": [],
        "loss_clip_low_frac": [],
        "loss_clip_high_frac": [],
        "loss_ratio_mean": [],
        "loss_clipped_ratio_mean": [],
        "loss_log_ratio_mean": [],
        "loss_surrogate_mean": [],
        "loss_surrogate_clipped_mean": [],
    }

    clip_low = 1.0 - clip_epsilon
    clip_high = 1.0 + clip_epsilon

    for seq_trainer_logprobs, seq_inference_logprobs, seq_advantages, seq_valid_token_mask in zip(
        trainer_logprobs, inference_logprobs, advantages, valid_token_mask
    ):
        log_ratio = seq_trainer_logprobs - seq_inference_logprobs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, min=clip_low, max=clip_high)
        surrogate_unclipped = ratio * seq_advantages
        surrogate_clipped = clipped_ratio * seq_advantages
        surrogate = torch.minimum(surrogate_unclipped, surrogate_clipped)

        preclip_mismatch = ratio - log_ratio - 1
        clipped_log_ratio = torch.log(clipped_ratio)
        postclip_mismatch = clipped_ratio - clipped_log_ratio - 1
        is_clipped = (ratio < clip_low) | (ratio > clip_high)

        totals["loss_grpo"] = totals["loss_grpo"] - surrogate[seq_valid_token_mask].sum()
        totals["loss_grpo_unclipped"] = totals["loss_grpo_unclipped"] - surrogate_unclipped[seq_valid_token_mask].sum()

        metric_lists["loss_kl_valid_mean"].append(preclip_mismatch[seq_valid_token_mask].detach().float().cpu())
        metric_lists["loss_kl_valid_clipped_mean"].append(postclip_mismatch[seq_valid_token_mask].detach().float().cpu())
        metric_lists["loss_clip_frac"].append(is_clipped[seq_valid_token_mask].detach().float().cpu())
        metric_lists["loss_clip_low_frac"].append((ratio[seq_valid_token_mask] < clip_low).detach().float().cpu())
        metric_lists["loss_clip_high_frac"].append((ratio[seq_valid_token_mask] > clip_high).detach().float().cpu())
        metric_lists["loss_ratio_mean"].append(ratio[seq_valid_token_mask].detach().float().cpu())
        metric_lists["loss_clipped_ratio_mean"].append(clipped_ratio[seq_valid_token_mask].detach().float().cpu())
        metric_lists["loss_log_ratio_mean"].append(log_ratio[seq_valid_token_mask].detach().float().cpu())
        metric_lists["loss_surrogate_mean"].append(surrogate_unclipped[seq_valid_token_mask].detach().float().cpu())
        metric_lists["loss_surrogate_clipped_mean"].append(surrogate_clipped[seq_valid_token_mask].detach().float().cpu())

    aggregated = {key: torch.cat(values) for key, values in metric_lists.items() if values}
    for key, value in totals.items():
        aggregated[key] = (value / loss_scale).detach().float().cpu().reshape(())
    return aggregated


def compute_eval_loss(
    model: torch.nn.Module,
    micro_batches: list[dict],
    loss_config,
    clip_epsilon: float,
    parallel_dims,
    device: torch.device,
    eval_tag: str | None = None,
) -> dict[str, float]:
    def _selective_log_softmax_eager(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        logprobs = logits.log_softmax(dim=-1)
        return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

    def _compute_entropy_eager(shifted_logits: torch.Tensor) -> torch.Tensor:
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        return torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)

    loss_scale = sum(micro_batch["loss_mask"].sum().item() for micro_batch in micro_batches)
    loss_scale = max(loss_scale, 1)

    accumulated_losses = {
        "loss_grpo": 0.0,
        "loss_grpo_unclipped": 0.0,
    }
    loss_tensors_by_key: dict[str, list[torch.Tensor]] = {}
    masked_advantages: list[torch.Tensor] = []
    valid_token_count = 0
    total_micro_batches = len(micro_batches)
    cp_enabled = parallel_dims.cp_enabled
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
    cp_group = parallel_dims.world_mesh["cp"].get_group() if cp_enabled else None
    cp_size = parallel_dims.cp

    with torch.no_grad():
        for idx, micro_batch in enumerate(micro_batches, start=1):
            logger.debug(f"Loss micro-batch {idx}/{total_micro_batches}")
            input_ids = micro_batch["input_ids"].to(device)
            position_ids = micro_batch["position_ids"].to(device)
            advantages = micro_batch["advantages"].to(device)
            loss_mask = micro_batch["loss_mask"].to(device)
            inference_logprobs = micro_batch["inference_logprobs"].to(device)
            masked_advantages.append(advantages[loss_mask].detach().float().reshape(-1).cpu())
            valid_token_count += int(loss_mask.sum().item())

            pixel_values = (
                micro_batch["pixel_values"].to(device) if micro_batch.get("pixel_values") is not None else None
            )
            image_grid_thw = (
                micro_batch["image_grid_thw"].to(device) if micro_batch.get("image_grid_thw") is not None else None
            )
            labels = shift_tensor_left(input_ids)
            if cp_enabled and pixel_values is not None:
                raise NotImplementedError("Context parallelism is not supported with VLM/multimodal training")

            if cp_enabled:
                from prime_rl.utils.cp import setup_cp_params, shard_for_cp

                input_ids, forward_position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size, cp_group)
                labels = shard_for_cp(labels, cp_rank=cp_rank, cp_world_size=cp_size)
            else:
                forward_position_ids = position_ids

            temperatures = micro_batch["temperatures"].to(device)
            if cp_enabled:
                from prime_rl.utils.cp import shard_for_cp

                temperatures = shard_for_cp(temperatures, cp_rank=cp_rank, cp_world_size=cp_size)

            out = forward(
                model,
                input_ids,
                forward_position_ids,
                labels=labels,
                temperature=temperatures,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

            if out.get("logprobs") is None:
                logits = out["logits"]
                scaled_logits = logits / temperatures.unsqueeze(-1)
                # Use eager kernels in landscape eval to avoid stale compiled graph artifacts.
                out["logprobs"] = _selective_log_softmax_eager(scaled_logits, labels)
                out["entropy"] = _compute_entropy_eager(scaled_logits)

            if cp_enabled:
                logprobs = dist_nn.all_gather(out["logprobs"], group=cp_group)
                out["logprobs"] = torch.cat(logprobs, dim=1)
                entropies = [torch.zeros_like(out["entropy"]) for _ in range(cp_size)]
                dist.all_gather(entropies, out["entropy"], group=cp_group)
                out["entropy"] = torch.cat(entropies, dim=1)

            vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
            pad_logprob_value = float(torch.log(torch.tensor(1.0 / vocab_size)).item())
            out["logprobs"] = shift_tensor_right(out["logprobs"], pad_value=pad_logprob_value)
            out["entropy"] = shift_tensor_right(
                out["entropy"], pad_value=torch.log(torch.tensor(float(vocab_size))).item()
            )
            response_lengths = get_response_lengths(position_ids)
            if idx == 1:
                tag_prefix = f"[{eval_tag}] " if eval_tag else ""
                trainer_logprobs_std = float(out["logprobs"].float().std(unbiased=False).item())
                logits_sample_std = float("nan")
                has_logits = out.get("logits") is not None
                if has_logits:
                    logits_sample_std = float(out["logits"].float().std(unbiased=False).item())
                pad_like_frac = float(
                    (out["logprobs"].float() - pad_logprob_value).abs().lt(1e-5).float().mean().item()
                )

                if has_logits and logits_sample_std < 1e-8:
                    raise RuntimeError(
                        f"{tag_prefix}Degenerate trainer logits detected in landscape eval: "
                        f"logits_sample_std={logits_sample_std:.8e}, pad_like_frac={pad_like_frac:.8e}, "
                        f"trainer_logprobs_std={trainer_logprobs_std:.8e}. "
                        "This usually indicates a broken LM head load/tie path."
                    )
                if pad_like_frac > 0.9999 and trainer_logprobs_std < 1e-8:
                    raise RuntimeError(
                        f"{tag_prefix}Trainer logprobs are effectively all pad/uniform values "
                        f"(pad_like_frac={pad_like_frac:.8e}, trainer_logprobs_std={trainer_logprobs_std:.8e}); "
                        "aborting sweep because losses are not informative."
                    )

            loss_metrics = _compute_grpo_loss_metrics(
                trainer_logprobs=out["logprobs"].squeeze().split(response_lengths),
                inference_logprobs=inference_logprobs.squeeze().split(response_lengths),
                advantages=advantages.squeeze().split(response_lengths),
                valid_token_mask=loss_mask.squeeze().split(response_lengths),
                loss_scale=loss_scale,
                clip_epsilon=clip_epsilon,
            )
            for key, tensor in loss_metrics.items():
                if key in accumulated_losses:
                    accumulated_losses[key] += float(tensor.item())
                    continue
                loss_tensors_by_key.setdefault(key, []).append(tensor.detach().float().reshape(-1).cpu())

    logger.debug(
        "Sum of avg losses over "
        f"{total_micro_batches} micro-batches: "
        f"grpo={accumulated_losses['loss_grpo']:.7f} "
        f"grpo_unclipped={accumulated_losses['loss_grpo_unclipped']:.7f}"
    )

    diagnostics = {key: 0.0 for key in LOSS_EVAL_COLUMNS}
    diagnostics.update(accumulated_losses)
    diagnostics["loss_kl_valid_mean"] = _mean_or_zero(loss_tensors_by_key.get("loss_kl_valid_mean", []))
    diagnostics["loss_kl_valid_clipped_mean"] = _mean_or_zero(
        loss_tensors_by_key.get("loss_kl_valid_clipped_mean", [])
    )
    diagnostics["loss_clip_frac"] = _mean_or_zero(loss_tensors_by_key.get("loss_clip_frac", []))
    diagnostics["loss_clip_low_frac"] = _mean_or_zero(loss_tensors_by_key.get("loss_clip_low_frac", []))
    diagnostics["loss_clip_high_frac"] = _mean_or_zero(loss_tensors_by_key.get("loss_clip_high_frac", []))
    diagnostics["loss_ratio_mean"] = _mean_or_zero(loss_tensors_by_key.get("loss_ratio_mean", []))
    diagnostics["loss_clipped_ratio_mean"] = _mean_or_zero(
        loss_tensors_by_key.get("loss_clipped_ratio_mean", [])
    )
    diagnostics["loss_log_ratio_mean"] = _mean_or_zero(loss_tensors_by_key.get("loss_log_ratio_mean", []))
    diagnostics["loss_surrogate_mean"] = _mean_or_zero(loss_tensors_by_key.get("loss_surrogate_mean", []))
    diagnostics["loss_surrogate_clipped_mean"] = _mean_or_zero(
        loss_tensors_by_key.get("loss_surrogate_clipped_mean", [])
    )

    all_masked_advantages = torch.cat(masked_advantages) if masked_advantages else torch.tensor([], dtype=torch.float32)
    if all_masked_advantages.numel() > 0:
        diagnostics["loss_batch_adv_mean"] = float(all_masked_advantages.mean().item())
        diagnostics["loss_batch_adv_std"] = float(all_masked_advantages.std(unbiased=False).item())
        diagnostics["loss_batch_adv_abs_mean"] = float(all_masked_advantages.abs().mean().item())
    diagnostics["loss_valid_tokens"] = float(valid_token_count)

    return diagnostics
