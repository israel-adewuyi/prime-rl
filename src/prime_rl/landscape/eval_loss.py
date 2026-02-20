import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from loguru import logger

from prime_rl.trainer.model import forward
from prime_rl.trainer.rl.loss import (
    compute_loss,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.trainer.utils import get_response_lengths

LOSS_DIAGNOSTIC_COLUMNS = (
    "loss_mismatch_kl_mean",
    "loss_masked_mismatch_kl_mean",
    "loss_unmasked_mismatch_kl_mean",
    "loss_teacher_kl_mean",
    "loss_is_masked_frac",
    "loss_is_masked_low_frac",
    "loss_is_masked_high_frac",
    "loss_sequence_masked_low_frac",
    "loss_sequence_masked_high_frac",
    "loss_geo_masked_low_frac",
    "loss_geo_masked_high_frac",
    "loss_geo_seq_ratio_mean",
    "loss_adv_mean",
    "loss_adv_std",
    "loss_adv_abs_mean",
    "loss_adv_nonzero_frac",
    "loss_mask_true_frac",
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


def compute_eval_loss(
    model: torch.nn.Module,
    micro_batches: list[dict],
    loss_config,
    parallel_dims,
    device: torch.device,
    eval_tag: str | None = None,
) -> tuple[float, dict[str, float]]:
    def _selective_log_softmax_eager(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        logprobs = logits.log_softmax(dim=-1)
        return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

    def _compute_entropy_eager(shifted_logits: torch.Tensor) -> torch.Tensor:
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        return torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)

    def _mean_or_zero(tensors: list[torch.Tensor]) -> float:
        if not tensors:
            return 0.0
        values = torch.cat(tensors)
        if values.numel() == 0:
            return 0.0
        return float(values.mean().item())

    if loss_config.ratio_type == "token":
        loss_scale = sum(micro_batch["loss_mask"].sum().item() for micro_batch in micro_batches)
    else:
        loss_scale = len(micro_batches)
    loss_scale = max(loss_scale, 1)

    losses = []
    loss_tensors_by_key: dict[str, list[torch.Tensor]] = {}
    masked_advantages: list[torch.Tensor] = []
    loss_mask_true_count = 0
    loss_mask_total_count = 0
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
            loss_mask_true_count += int(loss_mask.sum().item())
            loss_mask_total_count += int(loss_mask.numel())

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

            loss, loss_tensors = compute_loss(
                trainer_logprobs=out["logprobs"].squeeze().split(response_lengths),
                inference_logprobs=inference_logprobs.squeeze().split(response_lengths),
                teacher_logprobs=None,
                advantages=advantages.squeeze().split(response_lengths),
                loss_mask=loss_mask.squeeze().split(response_lengths),
                loss_config=loss_config,
                loss_scale=loss_scale,
            )
            for key, tensor in loss_tensors.items():
                loss_tensors_by_key.setdefault(key, []).append(tensor.detach().float().reshape(-1).cpu())
            losses.append(loss.detach().float().cpu().item())

    sum_loss = float(sum(losses))  # we already normalize in compute_loss
    logger.debug(f"Sum of avg loss over {total_micro_batches} micro-batches: {sum_loss:.7f}")

    diagnostics = {key: 0.0 for key in LOSS_DIAGNOSTIC_COLUMNS}
    diagnostics["loss_mismatch_kl_mean"] = _mean_or_zero(loss_tensors_by_key.get("mismatch_kl", []))
    diagnostics["loss_masked_mismatch_kl_mean"] = _mean_or_zero(loss_tensors_by_key.get("masked_mismatch_kl", []))
    diagnostics["loss_unmasked_mismatch_kl_mean"] = _mean_or_zero(
        loss_tensors_by_key.get("unmasked_mismatch_kl", [])
    )
    diagnostics["loss_teacher_kl_mean"] = _mean_or_zero(loss_tensors_by_key.get("teacher_kl", []))
    diagnostics["loss_is_masked_frac"] = _mean_or_zero(loss_tensors_by_key.get("is_masked", []))
    diagnostics["loss_is_masked_low_frac"] = _mean_or_zero(loss_tensors_by_key.get("is_masked_low", []))
    diagnostics["loss_is_masked_high_frac"] = _mean_or_zero(loss_tensors_by_key.get("is_masked_high", []))
    diagnostics["loss_sequence_masked_low_frac"] = _mean_or_zero(
        loss_tensors_by_key.get("sequence_masked_low", [])
    )
    diagnostics["loss_sequence_masked_high_frac"] = _mean_or_zero(
        loss_tensors_by_key.get("sequence_masked_high", [])
    )
    diagnostics["loss_geo_masked_low_frac"] = _mean_or_zero(loss_tensors_by_key.get("geo_masked_low", []))
    diagnostics["loss_geo_masked_high_frac"] = _mean_or_zero(loss_tensors_by_key.get("geo_masked_high", []))
    diagnostics["loss_geo_seq_ratio_mean"] = _mean_or_zero(loss_tensors_by_key.get("geo_seq_ratio", []))
    diagnostics["loss_mask_true_frac"] = float(loss_mask_true_count / max(loss_mask_total_count, 1))

    all_masked_advantages = torch.cat(masked_advantages) if masked_advantages else torch.tensor([], dtype=torch.float32)
    if all_masked_advantages.numel() > 0:
        diagnostics["loss_adv_mean"] = float(all_masked_advantages.mean().item())
        diagnostics["loss_adv_std"] = float(all_masked_advantages.std(unbiased=False).item())
        diagnostics["loss_adv_abs_mean"] = float(all_masked_advantages.abs().mean().item())
        diagnostics["loss_adv_nonzero_frac"] = float((all_masked_advantages != 0).float().mean().item())

    return sum_loss, diagnostics
