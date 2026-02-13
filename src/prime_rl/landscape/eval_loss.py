import math

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
) -> float:
    def _selective_log_softmax_eager(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        logprobs = logits.log_softmax(dim=-1)
        return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

    def _compute_entropy_eager(shifted_logits: torch.Tensor) -> torch.Tensor:
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        return torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)

    def _mean_or_zero(tensor: torch.Tensor) -> float:
        if tensor.numel() == 0:
            return 0.0
        return float(tensor.float().mean().item())

    def _model_param_canary() -> tuple[float, float, int]:
        canary_sum = torch.tensor(0.0, device=device)
        canary_sq_sum = torch.tensor(0.0, device=device)
        sampled = 0
        target_samples = 4096
        for param in model.parameters():
            if not param.is_floating_point():
                continue
            tensor = param.data
            if hasattr(tensor, "to_local"):
                tensor = tensor.to_local()
            if tensor.numel() == 0:
                continue
            flat = tensor.detach().float().reshape(-1)
            take = min(flat.numel(), target_samples - sampled)
            if take <= 0:
                break
            sample = flat[:take]
            canary_sum += sample.sum()
            canary_sq_sum += sample.pow(2).sum()
            sampled += take
            if sampled >= target_samples:
                break
        return float(canary_sum.item()), float(canary_sq_sum.item()), sampled

    def _sample_signature(tensor: torch.Tensor, max_elems: int = 4096) -> tuple[float, float, int]:
        if tensor.numel() == 0:
            return 0.0, 0.0, 0
        flat = tensor.detach().float().reshape(-1)
        sample = flat[: min(flat.numel(), max_elems)]
        return float(sample.sum().item()), float(sample.pow(2).sum().item()), int(sample.numel())

    if loss_config.ratio_type == "token":
        loss_scale = sum(micro_batch["loss_mask"].sum().item() for micro_batch in micro_batches)
    else:
        loss_scale = len(micro_batches)
    loss_scale = max(loss_scale, 1)

    losses = []
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
            pre_shift_logprobs = out["logprobs"]
            pre_shift_logprobs_sum = float(pre_shift_logprobs.float().sum().item())
            pre_shift_logprobs_mean = float(pre_shift_logprobs.float().mean().item())
            pre_shift_logprobs_std = float(pre_shift_logprobs.float().std(unbiased=False).item())
            pre_shift_logprobs_min = float(pre_shift_logprobs.float().min().item())
            pre_shift_logprobs_max = float(pre_shift_logprobs.float().max().item())
            out["logprobs"] = shift_tensor_right(
                out["logprobs"], pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item()
            )
            out["entropy"] = shift_tensor_right(
                out["entropy"], pad_value=torch.log(torch.tensor(float(vocab_size))).item()
            )
            response_lengths = get_response_lengths(position_ids)
            if idx == 1:
                tag_prefix = f"[{eval_tag}] " if eval_tag else ""
                trainer_logprobs_sum = float(out["logprobs"].float().sum().item())
                trainer_logprobs_mean = float(out["logprobs"].float().mean().item())
                trainer_logprobs_std = float(out["logprobs"].float().std(unbiased=False).item())
                inference_logprobs_sum = float(inference_logprobs.float().sum().item())
                inference_logprobs_mean = float(inference_logprobs.float().mean().item())
                inference_logprobs_std = float(inference_logprobs.float().std(unbiased=False).item())
                temperature_min = float(temperatures.float().min().item())
                temperature_mean = float(temperatures.float().mean().item())
                temperature_max = float(temperatures.float().max().item())
                param_canary_sum, param_canary_sq_sum, param_canary_count = _model_param_canary()
                logits_sig_sum = 0.0
                logits_sig_sq_sum = 0.0
                logits_sig_count = 0
                logits_sample_std = 0.0
                if out.get("logits") is not None:
                    logits_sig_sum, logits_sig_sq_sum, logits_sig_count = _sample_signature(out["logits"])
                    logits_sample_std = float(out["logits"].float().std(unbiased=False).item())
                pad_logprob_value = float(math.log(1.0 / vocab_size))
                pad_like_frac = float((out["logprobs"].float() - pad_logprob_value).abs().lt(1e-5).float().mean().item())
                loss_mask_true_frac = float(loss_mask.float().mean().item())
                response_len_min = min(response_lengths) if response_lengths else 0
                response_len_max = max(response_lengths) if response_lengths else 0
                response_len_mean = float(sum(response_lengths) / max(len(response_lengths), 1))
                logger.debug(
                    f"{tag_prefix}Eval fingerprint micro-batch {idx}/{total_micro_batches}: "
                    f"trainer_logprobs_sum={trainer_logprobs_sum:.8e} "
                    f"trainer_logprobs_mean={trainer_logprobs_mean:.8e} "
                    f"trainer_logprobs_std={trainer_logprobs_std:.8e} "
                    f"inference_logprobs_sum={inference_logprobs_sum:.8e} "
                    f"inference_logprobs_mean={inference_logprobs_mean:.8e} "
                    f"inference_logprobs_std={inference_logprobs_std:.8e} "
                    f"pre_shift_logprobs_sum={pre_shift_logprobs_sum:.8e} "
                    f"pre_shift_logprobs_mean={pre_shift_logprobs_mean:.8e} "
                    f"pre_shift_logprobs_std={pre_shift_logprobs_std:.8e} "
                    f"pre_shift_logprobs_min={pre_shift_logprobs_min:.8e} "
                    f"pre_shift_logprobs_max={pre_shift_logprobs_max:.8e} "
                    f"logits_sig_sum={logits_sig_sum:.8e} "
                    f"logits_sig_sq_sum={logits_sig_sq_sum:.8e} "
                    f"logits_sig_count={logits_sig_count} "
                    f"logits_sample_std={logits_sample_std:.8e} "
                    f"pad_logprob={pad_logprob_value:.8e} "
                    f"pad_like_frac={pad_like_frac:.8e} "
                    f"loss_mask_true_frac={loss_mask_true_frac:.8e} "
                    f"response_len_min={response_len_min} "
                    f"response_len_mean={response_len_mean:.8e} "
                    f"response_len_max={response_len_max} "
                    f"temperature_min={temperature_min:.8e} "
                    f"temperature_mean={temperature_mean:.8e} "
                    f"temperature_max={temperature_max:.8e} "
                    f"param_canary_sum={param_canary_sum:.8e} "
                    f"param_canary_sq_sum={param_canary_sq_sum:.8e} "
                    f"param_canary_count={param_canary_count}"
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
            num_masked_tokens = int(loss_tensors["is_masked"].sum().item()) if loss_tensors["is_masked"].numel() else 0
            num_loss_tokens = int(loss_tensors["is_masked"].numel())
            num_kept_tokens = max(num_loss_tokens - num_masked_tokens, 0)
            logger.debug(
                f"Loss diagnostics micro-batch {idx}/{total_micro_batches}: "
                f"loss_tokens={num_loss_tokens} kept_tokens={num_kept_tokens} "
                f"masked_frac={_mean_or_zero(loss_tensors['is_masked']):.6f} "
                f"token_low_frac={_mean_or_zero(loss_tensors['is_masked_low']):.6f} "
                f"token_high_frac={_mean_or_zero(loss_tensors['is_masked_high']):.6f} "
                f"seq_low_frac={_mean_or_zero(loss_tensors['sequence_masked_low']):.6f} "
                f"seq_high_frac={_mean_or_zero(loss_tensors['sequence_masked_high']):.6f} "
                f"geo_low_frac={_mean_or_zero(loss_tensors['geo_masked_low']):.6f} "
                f"geo_high_frac={_mean_or_zero(loss_tensors['geo_masked_high']):.6f} "
                f"mismatch_kl={_mean_or_zero(loss_tensors['mismatch_kl']):.6e} "
                f"geo_seq_ratio={_mean_or_zero(loss_tensors['geo_seq_ratio']):.6e}"
            )
            losses.append(loss.detach().float().cpu().item())

    mean_loss = float(sum(losses) / max(len(losses), 1))
    logger.debug(f"Loss over {total_micro_batches} micro-batches: {mean_loss:.6f}")
    return mean_loss
