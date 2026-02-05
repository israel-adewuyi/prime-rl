import asyncio
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from typing import Iterable

import tomli_w
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import verifiers as vf
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AutoProcessor

from prime_rl.landscape.config import LandscapeConfig
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.trajectories import branch_rollout, build_vlm_image_cache, interleave_rollout
from prime_rl.orchestrator.utils import compute_teacher_logprobs, get_sampling_args, set_semaphore
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.lora import save_lora_config
from prime_rl.trainer.model import forward, setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.rl.loss import (
    compute_entropy,
    compute_loss,
    selective_log_softmax,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.trainer.utils import get_response_lengths, setup_torch_distributed
from prime_rl.trainer.weights import gather_weights_on_master, get_adapter_state_dict, save_state_dict
from prime_rl.trainer.world import get_world
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.logger import intercept_verifiers_logging, setup_logger
from prime_rl.utils.pydantic_config import get_temp_toml_file, parse_argv
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import get_env_ids_to_install, get_log_dir, install_env
from prime_rl.utils.vf import generate_batch, get_completion_len
from prime_rl.utils.vlm import is_vlm_model


@dataclass(frozen=True)
class SweepPoint:
    alpha: float
    beta: float


def _iter_named_parameters(model: torch.nn.Module, param_filter: str) -> list[tuple[str, torch.nn.Parameter]]:
    if param_filter == "all":
        params = [(name, param) for name, param in model.named_parameters()]
    else:
        params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not params:
        raise ValueError("No parameters selected for perturbation")
    return params


def _check_single_device(params: Iterable[torch.nn.Parameter]) -> torch.device:
    device = None
    for param in params:
        if device is None:
            device = param.device
        elif param.device != device:
            raise ValueError("All parameters must be on the same device for landscape perturbations")
    assert device is not None
    return device


def _compute_global_scale(params: list[tuple[str, torch.nn.Parameter]], seed: int, epsilon: float) -> float:
    device = _check_single_device(param for _, param in params)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total_param_sq = torch.tensor(0.0, device=device)
    total_dir_sq = torch.tensor(0.0, device=device)
    for _, param in params:
        if not param.is_floating_point():
            continue
        total_param_sq += param.detach().float().pow(2).sum()
        direction = torch.randn_like(param, generator=generator)
        total_dir_sq += direction.float().pow(2).sum()
    if total_dir_sq.item() == 0.0:
        return 0.0
    scale = torch.sqrt(total_param_sq) / (torch.sqrt(total_dir_sq) + epsilon)
    return float(scale.item())


def _apply_delta(
    params: list[tuple[str, torch.nn.Parameter]],
    seed: int,
    scale: float,
    step: float,
    norm: str,
    epsilon: float,
    direction: dict[str, torch.Tensor] | None = None,
) -> None:
    if step == 0.0:
        return
    device = _check_single_device(param for _, param in params)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for name, param in params:
        if not param.is_floating_point():
            continue
        if direction is None:
            direction_tensor = torch.randn_like(param, generator=generator)
            if norm == "layer":
                param_norm = param.detach().float().norm()
                dir_norm = direction_tensor.float().norm()
                if dir_norm.item() == 0.0:
                    continue
                direction_tensor = direction_tensor * (param_norm / (dir_norm + epsilon))
            else:
                direction_tensor = direction_tensor * scale
        else:
            if name not in direction:
                raise ValueError(f"Direction is missing parameter: {name}")
            direction_tensor = direction[name].to(device=param.device, dtype=param.dtype)
        param.data.add_(direction_tensor, alpha=step)


def _load_direction_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.startswith("hf://"):
        hf_ref = path.removeprefix("hf://")
        parts = hf_ref.split("/", 2)
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        resolved = hf_hub_download(repo_id=repo_id, filename=filename)
        loaded = torch.load(resolved, map_location="cpu")
    else:
        loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            return loaded["state_dict"]
        if all(isinstance(k, str) for k in loaded.keys()):
            return loaded
    raise ValueError("Direction file must be a state dict keyed by parameter names")


def _prepare_direction_tensors(
    params: list[tuple[str, torch.nn.Parameter]],
    direction_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    direction = {}
    for name, param in params:
        if name not in direction_state:
            raise ValueError(f"Direction state dict is missing parameter: {name}")
        tensor = direction_state[name]
        if tensor.shape != param.shape:
            raise ValueError(f"Direction tensor shape mismatch for {name}: {tensor.shape} vs {param.shape}")
        direction[name] = tensor.to(device=param.device, dtype=param.dtype)
    return direction


def _write_weights(
    model: torch.nn.Module,
    weight_dir: Path,
    save_format: str,
    save_sharded: bool,
    lora_name: str | None,
    lora_config,
) -> None:
    shutil.rmtree(weight_dir, ignore_errors=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    world = get_world()
    if lora_name is None:
        state = gather_weights_on_master(model, world.is_master)
        if world.is_master:
            save_state_dict(state, weight_dir, save_format=save_format, save_sharded=save_sharded)
    else:
        adapter_state = get_adapter_state_dict(model, world.is_master)
        if world.is_master:
            save_state_dict(adapter_state, weight_dir, save_format=save_format, save_sharded=save_sharded, adapter=True)
            save_lora_config(lora_config, model, weight_dir)


def _micro_batch_to_tensor(micro_batch) -> dict:
    if micro_batch.lora_num_tokens is None:
        micro_batch.lora_num_tokens = [0]
        micro_batch.lora_num_tokens[0] = len(micro_batch.input_ids)
    return {
        "input_ids": torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
        "position_ids": torch.tensor(micro_batch.position_ids, dtype=torch.long).unsqueeze(0),
        "advantages": torch.tensor(micro_batch.advantages, dtype=torch.float).unsqueeze(0),
        "inference_logprobs": torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
        "teacher_logprobs": torch.tensor(micro_batch.teacher_logprobs, dtype=torch.float).unsqueeze(0)
        if micro_batch.teacher_logprobs is not None
        else None,
        "loss_mask": torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
        "temperatures": torch.tensor(micro_batch.temperatures, dtype=torch.float).unsqueeze(0),
        "lora_num_tokens": torch.tensor(micro_batch.lora_num_tokens, dtype=torch.int32),
        "pixel_values": torch.tensor(micro_batch.pixel_values, dtype=torch.float)
        if micro_batch.pixel_values is not None
        else None,
        "image_grid_thw": torch.tensor(micro_batch.image_grid_thw, dtype=torch.long)
        if micro_batch.image_grid_thw is not None
        else None,
    }


def _compute_loss(
    model: torch.nn.Module,
    micro_batches: list[dict],
    loss_config,
    parallel_dims,
    lora_enabled: bool,
) -> float:
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
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")
            inference_logprobs = micro_batch["inference_logprobs"].to("cuda")
            teacher_logprobs = (
                micro_batch["teacher_logprobs"].to("cuda") if micro_batch["teacher_logprobs"] is not None else None
            )
            pixel_values = (
                micro_batch["pixel_values"].to("cuda") if micro_batch.get("pixel_values") is not None else None
            )
            image_grid_thw = (
                micro_batch["image_grid_thw"].to("cuda") if micro_batch.get("image_grid_thw") is not None else None
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

            if lora_enabled:
                from prime_rl.trainer.models.layers.lora import set_lora_num_tokens

                lora_num_tokens = micro_batch["lora_num_tokens"].to("cuda")
                if cp_enabled:
                    from prime_rl.utils.cp import shard_for_cp

                    chunk_size = input_ids.shape[1]
                    cu_offsets = lora_num_tokens.cumsum(dim=0, dtype=torch.int32)
                    adjusted_cu = torch.clip(cu_offsets - chunk_size * cp_rank, min=0, max=chunk_size)
                    lora_num_tokens = torch.diff(
                        adjusted_cu, prepend=torch.tensor([0], device=adjusted_cu.device, dtype=adjusted_cu.dtype)
                    )
                set_lora_num_tokens(lora_num_tokens)

            temperatures = micro_batch["temperatures"].to("cuda")
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
                out["logprobs"] = selective_log_softmax(scaled_logits, labels)
                out["entropy"] = compute_entropy(scaled_logits)

            if cp_enabled:
                logprobs = dist_nn.all_gather(out["logprobs"], group=cp_group)
                out["logprobs"] = torch.cat(logprobs, dim=1)
                entropies = [torch.zeros_like(out["entropy"]) for _ in range(cp_size)]
                dist.all_gather(entropies, out["entropy"], group=cp_group)
                out["entropy"] = torch.cat(entropies, dim=1)

            vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
            out["logprobs"] = shift_tensor_right(
                out["logprobs"], pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item()
            )
            out["entropy"] = shift_tensor_right(
                out["entropy"], pad_value=torch.log(torch.tensor(float(vocab_size))).item()
            )

            response_lengths = get_response_lengths(position_ids)
            loss, _ = compute_loss(
                trainer_logprobs=out["logprobs"].squeeze().split(response_lengths),
                inference_logprobs=inference_logprobs.squeeze().split(response_lengths),
                teacher_logprobs=teacher_logprobs.squeeze().split(response_lengths)
                if teacher_logprobs is not None
                else None,
                advantages=advantages.squeeze().split(response_lengths),
                loss_mask=loss_mask.squeeze().split(response_lengths),
                loss_config=loss_config,
                loss_scale=loss_scale,
            )
            losses.append(loss.detach().float().cpu().item())

    mean_loss = float(sum(losses) / max(len(losses), 1))
    logger.debug(f"Loss over {total_micro_batches} micro-batches: {mean_loss:.6f}")
    return mean_loss


def _prepare_examples(config: LandscapeConfig):
    env_ids_to_install = set()
    env_ids_to_install.update(get_env_ids_to_install(config.orchestrator.env))
    for env_id in env_ids_to_install:
        install_env(env_id)

    env = vf.EnvGroup(
        envs=[vf.load_environment(env.id, **env.args) for env in config.orchestrator.env],
        env_names=[env.name or env.id for env in config.orchestrator.env],
        map_kwargs=dict(writer_batch_size=1),
    )
    env.set_max_seq_len(config.orchestrator.seq_len)
    if config.orchestrator.trajectory_strategy == "interleaved":
        env.set_interleaved_rollouts(True)
    if config.orchestrator.buffer.skip_verification:
        env.set_score_rollouts(False)

    dataset = env.get_dataset(seed=config.orchestrator.buffer.seed)
    buffer = Buffer(dataset, env.env_names, config.orchestrator.buffer)

    rollouts_per_example = config.sweep.rollouts_per_example or config.orchestrator.rollouts_per_example
    batch_size = config.sweep.batch_size or config.orchestrator.batch_size
    if batch_size % rollouts_per_example != 0:
        raise ValueError("batch_size must be divisible by rollouts_per_example")
    num_examples = config.sweep.num_examples or (batch_size // rollouts_per_example)
    examples = buffer.sample_examples(num_examples)
    return env, examples, rollouts_per_example


def _prepare_sweep_points(grid) -> list[SweepPoint]:
    alphas = torch.linspace(grid.alpha_min, grid.alpha_max, grid.alpha_steps).tolist()
    betas = torch.linspace(grid.beta_min, grid.beta_max, grid.beta_steps).tolist()
    return [SweepPoint(alpha=a, beta=b) for a in alphas for b in betas]


def _ensure_rollout_temperatures(rollouts: list[vf.State], default_temp: float) -> None:
    for rollout in rollouts:
        trajectory = rollout.get("trajectory") or []
        for step in trajectory:
            if "temperature" not in step:
                step["temperature"] = default_temp


def _format_message_content(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _format_messages(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = _format_message_content(msg.get("content", ""))
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _format_rollout_text(
    rollout: vf.State,
    alpha: float,
    beta: float,
) -> str:
    header = (
        f"=== alpha={alpha:.3f} beta={beta:.3f} | example_id={rollout.get('example_id')} | "
        f"task={rollout.get('task')} | reward={rollout.get('reward')} | "
        f"is_truncated={rollout.get('is_truncated')} | error={rollout.get('error')} ==="
    )
    blocks = [header]
    trajectory = rollout.get("trajectory") or []
    for idx, step in enumerate(trajectory, start=1):
        prompt = step.get("prompt") or []
        response = step.get("response")
        blocks.append(f"TURN {idx} PROMPT:")
        blocks.append(_format_messages(prompt))
        blocks.append("")
        blocks.append(f"TURN {idx} RESPONSE:")
        if response is None:
            blocks.append("")
        else:
            if hasattr(response, "model_dump"):
                response = response.model_dump()
            if isinstance(response, dict):
                choices = response.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    blocks.append(_format_message_content(message.get("content", "")))
                else:
                    blocks.append(_format_message_content(response))
            else:
                blocks.append(_format_message_content(response))
        blocks.append("\n---\n")
    return "\n".join(blocks)


def _write_metadata(config: LandscapeConfig, output_dir: Path) -> None:
    metadata_path = output_dir / config.sweep.metadata_file
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_dir": str(output_dir),
        "trainer": config.trainer.model_dump(mode="json"),
        "orchestrator": config.orchestrator.model_dump(mode="json"),
        "sweep": config.sweep.model_dump(mode="json"),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_result(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with open(output_path, "a", newline="") as f:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        else:
            with open(output_path, "r", newline="") as read_f:
                header = read_f.readline().strip()
            if header and header != ",".join(fieldnames):
                raise ValueError(
                    f"Existing results file has different header: {header}. "
                    f"Expected: {','.join(fieldnames)}. "
                    "Delete the file or change output_dir to continue."
                )
        writer.writerow(row)


def _append_rollouts(output_path: Path, rollouts: list[vf.State], alpha: float, beta: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for rollout in rollouts:
            f.write(_format_rollout_text(rollout, alpha, beta))
            f.write("\n")


async def _evaluate_point(
    *,
    config: LandscapeConfig,
    model: torch.nn.Module,
    parallel_dims,
    lora_enabled: bool,
    inference_pool,
    teacher_pool,
    env,
    examples,
    rollouts_per_example: int,
    sampling_args: dict,
    temperature: float,
    is_vlm: bool,
    processor,
    weight_dir: Path,
    results_path: Path,
    rollouts_path: Path,
    alpha: float,
    beta: float,
    baseline: bool,
    logger,
) -> None:
    start_time = time.perf_counter()
    lora_name = config.orchestrator.model.lora.name if config.orchestrator.model.lora else None
    _write_weights(
        model,
        weight_dir,
        save_format=config.trainer.weight_broadcast.save_format,
        save_sharded=config.trainer.weight_broadcast.save_sharded,
        lora_name=lora_name,
        lora_config=config.trainer.model.lora,
    )
    await inference_pool.update_weights(weight_dir, lora_name=lora_name)

    rollouts = await generate_batch(
        clients=inference_pool.clients,
        env=env,
        model_name=config.orchestrator.model.name,
        examples=examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        pbar_description=f"Rollouts (alpha={alpha:.3f}, beta={beta:.3f})",
    )
    _ensure_rollout_temperatures(rollouts, temperature)
    _append_rollouts(rollouts_path, rollouts, alpha, beta)
    logger.debug(f"Completed rollouts for alpha={alpha:.3f}, beta={beta:.3f}")

    rewards = [rollout["reward"] for rollout in rollouts]
    completion_lens = [get_completion_len(rollout) for rollout in rollouts]
    advantages = compute_advantages(
        rewards,
        completion_lens,
        rollouts_per_example,
        config.orchestrator.advantage,
    )
    logger.debug(f"Completed advantages for alpha={alpha:.3f}, beta={beta:.3f}")

    rollout_fn = interleave_rollout if config.orchestrator.trajectory_strategy == "interleaved" else branch_rollout
    vlm_cache = None
    if is_vlm:
        vlm_cache = build_vlm_image_cache(rollouts, processor)

    logger.debug(f"Building train examples for alpha={alpha:.3f}, beta={beta:.3f}")
    train_examples = []
    for rollout, advantage in zip(rollouts, advantages):
        if vlm_cache is not None:
            cached = vlm_cache.get(rollout["example_id"])
            samples = rollout_fn(rollout, cached_pixel_values=cached[0], cached_image_grid_thw=cached[1])
        else:
            samples = rollout_fn(rollout)
        if samples is None:
            continue
        for sample in samples:
            sample.advantage = advantage
            sample.reward = rollout["reward"]
        train_examples.extend(samples)
    logger.debug(f"Completed train examples for alpha={alpha:.3f}, beta={beta:.3f}")
    if not train_examples:
        raise ValueError("No training samples were produced from rollouts")

    logger.debug(f"Computing teacher logprobs for alpha={alpha:.3f}, beta={beta:.3f}")
    if config.orchestrator.teacher_model is not None:
        teacher_logprobs_list = await compute_teacher_logprobs(
            clients=teacher_pool.clients,
            model_name=config.orchestrator.teacher_model.model.name,
            samples=train_examples,
        )
        for sample, teacher_logprobs in zip(train_examples, teacher_logprobs_list):
            sample.teacher_logprobs = teacher_logprobs
    logger.debug(f"Completed teacher logprobs for alpha={alpha:.3f}, beta={beta:.3f}")

    micro_batches_grid = prepare_batch(
        rollouts=train_examples,
        seq_len=config.trainer.model.seq_len,
        pad_to_multiple_of=config.trainer.model.cp,
        num_train_workers=1,
        idxs=[0] * len(train_examples),
        num_loras=1,
    )
    if not micro_batches_grid or not micro_batches_grid[0]:
        raise ValueError("No micro-batches were created from training samples")
    micro_batches = [_micro_batch_to_tensor(mb) for mb in micro_batches_grid[0]]
    logger.debug(f"Completed micro-batches for alpha={alpha:.3f}, beta={beta:.3f}")

    loss_value = _compute_loss(
        model,
        micro_batches,
        config.trainer.loss,
        parallel_dims,
        lora_enabled=lora_enabled,
    )
    logger.debug(f"Completed loss for alpha={alpha:.3f}, beta={beta:.3f}")

    reward_mean = float(sum(rewards) / max(len(rewards), 1))
    reward_std = float(torch.tensor(rewards).float().std(unbiased=False).item()) if rewards else 0.0
    elapsed_s = time.perf_counter() - start_time

    row = {
        "alpha": alpha,
        "beta": beta,
        "loss": loss_value,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "num_rollouts": len(rollouts),
        "num_examples": len(examples),
        "elapsed_s": elapsed_s,
        "baseline": baseline,
    }
    _append_result(results_path, row)
    logger.info(
        f"alpha={alpha:.3f} beta={beta:.3f} loss={loss_value:.4f} reward_mean={reward_mean:.4f} baseline={baseline}"
    )


async def _run_sweep(
    config: LandscapeConfig,
    output_dir: Path,
    model: torch.nn.Module,
    params: list[tuple[str, torch.nn.Parameter]],
    parallel_dims,
    lora_enabled: bool,
    scale_delta: float,
    scale_eta: float,
    delta_direction: dict[str, torch.Tensor] | None,
    eta_direction: dict[str, torch.Tensor] | None,
    logger,
) -> None:
    logger.info(
        f"Running sweep with alpha={config.sweep.grid.alpha_min} to {config.sweep.grid.alpha_max} and beta={config.sweep.grid.beta_min} to {config.sweep.grid.beta_max}"
    )
    inference_pool = await setup_inference_pool(config.orchestrator.client, base_model=config.orchestrator.model.name)
    await inference_pool.wait_for_ready(config.orchestrator.model.name)
    teacher_pool = None
    if config.orchestrator.teacher_model is not None:
        teacher_pool = await setup_inference_pool(
            config.orchestrator.teacher_model.client, base_model=config.orchestrator.teacher_model.model.name
        )
        await teacher_pool.wait_for_ready(config.orchestrator.teacher_model.model.name)

    env, examples, rollouts_per_example = _prepare_examples(config)
    max_concurrent = config.orchestrator.max_concurrent
    if max_concurrent is None:
        max_concurrent = len(examples) * rollouts_per_example
    await set_semaphore(max_concurrent)
    temperature = compute_temperature(0, config.orchestrator.sampling, config.orchestrator.max_steps)
    sampling_args = get_sampling_args(config.orchestrator.sampling, temperature=temperature)

    is_vlm = is_vlm_model(config.orchestrator.model.name)
    processor = None
    if is_vlm:
        processor = AutoProcessor.from_pretrained(
            config.orchestrator.model.name, trust_remote_code=config.orchestrator.model.trust_remote_code, use_fast=True
        )

    sweep_points = _prepare_sweep_points(config.sweep.grid)
    if not sweep_points or sweep_points[0] != SweepPoint(alpha=0.0, beta=0.0):
        sweep_points = [SweepPoint(alpha=0.0, beta=0.0)] + sweep_points

    prev_alpha = 0.0
    prev_beta = 0.0
    weight_dir = output_dir / config.sweep.weights_dir
    results_path = output_dir / config.sweep.results_file
    rollouts_path = output_dir / config.sweep.rollouts_file

    await _evaluate_point(
        config=config,
        model=model,
        parallel_dims=parallel_dims,
        lora_enabled=lora_enabled,
        inference_pool=inference_pool,
        teacher_pool=teacher_pool,
        env=env,
        examples=examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        temperature=temperature,
        is_vlm=is_vlm,
        processor=processor,
        weight_dir=weight_dir,
        results_path=results_path,
        rollouts_path=rollouts_path,
        alpha=0.0,
        beta=0.0,
        baseline=True,
        logger=logger,
    )

    for point in sweep_points:
        delta_alpha = point.alpha - prev_alpha
        delta_beta = point.beta - prev_beta
        with torch.no_grad():
            _apply_delta(
                params,
                config.sweep.direction.seed_delta,
                scale_delta,
                delta_alpha,
                config.sweep.direction.norm,
                config.sweep.direction.epsilon,
                direction=delta_direction,
            )
            _apply_delta(
                params,
                config.sweep.direction.seed_eta,
                scale_eta,
                delta_beta,
                config.sweep.direction.norm,
                config.sweep.direction.epsilon,
                direction=eta_direction,
            )
        prev_alpha = point.alpha
        prev_beta = point.beta

        await _evaluate_point(
            config=config,
            model=model,
            parallel_dims=parallel_dims,
            lora_enabled=lora_enabled,
            inference_pool=inference_pool,
            teacher_pool=teacher_pool,
            env=env,
            examples=examples,
            rollouts_per_example=rollouts_per_example,
            sampling_args=sampling_args,
            temperature=temperature,
            is_vlm=is_vlm,
            processor=processor,
            weight_dir=weight_dir,
            results_path=results_path,
            rollouts_path=rollouts_path,
            alpha=point.alpha,
            beta=point.beta,
            baseline=False,
            logger=logger,
        )

    await inference_pool.stop()
    if teacher_pool is not None:
        await teacher_pool.stop()


def main() -> None:
    config = parse_argv(LandscapeConfig)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger_obj = setup_logger(
        config.log.level,
        log_file=output_dir / "logs" / "landscape.log" if config.log.file else None,
    )
    intercept_verifiers_logging(level=config.log.vf_level)
    logger_obj.info("Starting landscape sweep")

    inference_process: Popen | None = None
    log_dir = get_log_dir(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        if config.start_inference:
            inference_file = get_temp_toml_file()
            with open(inference_file, "wb") as f:
                json_config = config.inference.model_dump(exclude_none=True, mode="json")
                tomli_w.dump(json_config, f)

            inference_cmd = ["uv", "run", "inference", "@", inference_file.as_posix()]
            logger_obj.info(f"Starting inference process on GPU(s) {' '.join(map(str, config.inference_gpu_ids))}")
            logger_obj.debug(f"Inference start command: {' '.join(inference_cmd)}")
            with open(log_dir / "inference.stdout", "w") as log_file:
                inference_process = Popen(
                    inference_cmd,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.inference_gpu_ids))},
                    stdout=log_file,
                    stderr=log_file,
                )

        setup_torch_distributed(enable_gloo=config.trainer.model.fsdp_cpu_offload)
        torch.set_float32_matmul_precision("high")

        parallel_dims = get_parallel_dims(config.trainer.model)
        model = setup_model(config.trainer.model, parallel_dims, loading_from_checkpoint_later=False)
        model.eval()

        params = _iter_named_parameters(model, config.sweep.direction.param_filter)
        delta_direction = None
        eta_direction = None
        if config.sweep.direction.delta_path:
            delta_state = _load_direction_state_dict(config.sweep.direction.delta_path)
            delta_direction = _prepare_direction_tensors(params, delta_state)
        if config.sweep.direction.eta_path:
            eta_state = _load_direction_state_dict(config.sweep.direction.eta_path)
            eta_direction = _prepare_direction_tensors(params, eta_state)
        if config.sweep.direction.norm == "global":
            scale_delta = _compute_global_scale(
                params, config.sweep.direction.seed_delta, config.sweep.direction.epsilon
            )
            scale_eta = _compute_global_scale(params, config.sweep.direction.seed_eta, config.sweep.direction.epsilon)
        else:
            scale_delta = 1.0
            scale_eta = 1.0

        _write_metadata(config, output_dir)

        asyncio.run(
            _run_sweep(
                config,
                output_dir,
                model,
                params,
                parallel_dims,
                config.trainer.model.lora is not None,
                scale_delta,
                scale_eta,
                delta_direction,
                eta_direction,
                logger_obj,
            )
        )
    finally:
        if inference_process is not None and inference_process.poll() is None:
            inference_process.terminate()
            try:
                inference_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                inference_process.kill()


if __name__ == "__main__":
    main()
