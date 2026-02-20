import time
from dataclasses import dataclass
from pathlib import Path

import torch
import verifiers as vf
from transformers import AutoProcessor

from prime_rl.landscape.config import LandscapeConfig
from prime_rl.landscape.directions import apply_point, compute_parameter_delta_stats
from prime_rl.landscape.eval_loss import LOSS_DIAGNOSTIC_COLUMNS, compute_eval_loss, micro_batch_to_tensor
from prime_rl.landscape.io import append_result, append_sampled_prompts
from prime_rl.landscape.weights import write_weights
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.trajectories import branch_rollout, build_vlm_image_cache, interleave_rollout
from prime_rl.orchestrator.utils import get_sampling_args, set_semaphore
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.model import reshard_module
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import get_env_ids_to_install, install_env
from prime_rl.utils.vf import generate_batch, get_completion_len
from prime_rl.utils.vlm import is_vlm_model


@dataclass(frozen=True)
class SweepPoint:
    alpha: float
    beta: float


@dataclass(frozen=True)
class FixedOldPolicyBatch:
    micro_batches: list[dict]
    reward_mean: float
    reward_std: float
    num_rollouts: int
    num_train_samples: int
    adv_mean: float
    adv_std: float
    adv_abs_mean: float
    adv_nonzero_frac: float
    loss_mask_true_frac: float


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


def _compute_reward_stats(rollouts: list[vf.State]) -> tuple[float, float]:
    rewards = [rollout["reward"] for rollout in rollouts]
    reward_mean = float(sum(rewards) / max(len(rewards), 1))
    reward_std = float(torch.tensor(rewards).float().std(unbiased=False).item()) if rewards else 0.0
    return reward_mean, reward_std


def _build_train_examples(
    *,
    config: LandscapeConfig,
    rollouts: list[vf.State],
    rollouts_per_example: int,
    is_vlm: bool,
    processor,
) -> tuple[list, list[float]]:
    rewards = [rollout["reward"] for rollout in rollouts]
    completion_lens = [get_completion_len(rollout) for rollout in rollouts]
    advantages = compute_advantages(
        rewards,
        completion_lens,
        rollouts_per_example,
        config.orchestrator.advantage,
    )

    rollout_fn = interleave_rollout if config.orchestrator.trajectory_strategy == "interleaved" else branch_rollout
    vlm_cache = build_vlm_image_cache(rollouts, processor) if is_vlm else None

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

    if not train_examples:
        raise ValueError("No training samples were produced from rollouts")

    return train_examples, advantages


def _prepare_micro_batches(config: LandscapeConfig, train_examples: list) -> list[dict]:
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
    return [micro_batch_to_tensor(mb) for mb in micro_batches_grid[0]]


async def _generate_rollouts_with_current_weights(
    *,
    config: LandscapeConfig,
    model: torch.nn.Module,
    inference_pool,
    env,
    examples,
    rollouts_per_example: int,
    sampling_args: dict,
    temperature: float,
    weight_dir: Path,
    pbar_description: str,
) -> list[vf.State]:
    write_weights(
        model,
        weight_dir,
        save_format=config.trainer.weight_broadcast.save_format,
        save_sharded=config.trainer.weight_broadcast.save_sharded,
    )
    await inference_pool.update_weights(weight_dir)

    rollouts = await generate_batch(
        clients=inference_pool.clients,
        env=env,
        model_name=config.orchestrator.model.name,
        examples=examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        pbar_description=pbar_description,
    )
    _ensure_rollout_temperatures(rollouts, temperature)
    return rollouts


async def _collect_fixed_old_policy_batch(
    *,
    config: LandscapeConfig,
    model: torch.nn.Module,
    inference_pool,
    env,
    examples,
    rollouts_per_example: int,
    sampling_args: dict,
    temperature: float,
    is_vlm: bool,
    processor,
    weight_dir: Path,
    logger,
) -> FixedOldPolicyBatch:
    logger.info("Collecting fixed old-policy batch at alpha=0.000 beta=0.000")
    rollouts = await _generate_rollouts_with_current_weights(
        config=config,
        model=model,
        inference_pool=inference_pool,
        env=env,
        examples=examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        temperature=temperature,
        weight_dir=weight_dir,
        pbar_description="Rollouts (fixed old policy)",
    )

    train_examples, advantages = _build_train_examples(
        config=config,
        rollouts=rollouts,
        rollouts_per_example=rollouts_per_example,
        is_vlm=is_vlm,
        processor=processor,
    )
    micro_batches = _prepare_micro_batches(config, train_examples)
    reward_mean, reward_std = _compute_reward_stats(rollouts)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    adv_mean = float(advantages_tensor.mean().item()) if advantages else 0.0
    adv_std = float(advantages_tensor.std(unbiased=False).item()) if advantages else 0.0
    adv_abs_mean = float(advantages_tensor.abs().mean().item()) if advantages else 0.0
    adv_nonzero_frac = float((advantages_tensor != 0).float().mean().item()) if advantages else 0.0
    loss_mask_true_count = sum(int(micro_batch["loss_mask"].sum().item()) for micro_batch in micro_batches)
    loss_mask_total_count = sum(int(micro_batch["loss_mask"].numel()) for micro_batch in micro_batches)
    loss_mask_true_frac = float(loss_mask_true_count / max(loss_mask_total_count, 1))

    logger.info(
        f"Prepared fixed old-policy batch: num_rollouts={len(rollouts)} num_train_samples={len(train_examples)}"
    )
    logger.info(
        "Fixed old-policy diagnostics: "
        f"adv_mean={adv_mean:.8e} adv_std={adv_std:.8e} adv_nonzero_frac={adv_nonzero_frac:.6f} "
        f"loss_mask_true_frac={loss_mask_true_frac:.6f}"
    )
    if adv_nonzero_frac == 0.0:
        logger.warning(
            "Fixed old-policy diagnostics: all advantages are zero; loss can collapse to 0.0 unless loss.kl_tau > 0."
        )
    return FixedOldPolicyBatch(
        micro_batches=micro_batches,
        reward_mean=reward_mean,
        reward_std=reward_std,
        num_rollouts=len(rollouts),
        num_train_samples=len(train_examples),
        adv_mean=adv_mean,
        adv_std=adv_std,
        adv_abs_mean=adv_abs_mean,
        adv_nonzero_frac=adv_nonzero_frac,
        loss_mask_true_frac=loss_mask_true_frac,
    )


async def _evaluate_reward_online_point(
    *,
    config: LandscapeConfig,
    model: torch.nn.Module,
    inference_pool,
    env,
    examples,
    rollouts_per_example: int,
    sampling_args: dict,
    temperature: float,
    weight_dir: Path,
    alpha: float,
    beta: float,
) -> dict:
    start_time = time.perf_counter()
    rollouts = await _generate_rollouts_with_current_weights(
        config=config,
        model=model,
        inference_pool=inference_pool,
        env=env,
        examples=examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        temperature=temperature,
        weight_dir=weight_dir,
        pbar_description=f"Rollouts (alpha={alpha:.3f}, beta={beta:.3f})",
    )
    reward_mean, reward_std = _compute_reward_stats(rollouts)
    return {
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "num_rollouts": len(rollouts),
        "elapsed_reward_s": time.perf_counter() - start_time,
    }


def _is_origin(alpha: float, beta: float, tol: float = 1e-12) -> bool:
    return abs(alpha) <= tol and abs(beta) <= tol


async def run_sweep(
    config: LandscapeConfig,
    output_dir: Path,
    model: torch.nn.Module,
    params: list[tuple[str, torch.nn.Parameter]],
    base_tensors: dict[str, torch.Tensor],
    parallel_dims,
    delta_direction: dict[str, torch.Tensor],
    eta_direction: dict[str, torch.Tensor],
    compute_device: torch.device,
    logger,
) -> None:
    logger.info(
        "Running sweep with "
        f"alpha={config.sweep.grid.alpha_min} to {config.sweep.grid.alpha_max} and "
        f"beta={config.sweep.grid.beta_min} to {config.sweep.grid.beta_max}"
    )

    eval_mode = config.sweep.eval_mode
    run_loss_fixed_batch = eval_mode in ("loss_fixed_batch", "both")
    run_reward_online = eval_mode in ("reward_online", "both")

    inference_pool = await setup_inference_pool(
        config.orchestrator.client,
        base_model=config.orchestrator.model.name,
    )

    try:
        await inference_pool.wait_for_ready(config.orchestrator.model.name)

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
                config.orchestrator.model.name,
                trust_remote_code=config.orchestrator.model.trust_remote_code,
                use_fast=True,
            )

        sweep_points = _prepare_sweep_points(config.sweep.grid)

        weight_dir = output_dir / config.sweep.weights_dir
        results_path = output_dir / config.sweep.results_file
        sampled_prompts_path = output_dir / config.sweep.rollouts_file
        sampled_prompts_hash = append_sampled_prompts(sampled_prompts_path, examples)
        logger.info(
            f"Appended sampled prompts to {sampled_prompts_path} "
            f"(count={len(examples)} sha256={sampled_prompts_hash})"
        )

        fixed_old_batch = None
        if run_loss_fixed_batch:
            with torch.no_grad():
                apply_point(params, base_tensors, delta_direction, eta_direction, 0.0, 0.0)
                reshard_module(model)
            fixed_old_batch = await _collect_fixed_old_policy_batch(
                config=config,
                model=model,
                inference_pool=inference_pool,
                env=env,
                examples=examples,
                rollouts_per_example=rollouts_per_example,
                sampling_args=sampling_args,
                temperature=temperature,
                is_vlm=is_vlm,
                processor=processor,
                weight_dir=weight_dir,
                logger=logger,
            )

        for point in sweep_points:
            with torch.no_grad():
                apply_point(params, base_tensors, delta_direction, eta_direction, point.alpha, point.beta)
                # Ensure FSDP modules rebuild full params from the updated shards on the next forward.
                reshard_module(model)
                delta_l2_norm, delta_max_abs = compute_parameter_delta_stats(params, base_tensors)
            logger.debug(
                f"Applied perturbation alpha={point.alpha:.6f} beta={point.beta:.6f} "
                f"||theta-theta0||={delta_l2_norm:.8e} max_abs_delta={delta_max_abs:.8e}"
            )

            point_start = time.perf_counter()
            row = {
                "alpha": point.alpha,
                "beta": point.beta,
                "loss": None,
                "reward_mean": None,
                "reward_std": None,
                "num_rollouts": None,
                "reward_old_mean": fixed_old_batch.reward_mean if fixed_old_batch is not None else None,
                "reward_old_std": fixed_old_batch.reward_std if fixed_old_batch is not None else None,
                "num_rollouts_old": fixed_old_batch.num_rollouts if fixed_old_batch is not None else None,
                "num_train_samples_old": fixed_old_batch.num_train_samples if fixed_old_batch is not None else None,
                "adv_old_mean": fixed_old_batch.adv_mean if fixed_old_batch is not None else None,
                "adv_old_std": fixed_old_batch.adv_std if fixed_old_batch is not None else None,
                "adv_old_abs_mean": fixed_old_batch.adv_abs_mean if fixed_old_batch is not None else None,
                "adv_old_nonzero_frac": fixed_old_batch.adv_nonzero_frac if fixed_old_batch is not None else None,
                "loss_mask_old_true_frac": fixed_old_batch.loss_mask_true_frac if fixed_old_batch is not None else None,
                "num_examples": len(examples),
                "eval_mode": eval_mode,
                "baseline": _is_origin(point.alpha, point.beta),
                "elapsed_loss_s": None,
                "elapsed_reward_s": None,
                "elapsed_s": None,
                **{key: None for key in LOSS_DIAGNOSTIC_COLUMNS},
            }

            if run_loss_fixed_batch:
                assert fixed_old_batch is not None
                loss_start = time.perf_counter()
                loss, loss_diagnostics = compute_eval_loss(
                    model,
                    fixed_old_batch.micro_batches,
                    config.trainer.loss,
                    parallel_dims,
                    compute_device,
                    eval_tag=f"alpha={point.alpha:.6f},beta={point.beta:.6f}",
                )
                row["loss"] = loss
                row.update(loss_diagnostics)
                row["elapsed_loss_s"] = time.perf_counter() - loss_start

            if run_reward_online:
                reward_metrics = await _evaluate_reward_online_point(
                    config=config,
                    model=model,
                    inference_pool=inference_pool,
                    env=env,
                    examples=examples,
                    rollouts_per_example=rollouts_per_example,
                    sampling_args=sampling_args,
                    temperature=temperature,
                    weight_dir=weight_dir,
                    alpha=point.alpha,
                    beta=point.beta,
                )
                row.update(reward_metrics)

            row["elapsed_s"] = time.perf_counter() - point_start
            append_result(results_path, row)

            summary = [f"alpha={point.alpha:.3f}", f"beta={point.beta:.3f}"]
            if row["loss"] is not None:
                summary.append(f"loss={row['loss']:.4f}")
                summary.append(f"mismatch_kl={row['loss_mismatch_kl_mean']:.4f}")
            if row["reward_mean"] is not None:
                summary.append(f"reward_mean={row['reward_mean']:.4f}")
            logger.info(" ".join(summary))
    finally:
        await inference_pool.stop()
