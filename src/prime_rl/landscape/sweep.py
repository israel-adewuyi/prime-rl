import time
from dataclasses import dataclass
from pathlib import Path

import torch
import verifiers as vf
from transformers import AutoProcessor

from prime_rl.landscape.config import LandscapeConfig
from prime_rl.landscape.directions import apply_point
from prime_rl.landscape.eval_loss import compute_eval_loss, micro_batch_to_tensor
from prime_rl.landscape.io import append_result, append_rollouts
from prime_rl.landscape.weights import write_weights
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.trajectories import branch_rollout, build_vlm_image_cache, interleave_rollout
from prime_rl.orchestrator.utils import get_sampling_args, set_semaphore
from prime_rl.trainer.batch import prepare_batch
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import get_env_ids_to_install, install_env
from prime_rl.utils.vf import generate_batch, get_completion_len
from prime_rl.utils.vlm import is_vlm_model


@dataclass(frozen=True)
class SweepPoint:
    alpha: float
    beta: float


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


async def _evaluate_point(
    *,
    config: LandscapeConfig,
    model: torch.nn.Module,
    parallel_dims,
    inference_pool,
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
        pbar_description=f"Rollouts (alpha={alpha:.3f}, beta={beta:.3f})",
    )
    _ensure_rollout_temperatures(rollouts, temperature)
    if not baseline:
        append_rollouts(rollouts_path, rollouts, alpha, beta)
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
    micro_batches = [micro_batch_to_tensor(mb) for mb in micro_batches_grid[0]]
    logger.debug(f"Completed micro-batches for alpha={alpha:.3f}, beta={beta:.3f}")

    loss_value = compute_eval_loss(
        model,
        micro_batches,
        config.trainer.loss,
        parallel_dims,
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
    if not baseline:
        append_result(results_path, row)
    logger.info(
        f"alpha={alpha:.3f} beta={beta:.3f} loss={loss_value:.4f} "
        f"reward_mean={reward_mean:.4f} baseline={baseline}"
    )


async def run_sweep(
    config: LandscapeConfig,
    output_dir: Path,
    model: torch.nn.Module,
    params: list[tuple[str, torch.nn.Parameter]],
    base_tensors: dict[str, torch.Tensor],
    parallel_dims,
    delta_direction: dict[str, torch.Tensor],
    eta_direction: dict[str, torch.Tensor],
    logger,
) -> None:
    logger.info(
        f"Running sweep with alpha={config.sweep.grid.alpha_min} to {config.sweep.grid.alpha_max} and beta={config.sweep.grid.beta_min} to {config.sweep.grid.beta_max}"
    )
    inference_pool = await setup_inference_pool(
        config.orchestrator.client,
        base_model=config.orchestrator.model.name,
    )
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
    rollouts_path = output_dir / config.sweep.rollouts_file

    apply_point(params, base_tensors, delta_direction, eta_direction, 0.0, 0.0)
    await _evaluate_point(
        config=config,
        model=model,
        parallel_dims=parallel_dims,
        inference_pool=inference_pool,
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
        with torch.no_grad():
            apply_point(params, base_tensors, delta_direction, eta_direction, point.alpha, point.beta)

        await _evaluate_point(
            config=config,
            model=model,
            parallel_dims=parallel_dims,
            inference_pool=inference_pool,
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
