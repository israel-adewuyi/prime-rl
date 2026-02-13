import asyncio
import os
import subprocess
from pathlib import Path
from subprocess import Popen

import tomli_w
import torch
import torch.distributed as dist

from prime_rl.landscape.config import LandscapeConfig
from prime_rl.landscape.directions import (
    build_orthogonalized_direction_paths,
    build_random_direction,
    get_local_tensor,
    iter_named_parameters,
    load_direction_state_dict,
    log_direction_stats,
    orthogonalize_and_normalize_directions,
    prepare_direction_tensors,
    save_direction_state_dict,
    sanity_check_restore_base,
)
from prime_rl.landscape.io import write_metadata
from prime_rl.landscape.sweep import run_sweep
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.logger import intercept_verifiers_logging, setup_logger
from prime_rl.utils.pydantic_config import get_temp_toml_file, parse_argv
from prime_rl.utils.utils import get_log_dir


def _configure_trainer_cuda_visible_devices(config: LandscapeConfig, logger_obj) -> None:
    visible_devices = ",".join(map(str, config.trainer_gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    local_world_size_env = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size_env is not None and int(local_world_size_env) != len(config.trainer_gpu_ids):
        raise ValueError(
            f"LOCAL_WORLD_SIZE={local_world_size_env} but trainer_gpu_ids has {len(config.trainer_gpu_ids)} entries. "
            "When using torchrun, set trainer_gpu_ids length to match LOCAL_WORLD_SIZE."
        )

    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None and int(local_rank_env) >= len(config.trainer_gpu_ids):
        raise ValueError(
            f"LOCAL_RANK={local_rank_env} is out of range for trainer_gpu_ids length {len(config.trainer_gpu_ids)}"
        )

    logger_obj.info(f"Configured trainer visible GPU(s): {visible_devices}")


def _resolve_compute_device(params: list[tuple[str, torch.nn.Parameter]]) -> torch.device:
    for _, param in params:
        if param.is_floating_point():
            return get_local_tensor(param).device
    if not params:
        raise ValueError("No parameters selected for perturbation")
    return get_local_tensor(params[0][1]).device


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
        _configure_trainer_cuda_visible_devices(config, logger_obj)

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
        tie_word_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))
        logger_obj.info(f"Model tie_word_embeddings={tie_word_embeddings}")

        params = iter_named_parameters(model, config.sweep.direction.param_filter)
        compute_device = _resolve_compute_device(params)
        logger_obj.info(f"Landscape loss compute device: {compute_device}")
        base_tensors = {
            name: get_local_tensor(param).detach().clone() for name, param in params if param.is_floating_point()
        }

        delta_direction = None
        eta_direction = None
        if config.sweep.direction.orthogonalize_paths:
            if not config.sweep.direction.delta_path or not config.sweep.direction.eta_path:
                raise ValueError("orthogonalize_paths=true requires both sweep.direction.delta_path and eta_path")

            logger_obj.info("Orthogonalization toggle is enabled; loading delta/eta from configured paths")
            delta_state_raw = load_direction_state_dict(config.sweep.direction.delta_path)
            eta_state_raw = load_direction_state_dict(config.sweep.direction.eta_path)
            delta_loaded = prepare_direction_tensors(
                params,
                delta_state_raw,
                "delta",
                logger_obj,
                tie_word_embeddings=tie_word_embeddings,
            )
            eta_loaded = prepare_direction_tensors(
                params,
                eta_state_raw,
                "eta",
                logger_obj,
                tie_word_embeddings=tie_word_embeddings,
            )

            logger_obj.info("Starting orthogonalization + per-tensor normalization pipeline")
            delta_orth, eta_orth = orthogonalize_and_normalize_directions(
                params=params,
                delta_direction=delta_loaded,
                eta_direction=eta_loaded,
                epsilon=config.sweep.direction.epsilon,
                collinear_threshold=config.sweep.direction.collinear_threshold,
                fallback_seed=config.sweep.direction.seed_eta + 10_000,
                logger_obj=logger_obj,
            )

            sanity_check_restore_base(params, base_tensors, delta_orth, eta_orth, logger_obj)

            orth_delta_path, orth_eta_path = build_orthogonalized_direction_paths(
                output_dir=output_dir,
                orthogonalized_subdir=config.sweep.direction.orthogonalized_subdir,
                orthogonalized_suffix=config.sweep.direction.orthogonalized_suffix,
            )
            save_direction_state_dict(delta_orth, orth_delta_path, logger_obj)
            save_direction_state_dict(eta_orth, orth_eta_path, logger_obj)

            config.sweep.direction.delta_path = str(orth_delta_path)
            config.sweep.direction.eta_path = str(orth_eta_path)
            logger_obj.info(
                "Updated sweep.direction paths to generated orthogonalized files: "
                f"delta_path={config.sweep.direction.delta_path}, eta_path={config.sweep.direction.eta_path}"
            )

        if config.sweep.direction.delta_path:
            delta_state = load_direction_state_dict(config.sweep.direction.delta_path)
            delta_direction = prepare_direction_tensors(
                params,
                delta_state,
                "delta",
                logger_obj,
                tie_word_embeddings=tie_word_embeddings,
            )
        if config.sweep.direction.eta_path:
            eta_state = load_direction_state_dict(config.sweep.direction.eta_path)
            eta_direction = prepare_direction_tensors(
                params,
                eta_state,
                "eta",
                logger_obj,
                tie_word_embeddings=tie_word_embeddings,
            )

        if delta_direction is None:
            logger_obj.info("No delta_path configured; building random delta direction")
            delta_direction = build_random_direction(
                params,
                base_tensors,
                config.sweep.direction.seed_delta,
                config.sweep.direction.norm,
                config.sweep.direction.epsilon,
            )
        if eta_direction is None:
            logger_obj.info("No eta_path configured; building random eta direction")
            eta_direction = build_random_direction(
                params,
                base_tensors,
                config.sweep.direction.seed_eta,
                config.sweep.direction.norm,
                config.sweep.direction.epsilon,
            )

        floating_names = [name for name, param in params if param.is_floating_point()]
        log_direction_stats(
            names=floating_names,
            delta_direction=delta_direction,
            eta_direction=eta_direction,
            epsilon=config.sweep.direction.epsilon,
            label="Final directions entering sweep",
            logger_obj=logger_obj,
        )

        write_metadata(config, output_dir)

        asyncio.run(
            run_sweep(
                config,
                output_dir,
                model,
                params,
                base_tensors,
                parallel_dims,
                delta_direction,
                eta_direction,
                compute_device,
                logger_obj,
            )
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if inference_process is not None and inference_process.poll() is None:
            inference_process.terminate()
            try:
                inference_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                inference_process.kill()


if __name__ == "__main__":
    main()
