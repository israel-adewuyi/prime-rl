import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from torch.distributed.tensor import DTensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def iter_named_parameters(model: torch.nn.Module, param_filter: str) -> list[tuple[str, torch.nn.Parameter]]:
    if param_filter == "all":
        params = [(name, param) for name, param in model.named_parameters()]
    else:
        params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not params:
        raise ValueError("No parameters selected for perturbation")
    return params


def _check_single_device(params: Iterable[torch.nn.Parameter | torch.Tensor]) -> torch.device:
    device = None
    for param in params:
        local_param = get_local_tensor(param)
        if device is None:
            device = local_param.device
        elif local_param.device != device:
            raise ValueError("All parameters must be on the same device for landscape perturbations")
    assert device is not None
    return device


def _unwrap_param_tensor(param: torch.nn.Parameter | torch.Tensor) -> torch.Tensor:
    if isinstance(param, torch.nn.Parameter):
        return param.data
    return param


def _is_dtensor_param(param: torch.nn.Parameter | torch.Tensor) -> bool:
    return isinstance(_unwrap_param_tensor(param), DTensor)


def get_local_tensor(param: torch.nn.Parameter | torch.Tensor) -> torch.Tensor:
    tensor = _unwrap_param_tensor(param)
    if isinstance(tensor, DTensor):
        if hasattr(tensor, "to_local"):
            return tensor.to_local()
        return tensor._local_tensor
    return tensor


def _maybe_all_reduce(tensor: torch.Tensor) -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _maybe_all_reduce_max(tensor: torch.Tensor) -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)


def build_random_direction(
    params: list[tuple[str, torch.nn.Parameter]],
    base_tensors: dict[str, torch.Tensor],
    seed: int,
    norm: str,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    device = _check_single_device(get_local_tensor(param) for _, param in params)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    raw = {}
    total_param_sq = torch.tensor(0.0, device=device)
    total_dir_sq = torch.tensor(0.0, device=device)
    for name, param in params:
        if not param.is_floating_point():
            continue
        base_tensor = base_tensors[name]
        raw_dir = torch.randn_like(base_tensor, generator=generator)
        raw[name] = raw_dir
        if norm == "global":
            total_param_sq = total_param_sq + base_tensor.float().pow(2).sum()
            total_dir_sq = total_dir_sq + raw_dir.float().pow(2).sum()

    if norm == "global":
        _maybe_all_reduce(total_param_sq)
        _maybe_all_reduce(total_dir_sq)
        if total_dir_sq.item() == 0.0:
            raise ValueError("Direction has zero norm")
        scale = torch.sqrt(total_param_sq) / (torch.sqrt(total_dir_sq) + epsilon)

    direction = {}
    for name, param in params:
        if not param.is_floating_point():
            continue
        base_tensor = base_tensors[name]
        raw_dir = raw[name]
        if norm == "layer":
            param_sq = base_tensor.float().pow(2).sum()
            dir_sq = raw_dir.float().pow(2).sum()
            _maybe_all_reduce(param_sq)
            _maybe_all_reduce(dir_sq)
            if dir_sq.item() == 0.0:
                raise ValueError(f"Direction has zero norm for parameter {name}")
            scale = torch.sqrt(param_sq) / (torch.sqrt(dir_sq) + epsilon)
        direction[name] = raw_dir * scale
    return direction


def apply_point(
    params: list[tuple[str, torch.nn.Parameter]],
    base_tensors: dict[str, torch.Tensor],
    delta_direction: dict[str, torch.Tensor],
    eta_direction: dict[str, torch.Tensor],
    alpha: float,
    beta: float,
) -> None:
    for name, param in params:
        if not param.is_floating_point():
            continue
        base_tensor = base_tensors[name]
        delta = delta_direction[name]
        eta = eta_direction[name]
        updated = base_tensor + alpha * delta + beta * eta
        local_tensor = get_local_tensor(param)
        updated = updated.to(device=local_tensor.device, dtype=local_tensor.dtype)
        target_tensor = _unwrap_param_tensor(param)
        if isinstance(target_tensor, DTensor):
            updated = DTensor.from_local(updated, target_tensor.device_mesh, target_tensor.placements)
        target_tensor.copy_(updated)


def compute_parameter_delta_stats(
    params: list[tuple[str, torch.nn.Parameter]],
    base_tensors: dict[str, torch.Tensor],
) -> tuple[float, float]:
    device = _check_single_device(get_local_tensor(param) for _, param in params)
    total_delta_sq = torch.tensor(0.0, device=device)
    max_abs_delta = torch.tensor(0.0, device=device)
    for name, param in params:
        if not param.is_floating_point():
            continue
        local_tensor = get_local_tensor(param).detach()
        delta = (local_tensor - base_tensors[name]).float()
        total_delta_sq = total_delta_sq + delta.pow(2).sum()
        max_abs_delta = torch.maximum(max_abs_delta, delta.abs().max())
    _maybe_all_reduce(total_delta_sq)
    _maybe_all_reduce_max(max_abs_delta)
    l2_norm = torch.sqrt(torch.clamp(total_delta_sq, min=0.0)).item()
    return float(l2_norm), float(max_abs_delta.item())


def load_direction_state_dict(path: str) -> dict[str, torch.Tensor]:
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


def _find_direction_key_with_suffix(direction_state: dict[str, torch.Tensor], suffix: str) -> str | None:
    matches = [key for key in direction_state if key.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_tied_weight_alias(
    missing_name: str,
    direction_state: dict[str, torch.Tensor],
    tie_word_embeddings: bool,
) -> str | None:
    if not tie_word_embeddings:
        return None

    if missing_name.endswith("lm_head.weight"):
        if "model.embed_tokens.weight" in direction_state:
            return "model.embed_tokens.weight"
        return _find_direction_key_with_suffix(direction_state, "embed_tokens.weight")

    if missing_name.endswith("embed_tokens.weight"):
        if "lm_head.weight" in direction_state:
            return "lm_head.weight"
        return _find_direction_key_with_suffix(direction_state, "lm_head.weight")

    return None


def prepare_direction_tensors(
    params: list[tuple[str, torch.nn.Parameter]],
    direction_state: dict[str, torch.Tensor],
    direction_name: str,
    logger_obj,
    tie_word_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    selected_names = [name for name, _ in params]
    selected_name_set = set(selected_names)
    direction_keys = set(direction_state.keys())
    extras = sorted(direction_keys - selected_name_set)
    if extras:
        logger_obj.info(f"{direction_name} has {len(extras)} extra keys not in selected parameters; they will be ignored")
        logger_obj.debug(f"{direction_name} extra keys: {extras}")

    direction = {}
    for name, param in params:
        if name in direction_state:
            source_name = name
        else:
            alias_name = _resolve_tied_weight_alias(name, direction_state, tie_word_embeddings)
            if alias_name is None:
                raise ValueError(f"Direction state dict is missing parameter: {name}")
            source_name = alias_name
            logger_obj.info(
                f"{direction_name} missing {name}; using tied-weight alias from {alias_name} "
                f"(tie_word_embeddings={tie_word_embeddings})"
            )

        tensor = direction_state[source_name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Direction tensor for {source_name} (resolved for {name}) in {direction_name} is not a torch.Tensor"
            )
        local_tensor = get_local_tensor(param)
        if tensor.shape != local_tensor.shape:
            if _is_dtensor_param(param) and tensor.shape == param.shape:
                if dist.is_initialized() and dist.get_world_size() > 1:
                    raise ValueError(
                        f"Direction tensor for {name} has global shape {tensor.shape}, "
                        f"expected local shape {local_tensor.shape}."
                    )
            else:
                raise ValueError(f"Direction tensor shape mismatch for {name}: {tensor.shape} vs {local_tensor.shape}")
        direction[name] = tensor.to(device=local_tensor.device, dtype=torch.float32)
    return direction


def _global_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot = torch.dot(a.float(), b.float())
    _maybe_all_reduce(dot)
    return dot


def _global_norm(vector: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(_global_dot(vector, vector), min=0.0))


def _direction_stats(
    names: list[str],
    delta_direction: dict[str, torch.Tensor],
    eta_direction: dict[str, torch.Tensor],
    epsilon: float,
) -> tuple[float, float, float, float]:
    delta_vector = parameters_to_vector([delta_direction[name].float() for name in names])
    eta_vector = parameters_to_vector([eta_direction[name].float() for name in names])
    delta_norm = _global_norm(delta_vector).item()
    eta_norm = _global_norm(eta_vector).item()
    dot = _global_dot(delta_vector, eta_vector).item()
    cosine = dot / (delta_norm * eta_norm + epsilon)
    return delta_norm, eta_norm, dot, cosine


def log_direction_stats(
    names: list[str],
    delta_direction: dict[str, torch.Tensor],
    eta_direction: dict[str, torch.Tensor],
    epsilon: float,
    label: str,
    logger_obj,
) -> None:
    delta_norm, eta_norm, dot, cosine = _direction_stats(
        names=names,
        delta_direction=delta_direction,
        eta_direction=eta_direction,
        epsilon=epsilon,
    )
    logger_obj.info(f"{label}: ||delta||={delta_norm:.8e}, ||eta||={eta_norm:.8e}, dot={dot:.8e}, cos(theta)={cosine:.8e}")


def orthogonalize_and_normalize_directions(
    params: list[tuple[str, torch.nn.Parameter]],
    delta_direction: dict[str, torch.Tensor],
    eta_direction: dict[str, torch.Tensor],
    epsilon: float,
    collinear_threshold: float,
    fallback_seed: int,
    logger_obj,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    floating_names = [name for name, param in params if param.is_floating_point()]
    if not floating_names:
        raise ValueError("No floating-point parameters available for orthogonalization")

    logger_obj.info(f"Orthogonalizing {len(floating_names)} floating-point tensors")
    log_direction_stats(
        names=floating_names,
        delta_direction=delta_direction,
        eta_direction=eta_direction,
        epsilon=epsilon,
        label="Before orthogonalization",
        logger_obj=logger_obj,
    )

    delta_tensors = [delta_direction[name].float() for name in floating_names]
    eta_tensors = [eta_direction[name].float() for name in floating_names]
    delta_vector = parameters_to_vector(delta_tensors)
    eta_vector = parameters_to_vector(eta_tensors)

    delta_norm = _global_norm(delta_vector)
    if delta_norm.item() <= epsilon:
        raise ValueError("delta direction has near-zero norm; cannot orthogonalize")
    u1 = delta_vector / (delta_norm + epsilon)

    proj = _global_dot(eta_vector, u1)
    eta_orth = eta_vector - proj * u1
    eta_orth_norm = _global_norm(eta_orth)

    if eta_orth_norm.item() <= max(collinear_threshold, epsilon):
        logger_obj.warning(
            f"eta direction is nearly collinear with delta (||eta_orth||={eta_orth_norm.item():.8e}); "
            "sampling random orthogonal fallback"
        )
        generator = torch.Generator(device=eta_vector.device)
        generator.manual_seed(fallback_seed)
        random_vector = torch.randn_like(eta_vector, generator=generator)
        random_proj = _global_dot(random_vector, u1)
        eta_orth = random_vector - random_proj * u1
        eta_orth_norm = _global_norm(eta_orth)
        if eta_orth_norm.item() <= epsilon:
            raise ValueError("Random fallback direction also has near-zero norm after projection")

    u2 = eta_orth / (eta_orth_norm + epsilon)

    delta_orth_tensors = [torch.empty_like(tensor) for tensor in delta_tensors]
    eta_orth_tensors = [torch.empty_like(tensor) for tensor in eta_tensors]
    vector_to_parameters(u1, delta_orth_tensors)
    vector_to_parameters(u2, eta_orth_tensors)

    delta_orth = {name: tensor for name, tensor in zip(floating_names, delta_orth_tensors)}
    eta_orth_dict = {name: tensor for name, tensor in zip(floating_names, eta_orth_tensors)}

    log_direction_stats(
        names=floating_names,
        delta_direction=delta_orth,
        eta_direction=eta_orth_dict,
        epsilon=epsilon,
        label="After Gram-Schmidt",
        logger_obj=logger_obj,
    )

    return delta_orth, eta_orth_dict


def save_direction_state_dict(direction: dict[str, torch.Tensor], output_path: Path, logger_obj) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_direction = {name: tensor.detach().cpu() for name, tensor in direction.items()}
    torch.save(cpu_direction, output_path)
    logger_obj.info(f"Saved direction state dict to {output_path}")


def _infer_config_stem_from_argv() -> str:
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == "@" and idx + 1 < len(argv):
            return Path(argv[idx + 1]).stem
        if arg.startswith("@") and len(arg) > 1:
            return Path(arg[1:]).stem
    return "landscape"


def build_orthogonalized_direction_paths(output_dir: Path, orthogonalized_subdir: Path, orthogonalized_suffix: str) -> tuple[Path, Path]:
    config_stem = _infer_config_stem_from_argv()
    suffix = orthogonalized_suffix.strip()
    suffix_part = f"_{suffix}" if suffix else ""
    directions_dir = output_dir / orthogonalized_subdir
    delta_path = directions_dir / f"{config_stem}_delta{suffix_part}.pt"
    eta_path = directions_dir / f"{config_stem}_eta{suffix_part}.pt"
    return delta_path, eta_path


def sanity_check_restore_base(
    params: list[tuple[str, torch.nn.Parameter]],
    base_tensors: dict[str, torch.Tensor],
    delta_direction: dict[str, torch.Tensor],
    eta_direction: dict[str, torch.Tensor],
    logger_obj,
) -> None:
    with torch.no_grad():
        apply_point(params, base_tensors, delta_direction, eta_direction, alpha=1e-3, beta=-1e-3)
        apply_point(params, base_tensors, delta_direction, eta_direction, alpha=0.0, beta=0.0)

    max_abs_diff = 0.0
    for name, param in params:
        if not param.is_floating_point():
            continue
        restored = get_local_tensor(param).detach()
        expected = base_tensors[name]
        diff = (restored - expected).abs().max().item()
        if diff > max_abs_diff:
            max_abs_diff = diff
    logger_obj.info(f"Restore sanity check after perturbation: max_abs_diff={max_abs_diff:.8e}")
