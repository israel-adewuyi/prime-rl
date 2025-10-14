import pickle
import time
from collections import OrderedDict, defaultdict
from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import torch
import torch.distributed as dist
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from rich.text import Text
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num, format_time

DEFAULT_TIMEOUT = timedelta(seconds=600)


def setup_torch_distributed(timeout: timedelta = DEFAULT_TIMEOUT):
    torch.cuda.set_device(get_world().local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", torch.cuda.current_device()),
        timeout=timeout,
    )


def get_response_lengths(position_ids: torch.Tensor) -> list[int]:
    """
    Compute lengths of concatenated sequences from position_ids.

    Each sequence starts at 0 and increments. When position_ids resets to 0,
    it indicates the start of a new sequence. Trailing zeros (padding) are
    counted as part of the last sequence.

    Args:
        position_ids: Tensor of shape [total_seqlen]

    Returns:
        List of sequence lengths
    """
    position_ids = position_ids.flatten()

    boundaries = [0]  # Start of first sequence

    for i in range(1, len(position_ids)):
        if position_ids[i] == 0 and position_ids[i - 1] != 0:
            # This is a potential sequence boundary (0 after non-zero)
            # But only if the next element is 1 (indicating a new incrementing sequence)
            # Otherwise, this 0 is padding and belongs to current sequence
            if i + 1 < len(position_ids) and position_ids[i + 1] == 1:
                boundaries.append(i)

    # Calculate lengths based on boundaries
    lengths = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(position_ids)
        lengths.append(end - start)

    return lengths


def get_real_tensor(tensor: Tensor | DTensor) -> Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


OffloadedTensor: TypeAlias = list[tuple[Tensor, int]]


def offload_model_to_cpu(model: nn.Module) -> OffloadedTensor:
    """
    Retun a list of cpu tensor representing the model weight.
    Also reduce to 0 the gpu memory usage.
    """
    tensors_offloaded = []
    for param in chain(model.parameters(), model.buffers()):
        data = get_real_tensor(param.data)
        cpu_data = data.to("cpu", non_blocking=True)
        storage_size = data.untyped_storage().size()
        data.untyped_storage().resize_(1)
        tensors_offloaded.append((cpu_data, storage_size))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return tensors_offloaded


def copy_model_to_cpu(model: nn.Module) -> OffloadedTensor:
    """
    Retun a list of cpu tensor representing the model weight.
    Keep gpu memory intact.
    """

    tensors_offloaded = []
    for param in chain(model.parameters(), model.buffers()):
        data = get_real_tensor(param.data)
        cpu_data = data.to("cpu")
        storage_size = data.untyped_storage().size()
        tensors_offloaded.append((cpu_data, storage_size))

    return tensors_offloaded


def wake_up_model_from_cpu(model: nn.Module, tensors: OffloadedTensor):
    for param, (cpu_data, storage_size) in zip(chain(model.parameters(), model.buffers()), tensors):
        data = get_real_tensor(param.data)
        data.untyped_storage().resize_(storage_size)
        data.copy_(cpu_data, non_blocking=True)
    torch.cuda.synchronize()


def print_sample(input_ids: list[int], loss_mask: list[bool], tokenizer: PreTrainedTokenizer):
    """
    Visualize the loss mask of a tokenized sample using rich.
    Reference: https://huggingface.co/Qwen/Qwen3-8B/discussions/14
    """
    text = Text()
    for token, mask in zip(tokenizer.convert_ids_to_tokens(input_ids), loss_mask):
        text.append(token.replace("Ġ", " ").replace("Ċ", "\n"), style="cyan" if mask else "white")
    rich_print(text)


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """
    Print benchmark results as rich table. Shows formatted values for the
    training throughput and overall step time. First first N rows show the
    per-step values, and the last row shows the mean, std, min, and max values.
    """
    history.pop("step")
    assert all(len(v) for v in history.values()), "All metrics must have logged the same number of steps"

    # Turn metric history into pd.DataFrame
    df = pd.DataFrame(dict(history.items()))
    columns = {
        "perf/mfu": "MFU",
        "perf/throughput": "Throughput",
        "time/step": "Step Time",
        "perf/peak_memory": "Peak Memory",
    }
    df = df[columns.keys()].rename(columns=columns)
    df = df.iloc[1:]  # Exclude first row

    # Setup console
    console = Console()
    table = Table(title="Benchmark")

    # Add columns
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center", style="magenta")

    # Add formatted rows
    formatted_df = pd.DataFrame(columns=df.columns)
    formatted_df["MFU"] = df["MFU"].apply(lambda x: f"{format_num(x, precision=2)}%")
    formatted_df["Throughput"] = df["Throughput"].apply(lambda x: format_num(x, precision=2))
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    formatted_df["Peak Memory"] = df["Peak Memory"].apply(lambda x: f"{format_num(x, precision=1)} GiB")
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    table.add_row(*([""] * len(formatted_df.columns)))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame()
    formatted_mean_df["MFU"] = mean_df["MFU"].apply(lambda x: f"{format_num(x, precision=2)}%")
    formatted_mean_df["Throughput"] = mean_df["Throughput"].apply(format_num, precision=2)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    mean_row = (
        ["Overall"]
        + formatted_mean_df.T.apply(
            lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
        ).tolist()
        + [
            f"{format_num(mean_df['Peak Memory']['mean'], precision=1)} GiB ({mean_df['Peak Memory']['mean'] / (torch.cuda.mem_get_info()[1] / 1024**3) * 100:.1f}%)"
        ]
    )
    table.add_row(*mean_row)

    # Display table
    console.print(table)


def flexible_all_gather(tensor: Tensor) -> Tensor:
    """
    All-gather a 1D tensor between all ranks, with potentially different numbr of element per rank.
    Returns a tensor of shape (world_size * max_numel, dtype=tensor.dtype, device=tensor.device)
    """

    assert tensor.ndim == 1, "Can only flexibly all-gather 1D tensors"

    if dist.get_world_size() == 1:
        return tensor

    # Find the tensor with the most elements
    local_numel = tensor.numel()
    local_numel_tensor = torch.tensor(local_numel, device=tensor.device)
    all_numel_tensors = [torch.tensor(0, device=tensor.device) for _ in range(dist.get_world_size())]
    dist.all_gather(all_numel_tensors, local_numel_tensor)
    all_numels = [numel.item() for numel in all_numel_tensors]
    max_numel = int(max(all_numels))

    # Pad the tensor with zeros if it has less elements than the maximum
    if local_numel < max_numel:
        tensor = torch.cat([tensor, torch.zeros(max_numel - local_numel, dtype=tensor.dtype, device=tensor.device)])

    # All-gather the tensors
    all_tensors = [
        torch.zeros(max_numel, dtype=tensor.dtype, device=tensor.device) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_tensors, tensor)
    all_tensors_unpadded = torch.cat([tensor[:numel] for tensor, numel in zip(all_tensors, all_numels)])

    return all_tensors_unpadded


class Tensors(defaultdict):
    """A class to accumulate tensors and compute statistics (mean, median, std, min, max) across multiple steps and ranks."""

    def __init__(self):
        assert dist.is_initialized(), "Tensors requires a distributed environment"
        super().__init__(list)

    def compute_stats(self) -> dict[str, float | int]:
        """Synchronize the tensor statistic across all ranks for each key and compute relevant statistics."""

        metrics = {}
        for key in list(self.keys()):
            # All-gather tensors across steps and ranks (get global distribution)
            tensors = torch.cat(self.pop(key), dim=0).to("cuda")
            assert tensors.ndim == 1, "Can only aggregate 1D tensors"
            tensors = flexible_all_gather(tensors)
            assert tensors.ndim == 1, "Can only aggregate 1D tensors"

            # Compute relevant tensor statistics
            metrics[f"{key}/mean"] = tensors.mean().item()
            metrics[f"{key}/median"] = torch.median(tensors).item()
            metrics[f"{key}/std"] = tensors.std().item()
            metrics[f"{key}/min"] = tensors.min().item()
            metrics[f"{key}/max"] = tensors.max().item()

            # Add back all-gathered tensors to self
            self[key].append(tensors.tolist())

        return metrics


MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


class MemoryProfiler:
    def __init__(self, step_num: int, snapshot_path: Path):
        torch.cuda.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)
        self.logger = get_logger()
        snapshot_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = snapshot_path
        self.step_num = step_num

    def step(self):
        self.logger.info(f"Dumping memory snapshot at step {self.step_num} at {self.snapshot_path}")
        begin = time.monotonic()
        step_folder = self.snapshot_path / f"step_{self.step_num}"
        step_folder.mkdir(parents=True, exist_ok=True)
        file_path = step_folder / f"rank_{get_world().rank}.pickle"
        with open(file_path, "wb") as output:
            pickle.dump(torch.cuda.memory._snapshot(), output)
        self.logger.info(
            f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds, load {file_path} at https://docs.pytorch.org/memory_viz to visualize the memory usage"
        )
        self.step_num += 1


class GradientAccumulator:
    def __init__(
        self,
        beta: float,
        epsilon: float,
        save_interval: int,
        output_dir,
        model: nn.Module,
        tolerance: float = 1e-5,
        save_masks: bool = True,
        mask_save_interval: int = None,
        upload_to_hf: bool = False,
        hf_repo_id: str = None,
        hf_upload_interval: int = None,
        hf_private: bool = True,
    ):
        self.beta = beta
        self.epsilon = epsilon
        self.interval = save_interval
        self.output_dir = Path(output_dir)
        self.mask_tolerance = tolerance
        self.save_masks = save_masks
        self.mask_save_interval = mask_save_interval or save_interval
        self.upload_to_hf = upload_to_hf
        self.hf_repo_id = hf_repo_id
        self.hf_upload_interval = hf_upload_interval or mask_save_interval or save_interval
        self.hf_private = hf_private

        # Validate HF configuration
        if self.upload_to_hf and not self.hf_repo_id:
            raise ValueError("hf_repo_id must be provided when upload_to_hf is True")

        # Initialize accumulator
        self.acc = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Get HuggingFace-compatible name
                hf_name = get_fqns(model, name)
                assert len(hf_name) == 1, f"Expected single FQN, got {hf_name}"
                hf_name = next(iter(hf_name))

                self.acc[hf_name] = torch.zeros_like(param.data, requires_grad=False, device="cpu")

    def _compute_masks(self) -> OrderedDict:
        """Generate boolean masks based on gradient magnitudes."""
        masks = OrderedDict()
        for name, grad_ema in self.acc.items():
            # Create boolean mask: True if gradient magnitude > tolerance, False otherwise
            mask = grad_ema > self.mask_tolerance
            masks[name] = mask
        return masks

    def _compute_mask_stats(self, masks: OrderedDict) -> dict[str, float]:
        """Compute statistics about mask sparsity."""
        total_params = 0
        active_params = 0

        for name, mask in masks.items():
            layer_total = mask.numel()
            layer_active = mask.sum().item()

            total_params += layer_total
            active_params += layer_active

        active_fraction = active_params / total_params if total_params > 0 else 0.0
        sparsity = 1.0 - active_fraction

        return {
            "grad_mask/active_fraction": active_fraction,
            "grad_mask/active_count": active_params,
            "grad_mask/total_count": total_params,
            "grad_mask/sparsity": sparsity,
        }

    def step(self, model: nn.Module, step: int, monitor, logger):
        # Accumulate EMA of squared gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Translate to HuggingFace name
                hf_name = get_fqns(model, name)
                hf_name = next(iter(hf_name))

                # Compute on GPU, then immediately move to CPU
                grad_sq = (param.grad**2).detach().to("cpu")
                self.acc[hf_name] = self.beta * self.acc[hf_name] + (1 - self.beta) * grad_sq

        self.log(step, monitor, logger)

        if step % self.interval == 0 and step > 0:
            acc_path = self.output_dir / "grad_acc" / f"grad_ema_step_{step}.pt"
            self.save(acc_path)
            logger.info(f"Saved gradient EMA to {acc_path}")

        # Compute and save masks if enabled
        if self.save_masks and step % self.mask_save_interval == 0 and step > 0:
            masks = self._compute_masks()
            mask_path = self.output_dir / "grad_acc" / f"grad_mask_step_{step}.pt"
            self._save_masks(mask_path, masks)
            logger.info(f"Saved gradient masks to {mask_path}")

            # Upload to Hugging Face Hub if enabled
            if self.upload_to_hf and step % self.hf_upload_interval == 0:
                # Get model name from the first parameter (assuming all params have the same base model)
                model_name = "unknown_model"
                if self.acc:
                    first_param_name = next(iter(self.acc.keys()))
                    # Extract model name from parameter name (e.g., "model.embed_tokens.weight" -> "model")
                    model_name = first_param_name.split(".")[0] if "." in first_param_name else "model"

                # Prepare repository on first upload
                if step == self.hf_upload_interval:
                    if not self._prepare_hf_repo(logger):
                        logger.warning("Failed to prepare HF repository, skipping upload")
                        return
                    self._update_hf_readme(model_name, logger)

                # Create metadata and upload
                metadata = self._create_mask_metadata(masks, step, model_name)
                if self._upload_masks_to_hf(masks, metadata, step, logger):
                    logger.info(f"Successfully uploaded masks for step {step} to HF Hub")
                else:
                    logger.warning(f"Failed to upload masks for step {step} to HF Hub")

    def save(self, path, save_mask: bool = False):
        """Save gradient accumulation. Optionally save masks alongside gradients."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.acc, path)

        if save_mask and self.save_masks:
            # Save masks with similar filename pattern
            mask_path = path.parent / f"{path.stem.replace('grad_ema', 'grad_mask')}{path.suffix}"
            masks = self._compute_masks()
            self._save_masks(mask_path, masks)

    def _save_masks(self, path, masks: OrderedDict):
        """Save boolean masks to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(masks, path)

    def load_masks(self, path) -> OrderedDict:
        """Load boolean masks from disk."""
        return torch.load(path, map_location="cpu")

    def _create_mask_metadata(self, masks: OrderedDict, step: int, model_name: str) -> dict:
        """Create metadata for the masks."""
        total_params = sum(mask.numel() for mask in masks.values())
        active_params = sum(mask.sum().item() for mask in masks.values())
        active_fraction = active_params / total_params if total_params > 0 else 0.0

        return {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tolerance": self.mask_tolerance,
            "total_parameters": total_params,
            "active_parameters": active_params,
            "active_fraction": active_fraction,
            "sparsity": 1.0 - active_fraction,
            "base_model": model_name,
            "beta": self.beta,
            "epsilon": self.epsilon,
        }

    def _prepare_hf_repo(self, logger) -> bool:
        """Prepare HF repository for mask uploads."""
        try:
            from huggingface_hub import HfApi, create_repo

            api = HfApi()

            # Check if repo exists, create if it doesn't
            try:
                api.repo_info(self.hf_repo_id)
                logger.info(f"HF repository {self.hf_repo_id} already exists")
            except Exception:
                logger.info(f"Creating HF repository {self.hf_repo_id}")
                create_repo(repo_id=self.hf_repo_id, repo_type="model", private=self.hf_private, exist_ok=True)

            return True
        except ImportError:
            logger.error("huggingface_hub not available. Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"Failed to prepare HF repository: {e}")
            return False

    def _upload_masks_to_hf(self, masks: OrderedDict, metadata: dict, step: int, logger) -> bool:
        """Upload masks and metadata to Hugging Face Hub."""
        try:
            from huggingface_hub import upload_file

            # api = HfApi()

            # Upload masks file
            mask_filename = f"masks/step_{step}.pt"
            mask_path = self.output_dir / "grad_acc" / f"grad_mask_step_{step}.pt"

            if mask_path.exists():
                logger.info(f"Uploading masks to {self.hf_repo_id}/{mask_filename}")
                upload_file(
                    path_or_fileobj=str(mask_path),
                    path_in_repo=mask_filename,
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                )

            # Upload metadata
            import json
            import tempfile

            metadata_filename = f"metadata/step_{step}_info.json"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(metadata, f, indent=2)
                temp_path = f.name

            logger.info(f"Uploading metadata to {self.hf_repo_id}/{metadata_filename}")
            upload_file(
                path_or_fileobj=temp_path, path_in_repo=metadata_filename, repo_id=self.hf_repo_id, repo_type="model"
            )

            # Clean up temp file
            import os

            os.unlink(temp_path)

            return True

        except Exception as e:
            logger.error(f"Failed to upload masks to HF Hub: {e}")
            return False

    def _update_hf_readme(self, model_name: str, logger) -> bool:
        """Update or create README for the HF repository."""
        try:
            import os
            import tempfile

            # from huggingface_hub import HfApi, upload_file
            from huggingface_hub import upload_file

            # api = HfApi()

            # Create README content
            readme_content = f"""---
license: mit
tags:
- gradient-masks
- model-sparsity
- parameter-pruning
base_model: {model_name}
---

# Gradient Masks Repository

This repository contains gradient masks generated during training of `{model_name}`.

## Overview

Gradient masks are boolean tensors that indicate which parameters have significant gradients during training. These masks can be used to identify important parameters for fine-tuning or to create sparse models.

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Download masks for a specific step
mask_path = hf_hub_download(
    repo_id="{self.hf_repo_id}",
    filename="masks/step_1000.pt"
)
masks = torch.load(mask_path, map_location="cpu")

# Apply masks to a model
for name, param in model.named_parameters():
    if name in masks:
        mask = masks[name].to(param.device)
        param.requires_grad = mask  # Set requires_grad based on mask
```

## Mask Generation

- **Tolerance**: {self.mask_tolerance}
- **Beta (EMA decay)**: {self.beta}
- **Epsilon**: {self.epsilon}
- **Base Model**: {model_name}

## Files

- `masks/step_*.pt`: Boolean masks for each training step
- `metadata/step_*_info.json`: Metadata about each mask set

## License

This repository is licensed under the MIT License.
"""

            # Upload README
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(readme_content)
                temp_path = f.name

            logger.info(f"Updating README for {self.hf_repo_id}")
            upload_file(path_or_fileobj=temp_path, path_in_repo="README.md", repo_id=self.hf_repo_id, repo_type="model")

            # Clean up
            os.unlink(temp_path)
            return True

        except Exception as e:
            logger.error(f"Failed to update HF README: {e}")
            return False

    def log(self, step: int, monitor, logger):
        # Compute global mean RMS for logging
        rms_values = [torch.sqrt(self.acc[name] + self.epsilon).mean() for name in self.acc]
        global_rms = torch.stack(rms_values).mean().item()

        # Compute mask statistics
        masks = self._compute_masks()
        mask_stats = self._compute_mask_stats(masks)

        # Log gradient EMA
        logger.info(f"Step {step} | EMA RMS Mean: {global_rms:.4f}")
        monitor.log({"grad_ema/rms_mean": global_rms, "step": step})

        # Log mask statistics
        logger.info(
            f"Step {step} | Active Fraction: {mask_stats['grad_mask/active_fraction']:.4f} | Sparsity: {mask_stats['grad_mask/sparsity']:.4f}"
        )
        mask_stats["step"] = step
        monitor.log(mask_stats)


# Utility functions for loading masks from Hugging Face Hub


def load_masks_from_hf(repo_id: str, step: int, cache_dir: str = None) -> OrderedDict:
    """
    Load gradient masks from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID containing the masks
        step: Training step to load masks for
        cache_dir: Optional cache directory for downloads

    Returns:
        OrderedDict containing boolean masks for each parameter

    Example:
        >>> masks = load_masks_from_hf("username/model-gradient-masks", 1000)
        >>> for name, param in model.named_parameters():
        ...     if name in masks:
        ...         mask = masks[name].to(param.device)
        ...         param.requires_grad = mask
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")

    filename = f"masks/step_{step}.pt"
    mask_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", cache_dir=cache_dir)

    return torch.load(mask_path, map_location="cpu")


def apply_masks_to_model(model: nn.Module, masks: OrderedDict, device: str = "cuda") -> None:
    """
    Apply boolean masks to a model's parameters by registering gradient hooks.
    This enforces per-element sparsity by scaling gradients to zero where mask is False.

    Args:
        model: PyTorch model to apply masks to
        masks: OrderedDict containing boolean masks for each parameter
        device: Device to move masks to
    """
    logger = get_logger()
    applied_count = 0
    total_active = 0
    total_params = 0

    for name, param in model.named_parameters():
        clean_name = name
        if "_fsdp_wrapped_module." in name:
            clean_name = name.replace("_fsdp_wrapped_module.", "")

        # Skip FSDP flat parameters
        if "_flat_param" in name:
            logger.debug(f"Skipping FSDP flat parameter: {name}")
            continue

        # Check if we have a mask for this parameter
        if clean_name not in masks:
            continue

        mask = masks[clean_name].to(device)

        # Convert DTensor to regular tensor if needed (for FSDP)
        if hasattr(mask, "to_local"):
            mask = mask.to_local()

        # Get flat 1D indices of active elements
        mask_flat = mask.reshape(-1)
        active_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)

        # Ensure it's 1D
        if active_indices.dim() > 1:
            active_indices = active_indices.squeeze()
        if active_indices.dim() == 0:  # Scalar case
            active_indices = active_indices.unsqueeze(0)

        # Store indices and mask on parameter
        param._sparse_mask_indices = active_indices

        # Compute stats for logging
        active = mask.sum().item()
        total = mask.numel()
        total_active += active
        total_params += total
        applied_count += 1

        # Register hook to multiply grad by mask during backward
        def grad_hook_factory(full_name, msk):
            def hook(grad):
                if grad is not None:
                    # Element-wise multiplication: zeros grad where mask is False
                    return grad * msk
                return grad

            return hook

        param.register_hook(grad_hook_factory(name, mask))

        logger.debug(f"Applied mask to {name}: {active}/{total} parameters active ({active / total * 100:.1f}%)")

    # Log overall stats
    if applied_count > 0:
        active_fraction = total_active / total_params
        logger.info(
            f"Applied masks to {applied_count} parameters | Overall active fraction: {active_fraction:.4f} ({total_active}/{total_params})"
        )
    else:
        logger.warning("No matching masks found for model parameters")
