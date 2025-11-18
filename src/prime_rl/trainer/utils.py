import pickle
import time
from collections import OrderedDict, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from rich.text import Text
from torch import Tensor
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
        timeout=timedelta(seconds=1200),
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


def load_masks_from_hf(config) -> OrderedDict:
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

    if config.mask_format == "old":
        filename = f"masks/beta_{config.beta}_step_{config.step}_tolerance_{config.tolerance}.pt"
    else:
        filename = f"masks/beta_{config.beta}_batch_{config.batch}_step_{config.step}_active_{config.num_active}.pt"

    mask_path = hf_hub_download(repo_id=config.repo_id, filename=filename, repo_type="model", cache_dir=None)

    return torch.load(mask_path, map_location="cpu")


def verify_masking(model: nn.Module, masks: OrderedDict, num_params: int = 3):
    """
    Simple verification that masks are correctly applied to gradients.
    Checks that False mask positions have zero gradients.
    """
    logger = get_logger()

    logger.info("\n" + "=" * 60)
    logger.info("MASKING VERIFICATION")
    logger.info("=" * 60)

    checked = 0
    all_passed = True

    for name, param in model.named_parameters():
        if checked >= num_params or param.grad is None or name not in masks:
            continue

        # Get gradient and mask
        grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
        mask = masks[name].to(grad.device)

        if grad.shape != mask.shape:
            continue

        # Find where mask is False (should be zero)
        mask_bool = mask.bool() if mask.dtype == torch.bool else (mask != 0)
        masked_positions = ~mask_bool

        # Check if those positions are zero in gradient
        grads_at_masked = grad[masked_positions]
        num_masked = masked_positions.sum().item()
        num_zero = (grads_at_masked.abs() < 1e-10).sum().item()

        passed = num_zero == num_masked
        status = "✅" if passed else "❌"
        all_passed = all_passed and passed

        logger.info(f"{status} {name}: {num_zero}/{num_masked} masked positions are zero")

        if not passed:
            # Show a few non-zero values for debugging
            nonzero_vals = grads_at_masked[grads_at_masked.abs() >= 1e-10][:3]
            logger.info(f"   Sample non-zero values: {nonzero_vals}")

        checked += 1

    logger.info("=" * 60)
    logger.info(f"Overall: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    logger.info("=" * 60 + "\n")

    return all_passed


def mask_gradients_in_optimizer(optimizer, masks: OrderedDict, model: nn.Module, verify_first_step: bool = True):
    """Apply masks in optimizer step with optional one-time verification"""

    original_step = optimizer.step
    first_step = [True]  # Track if this is first step

    def step_with_masking(closure=None):
        # Apply masks
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    # Find parameter name
                    param_name = None
                    for name, p in model.named_parameters():
                        if p is param:
                            param_name = name
                            break

                    if param_name and param_name in masks:
                        mask = masks[param_name].to(param.grad.device)
                        if hasattr(param.grad, "to_local"):
                            param.grad.to_local().mul_(mask)
                        else:
                            param.grad.mul_(mask)

        # Verify on first step only
        if verify_first_step and first_step[0]:
            verify_masking(model, masks, num_params=5)
            first_step[0] = False

        return original_step(closure)

    optimizer.step = step_with_masking
