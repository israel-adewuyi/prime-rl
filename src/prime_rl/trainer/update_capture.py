"""Checkpoint artifacts for accumulated gradients and applied optimizer updates."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from prime_rl.configs.trainer import CheckpointConfig


def checkpoint_update_capture_enabled(config_ckpt: "CheckpointConfig | None", step: int, is_last_step: bool) -> bool:
    return bool(
        config_ckpt is not None
        and config_ckpt.save_gradients_and_updates
        and step > 0
        and (is_last_step or (config_ckpt.interval is not None and step % config_ckpt.interval == 0))
    )


def _cpu_copy(tensor: Tensor) -> Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().to(device="cpu", copy=True)


def _artifact_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


class CheckpointUpdateCapture:
    def __init__(self, model: nn.Module, step: int):
        if step < 1:
            raise ValueError("Checkpoint update capture requires a positive training step")
        self._parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        if not self._parameters:
            raise ValueError("Checkpoint update capture requires trainable parameters")
        if dist.is_initialized():
            if dist.get_world_size() != 1:
                raise ValueError("Checkpoint update capture requires exactly one trainer rank")
        elif any(parameter.device.type != "cpu" for parameter in self._parameters.values()):
            raise ValueError("GPU checkpoint update capture requires an initialized single-rank process group")

        self.step = step
        self._weight_snapshots = {name: _cpu_copy(parameter) for name, parameter in self._parameters.items()}
        self.batch_gradients = {
            name: None
            if parameter.grad is None
            else _cpu_copy(parameter.grad).to(dtype=_artifact_dtype(parameter.grad.dtype))
            for name, parameter in self._parameters.items()
        }
        self.parameter_metadata = {
            name: {
                "shape": list(parameter.shape),
                "parameter_dtype": str(parameter.dtype),
                "gradient_dtype": None if parameter.grad is None else str(parameter.grad.dtype),
            }
            for name, parameter in self._parameters.items()
        }
        self.applied_updates: dict[str, Tensor] = {}
        self._update_captured = False
        self._saved = False

    def capture_update(self) -> None:
        if self._update_captured:
            raise RuntimeError("The optimizer update has already been captured")
        for name, parameter in self._parameters.items():
            before = self._weight_snapshots[name]
            after = _cpu_copy(parameter)
            # Promote BF16/FP16 values before subtraction to retain their actual weight difference.
            dtype = _artifact_dtype(before.dtype)
            self._weight_snapshots[name] = after.to(dtype=dtype).sub_(before.to(dtype=dtype))
        self.applied_updates = self._weight_snapshots
        self._weight_snapshots = {}
        self._parameters.clear()
        self._update_captured = True

    def save(self, checkpoint_trainer_dir: Path) -> Path:
        if self._saved:
            raise RuntimeError("The gradient and update artifact has already been saved")
        if not self._update_captured:
            raise RuntimeError("Capture the optimizer update before saving its artifact")
        destination = checkpoint_trainer_dir / "gradient_update.pt"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite a gradient and update artifact: {destination}")
        checkpoint_trainer_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "step": self.step,
            "gradients_before_clipping": True,
            "update_definition": "after_minus_before",
            "batch_gradients": self.batch_gradients,
            "applied_updates": self.applied_updates,
            "parameter_metadata": self.parameter_metadata,
        }
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        self.batch_gradients.clear()
        self.applied_updates.clear()
        self.parameter_metadata.clear()
        self._saved = True
        return destination
