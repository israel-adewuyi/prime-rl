"""Capture the contribution of each prompt to a single-GPU training update."""

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import GradientCaptureConfig

if TYPE_CHECKING:
    from prime_rl.trainer.rl.data import TensorMicroBatch


def _cpu_copy(tensor: Tensor) -> Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().to(device="cpu", dtype=torch.float32, copy=True)


class PromptGradientCapture:
    def __init__(
        self,
        model: nn.Module,
        config: GradientCaptureConfig,
        output_dir: Path,
        step: int,
        normalizers: dict[str, float],
        reshard_after_forward: bool = True,
    ):
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Prompt gradient capture requires exactly one trainer GPU")
        self.model = model
        self.config = config
        self.step = step
        self.normalizers = normalizers
        directory = config.output_dir or Path("prompt_gradients")
        self.directory = (directory if directory.is_absolute() else output_dir / directory) / f"step_{step}"
        if self.directory.exists():
            raise FileExistsError(f"Refusing to overwrite prompt gradients: {self.directory}")
        self.reshard_policies: list[tuple[FSDPModule, bool]] = []
        seen_states = set()
        for module in model.modules():
            if not isinstance(module, FSDPModule):
                continue
            # FSDP has public setters but no getter; grouped heads may retain weights by design.
            state = module._get_fsdp_state()
            if id(state) in seen_states:
                continue
            seen_states.add(id(state))
            policies = {group._reshard_after_forward for group in state._fsdp_param_groups}
            if len(policies) > 1:
                raise ValueError("Prompt capture requires a uniform reshard policy within each FSDP module")
            self.reshard_policies.append((module, next(iter(policies), reshard_after_forward)))
        self.optimizer_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if not self.optimizer_params:
            raise ValueError("Prompt gradient capture requires trainable parameters")
        self.prompts: dict[str, dict] = {}
        self.parameter_metadata: dict[str, dict] = {}
        self.microbatches = 0

    def before_forward(self) -> None:
        if isinstance(self.model, FSDPModule):
            self.model.set_reshard_after_forward(False)

    def _compute_parameters(self) -> dict[str, nn.Parameter]:
        params = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        if params.keys() != self.optimizer_params.keys():
            raise RuntimeError("Compute and optimizer parameter names do not match")
        for name, param in params.items():
            if isinstance(param, DTensor) or not param.is_leaf:
                raise RuntimeError(f"Capture requires an unsharded computation leaf: {name}")
            if isinstance(self.model, FSDPModule) and param is self.optimizer_params[name]:
                raise RuntimeError(f"Capture selected the optimizer parameter instead of the computation leaf: {name}")
            if isinstance(self.model, FSDPModule) and param.grad is not None:
                raise RuntimeError(f"Computation parameter already has an accumulated gradient: {name}")
            self.parameter_metadata[name] = {
                "shape": list(param.shape),
                "compute_dtype": str(param.dtype),
                "optimizer_dtype": str(self.optimizer_params[name].dtype),
            }
        return params

    def _optimizer_gradients(self) -> dict[str, Tensor | None]:
        return {name: _cpu_copy(p.grad) if p.grad is not None else None for name, p in self.optimizer_params.items()}

    def _assert_unchanged(self, before: dict[str, Tensor | None]) -> None:
        for name, expected in before.items():
            actual = self.optimizer_params[name].grad
            if (actual is None) != (expected is None):
                raise RuntimeError(f"Gradient capture changed the optimizer gradient buffer for {name}")
            if expected is not None and not torch.equal(_cpu_copy(actual), expected):
                raise RuntimeError(f"Gradient capture modified the accumulated optimizer gradient for {name}")

    def capture_microbatch(self, loss: Tensor, sequence_losses: list[Tensor], micro_batch: "TensorMicroBatch") -> None:
        params = self._compute_parameters()
        lengths = micro_batch["sequence_lengths"]
        groups = micro_batch.get("group_ids") or [""] * len(lengths)
        trace_ids = micro_batch.get("trace_ids") or [""] * len(lengths)
        branch_indices = micro_batch.get("branch_indices") or [-1] * len(lengths)
        if not (len(sequence_losses) == len(groups) == len(trace_ids) == len(branch_indices) == len(lengths)):
            raise ValueError("Sequence losses and prompt metadata must align with sequence_lengths")
        active = micro_batch["loss_mask"].flatten().clone()
        for key in ("ce_weights", "ref_kl_weights"):
            if micro_batch[key] is not None:
                active |= micro_batch[key].flatten() != 0
        active_sequences = active.split(lengths)
        grouped_losses: dict[str, Tensor] = {}
        offset = 0
        for index, (group_id, seq_loss, length) in enumerate(zip(groups, sequence_losses, lengths, strict=True)):
            env_name = micro_batch["env_names"][offset]
            offset += length
            if not group_id:
                if bool(active_sequences[index].any()):
                    raise ValueError("A training sequence is missing its prompt-group ID")
                continue
            if group_id not in self.prompts:
                self.prompts[group_id] = {
                    "env_name": env_name,
                    "loss": 0.0,
                    "sequences": [],
                    "gradients": dict.fromkeys(params),
                }
            prompt = self.prompts[group_id]
            if prompt["env_name"] != env_name:
                raise ValueError(f"Prompt group {group_id} spans multiple environments")
            prompt["loss"] += float(seq_loss.detach())
            prompt["sequences"].append(
                {"trace_id": trace_ids[index], "branch_index": branch_indices[index], "length": length}
            )
            grouped_losses[group_id] = grouped_losses[group_id] + seq_loss if group_id in grouped_losses else seq_loss

        if self.config.verify:
            torch.testing.assert_close(torch.stack(sequence_losses).sum(), loss, rtol=1e-5, atol=1e-6)
        previous_gradients = self._optimizer_gradients() if self.config.verify else None
        if isinstance(self.model, FSDPModule):
            # Retain computation leaves and backward bookkeeping until the training backward.
            self.model.set_is_last_backward(False)
            self.model.set_requires_gradient_sync(False)
            self.model.set_reshard_after_backward(False)
        try:
            for group_id, prompt_loss in grouped_losses.items():
                grads = torch.autograd.grad(
                    prompt_loss, tuple(params.values()), retain_graph=True, create_graph=False, allow_unused=True
                )
                if all(grad is None for grad in grads):
                    raise RuntimeError(f"Prompt {group_id} is disconnected from all computation parameters")
                accumulated = self.prompts[group_id]["gradients"]
                for (name, param), grad in zip(params.items(), grads, strict=True):
                    if isinstance(self.model, FSDPModule) and param.grad is not None:
                        raise RuntimeError(f"autograd.grad populated the computation gradient buffer for {name}")
                    if grad is None:
                        continue
                    value = _cpu_copy(grad)
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError(f"Non-finite prompt gradient: group={group_id}, parameter={name}")
                    if accumulated[name] is None:
                        accumulated[name] = value
                    else:
                        accumulated[name].add_(value)
                if previous_gradients is not None:
                    self._assert_unchanged(previous_gradients)
                del grads, grad
        finally:
            if isinstance(self.model, FSDPModule):
                self.model.set_requires_gradient_sync(True)
                self.model.set_reshard_after_backward(True)
                self.model.set_is_last_backward(True)
                for module, reshard in self.reshard_policies:
                    module.set_reshard_after_forward(reshard, recurse=False)
        self.microbatches += 1

    def _verify_batch_gradient(self) -> dict:
        errors = {}
        for name, param in self.optimizer_params.items():
            contributions = [p["gradients"][name] for p in self.prompts.values() if p["gradients"][name] is not None]
            if param.grad is None:
                if contributions and any(bool(g.count_nonzero()) for g in contributions):
                    raise RuntimeError(f"Captured a gradient but normal backward left {name} unused")
                continue
            expected = _cpu_copy(param.grad)
            actual = torch.zeros_like(expected)
            for contribution in contributions:
                actual.add_(contribution)
            if not bool(torch.isfinite(expected).all()):
                raise RuntimeError(f"Non-finite batch gradient: {name}")
            error = (actual - expected).double()
            reference_norm = expected.double().norm().item()
            error_norm = error.norm().item()
            tolerance = self.config.verify_atol * math.sqrt(expected.numel()) + self.config.verify_rtol * reference_norm
            errors[name] = {
                "max_abs": error.abs().max().item(),
                "relative_l2": error_norm / max(reference_norm, 1e-30),
                "error_l2": error_norm,
                "tolerance_l2": tolerance,
            }
            if error_norm > tolerance:
                raise RuntimeError(f"Prompt gradients do not reconstruct {name}: {errors[name]}")
        return {"checked": True, "parameters": errors}

    def finish_step(self) -> Path:
        verification = self._verify_batch_gradient() if self.config.verify else {"checked": False}
        self.directory.mkdir(parents=True, exist_ok=False)
        groups = []
        for index, (group_id, prompt) in enumerate(self.prompts.items()):
            filename = f"prompt_{index:06d}.pt"
            payload = {
                "schema_version": 1,
                "step": self.step,
                "group_id": group_id,
                "normalizers": self.normalizers,
                "parameter_metadata": self.parameter_metadata,
                **prompt,
            }
            temporary = self.directory / f"{filename}.tmp"
            torch.save(payload, temporary)
            temporary.replace(self.directory / filename)
            groups.append(
                {
                    "group_id": group_id,
                    "file": filename,
                    "env_name": prompt["env_name"],
                    "sequence_count": len(prompt["sequences"]),
                }
            )
        manifest = {
            "schema_version": 1,
            "step": self.step,
            "microbatches": self.microbatches,
            "gradient_dtype": "float32",
            "normalization": "training_batch",
            "before_gradient_clipping": True,
            "normalizers": self.normalizers,
            "groups": groups,
            "verification": verification,
        }
        temporary_manifest = self.directory / "manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary_manifest.replace(self.directory / "manifest.json")
        self.prompts.clear()
        return self.directory
