"""Online rollout-gradient statistics for a single-GPU training update."""

import math
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import GradientCaptureConfig

if TYPE_CHECKING:
    from prime_rl.trainer.rl.data import TensorMicroBatch


def _cpu_copy(tensor: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().to(device="cpu", dtype=dtype, copy=True)


class GradientStatistics:
    """Keep one gradient sum and a scalar sum of squared rollout norms."""

    def __init__(self):
        self.gradient_sum: dict[str, Tensor | None] = {}
        self.squared_norm_sum = 0.0
        self.count = 0

    def update(self, gradients: dict[str, Tensor | None]) -> None:
        if self.count == 0:
            self.gradient_sum = dict.fromkeys(gradients)
        elif gradients.keys() != self.gradient_sum.keys():
            raise ValueError("Gradient parameter names changed within a capture step")
        squared_norms = []
        for name, gradient in gradients.items():
            if gradient is None:
                continue
            value = _cpu_copy(gradient, dtype=torch.float64)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite rollout gradient: {name}")
            squared_norms.append(torch.dot(value.reshape(-1), value.reshape(-1)).item())
            if self.gradient_sum[name] is None:
                self.gradient_sum[name] = value
            else:
                self.gradient_sum[name].add_(value)
        self.squared_norm_sum = math.fsum((self.squared_norm_sum, math.fsum(squared_norms)))
        if not math.isfinite(self.squared_norm_sum):
            raise ValueError("Non-finite sum of squared rollout-gradient norms")
        self.count += 1

    def clear(self) -> None:
        self.gradient_sum.clear()
        self.squared_norm_sum = 0.0
        self.count = 0

    def finalize(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {"snr/count": self.count}
        if self.count:
            sum_squared_norm = math.fsum(
                torch.dot(value.reshape(-1), value.reshape(-1)).item()
                for value in self.gradient_sum.values()
                if value is not None
            )
            if not math.isfinite(sum_squared_norm):
                raise ValueError("Non-finite norm of the accumulated rollout gradients")
            mean_squared_norm = sum_squared_norm / self.count**2
            metrics["snr/mean_norm"] = math.sqrt(mean_squared_norm)
            if self.count > 1:
                centered_sum = self.squared_norm_sum - sum_squared_norm / self.count
                # The subtraction can lose precision when every rollout has nearly the same gradient.
                roundoff = (
                    8
                    * self.count
                    * torch.finfo(torch.float64).eps
                    * max(self.squared_norm_sum, sum_squared_norm / self.count)
                )
                if centered_sum < -roundoff:
                    raise ValueError("Squared gradient norms produced a negative variance beyond roundoff")
                if abs(centered_sum) <= roundoff:
                    centered_sum = 0.0
                variance = centered_sum / (self.count - 1)
                metrics["snr/variance"] = variance
                if variance > 0:
                    metrics["snr/value"] = mean_squared_norm / variance
        self.clear()
        return metrics


class RolloutGradientCapture:
    def __init__(
        self,
        model: nn.Module,
        config: GradientCaptureConfig,
        reshard_after_forward: bool = True,
    ):
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Rollout gradient capture requires exactly one trainer GPU")
        self.model = model
        self.config = config
        self.statistics = GradientStatistics()
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
                raise ValueError("Rollout capture requires a uniform reshard policy within each FSDP module")
            self.reshard_policies.append((module, next(iter(policies), reshard_after_forward)))
        self.optimizer_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if not self.optimizer_params:
            raise ValueError("Rollout gradient capture requires trainable parameters")

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
        trace_ids = micro_batch.get("trace_ids") or [""] * len(lengths)
        if not (len(sequence_losses) == len(trace_ids) == len(lengths)):
            raise ValueError("Sequence losses and rollout metadata must align with sequence_lengths")
        active = micro_batch["loss_mask"].flatten().clone()
        for key in ("ce_weights", "ref_kl_weights"):
            if micro_batch[key] is not None:
                active |= micro_batch[key].flatten() != 0
        active_sequences = active.split(lengths)
        if self.config.verify:
            torch.testing.assert_close(torch.stack(sequence_losses).sum(), loss, rtol=1e-5, atol=1e-6)
        previous_gradients = self._optimizer_gradients() if self.config.verify else None
        if isinstance(self.model, FSDPModule):
            # Retain computation leaves and backward bookkeeping until the training backward.
            self.model.set_is_last_backward(False)
            self.model.set_requires_gradient_sync(False)
            self.model.set_reshard_after_backward(False)
        try:
            for sequence_loss, trace_id, active_sequence in zip(
                sequence_losses, trace_ids, active_sequences, strict=True
            ):
                if not trace_id and not bool(active_sequence.any()):
                    continue
                grads = torch.autograd.grad(
                    sequence_loss, tuple(params.values()), retain_graph=True, create_graph=False, allow_unused=True
                )
                if all(grad is None for grad in grads):
                    raise RuntimeError("Rollout loss is disconnected from all computation parameters")
                for name, param in params.items():
                    if isinstance(self.model, FSDPModule) and param.grad is not None:
                        raise RuntimeError(f"autograd.grad populated the computation gradient buffer for {name}")
                self.statistics.update(dict(zip(params, grads, strict=True)))
                if previous_gradients is not None:
                    self._assert_unchanged(previous_gradients)
                del grads
        finally:
            if isinstance(self.model, FSDPModule):
                self.model.set_requires_gradient_sync(True)
                self.model.set_reshard_after_backward(True)
                self.model.set_is_last_backward(True)
                for module, reshard in self.reshard_policies:
                    module.set_reshard_after_forward(reshard, recurse=False)

    def _verify_batch_gradient(self) -> None:
        for name, param in self.optimizer_params.items():
            actual = self.statistics.gradient_sum.get(name)
            if param.grad is None:
                if actual is not None and bool(actual.count_nonzero()):
                    raise RuntimeError(f"Captured a gradient but normal backward left {name} unused")
                continue
            expected = _cpu_copy(param.grad, dtype=torch.float64)
            if not bool(torch.isfinite(expected).all()):
                raise RuntimeError(f"Non-finite batch gradient: {name}")
            error = actual - expected if actual is not None else -expected
            error_norm = error.norm().item()
            tolerance = (
                self.config.verify_atol * math.sqrt(expected.numel()) + self.config.verify_rtol * expected.norm().item()
            )
            if error_norm > tolerance:
                raise RuntimeError(
                    f"Rollout gradients do not reconstruct {name}: error_l2={error_norm}, tolerance_l2={tolerance}"
                )

    def finish_step(self) -> dict[str, float | int]:
        if self.config.verify:
            self._verify_batch_gradient()
        return self.statistics.finalize()
