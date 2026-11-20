"""Online rollout-gradient statistics for a single-GPU training update."""

import math
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import GradientCaptureConfig
from prime_rl.trainer.rl.gradient_statistics import CUDAGradientStatistics, GradientStatistics

if TYPE_CHECKING:
    from prime_rl.trainer.rl.data import TensorMicroBatch


def _cpu_copy(tensor: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().to(device="cpu", dtype=dtype, copy=True)


def _trace_ids(batch: "TensorMicroBatch") -> list[str]:
    lengths = batch["sequence_lengths"]
    ids = batch["trace_ids"] if batch["trace_ids"] is not None else [""] * len(lengths)
    if len(ids) != len(lengths):
        raise ValueError("Rollout metadata must align with sequence_lengths")
    if "" in ids:
        active = batch["loss_mask"].flatten().clone()
        if batch["rl_weights"] is not None:
            active &= batch["rl_weights"].flatten() != 0
        for key in ("ce_weights", "ref_kl_weights"):
            if batch[key] is not None:
                active |= batch[key].flatten() != 0
        for trace_id, mask in zip(ids, active.split(lengths), strict=True):
            if not trace_id and mask.any():
                raise ValueError("A sequence with active loss is missing its rollout trace_id")
    return ids


class RolloutGradientCapture:
    def __init__(
        self,
        model: nn.Module,
        config: GradientCaptureConfig,
        micro_batches: list["TensorMicroBatch"],
        reshard_after_forward: bool = True,
    ):
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Rollout gradient capture requires exactly one trainer GPU")
        self.model = model
        self.config = config
        self.statistics = CUDAGradientStatistics() if config.statistics_device == "cuda" else GradientStatistics()
        self.trace_ids = [_trace_ids(batch) for batch in micro_batches]
        self.micro_step = 0
        self.last_microbatch = {
            trace_id: index for index, ids in enumerate(self.trace_ids) for trace_id in ids if trace_id
        }
        self.partial_gradients: dict[str, dict[str, Tensor | None]] = {}
        self.peak_partial_traces = 0
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

    def _record_gradients(self, trace_id: str, gradients: dict[str, Tensor | None], complete: bool) -> None:
        if complete and trace_id not in self.partial_gradients:
            self.statistics.update(gradients)
            return

        # A packed update can leave hundreds of rollouts unfinished. Keep their
        # FP32 sums on the host, moving only the completing rollout to the
        # statistics device. Single-microbatch rollouts take the fast path above.
        partial = self.partial_gradients.pop(trace_id, None)
        if partial is None:
            partial = dict.fromkeys(gradients)
        device = self.config.statistics_device if complete else "cpu"
        for name, gradient in gradients.items():
            previous = partial[name]
            if previous is not None:
                previous = previous.to(device)
                if gradient is not None:
                    previous.add_(gradient.detach().to(device=device, dtype=torch.float32))
                partial[name] = previous
            elif gradient is not None:
                partial[name] = gradient.detach().to(device=device, dtype=torch.float32, copy=True)
        if complete:
            self.statistics.update(partial)
        else:
            self.partial_gradients[trace_id] = partial
            self.peak_partial_traces = max(self.peak_partial_traces, len(self.partial_gradients))

    def capture_microbatch(self, loss: Tensor, sequence_losses: list[Tensor]) -> None:
        params = self._compute_parameters()
        rollout_losses: dict[str, Tensor] = {}
        for trace_id, branch_loss in zip(self.trace_ids[self.micro_step], sequence_losses, strict=True):
            if trace_id in rollout_losses:
                rollout_losses[trace_id] = rollout_losses[trace_id] + branch_loss
            elif trace_id:
                rollout_losses[trace_id] = branch_loss
        if self.config.verify:
            torch.testing.assert_close(torch.stack(sequence_losses).sum(), loss, rtol=1e-5, atol=1e-6)
        previous_gradients = self._optimizer_gradients() if self.config.verify else None
        if isinstance(self.model, FSDPModule):
            # Retain computation leaves and backward bookkeeping until the training backward.
            self.model.set_is_last_backward(False)
            self.model.set_requires_gradient_sync(False)
            self.model.set_reshard_after_backward(False)
        try:
            for trace_id, rollout_loss in rollout_losses.items():
                grads = torch.autograd.grad(
                    rollout_loss, tuple(params.values()), retain_graph=True, create_graph=False, allow_unused=True
                )
                if all(grad is None for grad in grads):
                    raise RuntimeError("Rollout loss is disconnected from all computation parameters")
                for name, param in params.items():
                    if isinstance(self.model, FSDPModule) and param.grad is not None:
                        raise RuntimeError(f"autograd.grad populated the computation gradient buffer for {name}")
                complete = self.last_microbatch[trace_id] == self.micro_step
                self._record_gradients(trace_id, dict(zip(params, grads, strict=True)), complete)
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
        self.micro_step += 1

    def _verify_batch_gradient(self) -> None:
        for name, param in self.optimizer_params.items():
            actual = self.statistics.gradient_sum.get(name)
            if actual is not None:
                actual = actual.cpu()
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
        if (
            self.micro_step != len(self.trace_ids)
            or self.partial_gradients
            or self.statistics.count != len(self.last_microbatch)
        ):
            raise RuntimeError("Gradient capture did not complete every expected rollout")
        if self.config.verify:
            self._verify_batch_gradient()
        return {**self.statistics.finalize(), "snr/peak_partial_traces": self.peak_partial_traces}
