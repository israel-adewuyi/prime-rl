"""Rollout-gradient statistics with FP32 buffers and FP64 scalar moments."""

import math

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor


def _squared_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value, dtype=torch.float64).square()


class GradientStatistics:
    """CPU reference: retain a gradient sum and a scalar sum of squared norms."""

    def __init__(self):
        self.gradient_sum: dict[str, Tensor | None] = {}
        self.squared_norm_sum: float | Tensor = 0.0
        self.count = 0

    def _check_names(self, gradients: dict[str, Tensor | None]) -> None:
        if self.count == 0:
            self.gradient_sum = dict.fromkeys(gradients)
        elif gradients.keys() != self.gradient_sum.keys():
            raise ValueError("Gradient parameter names changed within a capture step")

    def update(self, gradients: dict[str, Tensor | None]) -> None:
        self._check_names(gradients)
        squared_norms = []
        for name, gradient in gradients.items():
            if gradient is None:
                continue
            if isinstance(gradient, DTensor):
                gradient = gradient.to_local()
            value = gradient.detach().to(device="cpu", dtype=torch.float32, copy=self.gradient_sum[name] is None)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite rollout gradient: {name}")
            squared_norms.append(_squared_norm(value).item())
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

    def _moments(self) -> tuple[float, float]:
        sum_squared_norm = math.fsum(
            _squared_norm(value).item() for value in self.gradient_sum.values() if value is not None
        )
        return self.squared_norm_sum, sum_squared_norm

    def finalize(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {"snr/count": self.count}
        if self.count:
            squared_norm_sum, sum_squared_norm = self._moments()
            if not math.isfinite(squared_norm_sum):
                raise ValueError("Non-finite sum of squared rollout-gradient norms")
            if not math.isfinite(sum_squared_norm):
                raise ValueError("Non-finite norm of the accumulated rollout gradients")
            mean_squared_norm = sum_squared_norm / self.count**2
            metrics["snr/mean_norm"] = math.sqrt(mean_squared_norm)
            if self.count > 1:
                centered_sum = squared_norm_sum - sum_squared_norm / self.count
                # Nearly identical gradients can lose precision in this subtraction.
                roundoff = (
                    8
                    * self.count
                    * torch.finfo(torch.float32).eps
                    * max(squared_norm_sum, sum_squared_norm / self.count)
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


class CUDAGradientStatistics(GradientStatistics):
    """Accumulate on one GPU; transfer two FP64 scalars at finalization."""

    def update(self, gradients: dict[str, Tensor | None]) -> None:
        self._check_names(gradients)
        device = self.squared_norm_sum.device if isinstance(self.squared_norm_sum, Tensor) else None
        squared_norms = []
        for name, gradient in gradients.items():
            if gradient is None:
                continue
            if isinstance(gradient, DTensor):
                gradient = gradient.to_local()
            if gradient.device.type != "cuda" or (device is not None and gradient.device != device):
                raise ValueError("CUDA gradient statistics require gradients on a single GPU")
            device = gradient.device
            value = gradient.detach().to(dtype=torch.float32, copy=self.gradient_sum[name] is None)
            squared_norms.append(_squared_norm(value))
            if self.gradient_sum[name] is None:
                self.gradient_sum[name] = value
            else:
                self.gradient_sum[name].add_(value)
        if squared_norms:
            rollout_squared_norm = torch.stack(squared_norms).sum()
            if isinstance(self.squared_norm_sum, Tensor):
                self.squared_norm_sum.add_(rollout_squared_norm)
            else:
                self.squared_norm_sum = rollout_squared_norm
        self.count += 1

    def _moments(self) -> tuple[float, float]:
        if not isinstance(self.squared_norm_sum, Tensor):
            return 0.0, 0.0
        squared_norms = [_squared_norm(value) for value in self.gradient_sum.values() if value is not None]
        # Squared norms propagate NaN/Inf; finalize checks them before an optimizer update.
        values = torch.stack((self.squared_norm_sum, torch.stack(squared_norms).sum())).cpu().tolist()
        return values[0], values[1]
