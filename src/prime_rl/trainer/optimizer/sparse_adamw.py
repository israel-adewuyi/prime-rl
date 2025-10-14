from typing import Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


class SparseAdamW(Optimizer):
    """
    AdamW optimizer with sparse state storage
    Only stores optimizer states for unmasked elements
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimizationi step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if hasattr(p, "_sparse_mask_indices"):
                    indices = p._sparse_mask_indices
                    if indices.numel() > 0:
                        self._sparse_step(p, grad, indices, group)
                else:
                    raise ValueError("When using SparseAdamW, all param groups should have sparse mask indices")

        return loss

    def _sparse_step(
        self,
        p: Tensor,
        grad: Tensor,
        indices: Tensor,
        group: dict,
    ):
        """Update sparse params following AdamW implementation"""
        state = self.state[p]

        return state  # TODO: remove
