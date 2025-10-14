import math
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

    def _init_sparse_state(self, p: Tensor, indices: Tensor):
        """Initialize sparse optimizer states for a parameter"""
        state = self.state[p]
        if len(state) == 0:
            num_active = indices.numel()
            state["step"] = 0

            # Initialize states with the same dtype as parameter
            state["exp_avg"] = torch.zeros(num_active, dtype=p.dtype, device=p.device)
            state["exp_avg_sq"] = torch.zeros(num_active, dtype=p.dtype, device=p.device)
            state["sparse_indices"] = indices.clone()

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

        if len(state) == 0:
            self._init_sparse_state(p, indices)

        assert torch.equal(state["sparse_indices"], indices), "Mask indices changed between steps - not supported"

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        state["step"] += 1

        beta1, beta2 = group["betas"]

        grad_flat = grad.reshape(-1)
        grad_sparse = grad_flat[indices]

        # Update moment estimates
        exp_avg.mul_(beta1).add_(grad_sparse, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_sparse, grad_sparse, value=1 - beta2)

        # bias correction
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        step_size = group["lr"] / bias_correction1

        # Compute adaptive step
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
        update = exp_avg / denom

        # Apply weight decay (decoupled, AdamW-style) and update parameters
        p_flat = p.reshape(-1)
        if group["weight_decay"] != 0:
            update = update + group["weight_decay"] * p_flat[indices]

        p_flat[indices] = p_flat[indices] - step_size * update
