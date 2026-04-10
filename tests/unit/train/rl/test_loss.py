import pytest
import torch

from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import compute_entropy, compute_loss

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="token", mask_ratio_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_grpo_loss_matches_clipped_objective():
    trainer_logprobs = [torch.log(torch.tensor([1.0, 1.3], dtype=torch.float32, device="cuda"))]
    inference_logprobs = [torch.zeros(2, dtype=torch.float32, device="cuda")]
    advantages = [torch.ones(2, dtype=torch.float32, device="cuda")]
    loss_mask = [torch.ones(2, dtype=torch.bool, device="cuda")]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(type="grpo", clip_eps=0.2),
        loss_scale=2.0,
    )
    assert torch.isclose(loss, torch.tensor(-1.1, dtype=torch.float32, device="cuda"))


def test_gspo_loss():
    # Create list of tensors as expected by compute_loss (simulating split sequences)
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="sequence", mask_ratio_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)
