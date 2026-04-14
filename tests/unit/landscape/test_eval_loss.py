import math

import pytest
import torch

from prime_rl.landscape.eval_loss import _compute_grpo_loss_metrics


def test_compute_grpo_loss_metrics_matches_hand_computed_values() -> None:
    trainer_logprobs = [torch.tensor([-0.2, -0.7, -0.1], dtype=torch.float32)]
    inference_logprobs = [torch.tensor([-0.2, -1.2, -0.1], dtype=torch.float32)]
    advantages = [torch.tensor([0.3, 1.1, -0.2], dtype=torch.float32)]
    valid_token_mask = [torch.tensor([True, True, False])]

    metrics = _compute_grpo_loss_metrics(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        valid_token_mask=valid_token_mask,
        loss_scale=2,
        clip_epsilon=0.2,
    )

    ratio = math.exp(0.5)
    clipped_ratio = 1.2
    preclip_mismatch = ratio - 0.5 - 1.0
    postclip_mismatch = clipped_ratio - math.log(clipped_ratio) - 1.0

    assert metrics["loss_grpo"].item() == pytest.approx(-((0.3 + 1.2 * 1.1) / 2.0))
    assert metrics["loss_grpo_unclipped"].item() == pytest.approx(-((0.3 + ratio * 1.1) / 2.0))
    assert metrics["loss_kl_valid_mean"].mean().item() == pytest.approx(preclip_mismatch / 2.0)
    assert metrics["loss_kl_valid_clipped_mean"].mean().item() == pytest.approx(postclip_mismatch / 2.0)
    assert metrics["loss_clip_frac"].mean().item() == pytest.approx(0.5)
    assert metrics["loss_clip_low_frac"].mean().item() == pytest.approx(0.0)
    assert metrics["loss_clip_high_frac"].mean().item() == pytest.approx(0.5)
    assert metrics["loss_ratio_mean"].mean().item() == pytest.approx((1.0 + ratio) / 2.0)
    assert metrics["loss_clipped_ratio_mean"].mean().item() == pytest.approx((1.0 + clipped_ratio) / 2.0)
    assert metrics["loss_log_ratio_mean"].mean().item() == pytest.approx(0.25)
    assert metrics["loss_surrogate_mean"].mean().item() == pytest.approx((0.3 + ratio * 1.1) / 2.0)
    assert metrics["loss_surrogate_clipped_mean"].mean().item() == pytest.approx((0.3 + clipped_ratio * 1.1) / 2.0)


def test_compute_grpo_loss_metrics_clips_negative_advantages_correctly() -> None:
    trainer_logprobs = [torch.tensor([math.log(0.5)], dtype=torch.float32)]
    inference_logprobs = [torch.tensor([0.0], dtype=torch.float32)]
    advantages = [torch.tensor([-2.0], dtype=torch.float32)]
    valid_token_mask = [torch.tensor([True])]

    metrics = _compute_grpo_loss_metrics(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        valid_token_mask=valid_token_mask,
        loss_scale=1,
        clip_epsilon=0.2,
    )

    assert metrics["loss_grpo"].item() == pytest.approx(1.6)
    assert metrics["loss_grpo_unclipped"].item() == pytest.approx(1.0)
    assert metrics["loss_clip_frac"].mean().item() == pytest.approx(1.0)
    assert metrics["loss_clip_low_frac"].mean().item() == pytest.approx(1.0)
    assert metrics["loss_clip_high_frac"].mean().item() == pytest.approx(0.0)
    assert metrics["loss_ratio_mean"].mean().item() == pytest.approx(0.5)
    assert metrics["loss_clipped_ratio_mean"].mean().item() == pytest.approx(0.8)


def test_compute_grpo_loss_metrics_ignores_invalid_tokens() -> None:
    trainer_logprobs = [torch.tensor([0.4, -0.2], dtype=torch.float32)]
    inference_logprobs = [torch.tensor([0.0, -0.2], dtype=torch.float32)]
    advantages = [torch.tensor([1.0, 100.0], dtype=torch.float32)]
    valid_token_mask = [torch.tensor([True, False])]

    metrics = _compute_grpo_loss_metrics(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        valid_token_mask=valid_token_mask,
        loss_scale=1,
        clip_epsilon=0.2,
    )

    ratio = math.exp(0.4)
    clipped_ratio = 1.2
    preclip_mismatch = ratio - 0.4 - 1.0
    postclip_mismatch = clipped_ratio - math.log(clipped_ratio) - 1.0

    assert metrics["loss_grpo"].item() == pytest.approx(-(clipped_ratio * 1.0))
    assert metrics["loss_grpo_unclipped"].item() == pytest.approx(-(ratio * 1.0))
    assert metrics["loss_kl_valid_mean"].mean().item() == pytest.approx(preclip_mismatch)
    assert metrics["loss_kl_valid_clipped_mean"].mean().item() == pytest.approx(postclip_mismatch)
    assert metrics["loss_clip_frac"].mean().item() == pytest.approx(1.0)
