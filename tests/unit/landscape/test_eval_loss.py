import math

import pytest
import torch

from prime_rl.landscape.loss_utils import compute_landscape_loss_metrics
from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import compute_loss


def test_masked_regime_matches_original_compute_loss_token_mode() -> None:
    trainer_logprobs = [torch.tensor([-0.2, -0.7, -0.1], dtype=torch.float32)]
    inference_logprobs = [torch.tensor([-0.2, -1.2, -0.1], dtype=torch.float32)]
    advantages = [torch.tensor([0.3, 1.1, -0.2], dtype=torch.float32)]
    loss_mask = [torch.tensor([True, True, False])]
    loss_config = LossConfig(
        ratio_type="token",
        token_mask_low=0.5,
        token_mask_high=1.4,
        geo_mask_low=0.0,
        geo_mask_high=1000.0,
        sequence_mask_low=0.0,
        sequence_mask_high=1000.0,
        kl_tau=0.2,
    )

    original_loss, original_metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        loss_scale=2,
    )
    new_metrics = compute_landscape_loss_metrics(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        loss_scale=2,
        clip_epsilon=0.2,
    )

    assert new_metrics["loss_masked"].item() == pytest.approx(float(original_loss.item()))
    assert new_metrics["loss_shared_mismatch_kl_mean"].item() == pytest.approx(
        float(original_metrics["mismatch_kl"].mean().item())
    )
    assert new_metrics["loss_masked_kept_mismatch_kl_mean"].item() == pytest.approx(
        float(original_metrics["unmasked_mismatch_kl"].mean().item())
    )
    assert new_metrics["loss_masked_dropped_mismatch_kl_mean"].item() == pytest.approx(
        float(original_metrics["masked_mismatch_kl"].mean().item())
    )
    assert new_metrics["loss_masked_drop_frac"].mean().item() == pytest.approx(
        float(original_metrics["is_masked"].mean().item())
    )
    assert new_metrics["loss_masked_drop_token_low_frac"].mean().item() == pytest.approx(
        float(original_metrics["is_masked_low"].mean().item())
    )
    assert new_metrics["loss_masked_drop_token_high_frac"].mean().item() == pytest.approx(
        float(original_metrics["is_masked_high"].mean().item())
    )


def test_masked_regime_matches_original_compute_loss_sequence_mode() -> None:
    trainer_logprobs = [torch.tensor([-0.3, -0.4], dtype=torch.float32)]
    inference_logprobs = [torch.tensor([-0.6, -0.5], dtype=torch.float32)]
    advantages = [torch.tensor([0.7, -0.2], dtype=torch.float32)]
    loss_mask = [torch.tensor([True, True])]
    loss_config = LossConfig(
        ratio_type="sequence",
        token_mask_low=0.1,
        token_mask_high=1000.0,
        geo_mask_low=0.0,
        geo_mask_high=1000.0,
        sequence_mask_low=0.0,
        sequence_mask_high=1000.0,
        kl_tau=0.1,
    )

    original_loss, original_metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        loss_scale=1,
    )
    new_metrics = compute_landscape_loss_metrics(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        loss_scale=1,
        clip_epsilon=0.2,
    )

    assert new_metrics["loss_masked"].item() == pytest.approx(float(original_loss.item()))
    assert new_metrics["loss_shared_mismatch_kl_mean"].item() == pytest.approx(
        float(original_metrics["mismatch_kl"].mean().item())
    )
    assert new_metrics["loss_masked_dropped_mismatch_kl_mean"].item() == pytest.approx(
        float(original_metrics["masked_mismatch_kl"].mean().item())
    )
    assert new_metrics["loss_masked_drop_sequence_low_frac"].mean().item() == pytest.approx(
        float(original_metrics["sequence_masked_low"].mean().item())
    )
    assert new_metrics["loss_masked_drop_sequence_high_frac"].mean().item() == pytest.approx(
        float(original_metrics["sequence_masked_high"].mean().item())
    )


def test_compute_landscape_loss_metrics_token_regimes() -> None:
    base_logprob = math.log(0.5)
    trainer_logprobs = [torch.tensor([base_logprob, base_logprob], dtype=torch.float32)]
    inference_logprobs = [torch.tensor([base_logprob, base_logprob - math.log(3.0)], dtype=torch.float32)]
    advantages = [torch.ones(2, dtype=torch.float32)]
    loss_mask = [torch.tensor([True, True])]
    loss_config = LossConfig(
        ratio_type="token",
        token_mask_low=0.0,
        token_mask_high=2.0,
        geo_mask_low=0.0,
        geo_mask_high=1000.0,
        sequence_mask_low=0.0,
        sequence_mask_high=1000.0,
        kl_tau=0.0,
    )

    metrics = compute_landscape_loss_metrics(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        loss_scale=2,
        clip_epsilon=0.2,
    )

    assert "teacher_kl" not in metrics
    assert metrics["loss_masked"].item() == pytest.approx(-(1.0 * base_logprob) / 2.0)
    assert metrics["loss_vanilla"].item() == pytest.approx(-((1.0 + 3.0) * base_logprob) / 2.0)
    assert metrics["loss_clipped"].item() == pytest.approx(-((1.0 + 1.2) * base_logprob) / 2.0)
    assert metrics["loss_shared_mismatch_kl_mean"].item() == pytest.approx((3.0 - math.log(3.0) - 1.0) / 2.0)
    assert metrics["loss_shared_geo_seq_ratio_mean"].item() == pytest.approx(math.sqrt(3.0))
    assert metrics["loss_masked_keep_frac"].mean().item() == pytest.approx(0.5)
    assert metrics["loss_masked_drop_frac"].mean().item() == pytest.approx(0.5)
    assert metrics["loss_masked_drop_token_low_frac"].mean().item() == pytest.approx(0.0)
    assert metrics["loss_masked_drop_token_high_frac"].mean().item() == pytest.approx(0.5)
    assert metrics["loss_masked_drop_sequence_low_frac"].mean().item() == pytest.approx(0.0)
    assert metrics["loss_masked_drop_sequence_high_frac"].mean().item() == pytest.approx(0.0)
    assert metrics["loss_masked_drop_geo_low_frac"].mean().item() == pytest.approx(0.0)
    assert metrics["loss_masked_drop_geo_high_frac"].mean().item() == pytest.approx(0.0)
    assert metrics["loss_masked_kept_mismatch_kl_mean"].item() == pytest.approx(0.0)
    assert metrics["loss_masked_dropped_mismatch_kl_mean"].item() == pytest.approx(3.0 - math.log(3.0) - 1.0)
    assert metrics["loss_clipped_clip_frac"].mean().item() == pytest.approx(0.5)
    assert metrics["loss_clipped_clip_low_frac"].mean().item() == pytest.approx(0.0)
    assert metrics["loss_clipped_clip_high_frac"].mean().item() == pytest.approx(0.5)
    assert metrics["loss_clipped_ratio_preclip_mean"].item() == pytest.approx(2.0)
    assert metrics["loss_clipped_ratio_postclip_mean"].item() == pytest.approx(1.1)


def test_compute_landscape_loss_metrics_sequence_clipping_uses_sequence_ratio() -> None:
    base_logprob = math.log(0.5)
    trainer_logprobs = [torch.tensor([base_logprob, base_logprob], dtype=torch.float32)]
    inference_logprobs = [torch.tensor([base_logprob - math.log(3.0), base_logprob], dtype=torch.float32)]
    advantages = [torch.ones(2, dtype=torch.float32)]
    loss_mask = [torch.tensor([True, True])]
    loss_config = LossConfig(
        ratio_type="sequence",
        token_mask_low=0.0,
        token_mask_high=1000.0,
        geo_mask_low=0.0,
        geo_mask_high=1000.0,
        sequence_mask_low=0.0,
        sequence_mask_high=1000.0,
        kl_tau=0.0,
    )

    metrics = compute_landscape_loss_metrics(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        loss_scale=1,
        clip_epsilon=0.2,
    )

    assert metrics["loss_masked"].item() == pytest.approx(-(3.0 * 2.0 * base_logprob) / 2.0)
    assert metrics["loss_vanilla"].item() == pytest.approx(-(3.0 * 2.0 * base_logprob) / 2.0)
    assert metrics["loss_clipped"].item() == pytest.approx(-(1.2 * 2.0 * base_logprob) / 2.0)
    assert metrics["loss_clipped_clip_frac"].item() == pytest.approx(1.0)
    assert metrics["loss_clipped_clip_low_frac"].item() == pytest.approx(0.0)
    assert metrics["loss_clipped_clip_high_frac"].item() == pytest.approx(1.0)
    assert metrics["loss_clipped_ratio_preclip_mean"].item() == pytest.approx(3.0)
    assert metrics["loss_clipped_ratio_postclip_mean"].item() == pytest.approx(1.2)
