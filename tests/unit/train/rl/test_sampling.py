import pytest
import torch

from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import apply_top_k_top_p, compute_loss, selective_log_softmax


def test_apply_top_k_top_p_applies_top_k_filter():
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    filtered = apply_top_k_top_p(logits.clone(), top_k=2, top_p=1.0)

    assert torch.isneginf(filtered[0, 0, 0])
    assert torch.isneginf(filtered[0, 0, 1])
    assert filtered[0, 0, 2] == 3.0
    assert filtered[0, 0, 3] == 4.0


def test_apply_top_k_top_p_applies_top_p_filter_to_logits():
    logits = torch.tensor([[[0.0, 0.0, 10.0]]])

    filtered = apply_top_k_top_p(logits.clone(), top_k=-1, top_p=0.9)

    assert torch.isneginf(filtered[0, 0, 0])
    assert torch.isneginf(filtered[0, 0, 1])
    assert filtered[0, 0, 2] == 10.0


def test_apply_top_k_top_p_matches_vllm_top_k_before_top_p_order():
    logits = torch.tensor([[[10.0, 9.0, 8.0]]])

    filtered = apply_top_k_top_p(logits.clone(), top_k=2, top_p=0.7)

    assert filtered[0, 0, 0] == 10.0
    assert torch.isneginf(filtered[0, 0, 1])
    assert torch.isneginf(filtered[0, 0, 2])


def test_apply_top_k_top_p_preserves_packed_shape_and_selected_support_shape():
    logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])
    input_ids = torch.tensor([[0, 2]])

    filtered = apply_top_k_top_p(logits.clone(), top_k=1, top_p=1.0)
    selected_in_support = torch.isfinite(torch.gather(filtered, -1, input_ids.unsqueeze(-1)).squeeze(-1))

    assert logits.shape == filtered.shape
    assert selected_in_support.shape == input_ids.shape
    assert selected_in_support.tolist() == [[True, False]]


def test_apply_top_k_top_p_accepts_token_level_sampling_args():
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 10.0, 0.0]]])
    top_k = torch.tensor([[2, -1]])
    top_p = torch.tensor([[1.0, 0.9]])

    filtered = apply_top_k_top_p(logits.clone(), top_k=top_k, top_p=top_p)

    assert torch.isneginf(filtered[0, 0, 0])
    assert torch.isneginf(filtered[0, 0, 1])
    assert filtered[0, 0, 2] == 3.0
    assert filtered[0, 0, 3] == 4.0
    assert torch.isneginf(filtered[0, 1, 0])
    assert torch.isneginf(filtered[0, 1, 1])
    assert filtered[0, 1, 2] == 10.0
    assert torch.isneginf(filtered[0, 1, 3])


def test_apply_top_k_top_p_support_mask_is_autograd_safe():
    logits = torch.tensor([[[0.0, 0.0, 10.0]]], requires_grad=True)
    input_ids = torch.tensor([[2]])

    filtered = apply_top_k_top_p(logits, top_k=-1, top_p=0.9)
    loss = selective_log_softmax(filtered, input_ids).sum()
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("loss_config", [LossConfig(), LossConfig(type="grpo"), LossConfig(type="dppo")])
def test_compute_loss_keeps_strict_objective_but_excludes_outside_support_from_metrics(loss_config):
    trainer_logprobs = [torch.tensor([0.0, -float("inf")])]
    inference_logprobs = [torch.tensor([0.0, 0.0])]
    advantages = [torch.tensor([1.0, 1.0])]
    loss_mask = [torch.tensor([True, True])]
    selected_in_support = [torch.tensor([True, False])]

    loss, loss_tensors = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        loss_scale=1,
        selected_in_support=selected_in_support,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_tensors["mismatch_kl"]).all()
    assert torch.isclose(loss_tensors["support_mismatch_frac"], torch.tensor([0.5])).all()
    if "log_importance_ratio" in loss_tensors:
        assert torch.isfinite(loss_tensors["log_importance_ratio"]).all()
