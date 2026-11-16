import weakref

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import GradientCaptureConfig, IPOLossConfig
from prime_rl.trainer.rl.gradient_statistics import CUDAGradientStatistics, GradientStatistics
from prime_rl.trainer.rl.gradients import RolloutGradientCapture
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def statistics_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires a CUDA or ROCm GPU")
    return torch.device(request.param)


@pytest.fixture
def statistics(statistics_device):
    return CUDAGradientStatistics() if statistics_device.type == "cuda" else GradientStatistics()


def test_gradient_statistics_match_stacked_reference_with_unused_parameters(statistics, statistics_device):
    samples = [
        {"weight": torch.tensor([1.0, 2.0]), "bias": None, "unused": None},
        {"weight": torch.tensor([-3.0, 4.0]), "bias": torch.tensor([2.0]), "unused": None},
        {"weight": torch.tensor([5.0, -2.0]), "bias": torch.tensor([-1.0]), "unused": None},
    ]
    stacked = torch.tensor([[1.0, 2.0, 0.0], [-3.0, 4.0, 2.0], [5.0, -2.0, -1.0]], dtype=torch.float64)
    for sample in samples:
        statistics.update(
            {name: value.to(statistics_device) if value is not None else None for name, value in sample.items()}
        )

    assert statistics.count == 3
    torch.testing.assert_close(statistics.gradient_sum["weight"].cpu(), stacked[:, :2].sum(dim=0))
    torch.testing.assert_close(statistics.gradient_sum["bias"].cpu(), stacked[:, 2:].sum(dim=0))
    assert statistics.gradient_sum["unused"] is None
    assert float(statistics.squared_norm_sum) == pytest.approx(stacked.square().sum().item())
    mean = stacked.mean(dim=0)
    variance = (stacked - mean).square().sum().item() / (len(stacked) - 1)
    metrics = statistics.finalize()
    assert metrics["snr/count"] == 3
    assert metrics["snr/mean_norm"] == pytest.approx(mean.norm().item())
    assert metrics["snr/variance"] == pytest.approx(variance)
    assert metrics["snr/value"] == pytest.approx(mean.square().sum().item() / variance)
    assert statistics.gradient_sum == {}
    assert statistics.count == 0
    assert statistics.squared_norm_sum == 0.0


def test_gradient_statistics_omit_undefined_variance_for_zero_or_one_rollout(statistics, statistics_device):
    assert statistics.finalize() == {"snr/count": 0}
    statistics.update({"weight": None})
    statistics.update({"weight": None})
    assert statistics.finalize() == {"snr/count": 2, "snr/mean_norm": 0.0, "snr/variance": 0.0}
    statistics.update({"weight": torch.tensor([3.0, 4.0], device=statistics_device)})
    assert statistics.finalize() == {"snr/count": 1, "snr/mean_norm": 5.0}


@pytest.mark.parametrize("gradient", [[1.25, -2.5], [0.0, 0.0]])
def test_identical_rollout_gradients_have_zero_variance_and_no_snr(gradient, statistics, statistics_device):
    for _ in range(3):
        statistics.update({"weight": torch.tensor(gradient, device=statistics_device)})
    metrics = statistics.finalize()
    assert metrics["snr/count"] == 3
    assert metrics["snr/mean_norm"] == pytest.approx(torch.tensor(gradient, dtype=torch.float64).norm().item())
    assert metrics["snr/variance"] == 0.0
    assert "snr/value" not in metrics


def test_nearly_identical_rollout_gradients_retain_small_variance(statistics, statistics_device):
    samples = torch.tensor([[1.0, -2.0], [1.0001, -2.0001], [0.9999, -1.9999]], dtype=torch.float32)
    for sample in samples:
        statistics.update({"weight": sample.to(statistics_device)})
    mean = samples.double().mean(dim=0)
    variance = (samples.double() - mean).square().sum().item() / (len(samples) - 1)
    metrics = statistics.finalize()
    assert metrics["snr/variance"] > 0.0
    assert metrics["snr/variance"] == pytest.approx(variance, rel=1e-6, abs=0)
    assert metrics["snr/value"] == pytest.approx(mean.square().sum().item() / variance, rel=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_gradient_statistics_retain_only_the_sum_and_release_it_after_finalization(
    statistics, statistics_device, dtype
):
    first_gradient = torch.tensor([1.0, 2.0], requires_grad=True, device=statistics_device, dtype=dtype)
    first_reference = weakref.ref(first_gradient)
    statistics.update({"weight": first_gradient, "bias": torch.tensor([0.5], device=statistics_device)})
    first_gradient.detach().zero_()
    torch.testing.assert_close(statistics.gradient_sum["weight"].cpu(), torch.tensor([1.0, 2.0], dtype=torch.float64))
    del first_gradient
    assert first_reference() is None
    for index in range(100):
        statistics.update(
            {
                "weight": torch.tensor([float(index), 1.0], device=statistics_device),
                "bias": torch.tensor([0.5], device=statistics_device),
            }
        )
        assert set(statistics.gradient_sum) == {"weight", "bias"}
        assert sum(value.numel() for value in statistics.gradient_sum.values()) == 3
        for value in statistics.gradient_sum.values():
            assert value.device.type == statistics_device.type and value.dtype == torch.float64
            assert not value.requires_grad and value.grad_fn is None
    metrics = statistics.finalize()
    assert metrics["snr/count"] == 101
    assert statistics.gradient_sum == {}
    assert statistics.squared_norm_sum == 0.0
    assert statistics.count == 0
    statistics.update({"weight": torch.tensor([0.0, 0.0], device=statistics_device)})
    assert statistics.finalize() == {"snr/count": 1, "snr/mean_norm": 0.0}


def test_capture_counts_masked_real_rollouts_but_skips_padding(statistics_device):
    model = nn.Linear(2, 1, bias=False).to(statistics_device)
    capture = RolloutGradientCapture(
        model, GradientCaptureConfig(steps=[1], verify=True, statistics_device=statistics_device.type)
    )
    capture.before_forward()
    logprobs = model(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=statistics_device)).flatten()
    mask = torch.tensor([[True, False, False]], device=statistics_device)
    sequence_losses = []
    loss, _ = compute_loss(
        trainer_logprobs=list(logprobs.split(1)),
        inference_logprobs=list(logprobs.detach().split(1)),
        ref_logprobs=None,
        advantages=list(torch.ones(3, device=statistics_device).split(1)),
        loss_mask=list(mask.flatten().split(1)),
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=setup_rl_loss_fn(IPOLossConfig()),
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        per_sequence_losses=sequence_losses,
    )
    batch = {
        "sequence_lengths": [1, 1, 1],
        "trace_ids": ["rollout-1", "truncated-rollout-2", ""],
        "loss_mask": mask,
        "rl_weights": None,
        "ce_weights": None,
        "ref_kl_weights": None,
    }
    capture.capture_microbatch(loss, sequence_losses, batch)
    assert capture.statistics.count == 2
    assert model.weight.grad is None
    loss.backward()
    metrics = capture.finish_step()
    assert metrics["snr/count"] == 2
    assert metrics["snr/variance"] > 0.0
    assert metrics["snr/value"] == pytest.approx(0.5)
    assert capture.statistics.gradient_sum == {}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e308])
def test_gradient_statistics_reject_nonfinite_moments(statistics, statistics_device, value):
    with pytest.raises(ValueError, match="Non-finite"):
        statistics.update({"weight": torch.tensor([value], dtype=torch.float64, device=statistics_device)})
        statistics.finalize()


def test_cuda_gradient_statistics_reject_cpu_inputs():
    statistics = CUDAGradientStatistics()
    with pytest.raises(ValueError, match="single GPU"):
        statistics.update({"weight": torch.ones(2)})
