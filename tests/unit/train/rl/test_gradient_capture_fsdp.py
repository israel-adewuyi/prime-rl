from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.rl.data import TensorMicroBatch
from prime_rl.trainer.rl.gradients import RolloutGradientCapture
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
from prime_rl.trainer.update_capture import CheckpointUpdateCapture, checkpoint_update_capture_enabled
from prime_rl.trainer.utils import clip_grad_norm_
from prime_rl.utils.config import cli

pytestmark = [pytest.mark.gpu]

CONFIG_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "gradient_capture" / "fsdp.toml"
NORMALIZERS = {"rl": 8.0, "ce": 1.0, "ref_kl": 1.0}


@pytest.fixture
def trainer_config(tmp_path):
    return cli(TrainerConfig, args=["@", str(CONFIG_PATH), "--output-dir", str(tmp_path)])


@pytest.fixture
def single_gpu_process_group(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA or ROCm GPU")
    assert not dist.is_initialized(), "Run this test with ordinary pytest, without torchrun"
    torch.cuda.set_device(0)
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group(
        "nccl",
        init_method=(tmp_path / "process_group").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=60),
        device_id=torch.device("cuda", 0),
    )
    try:
        yield
    finally:
        dist.destroy_process_group()
        torch.set_float32_matmul_precision(previous_precision)


def _model(reshard_after_forward: bool | None, head_reshard_after_forward: bool | None = None) -> nn.Sequential:
    torch.manual_seed(19)
    model = nn.Sequential(nn.Linear(8, 12), nn.SiLU(), nn.Linear(12, 12), nn.SiLU(), nn.Linear(12, 7)).cuda()
    if reshard_after_forward is None:
        return model.to(torch.bfloat16)
    if head_reshard_after_forward is None:
        head_reshard_after_forward = reshard_after_forward
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(model[0], mp_policy=policy, reshard_after_forward=reshard_after_forward)
    fully_shard(model[4], mp_policy=policy, reshard_after_forward=head_reshard_after_forward)
    fully_shard(model, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    return model


def _reshard_policies(model):
    return {
        name: tuple(group._reshard_after_forward for group in module._get_fsdp_state()._fsdp_param_groups)
        for name, module in model.named_modules()
        if isinstance(module, FSDPModule)
    }


def _local(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _gradient_snapshot(parameters: dict[str, nn.Parameter]) -> dict[str, Tensor | None]:
    return {
        name: None if parameter.grad is None else _local(parameter.grad).detach().cpu().clone()
        for name, parameter in parameters.items()
    }


def _assert_gradients_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if expected[name] is None:
            assert actual[name] is None, name
        else:
            assert actual[name] is not None, name
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)


def _microbatches(step: int) -> list[tuple[Tensor, TensorMicroBatch]]:
    generator = torch.Generator().manual_seed(100 + step)
    batches = []
    layout = [
        ([2, 2, 3], [1.0, -0.25, 1.0]),
        ([2, 3, 2], [-0.25, -0.75, -0.75]),
    ]
    for microbatch_index, (lengths, sequence_advantages) in enumerate(layout):
        token_count = sum(lengths)
        input_ids = torch.randint(0, 7, (1, token_count), generator=generator).cuda()
        loss_mask = torch.tensor([[position > 0 for length in lengths for position in range(length)]], device="cuda")
        positions = torch.tensor([[position for length in lengths for position in range(length)]], device="cuda")
        advantages = torch.tensor(
            [
                [
                    advantage
                    for length, advantage in zip(lengths, sequence_advantages, strict=True)
                    for _ in range(length)
                ]
            ],
            device="cuda",
        )
        batch = TensorMicroBatch(
            input_ids=input_ids,
            position_ids=positions,
            advantages=advantages,
            inference_logprobs=torch.full((1, token_count), -2.0, device="cuda"),
            ref_logprobs=None,
            loss_mask=loss_mask,
            temperatures=torch.ones((1, token_count), device="cuda"),
            env_names=["test-env"] * token_count,
            sequence_lengths=lengths,
            trace_ids=[f"step-{step}-micro-{microbatch_index}-sequence-{i}" for i in range(len(lengths))],
            branch_indices=[0] * len(lengths),
            lora_num_tokens=torch.tensor([token_count], dtype=torch.int32),
            seq_lens=torch.tensor(lengths, dtype=torch.long),
            routed_experts=None,
            sampling_mask=None,
            mm_refs=None,
            mm_token_type_ids=None,
            rl_weights=None,
            ce_weights=None,
            ref_kl_weights=None,
        )
        features = torch.randn(token_count, 8, generator=generator).cuda().to(torch.bfloat16)
        batches.append((features, batch))
    return batches


def _loss(model, features, batch, config):
    logits = model(features).float()
    logprobs = logits.log_softmax(-1).gather(-1, batch["input_ids"].flatten().unsqueeze(-1)).squeeze(-1)
    lengths = batch["sequence_lengths"]
    sequence_losses = []
    loss, _ = compute_loss(
        trainer_logprobs=list(logprobs.split(lengths)),
        inference_logprobs=list(batch["inference_logprobs"].flatten().split(lengths)),
        ref_logprobs=None,
        advantages=list(batch["advantages"].flatten().split(lengths)),
        loss_mask=list(batch["loss_mask"].flatten().split(lengths)),
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=setup_rl_loss_fn(config.loss),
        rl_scale=NORMALIZERS["rl"],
        ce_scale=NORMALIZERS["ce"],
        ref_kl_scale=NORMALIZERS["ref_kl"],
        per_sequence_losses=sequence_losses,
    )
    torch.testing.assert_close(torch.stack(sequence_losses).sum(), loss)
    return loss, sequence_losses


def _rollout_gradient_reference(optimizer_parameters, batches, config):
    reference = _model(reshard_after_forward=None)
    reference.load_state_dict({name: _local(parameter).detach() for name, parameter in optimizer_parameters.items()})
    parameters = dict(reference.named_parameters())
    rollout_gradients = []
    for features, batch in batches:
        _, sequence_losses = _loss(reference, features, batch, config)
        for sequence_loss in sequence_losses:
            gradients = torch.autograd.grad(sequence_loss, tuple(parameters.values()), retain_graph=True)
            rollout_gradients.append(torch.cat([gradient.detach().double().cpu().flatten() for gradient in gradients]))
    return torch.stack(rollout_gradients)


@pytest.mark.parametrize("reshard_after_forward", [True, False])
def test_compute_parameters_are_connected_to_loss(single_gpu_process_group, trainer_config, reshard_after_forward):
    model = _model(reshard_after_forward)
    optimizer_parameters = dict(model.named_parameters())
    model.set_reshard_after_forward(False)
    features, batch = _microbatches(step=1)[0]
    loss, _ = _loss(model, features, batch, trainer_config)
    compute_parameters = dict(model.named_parameters())
    assert compute_parameters.keys() == optimizer_parameters.keys()
    for name, parameter in compute_parameters.items():
        assert parameter is not optimizer_parameters[name]
        assert isinstance(optimizer_parameters[name], DTensor)
        assert optimizer_parameters[name].dtype == torch.float32
        assert not isinstance(parameter, DTensor)
        assert parameter.is_leaf and parameter.requires_grad
        assert parameter.dtype == torch.bfloat16

    model.set_is_last_backward(False)
    model.set_requires_gradient_sync(False)
    model.set_reshard_after_backward(False)
    wrong_gradients = torch.autograd.grad(
        loss, tuple(optimizer_parameters.values()), allow_unused=True, retain_graph=True
    )
    assert all(gradient is None for gradient in wrong_gradients)
    gradients = torch.autograd.grad(loss, tuple(compute_parameters.values()), retain_graph=True, allow_unused=False)
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)
    assert all(parameter.grad is None for parameter in compute_parameters.values())
    assert all(parameter.grad is None for parameter in optimizer_parameters.values())
    model.set_is_last_backward(True)
    model.set_requires_gradient_sync(True)
    model.set_reshard_after_backward(True)
    model.set_reshard_after_forward(reshard_after_forward)
    loss.backward()
    for name, gradient in zip(compute_parameters, gradients, strict=True):
        torch.testing.assert_close(_local(optimizer_parameters[name].grad), gradient.float(), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("reshard_after_forward", "head_reshard_after_forward"),
    [(True, True), (False, False), (True, False)],
    ids=["reshard", "retain", "retain-head"],
)
def test_two_step_capture_matches_training_without_capture(
    single_gpu_process_group, trainer_config, reshard_after_forward, head_reshard_after_forward
):
    baseline = _model(reshard_after_forward, head_reshard_after_forward)
    captured = _model(reshard_after_forward, head_reshard_after_forward)
    expected_policies = {
        "": (reshard_after_forward,),
        "0": (reshard_after_forward,),
        "4": (head_reshard_after_forward,),
    }
    assert _reshard_policies(captured) == expected_policies
    baseline_parameters = dict(baseline.named_parameters())
    captured_parameters = dict(captured.named_parameters())
    baseline_optimizer = torch.optim.AdamW(baseline_parameters.values(), lr=trainer_config.optim.lr, weight_decay=0.0)
    captured_optimizer = torch.optim.AdamW(captured_parameters.values(), lr=trainer_config.optim.lr, weight_decay=0.0)
    assert trainer_config.gradient_capture.steps == [1, 2]
    assert trainer_config.max_steps == 3
    assert trainer_config.ckpt.interval == 2
    assert trainer_config.ckpt.save_gradients_and_updates is True
    statistics_steps = []
    checkpoint_steps = []
    for step in range(1, trainer_config.max_steps + 1):
        batches = _microbatches(step)
        capture = None
        if step in trainer_config.gradient_capture.steps:
            expected_rollout_gradients = _rollout_gradient_reference(captured_parameters, batches, trainer_config)
            capture = RolloutGradientCapture(
                model=captured,
                config=trainer_config.gradient_capture,
                reshard_after_forward=reshard_after_forward,
            )
        for features, batch in batches:
            baseline_loss, _ = _loss(baseline, features, batch, trainer_config)
            baseline_loss.backward()
            before = _gradient_snapshot(captured_parameters)
            if capture is not None:
                capture.before_forward()
                assert _reshard_policies(captured) == dict.fromkeys(expected_policies, (False,))
            captured_loss, sequence_losses = _loss(captured, features, batch, trainer_config)
            if capture is not None:
                capture.capture_microbatch(captured_loss, sequence_losses, batch)
                _assert_gradients_equal(_gradient_snapshot(captured_parameters), before)
            assert _reshard_policies(captured) == expected_policies
            captured_loss.backward()
            _assert_gradients_equal(_gradient_snapshot(captured_parameters), _gradient_snapshot(baseline_parameters))

        if capture is not None:
            assert capture.statistics.count == 6
            actual_sum = torch.cat([capture.statistics.gradient_sum[name].flatten() for name in captured_parameters])
            torch.testing.assert_close(actual_sum, expected_rollout_gradients.sum(dim=0), rtol=0, atol=0)
            assert capture.statistics.squared_norm_sum == pytest.approx(
                expected_rollout_gradients.square().sum().item()
            )
            metrics = capture.finish_step()
            mean = expected_rollout_gradients.mean(dim=0)
            variance = ((expected_rollout_gradients - mean).square().sum() / 5).item()
            assert metrics["snr/count"] == 6
            assert metrics["snr/mean_norm"] == pytest.approx(mean.norm().item(), rel=1e-10)
            assert metrics["snr/variance"] == pytest.approx(variance, rel=1e-10)
            assert metrics["snr/value"] == pytest.approx(mean.square().sum().item() / variance, rel=1e-10)
            assert capture.statistics.gradient_sum == {}
            assert capture.statistics.squared_norm_sum == 0.0
            assert capture.statistics.count == 0
            statistics_steps.append(step)

        checkpoint_capture = None
        if checkpoint_update_capture_enabled(trainer_config.ckpt, step, step == trainer_config.max_steps):
            checkpoint_capture = CheckpointUpdateCapture(captured, step)
        raw_gradients = _gradient_snapshot(captured_parameters)
        before_update = {
            name: _local(parameter).detach().cpu().clone() for name, parameter in captured_parameters.items()
        }
        raw_norm = sum(gradient.double().square().sum().item() for gradient in raw_gradients.values()) ** 0.5
        assert raw_norm > 0
        max_norm = min(0.1, raw_norm / 2)
        clip_grad_norm_(None, baseline, max_norm)
        clip_grad_norm_(None, captured, max_norm)
        clipped_gradients = _gradient_snapshot(captured_parameters)
        assert any(not torch.equal(raw_gradients[name], clipped_gradients[name]) for name in captured_parameters)
        _assert_gradients_equal(clipped_gradients, _gradient_snapshot(baseline_parameters))

        baseline_optimizer.step()
        captured_optimizer.step()
        if checkpoint_capture is not None:
            checkpoint_capture.capture_update()
        for name, parameter in captured_parameters.items():
            baseline_parameter = baseline_parameters[name]
            torch.testing.assert_close(_local(parameter), _local(baseline_parameter), rtol=0, atol=0, msg=name)
            expected_state = baseline_optimizer.state[baseline_parameter]
            actual_state = captured_optimizer.state[parameter]
            assert actual_state.keys() == expected_state.keys()
            for key in expected_state:
                torch.testing.assert_close(_local(actual_state[key]), _local(expected_state[key]), rtol=0, atol=0)
        baseline_optimizer.zero_grad(set_to_none=True)
        captured_optimizer.zero_grad(set_to_none=True)
        if checkpoint_capture is not None:
            checkpoint_directory = trainer_config.output_dir / "checkpoints" / f"step_{step}" / "trainer"
            destination = checkpoint_capture.save(checkpoint_directory)
            payload = torch.load(destination, map_location="cpu", weights_only=True)
            assert destination.name == "gradient_update.pt"
            assert payload["step"] == step
            assert payload["gradients_before_clipping"] is True
            assert payload["update_definition"] == "after_minus_before"
            _assert_gradients_equal(payload["batch_gradients"], raw_gradients)
            for name, parameter in captured_parameters.items():
                expected_update = _local(parameter).detach().cpu() - before_update[name]
                torch.testing.assert_close(payload["applied_updates"][name], expected_update, rtol=0, atol=0)
                assert payload["parameter_metadata"][name]["parameter_dtype"] == "torch.float32"
            checkpoint_steps.append(step)
        assert not list(trainer_config.output_dir.rglob("prompt_*.pt"))
    assert statistics_steps == [1, 2]
    assert checkpoint_steps == [2, 3]
    assert len(list(trainer_config.output_dir.rglob("gradient_update.pt"))) == 2
