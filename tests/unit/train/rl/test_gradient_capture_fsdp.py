import json
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
from prime_rl.trainer.rl.gradients import PromptGradientCapture
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
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
        ([2, 2, 3], ["prompt-a", "prompt-a", "prompt-b"], [1.0, -0.25, 1.0]),
        ([2, 3, 2], ["prompt-b", "prompt-a", "prompt-b"], [-0.25, -0.75, -0.75]),
    ]
    for microbatch_index, (lengths, group_ids, sequence_advantages) in enumerate(layout):
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
            group_ids=group_ids,
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


def _prompt_gradient_reference(optimizer_parameters, batches, config):
    reference = _model(reshard_after_forward=None)
    reference.load_state_dict({name: _local(parameter).detach() for name, parameter in optimizer_parameters.items()})
    parameters = dict(reference.named_parameters())
    accumulated = {}
    for features, batch in batches:
        _, sequence_losses = _loss(reference, features, batch, config)
        for group_id in dict.fromkeys(batch["group_ids"]):
            prompt_loss = sum(
                loss
                for loss, sequence_group in zip(sequence_losses, batch["group_ids"], strict=True)
                if sequence_group == group_id
            )
            gradients = torch.autograd.grad(prompt_loss, tuple(parameters.values()), retain_graph=True)
            target = accumulated.setdefault(
                group_id,
                {name: torch.zeros_like(p, device="cpu", dtype=torch.float32) for name, p in parameters.items()},
            )
            for name, gradient in zip(parameters, gradients, strict=True):
                target[name].add_(gradient.detach().float().cpu())
    return accumulated


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
    for step in range(1, trainer_config.max_steps + 1):
        batches = _microbatches(step)
        capture = None
        if step in trainer_config.gradient_capture.steps:
            expected_prompt_gradients = _prompt_gradient_reference(captured_parameters, batches, trainer_config)
            capture = PromptGradientCapture(
                model=captured,
                config=trainer_config.gradient_capture,
                output_dir=trainer_config.output_dir,
                step=step,
                normalizers=NORMALIZERS,
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
            destination = capture.finish_step()
            _assert_saved_gradients(destination, step, captured_parameters, expected_prompt_gradients, batches)
        else:
            capture_root = trainer_config.output_dir / trainer_config.gradient_capture.output_dir
            assert not (capture_root / f"step_{step}").exists()

        baseline_optimizer.step()
        captured_optimizer.step()
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


def _assert_saved_gradients(destination, step, parameters, expected_prompt_gradients, batches):
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["microbatches"] == 2
    assert manifest["verification"]["checked"] is True
    assert manifest["verification"]["parameters"].keys() == parameters.keys()
    for error in manifest["verification"]["parameters"].values():
        assert error["error_l2"] <= error["tolerance_l2"]
    assert {group["group_id"] for group in manifest["groups"]} == {"prompt-a", "prompt-b"}
    assert len(list(destination.glob("prompt_*.pt"))) == 2
    for group in manifest["groups"]:
        assert group["sequence_count"] == 3
        payload = torch.load(destination / group["file"], map_location="cpu", weights_only=True)
        assert payload["schema_version"] == 1
        assert payload["step"] == step
        assert payload["env_name"] == "test-env"
        assert payload["normalizers"] == NORMALIZERS
        assert len(payload["sequences"]) == 3
        expected_traces = {
            trace_id
            for _, batch in batches
            for group_id, trace_id in zip(batch["group_ids"], batch["trace_ids"], strict=True)
            if group_id == group["group_id"]
        }
        assert {sequence["trace_id"] for sequence in payload["sequences"]} == expected_traces
        assert payload["gradients"].keys() == parameters.keys()
        for name, gradient in payload["gradients"].items():
            assert gradient is not None
            assert gradient.device.type == "cpu" and gradient.dtype == torch.float32
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            assert payload["parameter_metadata"][name]["compute_dtype"] == "torch.bfloat16"
            assert payload["parameter_metadata"][name]["optimizer_dtype"] == "torch.float32"
            torch.testing.assert_close(gradient, expected_prompt_gradients[group["group_id"]][name], rtol=0, atol=0)
