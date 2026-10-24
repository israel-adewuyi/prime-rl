"""LFM2 checkpoint compatibility, packed boundaries, and repeated backward checks."""

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM as HFLfm2ForCausalLM

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, get_custom_causal_lm_cls, supports_custom_impl
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.lfm2.modeling_lfm2 import Lfm2TrainingShortConv


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def lfm2_device(request):
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("Requires a CUDA or ROCm GPU")
        from prime_rl.trainer.models.layers.attn import get_flash_attn_func

        get_flash_attn_func(2)
    return torch.device(request.param)


def tiny_lfm2_config(tied=True, attn="sdpa"):
    return Lfm2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        max_position_embeddings=128,
        block_auto_adjust_ff_dim=False,
        conv_L_cache=3,
        conv_bias=False,
        tie_word_embeddings=tied,
        attn_implementation=attn,
        pad_token_id=0,
        use_cache=False,
    )


@pytest.mark.parametrize("tied", [True, False])
def test_lfm2_hf_equivalence_and_packed_isolation(lfm2_device, tied):
    torch.manual_seed(61)
    dtype = torch.bfloat16 if lfm2_device.type == "cuda" else torch.float32
    config = tiny_lfm2_config(tied, "flash_attention_2" if lfm2_device.type == "cuda" else "sdpa")
    assert supports_custom_impl(config)
    assert get_custom_causal_lm_cls(config).__name__ == "Lfm2ForCausalLM"
    model = AutoModelForCausalLMPrimeRL.from_config(config, dtype=dtype).to(lfm2_device)
    assert isinstance(model.model.layers[0].conv, Lfm2TrainingShortConv)
    reference_config = copy.deepcopy(config)
    reference_config._attn_implementation = "sdpa"
    reference = HFLfm2ForCausalLM(reference_config).to(device=lfm2_device, dtype=dtype)
    reference.load_state_dict(model.state_dict(), strict=True)
    inject_prime_lm_head(model)

    tokens = torch.randint(1, 128, (1, 12), device=lfm2_device)
    lengths = torch.tensor([5, 7], device=lfm2_device)
    positions = torch.cat([torch.arange(5), torch.arange(7)]).unsqueeze(0).to(lfm2_device)
    actual = model(tokens, position_ids=positions, seq_lens=lengths)["logits"]
    expected = torch.cat([reference(part, use_cache=False).logits for part in tokens.split([5, 7], dim=1)], dim=1)
    atol, rtol = (2e-2, 5e-2) if lfm2_device.type == "cuda" else (2e-6, 2e-5)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    target = torch.randint(0, 128, (12,), device=lfm2_device)
    F.cross_entropy(actual.float().reshape(-1, 128), target).backward()
    F.cross_entropy(expected.float().reshape(-1, 128), target).backward()
    for name, parameter in model.named_parameters():
        other = reference.get_parameter(name)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other.grad, atol=atol, rtol=rtol, msg=name)
    changed = tokens.clone()
    changed[:, :5] = (changed[:, :5] + 1) % 128
    with torch.no_grad():
        isolated = model(changed, position_ids=positions, seq_lens=lengths)["logits"]
    torch.testing.assert_close(isolated[:, 5:], actual.detach()[:, 5:], atol=0, rtol=0)


@pytest.mark.parametrize("chunk_size", [None, 4])
def test_lfm2_prompt_derivatives_then_backward(lfm2_device, chunk_size):
    torch.manual_seed(62)
    dtype = torch.bfloat16 if lfm2_device.type == "cuda" else torch.float32
    config = tiny_lfm2_config(attn="flash_attention_2" if lfm2_device.type == "cuda" else "sdpa")
    model = AutoModelForCausalLMPrimeRL.from_config(config, dtype=dtype).to(lfm2_device)
    inject_prime_lm_head(model, chunk_size=chunk_size)
    params = list(model.parameters())
    tokens = torch.randint(1, 128, (1, 12), device=lfm2_device)
    labels = tokens.roll(-1, dims=1)
    outputs = model(
        tokens,
        labels=labels,
        temperature=torch.ones_like(tokens, dtype=torch.float32),
        seq_lens=torch.tensor([5, 7], device=lfm2_device),
    )
    if chunk_size is None:
        logprobs = outputs["logits"].float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    else:
        logprobs = outputs["logprobs"]
    losses = [-part.sum() / tokens.numel() for part in logprobs.split([5, 7], dim=1)]
    extracted = [torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False) for loss in losses]
    assert all(parameter.grad is None for parameter in params)
    sum(losses).backward()
    atol, rtol = (3e-3, 5e-2) if lfm2_device.type == "cuda" else (2e-6, 2e-5)
    for index, parameter in enumerate(params):
        expected = sum(grads[index].float() for grads in extracted)
        torch.testing.assert_close(parameter.grad.float(), expected, atol=atol, rtol=rtol)


def test_lfm2_meta_rotary_buffers_are_restored():
    config = tiny_lfm2_config()
    with torch.device("meta"):
        model = AutoModelForCausalLMPrimeRL.from_config(config)
    model.to_empty(device="cpu")
    model.init_buffers_post_meta()
    reference = AutoModelForCausalLMPrimeRL.from_config(copy.deepcopy(config))
    for name, buffer in model.model.rotary_emb.named_buffers():
        torch.testing.assert_close(buffer, reference.model.rotary_emb.get_buffer(name), atol=0, rtol=0)


@pytest.fixture
def lfm2_process_group(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA or ROCm GPU")
    assert not dist.is_initialized(), "Run with ordinary pytest, without torchrun"
    from prime_rl.trainer.models.layers.attn import get_flash_attn_func

    get_flash_attn_func(2)
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


@pytest.mark.gpu
def test_lfm2_fsdp_capture_preserves_training(lfm2_process_group, tmp_path):
    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.trainer.model import setup_fsdp
    from prime_rl.trainer.parallel_dims import ParallelDims
    from prime_rl.trainer.rl.gradients import PromptGradientCapture
    from prime_rl.utils.config import cli

    fixture = Path(__file__).resolve().parents[3] / "fixtures" / "gradient_capture" / "fsdp.toml"
    trainer = cli(TrainerConfig, args=["@", str(fixture), "--output-dir", str(tmp_path)])
    config = tiny_lfm2_config(tied=True, attn="flash_attention_2")
    torch.manual_seed(63)
    baseline = AutoModelForCausalLMPrimeRL.from_config(config, dtype=torch.float32).cuda()
    captured = AutoModelForCausalLMPrimeRL.from_config(copy.deepcopy(config), dtype=torch.float32).cuda()
    captured.load_state_dict(baseline.state_dict())
    dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)
    for model in (baseline, captured):
        inject_prime_lm_head(model, chunk_size=4)
        setup_fsdp(model, trainer.model, dims)
    baseline_params = dict(baseline.named_parameters())
    captured_params = dict(captured.named_parameters())
    baseline_optim = torch.optim.AdamW(baseline_params.values(), lr=trainer.optim.lr, weight_decay=0.0)
    captured_optim = torch.optim.AdamW(captured_params.values(), lr=trainer.optim.lr, weight_decay=0.0)

    for step in range(1, trainer.max_steps + 1):
        capture = None
        if step in trainer.gradient_capture.steps:
            capture = PromptGradientCapture(captured, trainer.gradient_capture, tmp_path, step, {"tokens": 24.0})
        for micro in range(2):
            generator = torch.Generator().manual_seed(step * 100 + micro)
            tokens = torch.randint(1, 128, (1, 12), generator=generator).cuda()
            lengths = torch.tensor([5, 7], device="cuda")
            positions = torch.cat([torch.arange(5), torch.arange(7)]).unsqueeze(0).cuda()
            batch = {
                "sequence_lengths": [5, 7],
                "group_ids": ["prompt-a", "prompt-b"],
                "trace_ids": [f"a-{micro}", f"b-{micro}"],
                "branch_indices": [0, 0],
                "env_names": ["lfm2-test"] * 12,
                "loss_mask": torch.ones_like(tokens, dtype=torch.bool),
                "ce_weights": None,
                "ref_kl_weights": None,
            }
            for model in (baseline, captured):
                if model is captured and capture is not None:
                    capture.before_forward()
                outputs = model(
                    tokens,
                    position_ids=positions,
                    labels=tokens.roll(-1, dims=1),
                    temperature=torch.ones_like(tokens, dtype=torch.float32),
                    seq_lens=lengths,
                )
                parts = outputs["logprobs"].split([5, 7], dim=1)
                sequence_losses = [-parts[0].sum() / 24, 0.5 * parts[1].sum() / 24]
                loss = sum(sequence_losses)
                if model is captured and capture is not None:
                    capture.capture_microbatch(loss, sequence_losses, batch)
                loss.backward()
            for name, param in captured_params.items():
                assert param.grad is not None and torch.isfinite(param.grad.to_local()).all(), name
                torch.testing.assert_close(
                    param.grad.to_local(), baseline_params[name].grad.to_local(), rtol=0, atol=0, msg=name
                )
        if capture is not None:
            directory = capture.finish_step()
            manifest = json.loads((directory / "manifest.json").read_text())
            assert manifest["verification"]["checked"]
            assert manifest["microbatches"] == 2
            assert {group["group_id"] for group in manifest["groups"]} == {"prompt-a", "prompt-b"}
        baseline_optim.step()
        captured_optim.step()
        for name, param in captured_params.items():
            torch.testing.assert_close(param.to_local(), baseline_params[name].to_local(), rtol=0, atol=0, msg=name)
        baseline_optim.zero_grad(set_to_none=True)
        captured_optim.zero_grad(set_to_none=True)
