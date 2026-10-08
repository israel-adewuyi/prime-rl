"""Dense Qwen3 correctness gates shared by CUDA and ROCm."""

import copy
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, get_custom_causal_lm_cls, supports_custom_impl
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head


def test_dense_imports_do_not_load_optional_backends():
    code = """
import sys

blocked = {"fla", "ring_flash_attn", "flash_attn", "flash_attn_interface", "torchao",
           "dion", "nixl", "deep_gemm", "deep_ep", "quack", "tilelang", "pynvml", "vllm", "prime_kernels"}
# Simulate an installation without optional backends, including on GPU hosts.
sys.modules.update(dict.fromkeys(blocked))
import transformers
from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.utils.cp import CPContext
config = transformers.Qwen3Config(hidden_size=64, intermediate_size=128,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    head_dim=32, vocab_size=128, attn_implementation="flash_attention_2")
model = AutoModelForCausalLMPrimeRL.from_config(config)
assert model.config.model_type == "qwen3"
assert not any(name.startswith("prime_rl.trainer.models.qwen3_5.modeling") for name in sys.modules)
assert not any(name.startswith("prime_rl.trainer.models.deepseek_v4.modeling") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


def test_qwen3_registry():
    config = Qwen3Config()
    assert supports_custom_impl(config)
    assert get_custom_causal_lm_cls(config).__name__ == "Qwen3ForCausalLM"


@pytest.fixture
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA or ROCm GPU")
    # A missing/broken selected backend is a failure on the target machine.
    from prime_rl.trainer.models.layers.attn import get_flash_attn_func

    get_flash_attn_func(2)
    return torch.device("cuda")


def make_model(tied: bool, device):
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=256,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
        attn_implementation="flash_attention_2",
    )
    model = AutoModelForCausalLMPrimeRL.from_config(config, dtype=torch.bfloat16).to(device)
    inject_prime_lm_head(model, chunk_size=None)
    return model


@pytest.mark.gpu
@pytest.mark.parametrize("tied", [True, False])
def test_qwen3_packed_logits_gradients_and_isolation(gpu, tied):
    torch.manual_seed(123)
    model = make_model(tied, gpu)
    reference_config = copy.deepcopy(model.config)
    reference_config._attn_implementation = "sdpa"
    reference = HFQwen3(reference_config).to(device=gpu, dtype=torch.bfloat16)
    reference.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.randint(0, 256, (1, 24), device=gpu)
    lengths = torch.tensor([9, 15], device=gpu)
    positions = torch.cat([torch.arange(9), torch.arange(15)]).unsqueeze(0).to(gpu)
    actual = model(tokens, position_ids=positions, seq_lens=lengths)["logits"]
    expected = torch.cat([reference(doc, use_cache=False).logits for doc in tokens.split([9, 15], dim=1)], dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    targets = torch.randint(0, 256, (24,), device=gpu)
    actual_loss = F.cross_entropy(actual.float().reshape(-1, 256), targets)
    expected_loss = F.cross_entropy(expected.float().reshape(-1, 256), targets)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-3, rtol=2e-3)
    actual_loss.backward()
    expected_loss.backward()
    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=3e-3, rtol=5e-2, msg=name)
    changed = tokens.clone()
    changed[:, :9] = (changed[:, :9] + 1) % 256
    with torch.no_grad():
        isolated = model(changed, position_ids=positions, seq_lens=lengths)["logits"]
    torch.testing.assert_close(isolated[:, 9:], actual.detach()[:, 9:], atol=0, rtol=0)


@pytest.mark.gpu
def test_qwen3_optimizer_checkpoint_resume(gpu, tmp_path):
    torch.manual_seed(321)
    model = make_model(False, gpu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    tokens = torch.randint(0, 256, (1, 16), device=gpu)
    lengths = torch.tensor([16], device=gpu)
    targets = torch.randint(0, 256, (16,), device=gpu)

    def step(m, opt):
        opt.zero_grad(set_to_none=True)
        logits = m(tokens, seq_lens=lengths)["logits"]
        loss = F.cross_entropy(logits.float().reshape(-1, 256), targets)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()

    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    step(model, optimizer)
    assert not torch.equal(before, model.model.layers[0].self_attn.q_proj.weight)
    path = tmp_path / "state.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    resumed = make_model(False, gpu)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3, foreach=False)
    state = torch.load(path, weights_only=True)
    resumed.load_state_dict(state["model"])
    resumed_optimizer.load_state_dict(state["optimizer"])
    step(model, optimizer)
    step(resumed, resumed_optimizer)
    for actual, expected in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
