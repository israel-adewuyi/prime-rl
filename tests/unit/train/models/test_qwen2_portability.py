"""Qwen2 architecture checks and packed training parity against Hugging Face."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as HFQwen2

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, get_custom_causal_lm_cls, supports_custom_impl
from prime_rl.trainer.models.layers.attn import FlashAttention
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head


def make_models(tied, device, dtype, **config_args):
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
        attn_implementation="flash_attention_2",
        **config_args,
    )
    model = AutoModelForCausalLMPrimeRL.from_config(config, dtype=dtype).to(device)
    inject_prime_lm_head(model, chunk_size=None)
    reference_config = deepcopy(config)
    reference_config._attn_implementation = "sdpa"
    reference = HFQwen2(reference_config).to(device=device, dtype=dtype)
    reference.load_state_dict(model.state_dict(), strict=True)
    return model, reference


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("sliding", [False, True])
def test_qwen2_architecture_and_hf_state_dict(tied, sliding):
    model, reference = make_models(
        tied,
        "cpu",
        torch.float32,
        use_sliding_window=sliding,
        sliding_window=4,
        max_window_layers=1,
        rope_parameters={"rope_type": "linear", "factor": 2.0, "rope_theta": 1000000.0},
    )
    assert supports_custom_impl(model.config)
    assert get_custom_causal_lm_cls(model.config) is type(model)
    assert model.state_dict().keys() == reference.state_dict().keys()
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == tied
    for actual, expected in zip(model.model.layers, reference.model.layers, strict=True):
        assert actual.self_attn.sliding_window == expected.self_attn.sliding_window
        assert actual.self_attn.q_proj.bias is not None
        assert actual.self_attn.k_proj.bias is not None
        assert actual.self_attn.v_proj.bias is not None
        assert actual.self_attn.o_proj.bias is None
        assert not actual.self_attn.use_qk_norm
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, reference.model.rotary_emb.inv_freq, atol=0, rtol=0)


def cpu_packed_sdpa(self, q, k, v, cu_seqlens, max_seqlen):
    """CPU substitute for the GPU kernel only; keeps the model and packing path intact."""
    documents = []
    boundaries = cu_seqlens.tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        query, key, value = [x[start:end].transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
        mask = None
        window = getattr(self, "sliding_window", None)
        if window is not None:
            pos = torch.arange(end - start)
            distance = pos[:, None] - pos[None, :]
            mask = (distance >= 0) & (distance < window)
        result = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            is_causal=mask is None,
            enable_gqa=True,
        )
        documents.append(result.squeeze(0).transpose(0, 1))
    return torch.cat(documents)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("device_name", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_qwen2_packed_logits_and_gradients(monkeypatch, tied, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires a CUDA or ROCm GPU")
    if device_name == "cpu":
        monkeypatch.setattr(FlashAttention, "_compute_attention", cpu_packed_sdpa)
    torch.manual_seed(123)
    dtype = torch.bfloat16 if device_name == "cuda" else torch.float32
    model, reference = make_models(tied, device_name, dtype)
    tokens = torch.randint(0, 256, (1, 24), device=device_name)
    lengths = torch.tensor([9, 15], device=device_name)
    positions = torch.cat([torch.arange(9), torch.arange(15)]).unsqueeze(0).to(device_name)
    actual = model(tokens, position_ids=positions, seq_lens=lengths)["logits"]
    expected = torch.cat([reference(doc, use_cache=False).logits for doc in tokens.split([9, 15], dim=1)], dim=1)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    targets = torch.randint(0, 256, (24,), device=device_name)
    actual_loss = F.cross_entropy(actual.float().reshape(-1, 256), targets)
    expected_loss = F.cross_entropy(expected.float().reshape(-1, 256), targets)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-3, rtol=2e-3)
    actual_loss.backward()
    expected_loss.backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        model.named_parameters(),
        reference.named_parameters(),
        strict=True,
    ):
        assert name == ref_name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad,
            ref_parameter.grad,
            atol=3e-3 if dtype == torch.bfloat16 else 1e-5,
            rtol=5e-2 if dtype == torch.bfloat16 else 1e-5,
            msg=lambda message: f"{name}\n{message}",
        )


@pytest.mark.gpu
def test_qwen2_packed_document_isolation():
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA or ROCm GPU")
    model, _ = make_models(True, "cuda", torch.bfloat16)
    tokens = torch.randint(0, 256, (1, 24), device="cuda")
    lengths = torch.tensor([9, 15], device="cuda")
    positions = torch.cat([torch.arange(9), torch.arange(15)]).unsqueeze(0).cuda()
    with torch.no_grad():
        original = model(tokens, position_ids=positions, seq_lens=lengths)["logits"]
        tokens[:, :9] = (tokens[:, :9] + 1) % 256
        changed = model(tokens, position_ids=positions, seq_lens=lengths)["logits"]
    torch.testing.assert_close(original[:, 9:], changed[:, 9:], atol=0, rtol=0)
