"""Hardware gate for filesystem reloads and trainer/inference log-probability agreement."""

import math

import pytest
import torch


@pytest.mark.gpu
def test_qwen3_filesystem_weight_refresh(tmp_path, monkeypatch):
    # ROCm workers must not inherit the parent's initialized HIP runtime via fork.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA or ROCm GPU and the matching vLLM build")

    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from vllm import LLM, SamplingParams

    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attn_implementation="sdpa",
    )
    reference = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    original_path = tmp_path / "original"
    reference.save_pretrained(original_path)
    token_ids = [10, 20, 30, 40, 50]
    with torch.no_grad():
        logits = reference(torch.tensor([token_ids]), use_cache=False).logits.float()
        expected = logits[0, :-1].log_softmax(-1).gather(1, torch.tensor(token_ids[1:])[:, None]).flatten()
        reference.lm_head.weight.zero_()
    updated_path = tmp_path / "updated"
    reference.save_pretrained(updated_path)

    llm = LLM(
        model=str(original_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=128,
        max_num_seqs=2,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=False,
        worker_extension_cls="prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
    )

    def prompt_logprobs():
        result = llm.generate(
            [{"prompt_token_ids": token_ids}],
            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1, detokenize=False),
            use_tqdm=False,
        )[0]
        return torch.tensor([result.prompt_logprobs[i][token].logprob for i, token in enumerate(token_ids) if i])

    before = prompt_logprobs()
    torch.testing.assert_close(before, expected, atol=2e-2, rtol=2e-3)
    llm.collective_rpc("update_weights_from_path", args=(str(updated_path),))
    after = prompt_logprobs()
    torch.testing.assert_close(after, torch.full_like(after, -math.log(256)), atol=2e-3, rtol=0)
    assert (after - before).abs().max() > 0.02
    llm.collective_rpc("update_weights_from_path", args=(str(original_path),))
    torch.testing.assert_close(prompt_logprobs(), before, atol=2e-3, rtol=0)
