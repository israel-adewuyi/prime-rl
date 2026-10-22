"""Report Qwen3 gradient discrepancies or ROCm memory API failures without changing tolerances."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def report_error(actual, expected, **context):
    a, b = actual.detach().float(), expected.detach().float()
    error = (a - b).abs()
    print(
        json.dumps(
            {
                **context,
                "max_abs": error.max().item(),
                "relative_l2": (error.norm() / b.norm().clamp_min(1e-12)).item(),
                "cosine": F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
                "failing_elements": (error > 3e-3 + 5e-2 * b.abs()).sum().item(),
                "elements": a.numel(),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            }
        )
    )


def report_memory():
    device = torch.cuda.current_device()
    torch.cuda.init()
    for name, operation in (
        ("cuda.mem_get_info", lambda: torch.cuda.mem_get_info(device)),
        ("accelerator.get_memory_info", lambda: torch.accelerator.get_memory_info(torch.device("cuda", device))),
    ):
        # Preserve both probe results when one API fails.
        try:
            print(json.dumps({"api": name, "result": operation()}))
        except RuntimeError as error:
            print(json.dumps({"api": name, "error": str(error)}))


def report_gradients(reference_backend):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    test_path = Path(__file__).resolve().parents[1] / "tests/unit/train/models/test_qwen3_portability.py"
    spec = importlib.util.spec_from_file_location("qwen3_portability", test_path)
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    for tied in (True, False):
        torch.manual_seed(123)
        model = tests.make_model(tied, torch.device("cuda"))
        config = copy.deepcopy(model.config)
        config._attn_implementation = "sdpa"
        reference = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
        reference.load_state_dict(model.state_dict(), strict=True)
        separate_model = copy.deepcopy(model)
        tokens = torch.randint(0, 256, (1, 24), device="cuda")
        lengths = torch.tensor([9, 15], device="cuda")
        positions = torch.cat([torch.arange(9), torch.arange(15)]).unsqueeze(0).cuda()
        actual = model(tokens, position_ids=positions, seq_lens=lengths)["logits"]
        backends = (
            [SDPBackend.MATH]
            if reference_backend == "math"
            else [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        )
        with sdpa_kernel(backends):
            expected = torch.cat([reference(doc, use_cache=False).logits for doc in tokens.split([9, 15], 1)], 1)
        targets = torch.randint(0, 256, (24,), device="cuda")
        separate = torch.cat(
            [
                separate_model(doc, seq_lens=torch.tensor([doc.shape[1]], device="cuda"))["logits"]
                for doc in tokens.split([9, 15], 1)
            ],
            1,
        )
        F.cross_entropy(actual.float().reshape(-1, 256), targets).backward()
        F.cross_entropy(expected.float().reshape(-1, 256), targets).backward()
        F.cross_entropy(separate.float().reshape(-1, 256), targets).backward()
        print(
            json.dumps(
                {
                    "tied": tied,
                    "reference_backend": reference_backend,
                    "logit_max_abs": (actual.float() - expected.float()).abs().max().item(),
                }
            )
        )
        for (name, parameter), (reference_name, other) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == reference_name
            report_error(parameter.grad, other.grad, parameter=name, tied=tied, comparison="packed_fa2_vs_hf_sdpa")
        for (name, parameter), (separate_name, other) in zip(
            model.named_parameters(), separate_model.named_parameters(), strict=True
        ):
            assert name == separate_name
            report_error(parameter.grad, other.grad, parameter=name, tied=tied, comparison="packed_fa2_vs_separate_fa2")
        with torch.no_grad():
            changed = tokens.clone()
            changed[:, :9] = (changed[:, :9] + 1) % 256
            isolated = model(changed, position_ids=positions, seq_lens=lengths)["logits"]
        print(
            json.dumps(
                {
                    "tied": tied,
                    "packed_vs_separate_max_abs": (actual.float() - separate.float()).abs().max().item(),
                    "document_isolation_max_abs": (actual[:, 9:].float() - isolated[:, 9:].float()).abs().max().item(),
                }
            )
        )


def report_attention(seed, layout):
    from prime_rl.trainer.models.layers.attn import get_flash_attn_func

    torch.manual_seed(seed)
    fa2 = get_flash_attn_func(2)
    lengths = [9, 15]
    boundaries = torch.tensor([0, 9, 24], device="cuda", dtype=torch.int32)
    packed = [
        torch.randn(24, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for heads in (4, 2, 2)
    ]
    if layout == "model":
        for index in (0, 1):
            packed[index] = packed[index].transpose(0, 1).contiguous().transpose(0, 1).detach().requires_grad_()
    print(json.dumps({"attention_layout": layout, "strides": [list(x.stride()) for x in packed], "seed": seed}))
    separate = [x.detach().clone().requires_grad_() for x in packed]
    reference = [x.detach().float().requires_grad_() for x in packed]
    upstream = torch.randn_like(packed[0])
    actual = fa2(*packed, boundaries, boundaries, 15, 15, causal=True)
    separate_outputs = []
    reference_outputs = []
    offset = 0
    for length in lengths:
        local_boundaries = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        q, k, v = (x[offset : offset + length] for x in separate)
        separate_outputs.append(fa2(q, k, v, local_boundaries, local_boundaries, length, length, causal=True))
        q, k, v = (x[offset : offset + length].transpose(0, 1) for x in reference)
        k, v = k.repeat_interleave(2, dim=0), v.repeat_interleave(2, dim=0)
        scores = (q @ k.transpose(-1, -2)) / (128**0.5)
        mask = torch.ones(length, length, device="cuda", dtype=torch.bool).tril()
        probabilities = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        reference_outputs.append((probabilities @ v).transpose(0, 1))
        offset += length
    independent = torch.cat(separate_outputs, dim=0)
    expected = torch.cat(reference_outputs, dim=0)
    actual.backward(upstream)
    independent.backward(upstream)
    expected.backward(upstream.float())
    for label, output in (("separate_fa2", independent), ("fp32_math", expected)):
        report_error(actual, output, comparison=f"attention_forward_packed_fa2_vs_{label}", seed=seed)
    for name, actual_input, separate_input, expected_input in zip(
        ("q", "k", "v"), packed, separate, reference, strict=True
    ):
        report_error(
            actual_input.grad,
            separate_input.grad,
            comparison="attention_backward_packed_fa2_vs_separate_fa2",
            input=name,
            seed=seed,
        )
        report_error(
            actual_input.grad,
            expected_input.grad,
            comparison="attention_backward_packed_fa2_vs_fp32_math",
            input=name,
            seed=seed,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["memory", "gradients", "attention"])
    parser.add_argument("--reference-backend", choices=["default", "math"], default="default")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--layout", choices=["contiguous", "model"], default="contiguous")
    args = parser.parse_args()
    print(json.dumps({"torch": torch.__version__, "hip": torch.version.hip}))
    if args.mode == "memory":
        report_memory()
    elif args.mode == "attention":
        report_attention(args.seed, args.layout)
    else:
        report_gradients(args.reference_backend)


if __name__ == "__main__":
    main()
