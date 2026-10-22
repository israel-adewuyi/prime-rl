"""Sanity tests for the prime-RL ``ServingTokens`` subclass.

The full happy-path is owned upstream by vLLM's
``vllm/entrypoints/serve/disagg`` test suite. We only cover the prime-RL
deltas here:
    * ``serialize_routed_experts`` round-trips a compact raw-byte payload.
    * The subclass overrides ``serve_tokens_full_generator`` without
      monkey-patching the parent.
    * ``post_process`` swaps in the compact routed_experts while preserving
      the rest of the upstream response (``usage`` included).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pybase64
import pytest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateResponse,
    GenerateResponseChoice,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo

from prime_rl.inference.vllm.routed_experts import serialize_routed_experts
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlServingTokens,
    _GenerateRoutedExpertsCapture,
)


def _decode_routed_experts(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=np.uint8,
    ).reshape(encoded["shape"])


async def _empty_request_outputs():
    if False:
        yield


def test_subclass_overrides_serve_tokens_full_generator():
    upstream = PrimeRlServingTokens.__mro__[1]
    assert PrimeRlServingTokens.serve_tokens_full_generator is not upstream.serve_tokens_full_generator


def test_serialize_routed_experts_uses_compact_raw_payload():
    routed_experts = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int64,
    )

    encoded = serialize_routed_experts(routed_experts)
    assert encoded is not None

    decoded = _decode_routed_experts(encoded)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, routed_experts)


def test_generate_response_post_process_preserves_prompt_metadata():
    compact_routed_experts = {"data": "AQID", "shape": [1, 1, 3], "start": 0}
    capture = _GenerateRoutedExpertsCapture(_empty_request_outputs())
    capture.routed_experts[0] = compact_routed_experts
    usage = UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7)
    response = GenerateResponse(
        request_id="request-id",
        model="test-model",
        choices=[
            GenerateResponseChoice(
                index=0,
                token_ids=[1, 2, 3],
                routed_experts="upstream-npy-payload",
            )
        ],
        usage=usage,
        prompt_token_ids=[10, 11, 12, 13],
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=1, length=2)]},
    )

    processed = capture.post_process(response)

    assert processed.choices[0].routed_experts == compact_routed_experts
    assert processed.model == "test-model"
    assert processed.usage == usage
    payload = processed.model_dump(mode="json")
    assert payload["choices"][0]["routed_experts"] == compact_routed_experts
    assert payload["prompt_token_ids"] == [10, 11, 12, 13]
    assert payload["mm_placeholders"] == {"image": [{"offset": 1, "length": 2}]}
    assert payload["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["model", "aux_output"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_generate_handler_routed_expert_config_layout(layout, enabled, monkeypatch):
    handler = object.__new__(PrimeRlServingTokens)
    flag = SimpleNamespace(enable_return_routed_experts=enabled)
    handler.model_config = flag if layout == "model" else SimpleNamespace()
    config = SimpleNamespace() if layout == "model" else SimpleNamespace(aux_output_config=flag)
    handler.engine_client = SimpleNamespace(vllm_config=config)
    request = SimpleNamespace(sampling_params=SimpleNamespace(routed_experts_prompt_start=2))
    outputs = _empty_request_outputs()
    response = GenerateResponse(
        request_id="request-id",
        model="test-model",
        choices=[GenerateResponseChoice(index=0, token_ids=[1, 2, 3])],
        usage=UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7),
        prompt_token_ids=[10, 11, 12, 13],
    )
    upstream = AsyncMock(return_value=response)
    monkeypatch.setattr(PrimeRlServingTokens.__mro__[1], "serve_tokens_full_generator", upstream)
    metadata = SimpleNamespace()
    result = await handler.serve_tokens_full_generator(request, outputs, "request-id", "test-model", metadata)
    args = upstream.await_args.args
    assert args[0] is request
    assert args[2:] == ("request-id", "test-model", metadata)
    if enabled:
        assert isinstance(args[1], _GenerateRoutedExpertsCapture)
        assert result is not response
        assert result.model_dump(exclude={"choices"}) == response.model_dump(exclude={"choices"})
        assert result.choices[0].token_ids == response.choices[0].token_ids
    else:
        assert args[1] is outputs
        assert result is response
