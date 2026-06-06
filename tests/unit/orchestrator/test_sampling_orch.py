from types import SimpleNamespace

from prime_rl.orchestrator.config import SamplingConfig
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.utils.vf import make_rollouts


def test_get_sampling_args_uses_configured_top_p_and_vllm_top_k_extra_body():
    sampling_args = get_sampling_args(SamplingConfig(temperature=0.7, top_p=0.9, top_k=8))

    assert sampling_args["temperature"] == 0.7
    assert sampling_args["top_p"] == 0.9
    assert sampling_args["extra_body"]["top_k"] == 8
    assert "top_k" not in sampling_args


def test_make_rollouts_extracts_sampling_metadata():
    generate_outputs = SimpleNamespace(
        example_id=[123],
        task=["dummy-task"],
        metrics={},
        metadata=SimpleNamespace(
            sampling_args={
                "temperature": 0.7,
                "top_p": 0.9,
                "extra_body": {"top_k": 8},
            }
        ),
    )
    processed_outputs = SimpleNamespace(
        prompt_ids=[[1, 2]],
        prompt_mask=[[0, 0]],
        completion_ids=[[3, 4]],
        completion_mask=[[1, 1]],
        completion_logprobs=[[-0.1, -0.2]],
        rewards=[1.0],
    )

    rollout = make_rollouts(
        generate_outputs=generate_outputs,
        processed_outputs=processed_outputs,
        advantages=[0.5],
        all_is_truncated=[False],
    )[0]

    assert rollout["temperature"] == 0.7
    assert rollout["top_p"] == 0.9
    assert rollout["top_k"] == 8
