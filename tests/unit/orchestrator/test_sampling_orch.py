from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.config import SamplingConfig
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.utils.vf import generate_group_with_per_rollout_sampling_args, make_rollouts


@pytest.fixture
def anyio_backend():
    return "asyncio"


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


def test_make_rollouts_prefers_state_sampling_args():
    generate_outputs = SimpleNamespace(
        example_id=[123],
        task=["dummy-task"],
        metrics={},
        state=[
            {
                "sampling_args": {
                    "temperature": 1.2,
                    "top_p": 0.9,
                    "extra_body": {"top_k": 4},
                }
            }
        ],
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

    assert rollout["temperature"] == 1.2
    assert rollout["top_p"] == 0.9
    assert rollout["top_k"] == 4


@pytest.mark.anyio
async def test_generate_group_with_per_rollout_sampling_args_preserves_order_and_args():
    class FakeRubric:
        def __init__(self):
            self.score_calls = []

        async def score_rollouts(self, **kwargs):
            self.score_calls.append(kwargs)
            return SimpleNamespace(
                reward=[0.25, 0.75],
                metrics={"score": [0.25, 0.75]},
            )

    class FakeEnv:
        env_id = "fake"
        env_args = {}

        def __init__(self):
            self.rubric = FakeRubric()
            self.rollout_sampling_args = []

        async def init_completion(self):
            return []

        async def init_state(self, prompt, completion, answer, task, info, example_id):
            return {
                "prompt": prompt,
                "completion": completion,
                "answer": answer,
                "task": task,
                "info": info,
                "example_id": example_id,
                "responses": [],
            }

        async def run_rollout(
            self,
            sem,
            client,
            model,
            prompt,
            completion,
            answer,
            state,
            task,
            info,
            example_id,
            sampling_args,
        ):
            self.rollout_sampling_args.append(sampling_args)
            state["sampling_args"] = sampling_args
            return [{"role": "assistant", "content": str(sampling_args["temperature"])}], state

    client = SimpleNamespace(base_url="http://localhost")
    env = FakeEnv()
    rollout_sampling_args = [
        {"temperature": 0.6, "top_p": 1.0, "extra_body": {"top_k": 2}},
        {"temperature": 1.4, "top_p": 1.0, "extra_body": {"top_k": 8}},
    ]

    outputs = await generate_group_with_per_rollout_sampling_args(
        client=client,
        env=env,
        model_name="model",
        problem={
            "prompt": [{"role": "user", "content": "hi"}],
            "answer": "",
            "task": "fake",
            "info": {},
            "example_id": 123,
        },
        rollout_sampling_args=rollout_sampling_args,
    )

    assert env.rollout_sampling_args == rollout_sampling_args
    assert outputs.example_id == [123, 123]
    assert [state["sampling_args"] for state in outputs.state] == rollout_sampling_args
    assert outputs.reward == [0.25, 0.75]
    assert len(env.rubric.score_calls) == 1
