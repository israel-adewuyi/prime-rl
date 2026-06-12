import asyncio
import copy
import time
from collections import defaultdict
from datetime import datetime
from itertools import cycle
from random import Random
from typing import TypedDict

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI

from prime_rl.orchestrator.utils import get_semaphore


ROLLOUT_TEMPERATURE_CHOICES = (0.6, 0.8, 1.0, 1.2, 1.4, 1.5)
ROLLOUT_TOP_K_CHOICES = (2, 4, 8)


def get_field(obj, field: str, default=None):
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def build_rollout_sampling_args(base_sampling_args: dict, rollouts_per_example: int, rng: Random) -> list[dict]:
    """Build per-rollout sampling args for a single prompt group."""
    rollout_sampling_args = []
    for _ in range(rollouts_per_example):
        sampling_args = copy.deepcopy(base_sampling_args)
        sampling_args["temperature"] = rng.choice(ROLLOUT_TEMPERATURE_CHOICES)
        sampling_args.setdefault("extra_body", {})["top_k"] = rng.choice(ROLLOUT_TOP_K_CHOICES)
        rollout_sampling_args.append(sampling_args)
    return rollout_sampling_args


def _extract_sampling_params(sampling_args: dict) -> tuple[float, float, int]:
    extra_body = sampling_args.get("extra_body", {})
    return sampling_args.get("temperature", 1.0), sampling_args.get("top_p", 1.0), extra_body.get("top_k", -1)


def merge_metadata(generate_metadata_list: list[vf.GenerateMetadata]) -> vf.GenerateMetadata:
    """Merge multiple GenerateMetadata into a single GenerateMetadata."""
    num_examples = len(generate_metadata_list)  # Assumes one generate metadata per example
    time_ms = max(metadata.time_ms for metadata in generate_metadata_list)
    avg_reward = sum(metadata.avg_reward for metadata in generate_metadata_list) / num_examples
    avg_metrics = {
        key: sum(metadata.avg_metrics[key] for metadata in generate_metadata_list) / num_examples
        for key in generate_metadata_list[0].avg_metrics
    }
    state_columns = []
    for metadata in generate_metadata_list:
        state_columns.extend(metadata.state_columns)
    return vf.GenerateMetadata(
        env_id=generate_metadata_list[0].env_id,
        env_args=generate_metadata_list[0].env_args,
        model=generate_metadata_list[0].model,
        base_url=generate_metadata_list[0].base_url,
        num_examples=num_examples,
        rollouts_per_example=generate_metadata_list[0].rollouts_per_example,
        sampling_args=generate_metadata_list[0].sampling_args,
        date=generate_metadata_list[0].date,
        time_ms=time_ms,
        avg_reward=avg_reward,
        avg_metrics=avg_metrics,
        state_columns=state_columns,
        path_to_save=generate_metadata_list[0].path_to_save,
    )


def merge_outputs(generate_outputs_list: list[vf.GenerateOutputs]) -> vf.GenerateOutputs:
    """Merge multiple GenerateOutputs into a single GenerateOutputs."""
    example_id, prompt, completion, answer, state, reward, info, task, metrics = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        defaultdict(list),
    )
    for generate_output in generate_outputs_list:
        example_id.extend(generate_output.example_id)
        prompt.extend(generate_output.prompt)
        completion.extend(generate_output.completion)
        answer.extend(generate_output.answer)
        state.extend(generate_output.state)
        reward.extend(generate_output.reward)
        info.extend(generate_output.info)
        task.extend(generate_output.task)
        for key, value in generate_output.metrics.items():
            metrics[key].extend(value)
    metadata = merge_metadata([generate_output.metadata for generate_output in generate_outputs_list])
    return vf.GenerateOutputs(
        prompt=prompt,
        completion=completion,
        answer=answer,
        state=state,
        reward=reward,
        info=info,
        task=task,
        metrics=metrics,
        metadata=metadata,
        example_id=example_id,
    )


async def generate_group(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    problem: dict,
    rollouts_per_example: int,
    sampling_args: dict,
    use_tqdm: bool = False,
) -> vf.GenerateOutputs:
    """Asynchronously generate and score rollouts for one problem."""
    semaphore = get_semaphore()
    return await env.generate(
        inputs=Dataset.from_list([problem] * rollouts_per_example),
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        semaphore=semaphore,
        use_tqdm=use_tqdm,
    )


async def generate_group_with_per_rollout_sampling_args(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    problem: dict,
    rollout_sampling_args: list[dict],
) -> vf.GenerateOutputs:
    """Generate one prompt group with a different sampling config per rollout."""
    from verifiers.utils.async_utils import maybe_semaphore
    from verifiers.utils.message_utils import cleanup_messages
    from verifiers.utils.path_utils import get_results_path

    prompt = cleanup_messages(copy.deepcopy(problem["prompt"]))
    answer = copy.deepcopy(problem.get("answer", ""))
    task = copy.deepcopy(problem.get("task", "default"))
    info = copy.deepcopy(problem.get("info", {}))
    if isinstance(info, str):
        import json

        info = json.loads(info) if info else {}
    example_id = problem["example_id"]

    prompts = [copy.deepcopy(prompt) for _ in rollout_sampling_args]
    completions = [await env.init_completion() for _ in rollout_sampling_args]
    answers = [copy.deepcopy(answer) for _ in rollout_sampling_args]
    tasks = [copy.deepcopy(task) for _ in rollout_sampling_args]
    infos = [copy.deepcopy(info) for _ in rollout_sampling_args]
    example_ids = [example_id for _ in rollout_sampling_args]
    states = [
        await env.init_state(prompt_i, completion_i, answer_i, task_i, info_i, example_id_i)
        for prompt_i, completion_i, answer_i, task_i, info_i, example_id_i in zip(
            prompts, completions, answers, tasks, infos, example_ids
        )
    ]

    semaphore = get_semaphore() or await maybe_semaphore(-1)
    start_time = time.time()
    rollout_tasks = [
        env.run_rollout(
            semaphore,
            client,
            model_name,
            prompt_i,
            completion_i,
            answer_i,
            state_i,
            task_i,
            info_i,
            example_id_i,
            sampling_args_i,
        )
        for prompt_i, completion_i, answer_i, state_i, task_i, info_i, example_id_i, sampling_args_i in zip(
            prompts, completions, answers, states, tasks, infos, example_ids, rollout_sampling_args
        )
    ]
    rollout_results = await asyncio.gather(*rollout_tasks)
    completions = [completion for completion, _ in rollout_results]
    states = [state for _, state in rollout_results]
    for state, sampling_args in zip(states, rollout_sampling_args):
        state["sampling_args"] = sampling_args

    rollout_scores = await env.rubric.score_rollouts(
        prompts=prompts,
        completions=completions,
        answers=answers,
        states=states,
        tasks=tasks,
        infos=infos,
        example_ids=example_ids,
        max_concurrent=-1,
        apply_weights=True,
        use_tqdm=False,
    )
    elapsed_ms = (time.time() - start_time) * 1000.0
    avg_reward = sum(rollout_scores.reward) / len(rollout_scores.reward) if rollout_scores.reward else 0.0
    avg_metrics = {
        name: sum(values) / len(values) if values else 0.0 for name, values in rollout_scores.metrics.items()
    }
    metadata = vf.GenerateMetadata(
        env_id=env.env_id,
        env_args=env.env_args,
        model=model_name,
        base_url=str(client.base_url),
        num_examples=1,
        rollouts_per_example=len(rollout_sampling_args),
        sampling_args=rollout_sampling_args[0],
        avg_reward=avg_reward,
        avg_metrics=avg_metrics,
        state_columns=[],
        path_to_save=get_results_path(env.env_id, model_name),
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        time_ms=elapsed_ms,
    )

    return vf.GenerateOutputs(
        prompt=prompts,
        completion=completions,
        answer=answers,
        state=states,
        reward=rollout_scores.reward,
        info=infos,
        task=tasks,
        metrics=rollout_scores.metrics,
        metadata=metadata,
        example_id=example_ids,
    )


async def generate_batch(
    clients: list[AsyncOpenAI],
    env: vf.Environment,
    model_name: str,
    problems: list[dict],
    rollouts_per_example: int,
    sampling_args: dict,
    pbar_description: str = "Generating rollouts",
) -> vf.GenerateOutputs:
    """Asynchronously generate and score rollouts for a list of problems."""
    from tqdm import tqdm

    pbar = tqdm(total=len(problems) * rollouts_per_example, desc=pbar_description)

    async def generate_group_with_progress(client, problem):
        """Generate rollouts for one problem and update progress."""
        result = await generate_group(
            client, env, model_name, problem, rollouts_per_example, sampling_args, use_tqdm=False
        )
        pbar.update(rollouts_per_example)
        return result

    try:
        generate_outputs_list: list[vf.GenerateOutputs] = await asyncio.gather(
            *[generate_group_with_progress(client, problem) for client, problem in zip(cycle(clients), problems)]
        )
    finally:
        pbar.close()

    return merge_outputs(generate_outputs_list)


# Non-batched version of vf.ProcessedOutputs
# Also includes advantage and example_id field
class Rollout(TypedDict):
    example_id: int
    task: str  # Typically the env name
    temperature: float
    top_p: float
    top_k: int
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    reward: float
    advantage: float
    is_truncated: bool
    metrics: dict[str, float]


def make_rollouts(
    generate_outputs: vf.GenerateOutputs,
    processed_outputs: vf.ProcessedOutputs,
    advantages: list[float],
    all_is_truncated: list[bool],
) -> list[Rollout]:
    """Processs vf.ProcessedOutputs to a list of rollouts."""
    metadata_sampling_args = get_field(get_field(generate_outputs, "metadata"), "sampling_args", {})
    states = get_field(generate_outputs, "state", [])

    rollouts = []
    for i, (
        example_id,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        completion_logprobs,
        reward,
        advantage,
        is_truncated,
        task,
    ) in enumerate(
        zip(
            generate_outputs.example_id,
            processed_outputs.prompt_ids,
            processed_outputs.prompt_mask,
            processed_outputs.completion_ids,
            processed_outputs.completion_mask,
            processed_outputs.completion_logprobs,
            processed_outputs.rewards,
            advantages,
            all_is_truncated,
            generate_outputs.task,
        )
    ):
        metrics = {k: v[i] for k, v in generate_outputs.metrics.items()}
        sampling_args = (
            get_field(states[i], "sampling_args", metadata_sampling_args) if i < len(states) else metadata_sampling_args
        )
        temperature, top_p, top_k = _extract_sampling_params(sampling_args)
        rollouts.append(
            Rollout(
                example_id=example_id,
                task=task,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_logprobs=completion_logprobs,
                reward=reward,
                metrics=metrics,
                advantage=advantage,
                is_truncated=is_truncated,
            )
        )

    return rollouts
