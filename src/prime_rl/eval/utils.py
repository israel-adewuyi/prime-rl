import asyncio
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import whoami
from openai import AsyncOpenAI
from prime_evals import AsyncEvalsClient
from verifiers import load_environment
from verifiers.types import GenerateOutputs
from verifiers.utils.eval_utils import get_hf_hub_dataset_name, make_dataset, sanitize_metadata, save_to_disk

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.orchestrator.config import ClientConfig, EvalConfig, EvalSamplingConfig, EvalSaveConfig, ModelConfig
from prime_rl.orchestrator.utils import parse_is_truncated_completions, parse_num_completion_tokens
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize, get_eval_dir, get_step_path
from prime_rl.utils.vf import generate_batch

def compute_pass_at_k(rewards: list[int]) -> dict[str, float]:
    total_attempts = len(rewards)

    if total_attempts == 0:
        return {"pass@1": 0.0}

    num_correct = sum(1 for reward in rewards if reward == 1.0)

    def pass_at_k(n: int, c: int, k: int) -> float:
        if c == 0:
            return 0.0
        if k > n:
            return 0.0
        if n - c < k:
            return 1.0

        # 1 - C(n-c, k) / C(n, k)
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    results = {}

    k = total_attempts
    while k >= 1:
        results[f"pass@{k}"] = float(pass_at_k(total_attempts, num_correct, k))
        k //= 2

    return results
# def compute_pass_at_k(rewards: list[int]) -> dict[str, float]:
#     total_attempts = len(rewards)
#     k = total_attempts // 2

#     if k == 0:
#         return {"pass@1": float(any(reward == 1.0 for reward in rewards))}

#     num_trials = 100
#     pass_rates = []

#     for _ in range(num_trials):
#         sampled_rewards = np.random.choice(rewards, size=k, replace=False)
#         pass_rate = float(any(reward == 1.0 for reward in sampled_rewards))
#         pass_rates.append(pass_rate)

#     return {f"pass@{k}": float(np.mean(pass_rates))}


def prepare_sampling_args(
    sampling_config: EvalSamplingConfig, client_config: ClientConfig, top_logprobs: int | None = None
) -> dict[str, Any]:
    """Prepare sampling args for the client."""
    # Initialize sampling args
    sampling_args: dict[str, Any] = {}

    # Apply sampling arguments, if specified
    if sampling_config.temperature is not None:
        sampling_args["temperature"] = sampling_config.temperature
    if sampling_config.max_tokens is not None:
        sampling_args["max_tokens"] = sampling_config.max_tokens
    if sampling_config.top_p is not None:
        sampling_args["top_p"] = sampling_config.top_p
    if sampling_config.reasoning_effort is not None:
        sampling_args["reasoning_effort"] = sampling_config.reasoning_effort

    if client_config.server_type == "vllm":
        # Always return logprobs and token IDs from vLLM server
        sampling_args["logprobs"] = True
        if top_logprobs is not None:
            sampling_args["top_logprobs"] = top_logprobs
        extra_body: dict[str, Any] = {"return_tokens_as_token_ids": True}

        # Apply vLLM-specific sampling arguments, if specified
        if sampling_config.top_k is not None:
            extra_body["top_k"] = sampling_config.top_k
        if sampling_config.min_p is not None:
            extra_body["min_p"] = sampling_config.min_p
        if sampling_config.min_tokens is not None:
            extra_body["min_tokens"] = sampling_config.min_tokens
        if sampling_config.repetition_penalty is not None:
            extra_body["repetition_penalty"] = sampling_config.repetition_penalty

        sampling_args["extra_body"] = extra_body

    return sampling_args


def _get_response_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _parse_vllm_token_id(token: Any) -> int | None:
    if not isinstance(token, str):
        return None
    prefix = "token_id:"
    if not token.startswith(prefix):
        return None
    try:
        return int(token.removeprefix(prefix))
    except ValueError:
        return None


def _extract_top_logprobs(top_logprobs: Any) -> list[dict[str, Any]] | None:
    if top_logprobs is None:
        return None

    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(top_logprobs):
        token = _get_response_field(candidate, "token")
        logprob = _get_response_field(candidate, "logprob")
        if logprob is None:
            continue
        rows.append(
            {
                "rank": rank,
                "token": token,
                "token_id": _parse_vllm_token_id(token),
                "logprob": logprob,
                "probability": math.exp(logprob),
            }
        )
    return rows


def extract_token_metadata(results: GenerateOutputs) -> list[dict[str, Any]]:
    """Extract sampled-token IDs, logprobs, and probabilities from vLLM chat responses."""
    rows: list[dict[str, Any]] = []

    for rollout_idx, state in enumerate(results.state):
        responses = _get_response_field(state, "responses", [])
        global_token_idx = 0

        for turn_idx, response in enumerate(responses):
            choices = _get_response_field(response, "choices", []) or []
            if len(choices) != 1:
                continue

            choice = choices[0]
            logprobs = _get_response_field(choice, "logprobs")
            content = _get_response_field(logprobs, "content") if logprobs is not None else None
            if content is None:
                continue

            for token_idx, token_logprob in enumerate(content):
                token = _get_response_field(token_logprob, "token")
                logprob = _get_response_field(token_logprob, "logprob")
                top_logprobs = _extract_top_logprobs(_get_response_field(token_logprob, "top_logprobs"))
                if logprob is None:
                    continue

                rows.append(
                    {
                        "rollout_idx": rollout_idx,
                        "example_id": results.example_id[rollout_idx],
                        "task": results.task[rollout_idx],
                        "reward": results.reward[rollout_idx],
                        "turn_idx": turn_idx,
                        "token_idx": token_idx,
                        "global_token_idx": global_token_idx,
                        "token": token,
                        "token_id": _parse_vllm_token_id(token),
                        "logprob": logprob,
                        "probability": math.exp(logprob),
                        "top_logprobs": top_logprobs,
                    }
                )
                global_token_idx += 1

    return rows


def filter_token_metadata_examples(
    token_rows: list[dict[str, Any]], example_ids: list[int], max_examples: int | None, seed: int
) -> list[dict[str, Any]]:
    if max_examples is None:
        return token_rows

    unique_example_ids = list(dict.fromkeys(example_ids))
    if max_examples >= len(unique_example_ids):
        return token_rows

    selected = set(random.Random(seed).sample(unique_example_ids, max_examples))
    return [row for row in token_rows if row["example_id"] in selected]


async def run_eval(
    clients: list[AsyncOpenAI],
    env_id: str,
    env_name: str | None,
    env_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    output_dir: Path,
    ckpt_step: int,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    client_config: ClientConfig,
    save_config: EvalSaveConfig,
    evals_client: AsyncEvalsClient,
    step: int | None = None,
) -> None:
    # Get the logger
    logger = get_logger()
    monitor = get_monitor()
    eval_start_time = time.perf_counter()

    # Load the eval environment
    env_name_or_id = env_name or env_id
    env = load_environment(env_id, **env_args)
    dataset = env.get_eval_dataset(n=num_examples)
    token_metadata_config = save_config.token_metadata
    top_logprobs = (
        token_metadata_config.top_logprobs
        if token_metadata_config is not None and token_metadata_config.enabled
        else None
    )
    sampling_args = prepare_sampling_args(sampling_config, client_config, top_logprobs=top_logprobs)

    logger.info(
        f"Evaluating {env_name_or_id} ({num_examples=}, {rollouts_per_example=}) {'with default args' if env_args == {} else f'with args {env_args}'}"
    )
    # Run async generation and scoring
    results: GenerateOutputs = await generate_batch(
        env=env,
        model_name=model_config.name,
        problems=dataset.to_list(),
        clients=clients,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        pbar_description=f"Evaluating {env_name_or_id}",
    )

    # Parse vLLM responses
    k = rollouts_per_example
    responses = [state["responses"] for state in results.state]
    results_df = pd.DataFrame(
        {
            "example_id": results.example_id,
            "reward": results.reward,
            "completion_len": parse_num_completion_tokens(responses),
            "is_truncated": parse_is_truncated_completions(responses),
        }
    )
    unique_rewards = results_df.reward.unique()
    best_at_k = results_df.groupby("example_id").reward.max().mean()
    could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
    if could_be_binary:
        pass_at_k = (
            results_df.groupby("example_id")
            .apply(lambda x: compute_pass_at_k(x.reward), include_groups=False)
            .apply(pd.Series)
        )
    else:
        pass_at_k = None
        logger.warning("Skipping computing pass@k rates because the task rewards appear to be non-binary")

    # Log statistics to console
    eval_time = time.perf_counter() - eval_start_time
    message = (
        f"Evaluated {env_name_or_id} in {eval_time:.2f}s "
        f"(Avg@{k}={results_df.reward.mean():.4f}, Best@{k}={best_at_k:.4f}"
    )
    if could_be_binary:
        assert pass_at_k is not None
        for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
            message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"
    message += f", Completion Length: {results_df.completion_len.mean():.2f} (±{results_df.completion_len.std():.2f}, ∈[{results_df.completion_len.min():.2f}, {results_df.completion_len.max():.2f}]), Truncated: {results_df.is_truncated.mean() * 100:.1f}%)"
    logger.success(message)

    # Log statistics to monitor
    eval_metrics = {
        f"avg@{k}": results_df.reward.mean(),
        f"best@{k}": best_at_k,
        "completion_len/avg": results_df.completion_len.mean().item(),
        "completion_len/max": results_df.completion_len.max().item(),
        "completion_len/min": results_df.completion_len.min().item(),
        "is_truncated/mean": results_df.is_truncated.mean().item(),
        "time": eval_time,
    }
    if could_be_binary:
        assert pass_at_k is not None
        eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
    eval_metrics = {**{f"eval/{env_name_or_id}/{k}": v for k, v in eval_metrics.items()}}
    eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step or ckpt_step})
    monitor.log(eval_metrics)

    # Save results
    if save_config.disk is not None or save_config.hf is not None or save_config.env_hub:
        dataset = make_dataset(results)
        metadata_dict = sanitize_metadata(results.metadata)

        if save_config.disk is not None:
            is_online = step is not None
            default_save_path = (
                get_step_path(get_eval_dir(output_dir), ckpt_step) / env_name_or_id
                if is_online
                else results.metadata.path_to_save
            )
            save_path = save_config.disk.path or default_save_path
            save_to_disk(dataset, metadata_dict, save_path)
            logger.info(f"Saved eval results for {env_name_or_id} to disk ({save_path})")

        if save_config.hf is not None:
            dataset_name = save_config.hf.dataset_name or get_hf_hub_dataset_name(results)
            dataset_subset = save_config.hf.dataset_subset or env.env_id
            dataset_split = save_config.hf.dataset_split or "evals"
            
            # Add string substitution for {step} placeholder  
            step_value = str(step) if step is not None else ckpt_step
            dataset_split = dataset_split.replace("{step}", step_value)
            
            dataset.push_to_hub(dataset_name, dataset_subset, split=dataset_split, private=save_config.hf.private)
            default_org = whoami().get("name", "")
            repo_name = dataset_name if "/" in dataset_name else f"{default_org}/{dataset_name}"
            logger.info(
                f"Pushed {'private' if save_config.hf.private else 'public'} eval results for {env_name_or_id} to HF Hub (https://huggingface.co/datasets/{repo_name})"
            )

        if save_config.env_hub:
            eval_name = f"{env_id}--{model_config.name.replace('/', '--')}"

            # Create evaluation for environment
            create_response = await evals_client.create_evaluation(
                name=eval_name,
                environments=[{"id": env_id}],
                model_name=model_config.name,
                framework="verifiers",
                metadata=metadata_dict,
                metrics=eval_metrics,
            )

            eval_id = create_response.get("evaluation_id")
            assert eval_id is not None

            # Push samples
            await evals_client.push_samples(eval_id, dataset.to_list())

            # Finalize evaluation
            await evals_client.finalize_evaluation(eval_id, metrics=eval_metrics)

            logger.info(f"Pushed eval results for {env_id} to Environment Hub (eval_id: {eval_id})")

    if save_config.token_metadata is not None and save_config.token_metadata.enabled:
        token_rows = extract_token_metadata(results)
        token_rows = filter_token_metadata_examples(
            token_rows=token_rows,
            example_ids=results.example_id,
            max_examples=save_config.token_metadata.max_examples,
            seed=save_config.token_metadata.seed,
        )
        token_path = save_config.token_metadata.path or (
            get_step_path(get_eval_dir(output_dir), ckpt_step) / env_name_or_id / "token_metadata"
        )
        Dataset.from_list(token_rows).save_to_disk(token_path)
        logger.info(
            f"Saved token metadata for {env_name_or_id} to disk ({token_path}, num_tokens={len(token_rows)})"
        )


async def run_evals(
    clients: list[AsyncOpenAI],
    eval_config: EvalConfig | OfflineEvalConfig,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    client_config: ClientConfig,
    evals_client: AsyncEvalsClient,
    output_dir: Path,
    ckpt_step: int,
    step: int | None = None,
):
    await asyncio.gather(
        *[
            run_eval(
                clients=clients,
                env_id=env.id,
                env_name=env.name,
                env_args=env.args,
                num_examples=env.num_examples or eval_config.num_examples,
                rollouts_per_example=env.rollouts_per_example or eval_config.rollouts_per_example,
                output_dir=output_dir,
                model_config=model_config,
                sampling_config=sampling_config,
                client_config=client_config,
                save_config=eval_config.save,
                evals_client=evals_client,
                ckpt_step=ckpt_step,
                step=step,
            )
            for env in eval_config.env
        ]
    )
