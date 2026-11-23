"""Generate complete two-turn AlphabetSort groups and pack their GRPO samples."""

import asyncio
import math
from contextvars import ContextVar

import verifiers.v1 as vf
from alphabet_sort.taskset import AlphabetSortEnv
from verifiers.v1.clients import ModelContext

from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.orchestrator.train_sink import _prune_zero_advantages
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.trainer.rl.data import DataLoader

turn_seeds = ContextVar("alphabet_sort_turn_seeds")


class SeededAlphabetSort(AlphabetSortEnv):
    async def run(self, task, agents):
        agent = agents.agent
        async with agent.interaction(task) as interaction:
            for prompt, seed in zip(task.data.info["user_turns"], turn_seeds.get(), strict=True):
                agent.ctx.sampling.seed = seed
                if (await interaction.turn(prompt)).terminated:
                    break


async def collect(capture, tasks, env, clients, source, max_inflight):
    permits = asyncio.Semaphore(max_inflight)
    model = clients.model_name

    async def episode(prompt_id, replica):
        task = tasks[prompt_id]
        seeds = tuple(
            capture.request_seed(prompt_id, replica, turn) for turn in range(len(task.data.info["user_turns"]))
        )
        turn_seeds.set(seeds)
        sampling = vf.SamplingConfig(**source.sampling.to_sampling_args())
        ctx = ModelContext(model, clients.train_client, sampling)
        async with permits:
            result = await env.run_episode(task, ctx)
            await clients.finish_sessions([trace.id for trace in result.traces])
        return result

    pending = {}
    async with env.serving(), asyncio.TaskGroup() as group:
        for prompt_id in tasks:
            for replica in range(capture.group_size):
                pending[prompt_id, replica] = group.create_task(episode(prompt_id, replica))
    return {slot: job.result() for slot, job in pending.items()}


async def score_and_pack(capture, prompt_ids, episodes, algorithm, packer):
    capture.check_scored(prompt_ids, episodes)
    samples = []
    trace_ids = set()
    for prompt_id in prompt_ids:
        group = [episodes[prompt_id, replica] for replica in range(capture.group_size)]
        for episode in group:
            traces = [trace for _, trace in iter_trainable_traces([episode])]
            if not episode.ok or len(traces) != 1 or len(episode.traces) != 1:
                raise ValueError(f"{capture.name}: incomplete rollout for prompt {prompt_id}: {episode.last_error}")
            trace = traces[0]
            if trace.id in trace_ids or trace.task.data.idx != prompt_id:
                raise ValueError(f"{capture.name}: duplicate or misassigned trace {trace.id}")
            if trace.rewards.get("alphabet_sort") is None or not math.isfinite(trace.reward):
                raise ValueError(f"{capture.name}: missing or nonfinite AlphabetSort reward on {trace.id}")
            trace_ids.add(trace.id)
        await algorithm.score_group(group)
        for episode in group:
            trace = episode.traces[0]
            for sample in trace_to_samples(trace, env_name="alphabet-sort"):
                stamp_loss_routing(sample, algorithm.action_loss_type)
                sample.temperatures = [capture.temperature] * len(sample.token_ids)
                if _prune_zero_advantages(sample):
                    samples.append(sample)

    nonzero = len({sample.trace_id for sample in samples})
    packed = packer.pack(samples)[0] if samples else []
    return [DataLoader.micro_batch_to_tensor(batch) for batch in packed], len(trace_ids), nonzero
