"""Fixed-prompt streaming benchmark; no harness or training work is included."""
import argparse
import asyncio
import json
import random
import statistics
import time
import tomllib
from pathlib import Path

import aiohttp


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("base_url")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    c = tomllib.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(c["seed"])
    prompts = []
    for _ in range(c["batch_requests"]):
        words = rng.sample(c["words"], c["words_per_prompt"])
        prompts.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSort these words alphabetically. Put the sorted words inside <answer> and </answer> tags: " + ", ".join(words) + "<|im_end|>\n<|im_start|>assistant\n")
    (args.output / "prompts.json").write_text(json.dumps(prompts))
    semaphore = asyncio.Semaphore(c["concurrency"])
    timeout = aiohttp.ClientTimeout(total=180)
    connector = aiohttp.TCPConnector(limit=c["concurrency"])
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def request(i, greedy=False):
            async with semaphore:
                body = dict(model=c["model"], prompt=prompts[i], max_tokens=c["max_tokens"], temperature=0.0 if greedy else c["temperature"], top_p=c["top_p"], seed=c["seed"] + i, stream=True, stream_options={"include_usage": True}, logprobs=1)
                start = time.perf_counter()
                first = None
                chunks = []
                usage = {}
                async with session.post(args.base_url + "/v1/completions", json=body) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                            continue
                        item = json.loads(line[6:])
                        if item.get("usage"):
                            usage = item["usage"]
                        if item.get("choices"):
                            choice = item["choices"][0]
                            if choice.get("text") and first is None:
                                first = time.perf_counter()
                            chunks.append(choice)
                end = time.perf_counter()
                return dict(i=i, elapsed=end-start, ttft=None if first is None else first-start, usage=usage, choices=chunks)

        async def batch(label, count, greedy=False):
            start = time.perf_counter()
            values = await asyncio.gather(*(request(i, greedy) for i in range(count)), return_exceptions=True)
            elapsed = time.perf_counter()-start
            rows = [x if isinstance(x, dict) else {"error": repr(x)} for x in values]
            (args.output / (label + ".json")).write_text(json.dumps(rows))
            good = [x for x in rows if "error" not in x]
            latencies = sorted(x["elapsed"] for x in good)
            ttft = [x["ttft"] for x in good if x["ttft"] is not None]
            tokens = sum(x["usage"].get("completion_tokens", 0) for x in good)
            summary = dict(label=label, count=count, errors=count-len(good), wall_seconds=elapsed, output_tokens=tokens, tokens_per_second=tokens/elapsed, median_request_seconds=statistics.median(latencies) if latencies else None, p95_request_seconds=latencies[int(.95*(len(latencies)-1))] if latencies else None, median_ttft_seconds=statistics.median(ttft) if ttft else None)
            print(json.dumps(summary), flush=True)
            return summary

        summaries = [await batch("warmup", c["warmup_requests"])]
        if args.profile:
            async with session.post(args.base_url + "/start_profile") as response:
                print("START_PROFILE", response.status, await response.text(), flush=True)
            summaries.append(await batch("profile", c["profile_requests"]))
            async with session.post(args.base_url + "/stop_profile") as response:
                print("STOP_PROFILE", response.status, await response.text(), flush=True)
        else:
            for repeat in range(c["repeats"]):
                summaries.append(await batch(f"repeat-{repeat}", c["batch_requests"]))
            summaries.append(await batch("greedy", c["parity_requests"], True))
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
