"""Expand the fixed-checkpoint AlphabetSort sweep."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Capture:
    checkpoint: str
    repeat: int
    batch_size: int
    temperature: float
    prompt_seed: int
    engine_seed: int
    prompt_count: int

    @property
    def group_size(self) -> int:
        return self.batch_size // self.prompt_count

    @property
    def name(self) -> str:
        return (
            f"{self.checkpoint}-b{self.batch_size}-p{self.prompt_count}-r{self.group_size}"
            f"-t{self.temperature:.1f}-rep{self.repeat:02d}"
        )

    def request_seed(self, prompt_id: int, rollout: int, turn: int) -> int:
        # Shared slots receive the same seed across conditions and reruns, regardless of dispatch order.
        slot = f"{self.engine_seed}/{prompt_id}/{rollout}/{turn}".encode("ascii")
        return int.from_bytes(hashlib.blake2b(slot, digest_size=8).digest(), "big") >> 1

    def check_scored(self, prompt_ids, slots) -> None:
        expected = {(prompt, i) for prompt in prompt_ids for i in range(self.group_size)}
        if len(slots) != self.batch_size or set(slots) != expected:
            raise ValueError(f"{self.name}: scored slots differ from the requested prompt groups")


def captures():
    with Path(__file__).with_name("sweep.toml").open("rb") as file:
        spec = tomllib.load(file)
    conditions = [(n, spec["prompt_sweep_temperature"]) for n in spec["prompt_counts"]]
    for temperature in spec["temperatures"]:
        condition = (spec["temperature_sweep_prompts"], temperature)
        if condition not in conditions:
            conditions.append(condition)
    for checkpoint in spec["checkpoints"]:
        seeds = zip(spec["prompt_seeds"], spec["engine_seeds"], strict=True)
        for repeat, (prompt_seed, engine_seed) in enumerate(seeds, 1):
            for size in spec["batch_sizes"]:
                for n, temperature in conditions:
                    if size % n:
                        raise ValueError(f"Batch size {size} cannot be split across {n} prompts")
                    yield Capture(checkpoint, repeat, size, temperature, prompt_seed, engine_seed, n)


if __name__ == "__main__":
    for capture in captures():
        print(capture.name)
