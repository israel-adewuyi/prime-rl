#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import tomllib  # Python 3.11+
from dataclasses import dataclass
from pathlib import Path

import tomli_w  # pip install tomli-w


@dataclass(frozen=True)
class RunSpec:
    model_name: str
    s: int  # the value that replaces s=100 in results/rollouts filenames


def deep_set(d: dict, keys: list[str], value):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def patch_s_token(filename: str, new_s: int) -> str:
    """
    Replace the first occurrence of s=<digits> with s=<new_s>.
    Expects filenames like: "..._s=100_..." or "..._s=100."
    """
    if "s=" not in filename:
        raise ValueError(f"Expected 's=' token in filename: {filename}")
    out, n = re.subn(r"s=\d+", f"s={new_s}", filename, count=1)
    if n != 1:
        raise ValueError(f"Failed to patch exactly one 's=...' token in: {filename}")
    return out


def main():
    base_cfg_path = Path("configs/landscape/alpha_sort.toml")
    gen_dir = Path("configs/landscape/_generated")
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Fill this with the checkpoints you want to sweep over.
    runs: list[RunSpec] = [
        RunSpec(model_name="israel-adewuyi/Qwen2.5-0.5B-Instruct-AlphabetSort-RL", s=100),
        RunSpec(model_name="israel-adewuyi/Qwen2.5-0.5B-Instruct-AlphabetSort-RL-step_50", s=50),
        RunSpec(model_name="israel-adewuyi/Qwen2.5-0.5B-Instruct-AlphabetSort-RL-step_150", s=150),
        RunSpec(model_name="Qwen/Qwen2.5-0.5B-Instruct", s=0),
    ]

    base_cfg = tomllib.loads(base_cfg_path.read_text())

    # Grab the original filenames once (so we always patch from the base template)
    base_results = base_cfg["sweep"]["results_file"]
    base_rollouts = base_cfg["sweep"]["rollouts_file"]

    for spec in runs:
        cfg = base_cfg.copy()

        # Update model names
        deep_set(cfg, ["trainer", "model", "name"], spec.model_name)
        deep_set(cfg, ["orchestrator", "model", "name"], spec.model_name)

        # Patch ONLY s=... in filenames
        cfg["sweep"]["results_file"] = patch_s_token(base_results, spec.s)
        cfg["sweep"]["rollouts_file"] = patch_s_token(base_rollouts, spec.s)

        out_cfg_path = gen_dir / f"AS_s={spec.s}.toml"
        out_cfg_path.write_text(tomli_w.dumps(cfg))

        # Your exact launch command:
        cmd = [
            "uv",
            "run",
            "landscape",
            "@",
            str(out_cfg_path),
            "--inference-gpu-ids",
            "1",
            "--log.level",
            "debug",
        ]

        print(f"\n=== RUN s={spec.s} model={spec.model_name} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, env=os.environ.copy())

    print("\nDone.")


if __name__ == "__main__":
    main()
