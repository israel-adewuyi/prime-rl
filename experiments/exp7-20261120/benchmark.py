"""Execute the accumulation benchmark grid declared in benchmark.toml."""

import argparse
import json
import shlex
import subprocess
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    config = tomllib.loads((here / "benchmark.toml").read_text())
    commands = []
    for parameters in config["parameters"]:
        for rollouts in config["rollouts"]:
            path = args.output_dir.resolve() / f"rollouts-{rollouts}-parameters-{parameters}.json"
            command = ["uv", "--no-config", "run", "--no-project", "--active", "python", "benchmarks/snr/run.py", str(path), "--parameters", str(parameters), "--rollouts", str(rollouts)]
            for key in ("tensor_numel", "pool_size", "dtype", "seed", "threads", "warmup", "repeats"):
                command.extend(["--" + key.replace("_", "-"), str(config[key])])
            command.extend(["--implementations", *config["implementations"]])
            commands.append(command)
    print(json.dumps({"source_toml": str(here / "benchmark.toml"), "commands": commands}), flush=True)
    if args.plan:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "commands.json").write_text(json.dumps(commands, indent=2))
    for command in commands:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=repo, check=True)
        print(f"Verified benchmark result: {command[7]}", flush=True)


if __name__ == "__main__":
    main()
