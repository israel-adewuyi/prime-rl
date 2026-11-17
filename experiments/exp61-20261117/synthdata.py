"""Wire the declared endpoint and benchmark commands from synth.toml."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import tomllib
import urllib.error
import urllib.request

PREFIX = ["uv", "--no-config", "run", "--no-project", "--active", "vllm"]


def cli_arguments(arguments, negative_booleans=False):
    result = []
    for key, value in arguments.items():
        if isinstance(value, bool):
            if value:
                result.append("--" + key)
            elif negative_booleans:
                result.append("--no-" + key)
        elif isinstance(value, list):
            result.extend(["--" + key, *map(str, value)])
        else:
            result.extend(["--" + key, json.dumps(value) if isinstance(value, dict) else str(value)])
    return result


def load_config(path):
    config = tomllib.loads(path.read_text())
    devices = config["runtime"]["gpus"].split(",")
    if len(set(devices)) != len(devices) or not all(device.isdigit() for device in devices):
        raise ValueError("runtime.gpus must contain distinct GPU ordinals")
    for layout in config["layouts"].values():
        if layout["tensor-parallel-size"] * layout["data-parallel-size"] != len(devices):
            raise ValueError("Each declared layout must use the declared GPU count")
    return config


def serve_command(config, layout, phase):
    arguments = {**config["server"]["base"], **config["server"][phase], **config["layouts"][layout]}
    model = config["model"]
    return PREFIX + ["serve", model["name"], "--revision", model["revision"],
        "--tokenizer-revision", model["revision"], "--port", str(config["runtime"]["port"]),
        *cli_arguments(arguments, negative_booleans=True)]


def benchmark_command(config, layout, concurrency, output):
    model = config["model"]
    cache = Path(os.environ.get("HF_HOME", "/mnt/shared/.cache/huggingface")) / "hub"
    tokenizer = cache / ("models--" + model["name"].replace("/", "--")) / "snapshots" / model["revision"]
    host = config["server"]["base"]["host"]
    base_url = f"http://{host}:{config['runtime']['port']}"
    return PREFIX + ["bench", "serve", "--base-url", base_url,
        "--model", model["name"], "--tokenizer", str(tokenizer),
        *cli_arguments(config["benchmark"]["arguments"]),
        "--max-concurrency", str(concurrency), "--result-dir", str(output),
        "--result-filename", f"{layout}-c{concurrency}.json"]


def runtime_environment(config):
    devices = config["runtime"]["gpus"]
    return dict(os.environ, **config["environment"],
        HIP_VISIBLE_DEVICES=devices, CUDA_VISIBLE_DEVICES=devices,
        HF_HOME=os.environ.get("HF_HOME", "/mnt/shared/.cache/huggingface"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "serve", "benchmark"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("synth.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output_dir.resolve()
    commands = {layout: dict(serve=serve_command(config, layout, "benchmark"), benchmarks=[
        benchmark_command(config, layout, concurrency, output / layout)
        for concurrency in config["benchmark"]["concurrency"]])
        for layout in config["benchmark"]["layouts"]}
    endpoint = serve_command(config, config["endpoint"]["layout"], "endpoint")
    record = dict(config=config, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        endpoint_command=endpoint, benchmark_commands=commands)
    print(json.dumps(record, indent=2), flush=True)
    if args.action == "plan":
        return
    environment = runtime_environment(config)
    if args.action == "serve":
        print(shlex.join(endpoint), flush=True)
        raise SystemExit(subprocess.call(endpoint, env=environment))
    output.mkdir(parents=True, exist_ok=False)
    (output / "commands.json").write_text(json.dumps(record, indent=2))
    runtime = config["runtime"]
    health_url = f"http://{config['server']['base']['host']}:{runtime['port']}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for layout, row in commands.items():
        directory = output / layout
        directory.mkdir()
        print(f"Starting {layout}; raw results and engine log: {directory}", flush=True)
        print(shlex.join(row["serve"]), flush=True)
        with (directory / "server.log").open("w") as log:
            server = subprocess.Popen(row["serve"], env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + runtime["startup_timeout_seconds"]
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"{layout} server exited {server.returncode}; see {directory / 'server.log'}")
                    try:
                        with opener.open(health_url, timeout=runtime["poll_seconds"]) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{layout} startup exceeded its configured timeout")
                    time.sleep(runtime["poll_seconds"])
                for command in row["benchmarks"]:
                    print(shlex.join(command), flush=True)
                    subprocess.run(command, env=environment, check=True)
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=runtime["shutdown_grace_seconds"])
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
    print(f"All declared benchmarks completed: {output}", flush=True)


if __name__ == "__main__":
    main()
