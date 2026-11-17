"""Wire a declared all-local retail evaluation and endpoint smoke from TOML."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import tomllib
import urllib.error
import urllib.request


def load_config(path):
    config = tomllib.loads(path.read_text())
    serving = tomllib.loads(path.with_name("synth.toml").read_text())
    for key in ["name", "revision"]:
        if config["model"][key] != serving["model"][key]:
            raise ValueError(f"Retail and serving model {key} differ")
    if config["model"]["alias"] not in serving["server"]["endpoint"]["served-model-name"]:
        raise ValueError("Retail model alias is not declared by the endpoint")
    return config


def cli_arguments(arguments):
    result = []
    for key, value in arguments.items():
        if isinstance(value, bool):
            if value:
                result.append("--" + key)
        elif isinstance(value, list):
            result.extend(["--" + key, *map(str, value)])
        else:
            result.extend(["--" + key, str(value)])
    return result


def tau_command(config, output):
    llm_arguments = dict(config["llm"], api_base=config["runtime"]["api_base"])
    judge = dict(config["judge"], model=config["roles"]["judge"], model_args=llm_arguments)
    return ["uv", "run", "--no-sync", "tau2", "run", *cli_arguments(config["evaluation"]),
        "--agent-llm", config["roles"]["agent"], "--agent-llm-args", json.dumps(llm_arguments),
        "--user-llm", config["roles"]["user"], "--user-llm-args", json.dumps(llm_arguments),
        "--communicate-judge", json.dumps(judge), "--save-to", str(output)]


def wait_for_endpoint(config, opener):
    runtime = config["runtime"]
    deadline = time.monotonic() + runtime["endpoint_wait_seconds"]
    print("Waiting for the configured retail endpoint", flush=True)
    while time.monotonic() < deadline:
        try:
            with opener.open(runtime["api_base"] + "/models", timeout=runtime["poll_seconds"]) as response:
                models = json.load(response)
            if any(model["id"] == config["model"]["alias"] for model in models["data"]):
                return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(runtime["poll_seconds"])
    raise TimeoutError("Retail endpoint did not become ready within its configured timeout")


def smoke_endpoint(config, output, opener):
    directory = output.parent / "endpoint-smoke" / str(time.time_ns())
    directory.mkdir(parents=True)
    common = dict(config["smoke"]["generation"], model=config["model"]["alias"])
    for name in ["chat", "tool"]:
        specification = config["smoke"][name]
        payload = dict(common, **{key: value for key, value in specification.items() if not key.startswith("expected_")})
        (directory / f"{name}-request.json").write_text(json.dumps(payload, indent=2))
        request = urllib.request.Request(config["runtime"]["api_base"] + "/chat/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=config["runtime"]["request_timeout_seconds"]) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            (directory / f"{name}-error.txt").write_bytes(error.read())
            raise
        (directory / f"{name}-response.json").write_text(json.dumps(result, indent=2))
        message = result["choices"][0]["message"]
        if name == "chat":
            if (message.get("content") or "").strip() != specification["expected_content"]:
                raise ValueError(f"Chat smoke did not match its configured expectation: {directory}")
        else:
            calls = message.get("tool_calls") or []
            if len(calls) != 1 or calls[0]["function"]["name"] != specification["expected_name"]:
                raise ValueError(f"Tool smoke did not select its configured function: {directory}")
            if json.loads(calls[0]["function"]["arguments"]) != specification["expected_arguments"]:
                raise ValueError(f"Tool smoke arguments differ from the configured expectation: {directory}")
        print(f"ENDPOINT_SMOKE_PASS {name}; evidence={directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("retail.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/shared/tau2-bench"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    if revision != config["framework"]["revision"]:
        raise ValueError("The installed tau-bench source differs from the configured revision")
    command = tau_command(config, args.output_dir.resolve())
    if args.dry_run:
        command.append("--dry-run")
    print(json.dumps(dict(config=config, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        framework_revision=revision, command=command), indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        wait_for_endpoint(config, opener)
        smoke_endpoint(config, args.output_dir.resolve(), opener)
    raise SystemExit(subprocess.call(command, cwd=args.repo, env=dict(os.environ, **config["environment"])))


if __name__ == "__main__":
    main()
