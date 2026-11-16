"""Plan or run Qwen3.8-27B text-serving comparisons on GPUs 4,5,6,7."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request

MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '37561816e66b5b9153b1b64c10968c18c1e6f4a0'
LAYOUTS = {'tp4': (4, 1), 'tp2-dp2': (2, 2), 'dp4': (1, 4)}
PREFIX = ['uv', '--no-config', 'run', '--no-project', '--active', 'vllm']


def serve_command(layout, port):
    tp, dp = LAYOUTS[layout]
    return PREFIX + ['serve', MODEL, '--revision', REVISION, '--tokenizer-revision', REVISION,
        '--served-model-name', 'qwen38-27b', '--host', '127.0.0.1', '--port', str(port),
        '--dtype', 'bfloat16', '--tensor-parallel-size', str(tp), '--data-parallel-size', str(dp),
        '--distributed-executor-backend', 'mp', '--max-model-len', '8192',
        '--max-num-seqs', '64', '--gpu-memory-utilization', '0.85', '--seed', '42',
        '--language-model-only']


def benchmark_command(layout, concurrency, output, port):
    tokenizer = Path(os.environ.get('HF_HOME', '/mnt/shared/.cache/huggingface')) / 'hub' / 'models--Qwen--Qwen3.8-27B' / 'snapshots' / REVISION
    return PREFIX + ['bench', 'serve', '--backend', 'openai-chat', '--base-url',
        f'http://127.0.0.1:{port}', '--endpoint', '/v1/chat/completions', '--model', MODEL,
        '--served-model-name', 'qwen38-27b', '--tokenizer', str(tokenizer), '--dataset-name', 'random',
        '--random-input-len', '512', '--random-output-len', '128', '--num-prompts', '128',
        '--max-concurrency', str(concurrency), '--request-rate', 'inf', '--ignore-eos',
        '--seed', '42', '--extra-body', json.dumps({'chat_template_kwargs': {'enable_thinking': False}}),
        '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,90,99',
        '--save-result', '--save-detailed', '--result-dir', str(output),
        '--result-filename', f'{layout}-c{concurrency}.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'benchmark', 'serve'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layout', choices=LAYOUTS, default='dp4')
    parser.add_argument('--port', type=int, default=8300)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    commands = {layout: dict(serve=serve_command(layout, args.port), benchmarks=[
        benchmark_command(layout, concurrency, output / layout, args.port)
        for concurrency in [1, 16, 64]]) for layout in LAYOUTS}
    print(json.dumps(dict(model=MODEL, revision=REVISION, physical_gpus='4,5,6,7', commands=commands), indent=2), flush=True)
    if args.action == 'plan':
        return
    env = dict(os.environ, HIP_VISIBLE_DEVICES='4,5,6,7', CUDA_VISIBLE_DEVICES='4,5,6,7',
        HF_HOME='/mnt/shared/.cache/huggingface', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    if args.action == 'serve':
        print(shlex.join(commands[args.layout]['serve']), flush=True)
        raise SystemExit(subprocess.call(commands[args.layout]['serve'], env=env))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'commands.json').write_text(json.dumps(commands, indent=2))
    for layout, record in commands.items():
        directory = output / layout
        directory.mkdir()
        print(f'Serving {layout}; live log and benchmark JSON: {directory}', flush=True)
        with (directory / 'server.log').open('w') as log:
            server = subprocess.Popen(record['serve'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 900
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f'{layout} server exited {server.returncode}; read {directory / "server.log"}')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'{layout} server startup exceeded 900 seconds')
                    time.sleep(2)
                for command in record['benchmarks']:
                    subprocess.run(command, env=env, check=True)
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
    print(f'Benchmark results: {output}. Start the selected final layout with the serve action.', flush=True)


if __name__ == '__main__':
    main()
