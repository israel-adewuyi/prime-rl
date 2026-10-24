#!/usr/bin/env bash
set -euo pipefail

: "${VIRTUAL_ENV:?Activate the provisioned ROCm environment first}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

exec uv --no-config run --no-project --active --no-sync rl \
    @ examples/prompt-gradient-capture/rl.toml "$@"
