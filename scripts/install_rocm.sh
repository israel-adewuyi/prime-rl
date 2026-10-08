#!/usr/bin/env bash
set -euo pipefail

# Usage: activate a dedicated ROCm environment, then bash scripts/install_rocm.sh sft|rl.
mode="${1:?Usage: bash scripts/install_rocm.sh sft|rl}"
case "$mode" in sft|rl) ;; *) echo "Expected sft or rl" >&2; exit 2 ;; esac
: "${VIRTUAL_ENV:?Activate a dedicated environment containing ROCm PyTorch first}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
runtime_python="$VIRTUAL_ENV/bin/python"
manifest_dir="$VIRTUAL_ENV/prime-rl-runtime"

for package in verifiers renderers pydantic-config; do
    if [[ ! -f "deps/$package/pyproject.toml" ]]; then
        echo "Initialize submodules first: git submodule update --init --recursive" >&2
        exit 1
    fi
done

uv --no-config run --no-project --active --no-sync python tools/check_rocm_runtime.py "$mode" "$manifest_dir"
uv --no-config pip install --python "$runtime_python" \
    --constraint "$manifest_dir/runtime-constraints.txt" \
    -r environments/rocm/pyproject.toml \
    -e . -e packages/prime-rl-configs -e deps/pydantic-config \
    -e 'deps/verifiers[harbor]' -e 'deps/renderers[multimodal]' \
    pytest pytest-asyncio packaging ninja setuptools wheel

# FA2's Triton AMD backend avoids CUDA cubins and CK architecture-specific builds.
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE FLASH_ATTENTION_FORCE_BUILD=TRUE \
    uv --no-config pip install --python "$runtime_python" \
    --constraint "$manifest_dir/runtime-constraints.txt" \
    --no-build-isolation --no-binary flash-attn 'flash-attn==2.8.3'

if [[ "$mode" == rl ]]; then
    uv --no-config pip install --python "$runtime_python" \
        --constraint "$manifest_dir/runtime-constraints.txt" -e deps/verifiers/environments/reverse_text
fi
uv --no-config pip check --python "$runtime_python"
uv --no-config pip freeze --python "$runtime_python" > "$manifest_dir/installed.txt"
uv --no-config run --no-project --active --no-sync python -c \
    'from prime_rl.trainer.model import setup_model; from prime_rl.trainer.optim import setup_optimizer; from prime_rl.trainer.models.layers.attn import get_flash_attn_func; get_flash_attn_func(2); print("Qwen3 trainer imports and FA2 backend loaded")'
