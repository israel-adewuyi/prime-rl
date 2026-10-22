"""Inspect the provisioned ROCm runtime before installing PrimeRL."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sft", "rl"])
    parser.add_argument("output", type=Path, help="Directory for runtime constraints and version manifest")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("The ROCm profile requires Linux and Python 3.12.")
    if torch.version.hip is None:
        raise RuntimeError("Provision ROCm PyTorch first; a CUDA/CPU PyTorch wheel cannot run this profile.")
    if torch.__version__.split("+")[0] != "2.13.0":
        raise RuntimeError(f"This checkout requires PyTorch 2.13.0; found {torch.__version__}.")
    if args.mode == "rl":
        version = importlib.metadata.version("vllm")
        if version.split("+")[0] != "0.30.0":
            raise RuntimeError(f"This checkout's vLLM extensions target 0.30.0; found {version}.")
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            raise RuntimeError("vLLM must be built for ROCm, not CUDA.")
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    # Preserve the compiled GPU stack and vLLM's provisioned Transformers version.
    # Application packages must remain resolvable against PrimeRL's dependencies.
    protected_packages = {
        "torch", "torchvision", "torchaudio", "triton", "triton-rocm", "vllm",
        "amd-quark", "amdsmi", "aiter", "conch-triton-kernels", "tilelang",
        "apache-tvm-ffi", "compressed-tensors", "xgrammar",
        "mooncake-transfer-engine-rocm", "transformers",
    }
    (args.output / "runtime-constraints.txt").write_text(
        "".join(
            f"{name}=={version}\n"
            for name, version in sorted(versions.items())
            if name.lower().replace("_", "-") in protected_packages
        ), encoding="utf-8"
    )
    manifest = {"python": platform.python_version(), "hip": torch.version.hip, "packages": versions}
    (args.output / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"torch": torch.__version__, "hip": torch.version.hip, "mode": args.mode}))


if __name__ == "__main__":
    main()
