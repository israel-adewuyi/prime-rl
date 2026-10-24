import logging.config
import os

from prime_rl.configs.inference import InferenceConfig


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    import torch

    if torch.version.hip is not None:
        if config.use_deep_gemm:
            raise ValueError("DeepGEMM is not supported by the ROCm Qwen3 profile.")
        if config.vllm.enable_lora:
            raise ValueError("The ROCm Qwen3 profile supports full-parameter training only.")
        if config.weight_broadcast.type != "filesystem":
            raise ValueError("The ROCm Qwen3 profile requires filesystem weight broadcasting.")
        if config.deployment.type == "disaggregated" or config.kv_cache_offload is not None:
            raise ValueError("The ROCm Qwen3 profile does not support disaggregated inference or KV-cache offload.")
        if config.vllm.quantization is not None:
            raise ValueError("The ROCm Qwen3 profile requires unquantized inference weights.")
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(config.vllm.model, trust_remote_code=config.vllm.trust_remote_code)
        if model_config.model_type not in {"qwen3", "lfm2"}:
            raise ValueError("The ROCm inference profile currently supports dense Qwen3 and LFM2 only.")

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Standard deployments use V2 for both capture modes. NIXL P/D also uses V2
    # for sampling-mask capture, but router replay remains on V1 because vLLM
    # rejects routed-expert capture with KV connectors and prime-rl's stitching
    # patch is V1-only. setdefault keeps an explicit env-var choice authoritative.
    if config.enable_return_sampling_mask:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    elif config.vllm.enable_return_routed_experts:
        use_v2_runner = config.deployment.type != "disaggregated"
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1" if use_v2_runner else "0")

    # vLLM 0.24.0 flipped VLLM_ENFORCE_STRICT_TOOL_CALLING's default to True, which
    # grammar-constrains generation (xgrammar structural tags) for tool_choice
    # "required"/named and strict tools — a sampling distribution the trainer never
    # sees. Keep it off so rollout logprobs stay faithful for importance ratios.
    os.environ.setdefault("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")

    deep_gemm_enabled = "1" if config.use_deep_gemm else "0"
    os.environ["VLLM_USE_DEEP_GEMM"] = deep_gemm_enabled
    os.environ["VLLM_MOE_USE_DEEP_GEMM"] = deep_gemm_enabled

    if config.vllm.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    if config.log.json_logging:
        # Route vLLM's stdlib loggers through a JSON formatter matching
        # trainer / orchestrator. The env var (not in-process dictConfig)
        # is what reaches vLLM's spawned workers.
        from prime_rl.inference.json_logging import build_dict_config, write_logging_config

        config_path = write_logging_config(config.log.level)
        # vLLM raises if VLLM_LOGGING_CONFIG_PATH is set while
        # VLLM_CONFIGURE_LOGGING=0 (its supported way to disable logger
        # setup). Force it on — opting into JSON logging is an explicit
        # request to configure vLLM's logger.
        os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
        os.environ["VLLM_LOGGING_CONFIG_PATH"] = str(config_path)
        logging.config.dictConfig(build_dict_config(config.log.level))
