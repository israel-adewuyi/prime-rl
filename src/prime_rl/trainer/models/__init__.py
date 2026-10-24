"""PrimeRL configurations and lazily resolved model implementations."""

from collections import OrderedDict
from importlib import import_module

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.afmoe import AfmoeConfig
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.laguna import LagunaConfig
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextConfig, Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig

# Make custom config discoverable by AutoConfig
AutoConfig.register("afmoe", AfmoeConfig, exist_ok=True)
AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoConfig.register("glm4_moe", Glm4MoeConfig, exist_ok=True)
AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig, exist_ok=True)
AutoConfig.register("gpt_oss", GptOssConfig, exist_ok=True)
AutoConfig.register("laguna", LagunaConfig, exist_ok=True)
AutoConfig.register("minimax_m2", MiniMaxM2Config, exist_ok=True)
AutoConfig.register("nemotron_h", NemotronHConfig, exist_ok=True)
AutoConfig.register("qwen3_moe", Qwen3MoeConfig, exist_ok=True)
AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
AutoConfig.register("qwen3_5_moe", Qwen3_5MoeConfig, exist_ok=True)
AutoConfig.register("qwen3_5_moe_text", Qwen3_5MoeTextConfig, exist_ok=True)
AutoConfig.register("qwen3_5_vision", Qwen3_5VisionConfig, exist_ok=True)
AutoConfig.register("qwen4_exp", Qwen3_8FlashNextConfig, exist_ok=True)
AutoConfig.register("qwen4_exp_text", Qwen3_8FlashNextTextConfig, exist_ok=True)

_CUSTOM_CAUSAL_LM_MODELS: tuple[
    tuple[type[PretrainedConfig], str],
    ...,
] = (
    (Lfm2Config, "Lfm2ForCausalLM"),
    (LlamaConfig, "LlamaForCausalLM"),
    (Qwen3Config, "Qwen3ForCausalLM"),
    (AfmoeConfig, "AfmoeForCausalLM"),
    (DeepseekV4Config, "DeepseekV4ForCausalLM"),
    (Glm4MoeConfig, "Glm4MoeForCausalLM"),
    (GlmMoeDsaConfig, "GlmMoeDsaForCausalLM"),
    (LagunaConfig, "LagunaForCausalLM"),
    (MiniMaxM2Config, "MiniMaxM2ForCausalLM"),
    (NemotronHConfig, "NemotronHForCausalLM"),
    (Qwen3MoeConfig, "Qwen3MoeForCausalLM"),
    (Qwen3_5TextConfig, "Qwen3_5ForCausalLM"),
    (Qwen3_5MoeTextConfig, "Qwen3_5ForCausalLM"),
    (Qwen3_5Config, "Qwen3_5ForCausalLM"),
    (Qwen3_5MoeConfig, "Qwen3_5ForCausalLM"),
    (Qwen3_8FlashNextTextConfig, "Qwen3_8FlashNextForCausalLM"),
    (Qwen3_8FlashNextConfig, "Qwen3_8FlashNextForCausalLM"),
    (GptOssConfig, "GptOssForCausalLM"),
)
_MODEL_MODULES = {
    "AfmoeForCausalLM": "afmoe",
    "DeepseekV4ForCausalLM": "deepseek_v4",
    "Glm4MoeForCausalLM": "glm4_moe",
    "GlmMoeDsaForCausalLM": "glm_moe_dsa",
    "GptOssForCausalLM": "gpt_oss",
    "LagunaForCausalLM": "laguna",
    "Lfm2ForCausalLM": "lfm2",
    "LlamaForCausalLM": "llama",
    "MiniMaxM2ForCausalLM": "minimax_m2",
    "NemotronHForCausalLM": "nemotron_h",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3_5ForCausalLM": "qwen3_5",
    "Qwen3_8FlashNextForCausalLM": "qwen3_8_flash_next",
    "Qwen3MoeForCausalLM": "qwen3_moe",
}


def _load_model_class(name: str) -> type[PreTrainedModelPrimeRL]:
    module = import_module(f"prime_rl.trainer.models.{_MODEL_MODULES[name]}")
    return getattr(module, name)


class _PrimeLazyAutoMapping(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type, attr):
        for config, _ in _CUSTOM_CAUSAL_LM_MODELS:
            if config.model_type == model_type and config.__name__ == attr:
                return config
        return _load_model_class(attr)


_CUSTOM_CAUSAL_LM_MAPPING = _PrimeLazyAutoMapping(
    OrderedDict((config.model_type, config.__name__) for config, _ in _CUSTOM_CAUSAL_LM_MODELS),
    OrderedDict((config.model_type, name) for config, name in _CUSTOM_CAUSAL_LM_MODELS),
)
_CUSTOM_CAUSAL_LM_BY_MODEL_TYPE = {config.model_type: name for config, name in _CUSTOM_CAUSAL_LM_MODELS}


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def get_custom_causal_lm_cls(model_config: PretrainedConfig) -> type[PreTrainedModelPrimeRL]:
    """Resolve the PrimeRL model class from a possibly non-PrimeRL config instance."""
    return _load_model_class(_CUSTOM_CAUSAL_LM_BY_MODEL_TYPE[model_config.model_type])


def supports_custom_impl(model_config: PretrainedConfig) -> bool:
    """Check if the model configuration supports the custom PrimeRL implementation.

    Args:
        model_config: The model configuration to check.

    Returns:
        True if the model supports custom implementation, False otherwise.
    """
    return type(model_config) in _CUSTOM_CAUSAL_LM_MAPPING


# Mapping from HF composite VLM model_type to custom PrimeRL class.
# Used by get_model() to dispatch VLMs that have a custom text model implementation.
# Points to the same unified class — the config drives text-only vs VLM behavior.
_CUSTOM_VLM_MAPPING: dict[str, str] = {
    "qwen3_5": "Qwen3_5ForCausalLM",
    "qwen3_5_moe": "Qwen3_5ForCausalLM",
    "qwen4_exp": "Qwen3_8FlashNextForCausalLM",
}


def get_custom_vlm_cls(model_config: PretrainedConfig) -> type | None:
    """Return the custom PrimeRL VLM class for this config, or None if unsupported."""
    name = _CUSTOM_VLM_MAPPING.get(getattr(model_config, "model_type", None))
    return _load_model_class(name) if name is not None else None


__all__ = [
    "AutoModelForCausalLMPrimeRL",
    "PreTrainedModelPrimeRL",
    "get_custom_causal_lm_cls",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]


def __getattr__(name: str):
    if name in _MODEL_MODULES:
        value = _load_model_class(name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
