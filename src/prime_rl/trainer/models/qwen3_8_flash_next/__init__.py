import sys

from transformers.utils import _LazyModule

_import_structure = {
    "attention": ["IndexedGatedAttention"],
    "configuration_qwen3_8_flash_next": ["Qwen3_8FlashNextConfig", "Qwen3_8FlashNextTextConfig"],
    "gated_delta_net": ["GatedDeltaNet"],
    "hyper_connection": ["ExpandedRMSNorm", "HyperConnection"],
    "modeling_qwen3_8_flash_next": [
        "Qwen3_8FlashNextDecoderLayer",
        "Qwen3_8FlashNextForCausalLM",
        "Qwen3_8FlashNextModel",
        "Qwen3_8FlashNextTextModel",
    ],
    "ngram_embedding": ["NGramEmbedding"],
    "position_learning": ["PositionLearningEnhancement"],
    "rotary_embedding": ["RotaryEmbedding"],
}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
