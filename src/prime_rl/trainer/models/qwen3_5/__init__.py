import sys

from transformers.utils import _LazyModule

_import_structure = {
    "configuration_qwen3_5": [
        "Qwen3_5Config",
        "Qwen3_5MoeConfig",
        "Qwen3_5MoeTextConfig",
        "Qwen3_5TextConfig",
        "Qwen3_5VisionConfig",
    ],
    "modeling_qwen3_5": ["Qwen3_5ForCausalLM", "Qwen3_5Model", "Qwen3_5PreTrainedModel"],
}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
