import sys

from transformers.utils import _LazyModule

_import_structure = {
    "configuration_qwen3_moe": ["Qwen3MoeConfig"],
    "modeling_qwen3_moe": ["Qwen3MoeForCausalLM", "Qwen3MoeModel", "Qwen3MoePreTrainedModel"],
}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
