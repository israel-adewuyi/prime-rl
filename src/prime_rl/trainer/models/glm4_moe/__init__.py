import sys

from transformers.utils import _LazyModule

_import_structure = {
    "configuration_glm4_moe": ["Glm4MoeConfig"],
    "modeling_glm4_moe": ["Glm4MoeForCausalLM", "Glm4MoeModel", "Glm4MoePreTrainedModel"],
}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
