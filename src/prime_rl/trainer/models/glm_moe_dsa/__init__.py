import sys

from transformers.utils import _LazyModule

_import_structure = {
    "configuration_glm_moe_dsa": ["GlmMoeDsaConfig"],
    "modeling_glm_moe_dsa": ["GlmMoeDsaForCausalLM", "GlmMoeDsaModel", "GlmMoeDsaPreTrainedModel"],
}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
