import sys

from transformers.utils import _LazyModule

_import_structure = {
    "configuration_deepseek_v4": ["DeepseekV4Config"],
    "modeling_deepseek_v4": ["DeepseekV4ForCausalLM"],
}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
