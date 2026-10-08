import sys

from transformers.utils import _LazyModule

_import_structure = {"modeling_llama": ["LlamaForCausalLM", "LlamaModel"]}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
