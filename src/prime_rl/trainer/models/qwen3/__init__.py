import sys

from transformers.utils import _LazyModule

_import_structure = {"modeling_qwen3": ["Qwen3ForCausalLM", "Qwen3Model", "Qwen3PreTrainedModel"]}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
