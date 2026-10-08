import sys

from transformers.utils import _LazyModule

_import_structure = {
    "configuration_minimax_m2": ["MiniMaxM2Config"],
    "modeling_minimax_m2": ["MiniMaxM2ForCausalLM", "MiniMaxM2Model", "MiniMaxM2PreTrainedModel"],
}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
