import sys

from transformers.utils import _LazyModule

_import_structure = {
    "configuration_nemotron_h": ["NemotronHConfig"],
    "modeling_nemotron_h": ["NemotronHForCausalLM", "NemotronHModel", "NemotronHPreTrainedModel"],
}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
