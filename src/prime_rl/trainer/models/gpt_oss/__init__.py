import sys

from transformers.utils import _LazyModule

_import_structure = {
    "configuration_gpt_oss": ["GptOssConfig"],
    "modeling_gpt_oss": ["GptOssForCausalLM", "GptOssModel", "GptOssPreTrainedModel"],
}

sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
