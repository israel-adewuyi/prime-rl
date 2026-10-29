"""Packed Qwen2/Qwen2.5 training using the shared dense transformer layers."""

from copy import copy

from transformers import Qwen2Config

from prime_rl.trainer.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel


class Qwen2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        # Qwen2 has Q/K/V biases, no output/MLP biases and no Q/K normalization.
        layer_config = copy(config)
        layer_config.attention_bias = True
        layer_config.mlp_bias = False
        super().__init__(layer_config, layer_idx)
        layer_types = getattr(config, "layer_types", None)
        sliding = (
            layer_types[layer_idx] == "sliding_attention"
            if layer_types is not None
            else config.use_sliding_window and layer_idx >= config.max_window_layers
        )
        self.self_attn.sliding_window = config.sliding_window if sliding else None


class Qwen2Model(LlamaModel):
    config_class = Qwen2Config
    decoder_layer_cls = Qwen2DecoderLayer
    _no_split_modules = ["Qwen2DecoderLayer"]
    _can_record_outputs = {"hidden_states": Qwen2DecoderLayer}

    def __init__(self, config: Qwen2Config):
        if config.attention_dropout != 0:
            raise ValueError("The packed Qwen2 implementation requires attention_dropout=0.")
        super().__init__(config)


class Qwen2ForCausalLM(LlamaForCausalLM):
    config_class = Qwen2Config
    model_cls = Qwen2Model
    _no_split_modules = ["Qwen2DecoderLayer"]
    _can_record_outputs = {"hidden_states": Qwen2DecoderLayer}
