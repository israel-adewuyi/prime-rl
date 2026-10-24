"""LFM2 training adapter with independent convolution state for packed sequences."""

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2Model, Lfm2RotaryEmbedding

from prime_rl.trainer.models.base import CPSupport, PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.utils.cp import CPContext
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class Lfm2TrainingShortConv(nn.Module):
    """Stateless PyTorch convolution using the native LFM2 checkpoint names."""

    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_L_cache,
            groups=config.hidden_size,
            padding=config.conv_L_cache - 1,
            bias=config.conv_bias,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.conv_bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.conv_bias)

    def forward(
        self,
        hidden_states: Tensor,
        past_key_values=None,
        attention_mask: Tensor | None = None,
        seq_idx: Tensor | None = None,
    ) -> Tensor:
        if past_key_values is not None or seq_idx is not None:
            raise ValueError("LFM2 training convolution requires separate uncached sequences")
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
        gate_in, gate_out, values = self.in_proj(hidden_states).transpose(1, 2).chunk(3, dim=1)
        convolved = self.conv(gate_in * values)[..., : hidden_states.shape[1]]
        return self.out_proj((gate_out * convolved).transpose(1, 2).contiguous())


class Lfm2TrainingModel(Lfm2Model):
    def __init__(self, config: Lfm2Config):
        super().__init__(config)
        self.cp_context = CPContext()
        for layer in self.layers:
            if not layer.is_attention_layer:
                layer.conv = Lfm2TrainingShortConv(config)

    def forward(
        self,
        input_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        if seq_lens_are_pre_shard or self.cp_context.cp_enabled:
            raise ValueError("LFM2 training requires cp=1")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        reference = input_ids if input_ids is not None else inputs_embeds
        if reference.shape[0] != 1:
            raise ValueError("LFM2 training expects one packed row")
        get_cu_seqlens_from_seq_lens(seq_lens, total_tokens=reference.shape[1])
        hidden_states = []
        offset = 0
        # Reset both attention and short-convolution history at every document boundary.
        for length in seq_lens.tolist():
            end = offset + length
            outputs = super().forward(
                input_ids=input_ids[:, offset:end] if input_ids is not None else None,
                inputs_embeds=inputs_embeds[:, offset:end] if inputs_embeds is not None else None,
                position_ids=position_ids[:, offset:end] if position_ids is not None else None,
                use_cache=False,
                return_dict=True,
            )
            hidden_states.append(outputs.last_hidden_state)
            offset = end
        return BaseModelOutputWithPast(last_hidden_state=torch.cat(hidden_states, dim=1))


class Lfm2ForCausalLM(PreTrainedModelPrimeRL):
    config_class = Lfm2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Lfm2DecoderLayer"]
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Lfm2Config):
        super().__init__(config)
        config.use_cache = False
        self.model = Lfm2TrainingModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def cp_support(cls, config) -> CPSupport:
        return CPSupport(frozenset(), "LFM2 training evaluates each packed sequence independently")

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return False

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> PrimeLmOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        return PrimeLmOutput(logits=self.lm_head(outputs.last_hidden_state))

    def init_buffers_post_meta(self) -> None:
        rotary = self.model.rotary_emb
        with rotary.inv_freq.device:
            initialized = Lfm2RotaryEmbedding(self.config)
        for name, buffer in rotary.named_buffers():
            buffer.copy_(initialized.get_buffer(name))
        rotary.attention_scaling = initialized.attention_scaling
