from transformers import AutoModelForCausalLM
from jaxtyping import Tuple
from torch import Tensor


class WeightInvestigator:
    def __init__(self, config: None):
        self.config = config
        self.model_1 = AutoModelForCausalLM.from_pretrained(config.model_1)
        self.model_2 = AutoModelForCausalLM.from_pretrained(config.model_2)

    def get_token_embed(self) -> Tuple[Tensor, Tensor]:
        pass

    def get_token_umembed(self) -> Tuple[Tensor, Tensor]:
        pass

    def get_attn(self) -> Tuple[Tensor, Tensor]:
        pass

    def get_attn_at_layer(self, layer: int) -> Tuple[Tensor, Tensor]:
        pass

    def get_mlp(self) -> Tuple[Tensor, Tensor]:
        pass

    def get_mlp_at_layer(self, layer: int) -> Tuple[Tensor, Tensor]:
        pass