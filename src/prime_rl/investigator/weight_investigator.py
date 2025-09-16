from transformers import AutoModelForCausalLM
from typing import Tuple
from torch import Tensor

from prime_rl.investigator.config import InvestigatorConfig
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.investigator.logger import setup_logger

import numpy as np

class WeightInvestigator:
    """
        Weights-based interp class.
        Loads two checkpoints of the same model and compare weights across both.
    """
    def __init__(self, config: None):
        self.config = config

        self.logger = setup_logger(config.log)
        self.logger.info("Starting investigator")

        self.logger.info(f"Loading first model at path {config.checkpoint_path_1}")
        self.model_1 = AutoModelForCausalLM.from_pretrained(config.checkpoint_path_1)
        self.logger.info(f"Loading second model at path {config.checkpoint_path_2}")
        self.model_2 = AutoModelForCausalLM.from_pretrained(config.checkpoint_path_2)

        self.logger.info("Generating stats")
        self.generate_stats()

    def generate_stats(self) -> None:
        tok_embeds = self.get_token_embed()
        diff_tensor = tok_embeds[0] - tok_embeds[1]
        self.logger.info(f"Difference in token embedding is {diff_tensor.abs().mean()}")

        for layer in range(self.model_1.config.num_hidden_layers):
            layer_diff = self.get_attn_diff_at_layer(layer)
            self.logger.info(f"Difference in attention at layer {layer} is {layer_diff}")

    def get_token_embed(self) -> Tuple[Tensor, Tensor]:
        # Get the token embedding layer 
        tok_embed_1 = self.model_1.model.embed_tokens.weight.data
        tok_embed_2 = self.model_2.model.embed_tokens.weight.data

        return (tok_embed_1, tok_embed_2)

    def get_token_umembed(self) -> Tuple[Tensor, Tensor]:
        # Get the token unembedding layer
        if model.config.tie_word_embeddings:
            tok_unembed_1 = self.model_1.model.embed_tokens.weight.data.T
            tok_unembed_2 = self.model_2.model.embed_tokens.weight.data.T
        else:
            tok_unembed_1 = self.model_1.model.unembed_tokens.weight.data
            tok_unembed_2 = self.model_2.model.unembed_tokens.weight.data

        return (tok_unembed_1, tok_unembed_2)

    def get_attn(self) -> Tuple[Tensor, Tensor]:
        pass

    def get_attn_diff_at_layer(self, layer: int) -> Tuple[Tensor, Tensor]:
        attn_1 = self.model_1.model.layers[layer].self_attn.state_dict()
        attn_2 = self.model_2.model.layers[layer].self_attn.state_dict()

        stats = []
        for key in attn_1:
            if key in attn_2:
                diff = (attn_1[key] - attn_2[key]).abs().mean()
                self.logger.debug(f"Data type of diff in get_attn_at_layer is {type(diff)}")
                stats.append(diff.item())

        return np.mean(np.array(stats))

    def get_mlp(self) -> Tuple[Tensor, Tensor]:
        pass

    def get_mlp_at_layer(self, layer: int) -> Tuple[Tensor, Tensor]:
        pass

def main():
    """Main entry-point for investigator. Run using `uv run investigator`"""
    config = parse_argv(InvestigatorConfig)
    WeightInvestigator(config)
    


if __name__ == "__main__":
    main()