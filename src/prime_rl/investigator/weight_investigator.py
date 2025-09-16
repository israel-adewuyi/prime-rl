from transformers import AutoModelForCausalLM
from typing import Tuple
from torch import Tensor

from prime_rl.investigator.config import InvestigatorConfig
from prime_rl.utils.pydantic_config import parse_argv

class WeightInvestigator:
    """
        Class to retrieve the weights of two models. Ideally, the model is the same, 
        but differs along traininig progress. 
    """
    def __init__(self, config: None):
        self.config = config
        self.model_1 = AutoModelForCausalLM.from_pretrained(config.checkpoint_path_1)
        self.model_2 = AutoModelForCausalLM.from_pretrained(config.checkpoint_path_2)

        self.generate_stats()

    def generate_stats(self) -> None:
        tok_embeds = self.get_token_embed()
        diff_tensor = tok_embeds[0] - tok_embeds[1]
        print(f"Difference across checkpoints is {diff_tensor.mean()}")

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

    def get_attn_at_layer(self, layer: int) -> Tuple[Tensor, Tensor]:
        pass

    def get_mlp(self) -> Tuple[Tensor, Tensor]:
        pass

    def get_mlp_at_layer(self, layer: int) -> Tuple[Tensor, Tensor]:
        pass

def main():
    """Main entry-point for investigator. Run using `uv run investigator`"""
    import asyncio

    asyncio.run(WeightInvestigator(parse_argv(InvestigatorConfig)))


if __name__ == "__main__":
    main()