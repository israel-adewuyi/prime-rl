from transformers import AutoModelForCausalLM
from typing import Tuple
from torch import Tensor
from jaxtyping import Float, Int

from prime_rl.investigator.config import InvestigatorConfig
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.investigator.logger import setup_logger
from prime_rl.investigator.utils import plot_weight_diffs

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

        self._check_one_to_one_mapping()

        self.logger.info("Generating stats")
        for key in self.model_1.state_dict().keys():
            self.get_module_stats(self.model_1.state_dict()[key], self.model_2.state_dict()[key])
            break
        # self.generate_stats()

    def _check_one_to_one_mapping(self) -> None:
        """This function checks if both models have a 1-1 mappping. 
           This is achieved by checking if the set of modules in both models are the same
        """
        set_model_a = set(self.model_1.state_dict().keys())
        set_model_b = set(self.model_2.state_dict().keys())

        if set_model_a != set_model_b:
            missing_in_a = set_model_b - set_model_a
            missing_in_b = set_model_a - set_model_b
            raise ValueError(
                f"Model state_dict mismatch.\n"
                f"Missing in model_1: {missing_in_a}\n"
                f"Missing in model_2: {missing_in_b}"
            )
        else:
            self.logger.info(f"All modules align in both models")

    def get_module_stats(
        self,
        module_a: Float[Tensor, "..."],
        module_b: Float[Tensor, "..."],
        tolerance: float = 1e-5
    ) -> Tuple[Int, Int]:
        """Given a tranformer module tensor, return the total number of params as well as the number of non_zero diffs"""
        diff = module_a - module_b
        num_non_zero = (diff > tolerance).int().sum()

        tensor_shape = list(diff.shape)
        num_params = tensor_shape[0]
        for idx in range(1, len(tensor_shape)):
            num_params *= tensor_shape[idx]
        
        self.logger.debug(f"Number of params in module is {num_params}")
        self.logger.debug(f"Number of non_zero params in module is {num_non_zero}")

        return num_non_zero, num_params
        
        

    

def main():
    """Main entry-point for investigator. Run using `uv run investigator`"""
    config = parse_argv(InvestigatorConfig)
    WeightInvestigator(config)
    

if __name__ == "__main__":
    main()