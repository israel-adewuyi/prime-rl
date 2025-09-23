from transformers import AutoModelForCausalLM
from typing import Tuple
from torch import Tensor

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
        # self.generate_stats()

    def _check_one_to_one_mapping():
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
        

    

def main():
    """Main entry-point for investigator. Run using `uv run investigator`"""
    config = parse_argv(InvestigatorConfig)
    WeightInvestigator(config)
    

if __name__ == "__main__":
    main()