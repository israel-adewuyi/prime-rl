from transformers import AutoModelForCausalLM
from typing import Tuple
from torch import Tensor
from jaxtyping import Float, Int

from prime_rl.investigator.config import InvestigatorConfig
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.logger import setup_logger
from prime_rl.investigator.utils import visualize_sparsity, tensor_to_serializable, get_name_of_run

import os
import json
import torch
import numpy as np

class WeightInvestigator:
    """
        Weights-based interp class.
        Loads two checkpoints of the same model and compare weights across both.
    """
    def __init__(self, config: None):
        self.config = config

        # self.logger = setup_logger(config.log)
        self.logger = setup_logger(
            config.log.level, log_file=config.output_dir / "logs" / "orchestrator.log" if config.log.file else None
        )
        self.logger.info("Starting investigator")

        self.logger.info(f"Loading first model at path {config.checkpoint_path_1}")
        self.model_1 = AutoModelForCausalLM.from_pretrained(config.checkpoint_path_1).to("cuda")
        self.logger.info(f"Loading second model at path {config.checkpoint_path_2}")
        self.model_2 = AutoModelForCausalLM.from_pretrained(config.checkpoint_path_2).to("cuda")

        self._check_one_to_one_mapping()

        self.name = get_name_of_run(config.checkpoint_path_1, config.checkpoint_path_2)

        self.logger.info(f"Generating stats for run {self.name} with configs {self.config}")

        update_sparsity = self.calculate_update_sparsity() 
        self.logger.debug(f"Update sparsity across the entire model is {update_sparsity}") 
        update_sparsity_dict = self.calculate_update_sparsity_across_layers() 
        self.logger.debug(f"Update sparsity layer_wise = {update_sparsity_dict}")
        update_sparsity_dict_across_submodules = self.calculate_update_sparsity_across_submodules()
        self.logger.debug(f"Update sparsity across submodules = {update_sparsity_dict_across_submodules}")

        if config.merge:
            self.logger.info("Merging and saving dicts")
            self.merge_and_save_dicts(self.name, update_sparsity, update_sparsity_dict, update_sparsity_dict_across_submodules)
        else:
            self.logger.info("Skipping the merge process.")

        if config.generate_charts:
            self.logger.info("Generating charts")
            visualize_sparsity()
        else:
            self.logger.info("Skipping the generation of charts")

    def merge_and_save_dicts(
        self,
        models_being_compared: str,
        update_sparsity: float,
        update_sparsity_dict: dict,
        update_sparsity_dict_across_submodules: dict,
        path: str = "test.json"
    ) -> None:
        # load existing file if available
        if os.path.exists(path):
            with open(path, "r") as file:
                all_data = json.load(file)
        else:
            all_data = {}
    
        # add current checkpoint
        all_data[models_being_compared] = {
            "global": update_sparsity,
            "layers": update_sparsity_dict,
            "submodules": update_sparsity_dict_across_submodules,
        }
    
        with open(path, "w") as file:
            serializable_data = tensor_to_serializable(all_data)
            json.dump(serializable_data, file, indent=4)

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

    def calculate_update_sparsity(self) -> Float:
        total_num_non_zero, total_num_params = 0, 0
        for key in self.model_1.state_dict().keys():
            num_non_zero, num_params = self.get_module_stats(self.model_1.state_dict()[key], self.model_2.state_dict()[key])
            total_num_non_zero += num_non_zero
            total_num_params += num_params

        return 1 - (total_num_non_zero / total_num_params)

    def calculate_update_sparsity_across_layers(self) -> dict: 
        """ Calculate update sparsity for each layer (grouped by prefix in state_dict). 
        Returns a dict {layer_name: sparsity_value}. 
        """ 
        layer_stats = {} 
        for key in self.model_1.state_dict().keys(): 
            module_a = self.model_1.state_dict()[key] 
            module_b = self.model_2.state_dict()[key] 
            if "layers.17" in key: 
                self.logger.debug(key) 
            num_non_zero, num_params = self.get_module_stats(module_a, module_b) 
            parts = key.split(".") 
            if "layers" in parts: 
                layer_idx = parts.index("layers") 
                layer_name = ".".join(parts[: layer_idx + 2]) 
            else:  
                layer_name = key 
                
            if layer_name not in layer_stats: 
                layer_stats[layer_name] = {"non_zero": 0, "params": 0} 
                layer_stats[layer_name]["non_zero"] += num_non_zero 
                layer_stats[layer_name]["params"] += num_params 
                        
        # compute update sparsity per layer 
        sparsity_per_layer = {} 
        for layer, stats in layer_stats.items(): 
            sparsity = 1 - (stats["non_zero"] / stats["params"])
            sparsity_per_layer[layer] = sparsity 
            self.logger.debug(f"Layer {layer}: Update sparsity = {sparsity:.4f}") 
        return sparsity_per_layer

    def calculate_update_sparsity_across_submodules(self, tolerance: float = 1e-5) -> dict:
        """
        Calculate update sparsity for each parameter tensor (weight/bias) inside each submodule.
        Returns a dict:
        {
            "model.layers.17.self_attn.q_proj.weight": sparsity_value,
            "model.layers.17.self_attn.q_proj.bias": sparsity_value,
            ...
        }
        """
        submodule_stats = {}

        for key in self.model_1.state_dict().keys():
            module_a = self.model_1.state_dict()[key]
            module_b = self.model_2.state_dict()[key]
            num_non_zero, num_params = self.get_module_stats(module_a, module_b)
            if key not in submodule_stats:
                submodule_stats[key] = {"non_zero": 0, "params": 0}
            submodule_stats[key]["non_zero"] += num_non_zero
            submodule_stats[key]["params"] += num_params

        # compute sparsity per submodule (weight/bias included)
        sparsity_per_submodule = {}
        for key, stats in submodule_stats.items():
            sparsity = 1 - (stats["non_zero"] / stats["params"])
            sparsity_per_submodule[key] = sparsity
            self.logger.debug(f"{key}: Update sparsity = {sparsity:.4f}")

        return sparsity_per_submodule

def main():
    """Main entry-point for investigator. Run using `uv run investigator`"""
    config = parse_argv(InvestigatorConfig)
    WeightInvestigator(config)
    

if __name__ == "__main__":
    main()