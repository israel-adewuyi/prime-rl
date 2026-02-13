import shutil
from pathlib import Path

import torch

from prime_rl.trainer.weights import gather_weights_on_master, save_state_dict
from prime_rl.trainer.world import get_world


def write_weights(model: torch.nn.Module, weight_dir: Path, save_format: str, save_sharded: bool) -> None:
    shutil.rmtree(weight_dir, ignore_errors=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    world = get_world()
    state = gather_weights_on_master(model, world.is_master)
    if world.is_master:
        save_state_dict(state, weight_dir, save_format=save_format, save_sharded=save_sharded)
