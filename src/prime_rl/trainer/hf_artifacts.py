import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch
from huggingface_hub import HfApi
from torch import Tensor, nn
from transformers import PreTrainedTokenizer

from prime_rl.trainer.rl.config import HFArtifactsConfig
from prime_rl.trainer.weights import (
    convert_tt_to_hf_moe,
    gather_weights_on_master,
    has_tt_moe_layers,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

MAX_UPLOAD_RETRIES = 3


class HFArtifactsManager:
    def __init__(self, config: HFArtifactsConfig):
        self.config = config
        self.api = HfApi()
        self.logger = get_logger()
        self.world = get_world()
        if self.world.is_master:
            self.api.create_repo(repo_id=config.repo_id, repo_type="model", exist_ok=True)

    def should_save(self, step: int) -> bool:
        return step % self.config.interval == 0

    def _upload_folder(self, step_path: str, artifact: str, folder_path: Path) -> bool:
        for attempt in range(MAX_UPLOAD_RETRIES + 1):
            try:
                self.api.upload_folder(
                    repo_id=self.config.repo_id,
                    repo_type="model",
                    folder_path=str(folder_path),
                    path_in_repo=f"{step_path}/{artifact}",
                    commit_message=f"Upload {self.config.algo} {artifact} at step {step_path.split('_')[-1]}",
                )
                return True
            except Exception as e:
                if attempt == MAX_UPLOAD_RETRIES:
                    self.logger.error(f"Failed to upload HF artifact '{artifact}' to {step_path}: {e}")
                    return False
                delay = 2**attempt
                self.logger.warning(
                    f"HF upload failed for '{artifact}' to {step_path} (attempt {attempt + 1}/{MAX_UPLOAD_RETRIES + 1}): {e}. Retrying in {delay}s"
                )
                time.sleep(delay)

    def _upload_metadata(self, step_path: str, metadata_path: Path, step: int) -> bool:
        for attempt in range(MAX_UPLOAD_RETRIES + 1):
            try:
                self.api.upload_file(
                    repo_id=self.config.repo_id,
                    repo_type="model",
                    path_or_fileobj=str(metadata_path),
                    path_in_repo=f"{step_path}/metadata.json",
                    commit_message=f"Upload {self.config.algo} metadata at step {step}",
                )
                return True
            except Exception as e:
                if attempt == MAX_UPLOAD_RETRIES:
                    self.logger.error(f"Failed to upload HF metadata to {step_path}: {e}")
                    return False
                delay = 2**attempt
                self.logger.warning(
                    f"HF metadata upload failed to {step_path} (attempt {attempt + 1}/{MAX_UPLOAD_RETRIES + 1}): {e}. Retrying in {delay}s"
                )
                time.sleep(delay)

    def save(
        self,
        step: int,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        pre_weights: dict[str, Tensor],
        grads: dict[str, Tensor],
    ) -> None:
        post_weights = gather_weights_on_master(model, self.world.is_master, dtype=torch.bfloat16)
        if not self.world.is_master:
            return
        delta = {key: post_weights[key] - pre_weights[key] for key in pre_weights}
        weights_to_save = dict(post_weights)
        if has_tt_moe_layers(weights_to_save):
            convert_tt_to_hf_moe(weights_to_save)
        with TemporaryDirectory() as tmp_dir:
            step_dir = Path(tmp_dir)
            step_path = f"{self.config.algo}_steps/step_{step}"
            weights_dir = step_dir / "weights"
            grads_dir = step_dir / "grads"
            delta_dir = step_dir / "delta"
            save_state_dict(weights_to_save, weights_dir)
            save_state_dict(dict(grads), grads_dir)
            save_state_dict(delta, delta_dir)
            model.config.save_pretrained(weights_dir)
            if model.generation_config:
                model.generation_config.save_pretrained(weights_dir)
            tokenizer.save_pretrained(weights_dir)
            self.logger.info(f"Uploading HF artifacts at step {step}")
            uploaded = []
            failed = []
            for artifact, folder in (("weights", weights_dir), ("grads", grads_dir), ("delta", delta_dir)):
                if self._upload_folder(step_path, artifact, folder):
                    uploaded.append(artifact)
                else:
                    failed.append(artifact)
            metadata_path = step_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps({"step": step, "algo": self.config.algo, "uploaded": uploaded, "failed": failed}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            self._upload_metadata(step_path, metadata_path, step)
