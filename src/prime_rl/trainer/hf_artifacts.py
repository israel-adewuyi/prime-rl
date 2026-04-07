import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from torch import Tensor, nn

from prime_rl.configs.trainer import HFArtifactsConfig
from prime_rl.trainer.weights import gather_weights_on_master, save_state_dict
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class HFArtifactsManager:
    def __init__(self, config: HFArtifactsConfig):
        self.api = HfApi()
        self.config = config
        self.logger = get_logger()
        self.world = get_world()
        if self.world.is_master:
            self.api.create_repo(repo_id=config.repo_id, repo_type="model", exist_ok=True)

    def save(self, step: int, model: nn.Module, pre_weights: dict[str, Tensor], grads: dict[str, Tensor]) -> None:
        post_weights = gather_weights_on_master(model, self.world.is_master, dtype=None)
        if not self.world.is_master:
            return

        delta = {key: post_weights[key] - pre_weights[key] for key in pre_weights}
        step_path = f"steps/{step:06d}"
        with tempfile.TemporaryDirectory(prefix="prime-rl-hf-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for name, state_dict in (("weights", post_weights), ("grad", grads), ("delta", delta)):
                artifact_dir = tmp_path / name
                save_state_dict(state_dict, artifact_dir)
                self.api.upload_folder(
                    folder_path=str(artifact_dir),
                    path_in_repo=f"{step_path}/{name}",
                    repo_id=self.config.repo_id,
                    repo_type="model",
                    commit_message=f"Upload {name} for step {step}",
                )
                shutil.rmtree(artifact_dir)

            meta_path = tmp_path / "meta.json"
            meta_path.write_text(json.dumps({"step": step}) + "\n", encoding="utf-8")
            self.api.upload_file(
                path_or_fileobj=str(meta_path),
                path_in_repo=f"{step_path}/meta.json",
                repo_id=self.config.repo_id,
                repo_type="model",
                commit_message=f"Upload metadata for step {step}",
            )

            complete_path = tmp_path / "COMPLETE"
            complete_path.write_text("", encoding="utf-8")
            self.api.upload_file(
                path_or_fileobj=str(complete_path),
                path_in_repo=f"{step_path}/COMPLETE",
                repo_id=self.config.repo_id,
                repo_type="model",
                commit_message=f"Mark step {step} complete",
            )
            self.logger.info(f"Uploaded HF artifacts for step {step} to {self.config.repo_id}")
