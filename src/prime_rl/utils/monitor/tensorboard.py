import json
import os
import time
from pathlib import Path, PosixPath
from typing import Any

import numpy as np
import pandas as pd
import verifiers as vf
from torch.utils.tensorboard import SummaryWriter
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import WandbConfig, WandbWithExtrasConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor
from prime_rl.utils.pydantic_config import BaseSettings


class TensorboardMonitor(Monitor):
    """Logs metrics and samples to TensorBoard."""

    def __init__(
        self,
        config: WandbConfig | WandbWithExtrasConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseSettings | None = None,
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0
        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return

        assert config is not None
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")

        log_dir = output_dir / "runs" if output_dir else Path("runs")
        if config.name:
            log_dir = log_dir / config.name
        log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(log_dir))

        if run_config:
            config_data = _convert_posix_to_str(run_config.model_dump())
            config_str = json.dumps(config_data, indent=2)
            self.writer.add_text("config", f"```json\n{config_str}\n```", 0)

        if config is not None and isinstance(config, WandbWithExtrasConfig) and config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.tokenizer = tokenizer
                self.samples = []

            if config.log_extras.distributions:
                self.last_log_distributions_step = -1
                self.distributions = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.history.append(metrics)
        if not self.is_master:
            return
        if not self.enabled:
            return

        resolved_step = step if step is not None else metrics.get("step", None)
        for key, value in metrics.items():
            if key == "step":
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, resolved_step)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        self.writer.add_scalar(f"{key}/{sub_key}", sub_value, resolved_step)

    def log_samples(self, rollouts: list[vf.State], step: int) -> None:
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            return

        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for sample logging"

        self.logger.info(f"Logging samples to TensorBoard at step {step}")
        start_time = time.perf_counter()

        sample_texts: list[str] = []
        for rollout in rollouts:
            trajectory = rollout.get("trajectory", [])
            if not trajectory:
                continue
            last_step = trajectory[-1]
            tokens = last_step["tokens"]
            full_ids = tokens["prompt_ids"] + tokens["completion_ids"]
            messages_text = self.tokenizer.decode(full_ids)
            sample = {
                "step": step,
                "task": rollout.get("task"),
                "example_id": rollout.get("example_id"),
                "reward": rollout.get("reward"),
                "messages": messages_text,
                "input_ids": str(full_ids),
            }
            self.samples.append(sample)

            sample_texts.append(
                "\n".join(
                    [
                        f"### Example {sample['example_id']}",
                        f"**Task:** {sample['task']}",
                        f"**Reward:** {sample['reward']}",
                        "",
                        "**Messages:**",
                        "```",
                        messages_text,
                        "```",
                        "---",
                    ]
                )
            )

        if sample_texts:
            self.writer.add_text(f"samples/step_{step}", "\n".join(sample_texts), step)

        self.last_log_samples_step = step
        self.logger.debug(
            f"Logged samples at step {step} to TensorBoard in {time.perf_counter() - start_time:.2f}s"
        )

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.distributions
            or step % self.config.log_extras.interval != 0
        ):
            return
        assert self.last_log_distributions_step <= step, "Step must be greater than last logged step"
        self.logger.info(f"Logging distributions for keys {list(distributions.keys())} to TensorBoard at step {step}")

        start_time = time.perf_counter()
        for key, values in distributions.items():
            values_array = np.array(values)
            self.writer.add_histogram(f"distributions/{key}", values_array, step)

        self.distributions.append({"step": step, **distributions})
        self.last_log_distributions_step = step
        self.logger.debug(
            f"Logged distributions at step {step} to TensorBoard in {time.perf_counter() - start_time:.2f}s"
        )

    def log_final_samples(self) -> None:
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
        ):
            return
        if self.samples and self.output_dir:
            self.logger.info("Saving final samples to CSV")
            output_path = self.output_dir / "final_samples.csv"
            pd.DataFrame(self.samples).to_csv(output_path, index=False)
            self.logger.info(f"Saved final samples to {output_path}")

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        if not self.is_master or not self.enabled:
            return
        assert self.output_dir is not None, "Output directory is required for saving final summary"

        summary: dict[str, Any] = {}
        for entry in self.history:
            for key, value in entry.items():
                if isinstance(value, (int, float)):
                    summary[f"final_{key}"] = value

        dir_path = self.output_dir / "tensorboard_summary"
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / filename, "w") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        if self.is_master and self.enabled:
            self.writer.close()


def _convert_posix_to_str(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _convert_posix_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_posix_to_str(v) for v in obj]
    if isinstance(obj, PosixPath):
        return str(obj)
    return obj
