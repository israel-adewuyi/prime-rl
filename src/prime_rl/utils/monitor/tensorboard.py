import json
import os
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
import verifiers as vf
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.shared import TensorBoardConfig
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor, sample_items_for_logging


def get_summary_writer_cls():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires the `tensorboard` package. "
            "Add it to `pyproject.toml` and install dependencies before enabling `[tensorboard]`."
        ) from exc

    return SummaryWriter


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


class TensorBoardMonitor(Monitor):
    """Logs metrics, samples, and summaries to TensorBoard event files."""

    def __init__(
        self,
        config: TensorBoardConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseConfig | None = None,
        process_name: str | None = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.run_config = run_config
        self.process_name = process_name
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.last_step = 0

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0

        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(
                    f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})"
                )
            return

        assert self.config is not None
        writer_cls = get_summary_writer_cls()
        self.log_dir = self._resolve_log_dir()
        self.logger.info(f"Initializing {self.__class__.__name__} ({self.log_dir})")
        self.writer = writer_cls(
            log_dir=str(self.log_dir),
            max_queue=self.config.max_queue,
            flush_secs=self.config.flush_secs,
        )

        if self.run_config is not None:
            self.writer.add_text(
                "config",
                _json_dumps(self.run_config.model_dump(mode="json")),
                global_step=0,
            )

    def _resolve_log_dir(self) -> Path:
        assert self.config is not None

        base_run_name = self.config.run_name
        if base_run_name is None:
            if self.output_dir is None:
                base_run_name = "run"
            else:
                base_run_name = self.output_dir.name

        run_name = f"{self.process_name}_{base_run_name}" if self.process_name else base_run_name
        if self.config.log_dir is not None:
            return self.config.log_dir / run_name
        if self.output_dir is None:
            raise ValueError("TensorBoard logging requires either `tensorboard.log_dir` or `output_dir`.")
        return self.output_dir / "tensorboard" / run_name

    def _extras_enabled(self, extra_name: str, step: int) -> bool:
        if not self.is_master or not self.enabled:
            return False
        assert self.config is not None
        if self.config.log_extras is None:
            return False
        if not getattr(self.config.log_extras, extra_name):
            return False
        return step % self.config.log_extras.interval == 0

    def _format_sample_block(self, sample: dict[str, Any]) -> str:
        return (
            f"task: {sample['task']}\n"
            f"example_id: {sample['example_id']}\n"
            f"reward: {sample['reward']}\n"
            f"messages:\n{sample['messages']}\n"
            f"input_ids: {sample['input_ids']}"
        )

    def _format_eval_sample_block(self, sample: dict[str, Any]) -> str:
        return (
            f"env: {sample['env']}\n"
            f"task: {sample['task']}\n"
            f"example_id: {sample['example_id']}\n"
            f"reward: {sample['reward']}\n"
            f"completion:\n{sample['completion']}"
        )

    def _build_final_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"step": self.last_step}
        for metrics in self.history:
            for key, value in metrics.items():
                if key == "step":
                    continue
                if isinstance(value, Number | str | bool):
                    summary[key] = value
        return summary

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.history.append(metrics)
        self.last_step = step
        if not self.is_master or not self.enabled:
            return

        for key, value in metrics.items():
            if key == "step" or not isinstance(value, Number):
                continue
            self.writer.add_scalar(key, value, global_step=step)

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        if not self._extras_enabled("samples", step):
            return

        assert self.config is not None
        assert self.config.log_extras is not None
        assert self.tokenizer is not None, "Tokenizer is required for TensorBoard sample logging"

        rollouts = sample_items_for_logging(rollouts, self.config.log_extras.sample_ratio)
        if not rollouts:
            return

        logged_samples: list[dict[str, Any]] = []
        for rollout in rollouts:
            trajectory = rollout["trajectory"]
            if not trajectory:
                continue
            tokens = trajectory[-1]["tokens"]
            full_ids = tokens["prompt_ids"] + tokens["completion_ids"]
            sample = {
                "task": rollout.get("task"),
                "example_id": rollout["example_id"],
                "reward": rollout["reward"],
                "messages": self.tokenizer.decode(full_ids),
                "input_ids": str(full_ids),
            }
            self.samples.append(sample)
            logged_samples.append(sample)

        if not logged_samples:
            return

        body = "\n\n".join(self._format_sample_block(sample) for sample in logged_samples)
        self.writer.add_text("samples", body, global_step=step)

    def log_eval_samples(self, rollouts: list[vf.RolloutOutput], env_name: str, step: int) -> None:
        if not self._extras_enabled("samples", step):
            return

        logged_samples: list[dict[str, Any]] = []
        for rollout in rollouts:
            completion = rollout.get("completion")
            if not completion:
                continue
            if isinstance(completion, list):
                assert self.tokenizer is not None, "Tokenizer is required for chat-format eval sample logging"
                completion = self.tokenizer.apply_chat_template(completion, tokenize=False)

            logged_samples.append(
                {
                    "env": env_name,
                    "task": rollout.get("task"),
                    "example_id": rollout["example_id"],
                    "reward": rollout["reward"],
                    "completion": completion,
                }
            )

        if not logged_samples:
            return

        body = "\n\n".join(self._format_eval_sample_block(sample) for sample in logged_samples)
        self.writer.add_text(f"eval/samples/{env_name}", body, global_step=step)

    def log_final_samples(self) -> None:
        if not self.is_master or not self.enabled or not self.samples:
            return

        body = "\n\n".join(self._format_sample_block(sample) for sample in self.samples)
        self.writer.add_text("final_samples", body, global_step=self.last_step)

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        if not self.is_master or not self.enabled:
            return

        summary = self._build_final_summary()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        self.writer.add_text(
            "final_summary",
            _json_dumps(summary),
            global_step=self.last_step,
        )

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if not self._extras_enabled("distributions", step):
            return

        for key, values in distributions.items():
            if not values:
                continue
            self.writer.add_histogram(key, np.asarray(values, dtype=float), global_step=step)

    def close(self) -> None:
        if not self.is_master or not self.enabled:
            return

        self.writer.flush()
        self.writer.close()
