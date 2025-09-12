import asyncio
import json
import os
import random
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import aiohttp
import pandas as pd
import psutil
import pynvml
import trackio as wandb
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import (
    APIMonitorConfig,
    BaseConfig,
    FileMonitorConfig,
    MultiMonitorConfig,
    SocketMonitorConfig,
    WandbMonitorConfig,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pydantic_config import BaseSettings


class Monitor(ABC):
    """Base class for logging metrics to a single monitoring type (e.g. file, socket, API)."""

    def __init__(self, config: BaseConfig, task_id: str | None = None):
        self.config = config
        self.lock = threading.Lock()
        self.metadata = {"task_id": task_id}
        self.has_metadata = any(self.metadata.values())
        self.logger = get_logger()
        if not self.has_metadata:
            self.logger.warning(
                "No run metadata found. This is fine for local runs, but unexpected when contributing to a public run."
            )
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")

    def _serialize_metrics(self, metrics: dict[str, Any]) -> str:
        if self.has_metadata:
            metrics.update(self.metadata)
        return json.dumps(metrics)

    @abstractmethod
    def log(self, metrics: dict[str, Any]) -> None: ...


class FileMonitor(Monitor):
    """Logs to a file. Used for debugging."""

    def __init__(self, config: FileMonitorConfig, task_id: str | None = None):
        super().__init__(config, task_id)
        self.file_path = config.path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        with self.lock:
            try:
                with open(self.file_path.as_posix(), "a") as f:
                    f.write(self._serialize_metrics(metrics) + "\n")
                self.logger.debug(f"Logged successfully to {self.file_path}")
            except Exception as e:
                self.logger.error(f"Failed to log metrics to {self.file_path}: {e}")


class SocketMonitor(Monitor):
    """Logs to a Unix socket. Previously called `PrimeMetrics`."""

    def __init__(self, config: SocketMonitorConfig, task_id: str | None = None):
        super().__init__(config, task_id)
        self.socket_path = config.path

    def log(self, metrics: dict[str, Any]) -> None:
        with self.lock:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(self.socket_path.as_posix())
                    sock.sendall(self._serialize_metrics(metrics).encode())
                self.logger.debug(f"Logged successfully to {self.socket_path}")
            except Exception as e:
                self.logger.error(f"Failed to log metrics to {self.socket_path}: {e}")


class APIMonitor(Monitor):
    """Logs to an API via HTTP. Previously called `HttpMonitor`."""

    def __init__(self, config: APIMonitorConfig, task_id: str | None = None):
        super().__init__(config, task_id)
        self.url = config.url
        self.auth_token = config.auth_token

    def log(self, metrics: dict[str, Any]) -> None:
        """Logs metrics to the server"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }
        payload = {"metrics": self._serialize_metrics(metrics)}

        async def _post_metrics():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.url, json=payload, headers=headers) as response:
                        if response is not None:
                            response.raise_for_status()
                    self.logger.debug(f"Logged successfully to server {self.url}")
            except Exception as e:
                self.logger.error(f"Failed to log metrics to {self.url}: {e}")

        asyncio.run(_post_metrics())


class WandbMonitor(Monitor):
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbMonitorConfig,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        task_id: str | None = None,
        run_config: BaseSettings | None = None,
    ):
        super().__init__(config, task_id)
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("DP_RANK", "0")))
        self.config = config
        self.output_dir = output_dir
        self.is_master = rank == 0
        if not self.is_master:
            self.logger.warning(f"Skipping WandbMonitor initialization from non-master rank ({rank})")
            return
        self._maybe_overwrite_wandb_command()
        self.wandb = wandb.init(
            project=config.project,
            name=config.name,
            resume="allow",
            config=run_config.model_dump() if run_config else None,
        )

        # Optionally, initialize sample logging attributes
        if config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.samples_cols = [
                    "step",
                    "tag",
                    "problem_id",
                    "sample_id",
                    "num_input_tokens",
                    "num_output_tokens",
                    "input_tokens",
                    "output_tokens",
                    "prompt",
                    "completion",
                    "reward",
                    "advantage",
                ]
                self.samples_table = wandb.Table(
                    columns=self.samples_cols,
                    log_mode="INCREMENTAL",
                )
                self.tokenizer = tokenizer
                self.samples = []

            if config.log_extras.distributions:
                self.last_log_distributions_step = -1
                # Incremental table is initialized dynamically in `log_distributions`
                self.distributions_table = None
                self.distributions = []

    def _maybe_overwrite_wandb_command(self) -> None:
        """Overwrites sys.argv with the start command if it is set in the environment variables."""
        wandb_args = os.environ.get("WANDB_ARGS", None)
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

    def log(self, metrics: dict[str, Any]) -> None:
        if not self.is_master:
            return
        wandb.log(metrics)

    def _select_problem_samples(
        self, 
        input_tokens: list[list[int]], 
        output_tokens: list[list[int]], 
        rollouts_per_problem: int
    ) -> dict[str, int]:
        """Select representative problem samples (min, max, random length).
        
        Args:
            input_tokens: List of input token sequences
            output_tokens: List of output token sequences  
            rollouts_per_problem: Number of rollouts per problem
            
        Returns:
            Dictionary mapping sample tags to problem IDs
        """
        batch_size = len(input_tokens)
        num_problems = batch_size // rollouts_per_problem
        
        # Compute per-problem statistics
        per_problem_tokens = defaultdict(list)
        token_sequences = [input_tokens[i] + output_tokens[i] for i in range(batch_size)]
        
        for i, token_sequence in enumerate(token_sequences):
            problem_id = i // rollouts_per_problem
            per_problem_tokens[problem_id].append(token_sequence)
            
        assert len(per_problem_tokens) == num_problems
        assert list(per_problem_tokens.keys()) == list(range(num_problems))

        # Calculate average sequence length per problem
        per_problem_seq_len = {
            problem_id: sum(len(token_sequence) for token_sequence in tokens) / len(tokens) 
            for problem_id, tokens in per_problem_tokens.items()
        }
        
        self.logger.debug(f"Per-problem sequence lengths: {per_problem_seq_len}")
        
        # Select representative problems
        min_len_problem_id = min(per_problem_seq_len.items(), key=lambda problem_id_and_length: problem_id_and_length[1])[0]
        max_len_problem_id = max(per_problem_seq_len.items(), key=lambda problem_id_and_length: problem_id_and_length[1])[0]
        random_problem_id = random.choice(list(range(num_problems)))
        
        problem_ids = {
            "min_len": min_len_problem_id,
            "max_len": max_len_problem_id,
            "random": random_problem_id,
        }
        
        self.logger.debug(f"Selected problem samples: {problem_ids}")
        return problem_ids

    def _create_sample_data(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]], 
        rewards: list[float],
        advantages: list[float],
        problem_ids: dict[str, int],
        rollouts_per_problem: int,
        step: int
    ) -> list[dict[str, Any]]:
        """Create sample data dictionaries for selected problems.
        
        Args:
            input_tokens: List of input token sequences
            output_tokens: List of output token sequences
            rewards: List of rewards for each sample
            advantages: List of advantages for each sample
            problem_ids: Dictionary mapping sample tags to problem IDs
            rollouts_per_problem: Number of rollouts per problem
            step: Current training step
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        for tag, problem_id in problem_ids.items():
            start_idx = problem_id * rollouts_per_problem
            
            for sample_id in range(start_idx, start_idx + rollouts_per_problem):
                sample = {
                    "step": step,
                    "tag": tag,
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "num_input_tokens": len(input_tokens[sample_id]),
                    "num_output_tokens": len(output_tokens[sample_id]),
                    "input_tokens": str(input_tokens[sample_id]),
                    "output_tokens": str(output_tokens[sample_id]),
                    "prompt": self.tokenizer.decode(input_tokens[sample_id]),
                    "completion": self.tokenizer.decode(output_tokens[sample_id]),
                    "reward": float(rewards[sample_id]),
                    "advantage": float(advantages[sample_id]),
                }
                
                # Verify column order matches expected structure
                assert list(sample.keys()) == self.samples_cols, (
                    "Sample column order must match self.samples_cols"
                )
                samples.append(sample)
                
        return samples

    def _save_samples_to_local(self, df: pd.DataFrame, step: int) -> None:
        """Save samples DataFrame to local CSV file with append functionality.
        
        Args:
            df: DataFrame containing sample data
            step: Current training step for logging
        """
        if not hasattr(self, 'output_dir') or self.output_dir is None:
            self.logger.warning("No output directory configured, skipping local sample save")
            return
            
        try:
            samples_file = self.output_dir / "samples_log.csv"
            
            # Create directory if it doesn't exist
            samples_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to existing file or create new one
            if samples_file.exists():
                df.to_csv(samples_file, mode='a', header=False, index=False)
                self.logger.debug(f"Appended {len(df)} samples to {samples_file}")
            else:
                df.to_csv(samples_file, mode='w', header=True, index=False)
                self.logger.debug(f"Created new samples log file: {samples_file} with {len(df)} samples")
                
        except Exception as e:
            self.logger.error(f"Failed to save samples to local file: {e}")

    def log_samples(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        rollouts_per_problem: int,
        step: int,
    ) -> None:
        """Log prompt/response samples to W&B table and local file.

        Args:
            input_tokens: List of input token sequences
            output_tokens: List of output token sequences
            rewards: List of rewards for each sample
            advantages: List of advantages for each sample
            rollouts_per_problem: Number of rollouts per problem
            step: Current training step
        """
        if not self.is_master:
            return
            
        if (
            not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            return
            
        # Validate required attributes
        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for sample logging"
        
        self.logger.info(f"Logging samples at step {step}")
        start_time = time.time()
        
        try:
            # Select representative problem samples
            problem_ids = self._select_problem_samples(
                input_tokens, output_tokens, rollouts_per_problem
            )
            
            # Create sample data
            samples = self._create_sample_data(
                input_tokens, output_tokens, rewards, advantages,
                problem_ids, rollouts_per_problem, step
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(samples)
            
            # Log to W&B/trackio
            wandb.log({"samples": wandb.Table(dataframe=df)})
            
            # Save to local file
            self._save_samples_to_local(df, step)
            
            # Update tracking
            self.last_log_samples_step = step
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Successfully logged {len(samples)} samples at step {step} in {elapsed_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to log samples at step {step}: {e}")
            raise

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if not self.is_master:
            return
        if (
            not self.config.log_extras
            or not self.config.log_extras.distributions
            or step % self.config.log_extras.interval != 0
        ):
            return
        assert self.last_log_distributions_step <= step, "Step must be greater than last logged step"
        self.logger.info(f"Logging distributions for keys {list(distributions.keys())} to W&B table at step {step}")

        # Initialize incremental table if not already done
        if self.distributions_table is None:
            self.distributions_cols = list(distributions.keys())
            self.distributions_table = wandb.Table(
                columns=["step"] + self.distributions_cols,
                log_mode="INCREMENTAL",
            )
        assert self.distributions_cols == list(distributions.keys()), (
            "Columns in the table must be the same across all steps"
        )

        # Append to distributions
        start_time = time.time()
        row = {"step": step, **distributions}
        # self.distributions.append(row)
        # self.distributions_table.add_data(*row.values())
        # wandb.log({"distributions": self.distributions_table}, step=step)
        self.last_log_distributions_step = step
        self.logger.debug(f"Logged distributions at step {step} to W&B table in {time.time() - start_time:.2f}s")

    def log_final_samples(self) -> None:
        """Log final samples to W&B table."""
        if not self.is_master:
            return
        if not self.config.log_extras or not self.config.log_extras.samples:
            return
        self.logger.debug("Logging final samples to W&B table")
        df = pd.DataFrame(self.samples)
        table = wandb.Table(dataframe=df)
        # wandb.log({"final-samples": table})

    def log_final_distributions(self) -> None:
        """Log final distributions to W&B table."""
        if not self.is_master:
            return
        if not self.config.log_extras or not self.config.log_extras.distributions:
            return
        self.logger.debug("Logging final distributions to W&B table")
        df = pd.DataFrame(self.distributions)
        table = wandb.Table(dataframe=df)
        # wandb.log({"final-distributions": table})


MonitorType = Literal["file", "socket", "api", "wandb"]


class MultiMonitor:
    """
    Log progress, performance, and system metrics to multiple (configurable) outputs.
    """

    def __init__(
        self,
        config: MultiMonitorConfig,
        output_dir: Path | None = None,
        task_id: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseSettings | None = None,
    ):
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self.output_dir = output_dir
        # Initialize outputs
        self.outputs: dict[MonitorType, Monitor] = {}
        self.wandb = None
        if config.file:
            self.outputs["file"] = FileMonitor(config.file, task_id)
        if config.socket:
            self.outputs["socket"] = SocketMonitor(config.socket, task_id)
        if config.api:
            self.outputs["api"] = APIMonitor(config.api, task_id)
        if config.wandb:
            self.wandb = WandbMonitor(config.wandb, output_dir, tokenizer, task_id, run_config=run_config)
            self.outputs["wandb"] = self.wandb

        self.disabled = len(self.outputs) == 0

        # Start metrics collection thread, if system_log_frequency is greater than 0
        if config.system_log_frequency > 0:
            self.logger.info(f"Starting thread to log system metrics every {config.system_log_frequency}s")
            self._system_log_frequency = config.system_log_frequency
            self._has_gpu = self._set_has_gpu()
            self._thread = None
            self._stop_event = threading.Event()
            self._start_metrics_thread()

    def log(
        self,
        metrics: dict[str, Any],
        exclude: list[MonitorType] = [],
    ) -> None:
        """Logs metrics to all outputs."""
        self.history.append(metrics)
        if self.disabled:
            return
        self.logger.debug(f"Logging metrics: {metrics}")
        for output_type, output in self.outputs.items():
            if output_type not in exclude:
                output.log(metrics)

    def _set_has_gpu(self) -> bool:
        """Determines if a GPU is available at runtime"""
        try:
            pynvml.nvmlInit()
            pynvml.nvmlDeviceGetHandleByIndex(0)  # Check if at least one GPU exists
            return True
        except pynvml.NVMLError:
            return False

    def _start_metrics_thread(self):
        """Starts the system metrics logging thread"""
        assert self._thread is None, "Metrics thread already started"
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._log_system_metrics, daemon=True)
        self._thread.start()

    def _stop_metrics_thread(self):
        """Stops the system metrics logging thread"""
        assert self._thread is not None, "Metrics thread not started"
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _log_system_metrics(self):
        """Loop that periodically logs system metrics."""
        assert self._thread is not None, "Metrics thread not started"
        while not self._stop_event.is_set():
            metrics = {
                "system/cpu_percent": psutil.cpu_percent(),
                "system/memory_percent": psutil.virtual_memory().percent,
                "system/memory_usage": psutil.virtual_memory().used,
                "system/memory_total": psutil.virtual_memory().total,
            }

            if self._has_gpu:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    metrics.update(
                        {
                            f"system/gpu_{i}_memory_used": info.used,
                            f"system/gpu_{i}_memory_total": info.total,
                            f"system/gpu_{i}_utilization": gpu_util.gpu,
                        }
                    )

            self.log(metrics, exclude=["wandb"])
            time.sleep(self._system_log_frequency)

    def __del__(self):
        # Need to check hasattr because __del__ sometime delete attributes before
        if hasattr(self, "_thread") and self._thread is not None:
            self._stop_metrics_thread()


_MONITOR: MultiMonitor | None = None


def get_monitor() -> MultiMonitor:
    """Returns the global monitor."""
    global _MONITOR
    if _MONITOR is None:
        raise RuntimeError("Monitor not initialized. Please call `setup_monitor` first.")
    return _MONITOR


def setup_monitor(
    config: MultiMonitorConfig,
    output_dir: Path | None = None,
    task_id: str | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    run_config: BaseSettings | None = None,
) -> MultiMonitor:
    """Sets up a monitor to log metrics to multiple specified outputs."""
    global _MONITOR
    if _MONITOR is not None:
        raise RuntimeError(
            "Monitor already initialized. Please call `setup_monitor` only once."
        )
    _MONITOR = MultiMonitor(config, output_dir, task_id, tokenizer, run_config)
    return _MONITOR
