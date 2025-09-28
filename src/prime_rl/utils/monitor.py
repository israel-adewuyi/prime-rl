import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import pynvml
import trackio as wandb
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import WandbMonitorConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pydantic_config import BaseSettings


class WandbMonitor:
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbMonitorConfig | None,
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
        self._maybe_overwrite_wandb_command()
        self.wandb = wandb.init(
            project=config.project,
            name=config.name,
            space_id=config.space_id,
            dataset_id=config.dataset_id,
            resume="allow",
            config=run_config.model_dump() if run_config else None,
        )

        # Optionally, initialize sample logging attributes
        if config is not None and config.log_extras:
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

            if config is not None and config.log_extras.distributions:
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
        self.history.append(metrics)
        if not self.is_master:
            return
        if not self.enabled:
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
            not self.config
            or not self.config.log_extras
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
        if not self.config or not self.config.log_extras or not self.config.log_extras.samples:
            return
        self.logger.info("Logging final samples to W&B table")
        df = pd.DataFrame(self.samples)
        table = wandb.Table(dataframe=df)
        # wandb.log({"final-samples": table})

    def log_final_distributions(self) -> None:
        """Log final distributions to W&B table."""
        if not self.is_master:
            return
        if not self.config or not self.config.log_extras or not self.config.log_extras.distributions:
            return
        self.logger.info("Logging final distributions to W&B table")
        df = pd.DataFrame(self.distributions)
        table = wandb.Table(dataframe=df)
        # wandb.log({"final-distributions": table})

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to W&B table."""
        if not self.is_master or not self.enabled:
            return
        self.logger.info("Saving final summary to file")
        assert self.output_dir is not None, "Output directory is required for saving final summary"
        dir_path = self.output_dir / f"run-{self.wandb.id}"
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / filename, "w") as f:
            json.dump(wandb.summary._as_dict(), f)


_MONITOR: WandbMonitor | None = None


def get_monitor() -> WandbMonitor:
    """Returns the global monitor."""
    global _MONITOR
    if _MONITOR is None:
        raise RuntimeError("WandbMonitor not initialized. Please call `setup_monitor` first.")
    return _MONITOR


def setup_monitor(
    config: WandbMonitorConfig | None,
    output_dir: Path | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    run_config: BaseSettings | None = None,
) -> WandbMonitor:
    """Sets up a monitor to log metrics to W&B."""
    global _MONITOR
    if _MONITOR is not None:
        raise RuntimeError("WandbMonitor already initialized. Please call `setup_monitor` only once.")
    _MONITOR = WandbMonitor(config=config, output_dir=output_dir, tokenizer=tokenizer, run_config=run_config)
    return _MONITOR
