from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.utils.config import LogConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class GridConfig(BaseConfig):
    alpha_min: Annotated[float, Field(description="Minimum alpha value.")] = -1.0
    alpha_max: Annotated[float, Field(description="Maximum alpha value.")] = 1.0
    alpha_steps: Annotated[int, Field(ge=1, description="Number of alpha points.")] = 21

    beta_min: Annotated[float, Field(description="Minimum beta value.")] = -1.0
    beta_max: Annotated[float, Field(description="Maximum beta value.")] = 1.0
    beta_steps: Annotated[int, Field(ge=1, description="Number of beta points.")] = 21

    @model_validator(mode="after")
    def validate_steps(self):
        if self.alpha_steps < 1 or self.beta_steps < 1:
            raise ValueError("alpha_steps and beta_steps must be >= 1")
        return self


class DirectionConfig(BaseConfig):
    seed_delta: Annotated[int, Field(description="Seed for the delta direction.")] = 0
    seed_eta: Annotated[int, Field(description="Seed for the eta direction.")] = 1
    norm: Annotated[
        Literal["layer", "global"],
        Field(description="Normalization strategy for random directions."),
    ] = "layer"
    param_filter: Annotated[
        Literal["trainable", "all"],
        Field(description="Which parameters to perturb."),
    ] = "trainable"
    epsilon: Annotated[float, Field(ge=0, description="Numerical stability for norm scaling.")] = 1e-12
    delta_path: Annotated[
        str | None,
        Field(description="Path or hf:// repo path to a .pt state dict for delta direction."),
    ] = None
    eta_path: Annotated[
        str | None,
        Field(description="Path or hf:// repo path to a .pt state dict for eta direction."),
    ] = None


class SweepConfig(BaseConfig):
    grid: GridConfig = GridConfig()
    direction: DirectionConfig = DirectionConfig()

    num_examples: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of examples to sample once and reuse. Defaults to batch_size/rollouts_per_example.",
        ),
    ] = None

    batch_size: Annotated[
        int | None,
        Field(
            ge=1,
            description="Total number of rollouts to generate per grid point. Defaults to orchestrator.batch_size.",
        ),
    ] = None

    rollouts_per_example: Annotated[
        int | None,
        Field(
            ge=1,
            description="Rollouts per example. Defaults to orchestrator.rollouts_per_example.",
        ),
    ] = None

    weights_dir: Annotated[
        Path,
        Field(description="Subdirectory (relative to output_dir) for temporary weights."),
    ] = Path("weights")

    results_file: Annotated[
        Path,
        Field(description="CSV file (relative to output_dir) for landscape results."),
    ] = Path("landscape.csv")

    metadata_file: Annotated[
        Path,
        Field(description="JSON file (relative to output_dir) for sweep metadata."),
    ] = Path("metadata.json")

    rollouts_file: Annotated[
        Path,
        Field(description="Text file (relative to output_dir) for human-readable rollouts."),
    ] = Path("rollouts.txt")


class LandscapeConfig(BaseSettings):
    trainer: RLTrainerConfig = RLTrainerConfig()
    orchestrator: OrchestratorConfig = OrchestratorConfig()
    inference: InferenceConfig = InferenceConfig()

    output_dir: Annotated[
        Path,
        Field(description="Directory to write landscape outputs."),
    ] = Path("outputs/landscape")

    log: LogConfig = LogConfig()
    sweep: SweepConfig = SweepConfig()

    start_inference: Annotated[
        bool,
        Field(description="Whether to start the inference server automatically."),
    ] = True

    inference_gpu_ids: Annotated[
        list[int],
        Field(description="GPU IDs to use for the inference server."),
    ] = [0]

    @model_validator(mode="after")
    def validate_shared_model(self):
        if self.trainer.model.name != self.orchestrator.model.name:
            raise ValueError(
                f"trainer.model.name ({self.trainer.model.name}) must match orchestrator.model.name ({self.orchestrator.model.name})"
            )
        if self.trainer.model.seq_len < self.orchestrator.seq_len:
            raise ValueError(
                f"trainer.model.seq_len ({self.trainer.model.seq_len}) must be >= orchestrator.seq_len ({self.orchestrator.seq_len})"
            )
        self.inference.model.name = self.trainer.model.name
        if self.trainer.weight_broadcast.type != "filesystem":
            raise ValueError("landscape requires trainer.weight_broadcast.type='filesystem'")
        if self.trainer.model.lora is not None and self.orchestrator.model.lora is None:
            raise ValueError("trainer.model.lora is set but orchestrator.model.lora is not")
        if self.trainer.model.lora is not None and self.orchestrator.model.lora is not None:
            trainer_name = self.trainer.model.lora.name
            orchestrator_name = self.orchestrator.model.lora.name
            if trainer_name and orchestrator_name and trainer_name != orchestrator_name:
                raise ValueError(
                    f"trainer.model.lora.name ({trainer_name}) must match orchestrator.model.lora.name ({orchestrator_name})"
                )
            if not self.inference.enable_lora:
                self.inference.enable_lora = True
            if self.inference.max_lora_rank is None:
                self.inference.max_lora_rank = self.trainer.model.lora.rank
        return self

    @model_validator(mode="after")
    def auto_setup_inference_dp(self):
        if self.inference is not None:
            tp = self.inference.parallel.tp
            if len(self.inference_gpu_ids) != self.inference.parallel.dp * tp:
                if len(self.inference_gpu_ids) % tp != 0:
                    raise ValueError("Number of inference GPUs must be divisible by inference.parallel.tp")
                self.inference.parallel.dp = len(self.inference_gpu_ids) // tp
        return self
