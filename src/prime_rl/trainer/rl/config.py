from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from prime_rl.trainer.config import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    WeightCheckpointConfig,
)
from prime_rl.utils.config import LogConfig, WandbMonitorConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class LossConfig(BaseModel):
    """Base config for loss."""

    ratio_type: Annotated[Literal["token", "sequence"], Field(description="Type of importance ratio to use.")] = "token"
    ratio_length_norm: Annotated[
        bool, Field(description="Whether to normalize the importance ratio by the sequence length.")
    ] = False

    clip_ratio: Annotated[float, Field(ge=0)] = 8.0


class FakeDataLoaderConfig(BaseConfig):
    """Configures a fake data loader sampling random micro batches for debugging."""

    micro_batch_size: Annotated[int, Field(ge=1)] = 1
    batch_size: Annotated[int, Field(ge=1)] = 2
    seq_len: Annotated[int, Field(ge=1)] = 128

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self


class DataLoaderConfig(BaseConfig):
    """Configures the data loader used for training."""

    fake: Annotated[FakeDataLoaderConfig | None, Field(description="Whether to use a fake data loader.")] = None


class GradientAccumulatorConfig(BaseConfig):
    """Configures the gradient accumulator class"""

    beta: Annotated[float, Field(description="Decay rate of previous gradient")] = 0.99
    epsilon: Annotated[float, Field(description="epsilon term for numeric stability when logging")] = 1e-8
    grad_save_interval: Annotated[
        int | None,
        Field(
            description="How often should the current accumulated grad be saved? If None, will not save the gradient EMA."
        ),
    ] = None
    tolerance: Annotated[
        list[float], Field(description="Threshold for determining if a parameter is active (for masking)")
    ] = [1e-5]
    save_masks: Annotated[bool, Field(description="Whether to save binary masks")] = True
    mask_save_interval: Annotated[
        int | None, Field(description="How often should masks be saved? If None, uses save_interval")
    ] = None
    upload_to_hf: Annotated[bool, Field(description="Whether to upload masks to Hugging Face Hub")] = False
    hf_repo_id: Annotated[str | None, Field(description="Hugging Face repo ID for uploading masks")] = None
    hf_upload_interval: Annotated[
        int | None, Field(description="Upload interval for HF Hub (defaults to mask_save_interval)")
    ] = None
    hf_private: Annotated[bool, Field(description="Whether to make HF repo private")] = True


class MaskLoadingConfig(BaseConfig):
    """Configures mask loading from Hugging Face Hub"""

    enabled: Annotated[bool, Field(description="Whether to load masks from HF Hub")] = False
    hf_repo_id: Annotated[str | None, Field(description="Hugging Face repository ID containing the masks")] = None
    step: Annotated[int | None, Field(description="Training step to load masks for")] = None


class RLTrainerConfig(BaseSettings):
    """Configures the RL trainer"""

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The data configuration
    data: DataLoaderConfig = DataLoaderConfig()

    # The loss configuration
    loss: LossConfig = LossConfig()

    # The optimizer configuration
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()

    # The learning rate scheduler configuration
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    # The weight checkpoint configuration
    weights: WeightCheckpointConfig = WeightCheckpointConfig()

    # The gradient accumulation config
    grad_acc: GradientAccumulatorConfig | None = None

    # The mask loading config
    mask_loading: MaskLoadingConfig | None = None

    # The logging configuration
    log: LogConfig = LogConfig()

    # The wandb configuration
    wandb: WandbMonitorConfig | None = None

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

    max_steps: Annotated[
        int | None,
        Field(
            description="Maximum number of steps to run training for. If None, will run indefinitely.",
        ),
    ] = None

    async_level: Annotated[
        int,
        Field(
            ge=0,
            description="Maximum number of steps that inference can be ahead of training. Determines how 'off-policy' the inference engines can be. Higher values yield better throughput through async execution, but may yield lower powerofrmance. If 0, will be fully synchronous.",
        ),
    ] = 2

    memory_profiler_path: Annotated[Path | None, Field(description="Path to write memory profile to.")] = None

    recompute_logprobs: Annotated[
        bool,
        Field(
            description="Whether to recompute the logprobs. If True, will always recompute logprobs and overwrite those found in the training batch.",
        ),
    ] = False

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5 and use fake data.",
        ),
    ] = False

    trace_path: Annotated[Path | None, Field(description="Path to write pytorch profiler trace to.")] = None

    dist_timeout_seconds: Annotated[
        int,
        Field(
            description="Timeout in seconds for torch distributed ops. Defaults to 600 seconds.",
        ),
    ] = 600

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.max_steps = 4  # 1 Warmup + 3 Benchmark
            if not self.data.fake:
                self.data.fake = FakeDataLoaderConfig()
            if self.wandb:  # Do not log extras
                self.wandb.log_extras = None
            if self.ckpt:  # Do not checkpoint
                self.ckpt = None
        return self

    @model_validator(mode="after")
    def disable_logging_wandb_samples(self):
        if self.wandb and self.wandb.log_extras:
            self.wandb.log_extras.samples = False
        return self

    @model_validator(mode="after")
    def dont_do_massive_traces(self):
        if self.trace_path:
            if self.max_steps is None:
                raise ValueError("Must specify max_steps when tracing")
            if self.max_steps >= 10:
                raise ValueError(
                    "Tracing more than 10 steps is not recommended as your trace will be massive. Remove this line if you really want to trace more steps."
                )
        return self

    @model_validator(mode="after")
    def validate_lora_adapter_saving(self):
        if self.weights and self.weights.save_adapter_separately:
            lora_enabled = self.model and self.model.experimental and self.model.experimental.lora
            if not lora_enabled:
                raise ValueError(
                    "save_adapter_separately=True requires LoRA to be enabled. "
                    "Set model.experimental.lora or disable save_adapter_separately."
                )
        return self

    @model_validator(mode="after")
    def validate_mask_operations(self):
        """Ensure mask loading and saving are mutually exclusive operations."""
        has_mask_loading = self.mask_loading is not None and self.mask_loading.enabled
        has_mask_saving = self.grad_acc is not None and self.grad_acc.save_masks

        if has_mask_loading and has_mask_saving:
            raise ValueError(
                "Cannot both load masks and save masks in the same run. "
                "Use separate training runs for these operations. "
                "Set either mask_loading.enabled=false or grad_acc.save_masks=false."
            )

        # Validate mask loading configuration
        if has_mask_loading:
            if not self.mask_loading.hf_repo_id:
                raise ValueError("mask_loading.hf_repo_id must be specified when mask_loading.enabled=true")
            if self.mask_loading.step is None:
                raise ValueError("mask_loading.step must be specified when mask_loading.enabled=true")

        return self
