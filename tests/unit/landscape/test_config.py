import pytest
from pydantic import ValidationError

from prime_rl.landscape.config import LandscapeConfig


def test_landscape_rejects_teacher_model() -> None:
    with pytest.raises(ValidationError, match="landscape does not support orchestrator.teacher_model"):
        LandscapeConfig(orchestrator={"teacher_model": {}})


def test_landscape_rejects_lora() -> None:
    with pytest.raises(ValidationError, match="landscape does not support LoRA"):
        LandscapeConfig(
            trainer={"model": {"lora": {"rank": 8}}},
            orchestrator={"model": {"lora": {"rank": 8}}},
        )


def test_landscape_rejects_inference_lora() -> None:
    with pytest.raises(ValidationError, match="landscape does not support inference.enable_lora"):
        LandscapeConfig(inference={"enable_lora": True})


def test_landscape_eval_mode_defaults_to_loss_fixed_batch() -> None:
    config = LandscapeConfig()
    assert config.sweep.eval_mode == "loss_fixed_batch"


def test_landscape_rejects_invalid_eval_mode() -> None:
    with pytest.raises(ValidationError):
        LandscapeConfig(sweep={"eval_mode": "invalid"})
