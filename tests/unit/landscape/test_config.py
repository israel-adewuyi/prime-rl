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
