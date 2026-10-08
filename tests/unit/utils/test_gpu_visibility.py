import pytest
import torch

from prime_rl.utils.process import get_physical_gpu_ids, gpu_visibility_env, partition_gpus


@pytest.mark.parametrize(
    "hip,cuda_visible,hip_visible,expected",
    [
        (None, "2,5", None, [2, 5]),
        ("7.2", "2,5", None, [2, 5]),
        ("7.2", "2,5", "1,3", [1, 3]),
        ("7.2", "-1", None, []),
        ("7.2", "2,5", "", []),
    ],
)
def test_gpu_visibility_precedence(monkeypatch, hip, cuda_visible, hip_visible, expected):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    if hip_visible is not None:
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", hip_visible)
    assert get_physical_gpu_ids() == expected


def test_rocr_namespace_and_component_partition(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,7")
    inference, trainer = partition_gpus(1, 1)
    assert inference == [0]
    assert trainer == [1]
    assert gpu_visibility_env(trainer) == {"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "1"}
    with pytest.raises(ValueError, match="only 2"):
        partition_gpus(1, 2)


def test_cuda_component_visibility(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert gpu_visibility_env([2, 5]) == {"CUDA_VISIBLE_DEVICES": "2,5"}
