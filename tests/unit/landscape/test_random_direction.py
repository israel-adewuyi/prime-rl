import pytest
import torch

from prime_rl.landscape.directions import build_random_direction


def test_build_random_direction_is_seeded() -> None:
    params = [
        ("w1", torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))),
        ("w2", torch.nn.Parameter(torch.tensor([1.5, -2.5, 0.5], dtype=torch.float32))),
    ]
    base_tensors = {name: param.detach().clone() for name, param in params}

    direction_a = build_random_direction(params, base_tensors, seed=123, epsilon=1e-12)
    direction_b = build_random_direction(params, base_tensors, seed=123, epsilon=1e-12)
    direction_c = build_random_direction(params, base_tensors, seed=456, epsilon=1e-12)

    for name, _ in params:
        assert torch.allclose(direction_a[name], direction_b[name])

    assert any(not torch.allclose(direction_a[name], direction_c[name]) for name, _ in params)


def test_build_random_direction_matches_each_tensor_norm() -> None:
    params = [
        ("small", torch.nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))),
        ("large", torch.nn.Parameter(torch.tensor([[3.0, 4.0], [0.0, 12.0]], dtype=torch.float32))),
    ]
    base_tensors = {name: param.detach().clone() for name, param in params}

    direction = build_random_direction(params, base_tensors, seed=7, epsilon=1e-12)

    for name, _ in params:
        base_norm = base_tensors[name].float().norm(p=2).item()
        direction_norm = direction[name].float().norm(p=2).item()
        assert direction_norm == pytest.approx(base_norm, rel=1e-5, abs=1e-7)
