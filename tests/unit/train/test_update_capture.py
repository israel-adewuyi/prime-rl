import weakref

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.update_capture import CheckpointUpdateCapture, checkpoint_update_capture_enabled


def test_checkpoint_capture_preserves_preclip_gradients_and_actual_adamw_update(tmp_path):
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [0.5, 1.0, -1.5]]))
        model.bias.copy_(torch.tensor([0.25, -0.5]))
    model.register_parameter("unused", nn.Parameter(torch.tensor([3.0, 4.0])))
    model.register_parameter("frozen", nn.Parameter(torch.tensor([5.0]), requires_grad=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.2)
    inputs = torch.tensor([[2.0, -1.0, 3.0], [-1.0, 2.0, 1.0]])
    model(inputs).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model(inputs).square().sum().backward()
    model(inputs * 2).sum().backward()
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    raw_gradients = {
        name: None if parameter.grad is None else parameter.grad.clone() for name, parameter in parameters.items()
    }
    capture = CheckpointUpdateCapture(model, step=2)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.15)
    assert total_norm > 0.15
    assert not torch.equal(model.weight.grad, raw_gradients["weight"])
    optimizer.step()
    expected_updates = {name: parameter.detach() - before[name] for name, parameter in parameters.items()}
    capture.capture_update()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.weight.add_(100.0)

    destination = capture.save(tmp_path / "trainer")
    assert destination.name == "gradient_update.pt"
    assert not destination.with_suffix(".pt.tmp").exists()
    payload = torch.load(destination, weights_only=True)
    assert payload["schema_version"] == 1
    assert payload["step"] == 2
    assert payload["gradients_before_clipping"] is True
    assert payload["update_definition"] == "after_minus_before"
    assert payload["batch_gradients"].keys() == parameters.keys()
    assert payload["applied_updates"].keys() == parameters.keys()
    assert payload["batch_gradients"]["unused"] is None
    assert not payload["applied_updates"]["unused"].any()
    for name in parameters:
        gradient = payload["batch_gradients"][name]
        if raw_gradients[name] is not None:
            assert gradient.device.type == "cpu" and gradient.dtype == torch.float32
            torch.testing.assert_close(gradient, raw_gradients[name], rtol=0, atol=0)
        update = payload["applied_updates"][name]
        assert update.device.type == "cpu" and update.dtype == torch.float32
        torch.testing.assert_close(update, expected_updates[name], rtol=0, atol=0)
        assert payload["parameter_metadata"][name]["shape"] == list(parameters[name].shape)
        assert payload["parameter_metadata"][name]["parameter_dtype"] == "torch.float32"
    assert payload["applied_updates"]["weight"].abs().sum() > 0


@pytest.mark.parametrize(
    ("dtype", "before", "after", "expected_dtype"),
    [
        (torch.bfloat16, 2**-8, 2.0, torch.float32),
        (torch.float64, 1e12, 1e12 + 0.25, torch.float64),
    ],
)
def test_update_difference_preserves_source_precision(tmp_path, dtype, before, after, expected_dtype):
    model = nn.ParameterDict({"weight": nn.Parameter(torch.tensor([before], dtype=dtype))})
    model["weight"].grad = torch.tensor([0.25], dtype=dtype)
    capture = CheckpointUpdateCapture(model, step=1)
    with torch.no_grad():
        model["weight"].fill_(after)
    capture.capture_update()
    payload = torch.load(capture.save(tmp_path), weights_only=True)
    update = payload["applied_updates"]["weight"]
    assert update.dtype == expected_dtype
    torch.testing.assert_close(update, torch.tensor([after - before], dtype=expected_dtype), rtol=0, atol=0)
    assert payload["batch_gradients"]["weight"].dtype == expected_dtype


def test_update_capture_releases_parameter_references_and_saved_buffers(tmp_path):
    model = nn.Linear(2, 1, bias=False)
    parameter = weakref.ref(model.weight)
    capture = CheckpointUpdateCapture(model, step=1)
    with pytest.raises(RuntimeError, match="before saving"):
        capture.save(tmp_path)
    del model
    assert parameter() is not None
    capture.capture_update()
    assert parameter() is None
    assert capture._weight_snapshots == {}
    with pytest.raises(RuntimeError, match="already been captured"):
        capture.capture_update()
    capture.save(tmp_path)
    assert capture.batch_gradients == {}
    assert capture.applied_updates == {}
    assert capture.parameter_metadata == {}
    with pytest.raises(RuntimeError, match="already been saved"):
        capture.save(tmp_path)


@pytest.mark.parametrize(
    ("enabled", "interval", "step", "is_last_step", "expected"),
    [
        (False, 2, 2, False, False),
        (False, 2, 3, True, False),
        (True, 2, 1, False, False),
        (True, 2, 2, False, True),
        (True, 2, 3, True, True),
        (True, None, 2, False, False),
        (True, None, 3, True, True),
        (True, 2, 0, False, False),
    ],
)
def test_checkpoint_update_capture_schedule(enabled, interval, step, is_last_step, expected):
    config = CheckpointConfig(save_gradients_and_updates=enabled, interval=interval)
    assert checkpoint_update_capture_enabled(config, step, is_last_step) is expected


def test_checkpoint_update_capture_is_disabled_without_opt_in():
    assert CheckpointConfig().save_gradients_and_updates is False
    assert checkpoint_update_capture_enabled(None, step=2, is_last_step=True) is False
    assert checkpoint_update_capture_enabled(CheckpointConfig(interval=2), step=2, is_last_step=True) is False
