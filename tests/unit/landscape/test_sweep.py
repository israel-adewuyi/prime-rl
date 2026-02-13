import asyncio
from pathlib import Path
from types import SimpleNamespace

import torch

from prime_rl.landscape import sweep as sweep_module


class _DummyLogger:
    def info(self, _msg: str) -> None:
        return

    def debug(self, _msg: str) -> None:
        return


class _DummyInferencePool:
    def __init__(self) -> None:
        self.clients = []
        self.stopped = False

    async def wait_for_ready(self, _model_name: str) -> None:
        return

    async def stop(self) -> None:
        self.stopped = True


class _Harness:
    def __init__(self, monkeypatch, mode: str) -> None:
        self.calls = {"collect": 0, "loss": 0, "reward": 0}
        self.rows = []
        self.pool = _DummyInferencePool()

        async def fake_setup_inference_pool(_client, base_model: str):
            assert base_model == "dummy-model"
            return self.pool

        async def fake_set_semaphore(_max_concurrent: int) -> None:
            return

        async def fake_collect_fixed_old_policy_batch(**_kwargs):
            self.calls["collect"] += 1
            return sweep_module.FixedOldPolicyBatch(
                micro_batches=[{"loss_mask": torch.tensor([[True]])}],
                reward_mean=0.5,
                reward_std=0.1,
                num_rollouts=4,
            )

        async def fake_evaluate_reward_online_point(**_kwargs):
            self.calls["reward"] += 1
            return {
                "reward_mean": 0.2 * self.calls["reward"],
                "reward_std": 0.0,
                "num_rollouts": 8,
                "elapsed_reward_s": 0.01,
            }

        def fake_compute_eval_loss(_model, _micro_batches, _loss_config, _parallel_dims):
            self.calls["loss"] += 1
            return 1.0 + self.calls["loss"]

        def fake_append_result(_path: Path, row: dict) -> None:
            self.rows.append(dict(row))

        monkeypatch.setattr(sweep_module, "setup_inference_pool", fake_setup_inference_pool)
        monkeypatch.setattr(sweep_module, "set_semaphore", fake_set_semaphore)
        monkeypatch.setattr(sweep_module, "compute_temperature", lambda *_args, **_kwargs: 1.0)
        monkeypatch.setattr(sweep_module, "get_sampling_args", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(sweep_module, "is_vlm_model", lambda _name: False)
        monkeypatch.setattr(sweep_module, "_prepare_examples", lambda _config: (object(), [1, 2], 1))
        monkeypatch.setattr(
            sweep_module,
            "_prepare_sweep_points",
            lambda _grid: [sweep_module.SweepPoint(0.0, 0.0), sweep_module.SweepPoint(1.0, 0.0)],
        )
        monkeypatch.setattr(sweep_module, "apply_point", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(sweep_module, "_collect_fixed_old_policy_batch", fake_collect_fixed_old_policy_batch)
        monkeypatch.setattr(sweep_module, "_evaluate_reward_online_point", fake_evaluate_reward_online_point)
        monkeypatch.setattr(sweep_module, "compute_eval_loss", fake_compute_eval_loss)
        monkeypatch.setattr(sweep_module, "append_result", fake_append_result)

        self.config = self._build_config(mode)

    @staticmethod
    def _build_config(mode: str):
        grid = SimpleNamespace(alpha_min=-1.0, alpha_max=1.0, alpha_steps=2, beta_min=0.0, beta_max=0.0, beta_steps=1)
        sweep = SimpleNamespace(
            grid=grid,
            eval_mode=mode,
            weights_dir=Path("weights"),
            results_file=Path("landscape.csv"),
        )
        orchestrator = SimpleNamespace(
            client=object(),
            model=SimpleNamespace(name="dummy-model", trust_remote_code=False),
            max_concurrent=None,
            sampling=SimpleNamespace(),
            max_steps=1,
        )
        trainer = SimpleNamespace(loss=SimpleNamespace())
        return SimpleNamespace(sweep=sweep, orchestrator=orchestrator, trainer=trainer)

    def run(self) -> None:
        asyncio.run(
            sweep_module.run_sweep(
                config=self.config,
                output_dir=Path("/tmp"),
                model=torch.nn.Linear(1, 1),
                params=[],
                base_tensors={},
                parallel_dims=SimpleNamespace(),
                delta_direction={},
                eta_direction={},
                logger=_DummyLogger(),
            )
        )


def test_run_sweep_loss_fixed_batch_mode(monkeypatch) -> None:
    harness = _Harness(monkeypatch, mode="loss_fixed_batch")
    harness.run()

    assert harness.calls == {"collect": 1, "loss": 2, "reward": 0}
    assert len(harness.rows) == 2
    assert harness.pool.stopped
    for row in harness.rows:
        assert row["loss"] is not None
        assert row["reward_mean"] is None
        assert row["reward_old_mean"] == 0.5
        assert row["reward_old_std"] == 0.1
        assert row["num_rollouts_old"] == 4
        assert row["eval_mode"] == "loss_fixed_batch"


def test_run_sweep_reward_online_mode(monkeypatch) -> None:
    harness = _Harness(monkeypatch, mode="reward_online")
    harness.run()

    assert harness.calls == {"collect": 0, "loss": 0, "reward": 2}
    assert len(harness.rows) == 2
    assert harness.pool.stopped
    for row in harness.rows:
        assert row["loss"] is None
        assert row["reward_mean"] is not None
        assert row["reward_old_mean"] is None
        assert row["reward_old_std"] is None
        assert row["num_rollouts_old"] is None
        assert row["eval_mode"] == "reward_online"


def test_run_sweep_both_mode(monkeypatch) -> None:
    harness = _Harness(monkeypatch, mode="both")
    harness.run()

    assert harness.calls == {"collect": 1, "loss": 2, "reward": 2}
    assert len(harness.rows) == 2
    assert harness.pool.stopped
    for row in harness.rows:
        assert row["loss"] is not None
        assert row["reward_mean"] is not None
        assert row["reward_old_mean"] == 0.5
        assert row["reward_old_std"] == 0.1
        assert row["num_rollouts_old"] == 4
        assert row["eval_mode"] == "both"
