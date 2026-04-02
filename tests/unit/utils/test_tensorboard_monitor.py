import json
from pathlib import Path

from prime_rl.configs.shared import LogExtrasConfig, TensorBoardConfig
from prime_rl.utils.monitor.tensorboard import TensorBoardMonitor


class FakeWriter:
    def __init__(self, log_dir: str, max_queue: int, flush_secs: int):
        self.log_dir = log_dir
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.scalars: list[tuple[str, float, int]] = []
        self.text: list[tuple[str, str, int]] = []
        self.histograms: list[tuple[str, list[float], int]] = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        self.scalars.append((tag, scalar_value, global_step))

    def add_text(self, tag: str, text_string: str, global_step: int):
        self.text.append((tag, text_string, global_step))

    def add_histogram(self, tag: str, values: list[float], global_step: int):
        self.histograms.append((tag, list(values), global_step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return " ".join(map(str, token_ids))

    def apply_chat_template(self, completion, tokenize: bool = False) -> str:
        assert not tokenize
        return " | ".join(message["content"] for message in completion)


def test_tensorboard_monitor_uses_process_prefixed_run_dirs_and_logs(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "prime_rl.utils.monitor.tensorboard.get_summary_writer_cls",
        lambda: FakeWriter,
    )

    config = TensorBoardConfig(
        run_name="292b",
        log_dir=tmp_path,
        log_extras=LogExtrasConfig(interval=1, sample_ratio=1.0),
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    monitor = TensorBoardMonitor(
        config=config,
        output_dir=output_dir,
        tokenizer=FakeTokenizer(),
        process_name="orch",
    )

    assert monitor.log_dir == tmp_path / "orch_292b"

    monitor.log({"reward/all/mean": 1.5, "step": 3}, step=3)
    monitor.log_distributions({"advantages": [0.1, 0.2]}, step=3)
    monitor.log_samples(
        [
            {
                "trajectory": [{"tokens": {"prompt_ids": [1, 2], "completion_ids": [3, 4]}}],
                "task": "alphabet-sort",
                "example_id": "ex-1",
                "reward": 1.0,
            }
        ],
        step=3,
    )
    monitor.log_eval_samples(
        [
            {
                "task": "alphabet-sort",
                "example_id": "ex-2",
                "reward": 0.5,
                "completion": [{"content": "sorted"}],
            }
        ],
        env_name="alphabet-sort",
        step=3,
    )
    monitor.log_final_samples()
    monitor.save_final_summary()
    monitor.close()

    assert monitor.writer.scalars == [("reward/all/mean", 1.5, 3)]
    assert monitor.writer.histograms == [("advantages", [0.1, 0.2], 3)]
    assert any(
        tag == "samples" and "example_id: ex-1" in body
        for tag, body, _ in monitor.writer.text
    )
    assert any(
        tag == "eval/samples/alphabet-sort" and "completion:\nsorted" in body
        for tag, body, _ in monitor.writer.text
    )
    assert any(tag == "final_samples" for tag, _, _ in monitor.writer.text)
    assert any(tag == "final_summary" for tag, _, _ in monitor.writer.text)
    assert monitor.writer.flushed
    assert monitor.writer.closed

    summary = json.loads((monitor.log_dir / "final_summary.json").read_text(encoding="utf-8"))
    assert summary["reward/all/mean"] == 1.5
    assert summary["step"] == 3


def test_tensorboard_monitor_defaults_run_name_from_output_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "prime_rl.utils.monitor.tensorboard.get_summary_writer_cls",
        lambda: FakeWriter,
    )

    output_dir = tmp_path / "my-run"
    output_dir.mkdir()
    monitor = TensorBoardMonitor(
        config=TensorBoardConfig(log_extras=None),
        output_dir=output_dir,
        process_name="train",
    )

    assert monitor.log_dir == output_dir / "tensorboard" / "train_my-run"
