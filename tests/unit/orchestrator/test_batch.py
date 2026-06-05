from unittest.mock import MagicMock

import pytest
import torch
from transformers import PreTrainedTokenizer

from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.utils.vf import Rollout


def _make_rollout(example_id: int) -> Rollout:
    prompt_ids = [example_id, example_id + 1]
    completion_ids = [example_id + 2, example_id + 3]
    return {
        "example_id": example_id,
        "task": "dummy-task",
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 8,
        "prompt_ids": prompt_ids,
        "prompt_mask": [1] * len(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": [1] * len(completion_ids),
        "completion_logprobs": [0.0] * len(completion_ids),
        "reward": 0.0,
        "advantage": 1.0,
        "is_truncated": False,
        "metrics": {},
    }


@pytest.mark.parametrize(
    ("rollout_count", "num_train_workers", "expected_batches_per_worker"), [(4, 2, 2), (5, 2, 3), (7, 1, 7), (11, 4, 3)]
)
def test_prepare_batch_balances_micro_batches_across_workers(
    rollout_count, num_train_workers, expected_batches_per_worker
):
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    rollouts = [_make_rollout(i) for i in range(rollout_count)]

    batches_per_gpu = prepare_batch(
        rollouts=rollouts,
        tokenizer=tokenizer,
        seq_len=4,
        num_train_workers=num_train_workers,
    )

    assert all(len(worker_batches) == expected_batches_per_worker for worker_batches in batches_per_gpu)

    flat_batches = [batch for worker_batches in batches_per_gpu for batch in worker_batches]
    assert len(rollouts) <= len(flat_batches) < len(rollouts) + num_train_workers

    # Verify real rollouts have expected non-zero advantages and loss mask
    for batch in flat_batches[: len(rollouts)]:
        assert torch.count_nonzero(batch["advantages"]) == 4
        assert torch.count_nonzero(batch["loss_mask"]) == 4
        assert batch["temperature"] == 0.5
        assert batch["top_p"] == 0.9
        assert batch["top_k"] == 8

    # Verify padded batches have zero advantages and loss mask
    for batch in flat_batches[len(rollouts) :]:
        assert torch.count_nonzero(batch["advantages"]) == 0
        assert torch.count_nonzero(batch["loss_mask"]) == 0
        assert batch["temperature"] == 0.5
        assert batch["top_p"] == 0.9
        assert batch["top_k"] == 8


def test_prepare_batch_rejects_mixed_sampling_args_in_packed_micro_batch():
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    rollouts = [_make_rollout(0), _make_rollout(1)]
    rollouts[1]["top_p"] = 0.8

    with pytest.raises(AssertionError, match="shared sampling arguments"):
        prepare_batch(
            rollouts=rollouts,
            tokenizer=tokenizer,
            seq_len=8,
            num_train_workers=1,
        )
