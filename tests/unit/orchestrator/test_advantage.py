import torch

from prime_rl.orchestrator.advantage import compute_advantage
from prime_rl.orchestrator.config import AdvantageConfig


def test_grpo_advantage_standardizes_group_rewards():
    rewards = torch.tensor([1.0, 2.0, 5.0])
    lengths = torch.tensor([4, 4, 4])

    advantages = compute_advantage(rewards, lengths, AdvantageConfig(type="grpo"))

    expected = (rewards - rewards.mean()) / rewards.std(unbiased=False)
    assert torch.allclose(advantages, expected)
