import torch

from prime_rl.orchestrator.advantage import compute_advantage
from prime_rl.orchestrator.config import AdvantageConfig


def test_grpo_advantage_standardizes_group_rewards():
    rewards = torch.tensor([1.0, 2.0, 5.0])
    lengths = torch.tensor([4, 4, 4])

    advantages = compute_advantage(rewards, lengths, AdvantageConfig(type="grpo"))

    expected = (rewards - rewards.mean()) / rewards.std(unbiased=False)
    assert torch.allclose(advantages, expected)


def test_max_rl_advantage_normalizes_by_group_mean():
    rewards = torch.tensor([0.0, 1.0, 1.0])
    lengths = torch.tensor([4, 4, 4])

    advantages = compute_advantage(rewards, lengths, AdvantageConfig(type="max_rl"))

    expected = (rewards - rewards.mean()) / rewards.mean()
    assert torch.allclose(advantages, expected)
