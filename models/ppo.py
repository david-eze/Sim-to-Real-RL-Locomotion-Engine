

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from config import EnvConfig, PPOConfig
from models.actor_critic import ActorCriticPPO


@dataclass
class RolloutBuffer:

    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.states, self.actions = [], []
        self.rewards, self.values, self.log_probs, self.dones = [], [], [], []
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        return len(self.rewards)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            value: float, log_prob: float, done: bool) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))

    def compute_gae(self, last_val: float, done: bool, gamma: float,
                    gae_lambda: float) -> None:
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0.0
        next_value = float(last_val)
        next_nonterminal = 0.0 if done else 1.0
        for index in range(len(self.rewards) - 1, -1, -1):
            delta = (self.rewards[index] + gamma * next_value * next_nonterminal
                     - self.values[index])
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[index] = gae
            next_value = self.values[index]
            next_nonterminal = 0.0 if self.dones[index] else 1.0
        self.advantages = advantages
        self.returns = advantages + np.asarray(self.values, dtype=np.float32)


class PPOTrainer:

    def __init__(self, policy: ActorCriticPPO, env_config: EnvConfig,
                 ppo_config: PPOConfig, device: torch.device) -> None:
        self.policy = policy.to(device)
        self.config = ppo_config
        self.device = device
        self.buffer = RolloutBuffer()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=ppo_config.learning_rate, eps=1e-5
        )

    def update_learning_rate(self, progress_remaining: float) -> None:
        if self.config.lr_schedule == "linear":
            learning_rate = self.config.learning_rate * max(0.0, progress_remaining)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

    def train_step(self) -> Dict[str, float]:
        if not len(self.buffer) or self.buffer.advantages is None:
            raise RuntimeError("Compute GAE before calling train_step().")

        states = torch.as_tensor(np.asarray(self.buffer.states), device=self.device)
        actions = torch.as_tensor(np.asarray(self.buffer.actions), device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self.buffer.log_probs), device=self.device).unsqueeze(1)
        advantages = torch.as_tensor(self.buffer.advantages, device=self.device).unsqueeze(1)
        returns = torch.as_tensor(self.buffer.returns, device=self.device).unsqueeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                   "approx_kl": 0.0, "clip_fraction": 0.0}
        updates = 0
        sample_count = len(self.buffer)
        for _ in range(self.config.n_epochs):
            for indices in torch.randperm(sample_count, device=self.device).split(self.config.batch_size):
                new_log_probs, entropy, values = self.policy.evaluate_actions(states[indices], actions[indices])
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(
                    ratio * advantages[indices],
                    ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range) * advantages[indices],
                ).mean()
                value_loss = 0.5 * (returns[indices] - values).square().mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy_mean.item()
                metrics["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                metrics["clip_fraction"] += ((ratio - 1).abs() > self.config.clip_range).float().mean().item()
                updates += 1
                if metrics["approx_kl"] / updates > self.config.target_kl:
                    break
            if updates and metrics["approx_kl"] / updates > self.config.target_kl:
                break

        self.buffer.clear()
        return {name: value / max(updates, 1) for name, value in metrics.items()}
