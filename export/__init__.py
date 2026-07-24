"""
Models package init file.

When you write `from models import ActorCriticPPO`, Python looks here first.
This file bundles together the Neural Network architecture AND the PPO training engine.
"""

from models.actor_critic import ActorCriticPPO, ActorNetwork, CriticNetwork  # The AI brain (muscles + judge)
from models.ppo import PPOTrainer                                              # The engine that improves the brain

# __all__ lists the public classes that can be imported from this package
__all__ = ["ActorCriticPPO", "ActorNetwork", "CriticNetwork", "PPOTrainer"]
