"""PyTorch Actor-Critic Neural Network Architectures with LayerNorm & Orthogonal Initialization."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Dict, Any, Optional

from config import PPOConfig


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    Orthogonal Weight Initialization:
    Sets starting weights cleanly so the neural network doesn't start with wild, chaotic guesses.
    """
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    r"""
    The Actor (The Robot Muscle Controller):
    Takes in 14 sensor inputs -> Outputs 4 motor torque signals for the hips & knees.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_layer_norm: bool = True,
        ortho_init: bool = True
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm
        
        # Build neural network layers (2 hidden layers of 256 neurons each)
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            if ortho_init:
                linear = layer_init(linear, std=np.sqrt(2)) # Clean starting weight setup
            layers.append(linear)
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim)) # Keeps numbers normalized so gradients don't explode
                
            layers.append(nn.Tanh()) # Activation function (squashes numbers into smooth -1 to +1 curves)
            in_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Output layer for motor mean actions (Gain 0.01 makes initial motor commands small & safe)
        self.mu_layer = nn.Linear(in_dim, action_dim)
        if ortho_init:
            self.mu_layer = layer_init(self.mu_layer, std=0.01)
            
        # Learnable standard deviation parameter (how much the AI experiments / tries random tweaks)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass sensor inputs through network to get motor mean and randomness range."""
        features = self.backbone(state)
        mu = torch.tanh(self.mu_layer(features)) # Tanh keeps motor commands strictly inside [-1.0, 1.0]
        
        # Clamp randomness range to prevent math errors (e.g. dividing by zero or infinite variance)
        log_std_clamped = torch.clamp(self.log_std, min=-20.0, max=2.0)
        std = torch.exp(log_std_clamped).expand_as(mu)
        
        return mu, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """Create Gaussian (Bell Curve) probability distribution to sample motor actions from."""
        mu, std = self.forward(state)
        return Normal(mu, std)


class CriticNetwork(nn.Module):
    """
    The Critic (The AI Judge):
    Takes in 14 sensor inputs -> Predicts 1 single value number V(s): "How well is the robot doing right now?"
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_layer_norm: bool = True,
        ortho_init: bool = True
    ):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            if ortho_init:
                linear = layer_init(linear, std=np.sqrt(2))
            layers.append(linear)
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
                
            layers.append(nn.Tanh())
            in_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Output layer producing single score estimate V(s)
        self.val_layer = nn.Linear(in_dim, 1)
        if ortho_init:
            self.val_layer = layer_init(self.val_layer, std=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict expected score V(s) for the current robot state."""
        features = self.backbone(state)
        val = self.val_layer(features)
        return val


class ActorCriticPPO(nn.Module):
    """Combines Actor (Muscles) and Critic (Judge) into one complete PyTorch AI brain."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[PPOConfig] = None):
        super().__init__()
        self.config = config or PPOConfig()
        
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            use_layer_norm=self.config.use_layer_norm,
            ortho_init=self.config.ortho_init
        )
        
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_dims=self.config.hidden_dims,
            use_layer_norm=self.config.use_layer_norm,
            ortho_init=self.config.ortho_init
        )

    def step(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decision step used when playing in the game:
        Returns chosen motor action, probability of choosing it, entropy (exploration), and Judge's score.
        """
        dist = self.actor.get_distribution(state)
        action = dist.sample() # Pick motor torque from bell curve
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        val = self.critic(state)
        
        return action, log_prob, entropy, val

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used during brain updates to calculate policy log probabilities and Judge score errors."""
        dist = self.actor.get_distribution(state)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        val = self.critic(state)
        
        return log_prob, entropy, val
