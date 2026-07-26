import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Dict, Any, Optional

from config import PPOConfig


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
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
        
        self.mu_layer = nn.Linear(in_dim, action_dim)
        if ortho_init:
            self.mu_layer = layer_init(self.mu_layer, std=0.01)
            
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        mu = torch.tanh(self.mu_layer(features))
        
        log_std_clamped = torch.clamp(self.log_std, min=-20.0, max=2.0)
        std = torch.exp(log_std_clamped).expand_as(mu)
        
        return mu, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        mu, std = self.forward(state)
        return Normal(mu, std)


class CriticNetwork(nn.Module):
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
        
        self.val_layer = nn.Linear(in_dim, 1)
        if ortho_init:
            self.val_layer = layer_init(self.val_layer, std=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.backbone(state)
        val = self.val_layer(features)
        return val


class ActorCriticPPO(nn.Module):
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
        dist = self.actor.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        val = self.critic(state)
        
        return action, log_prob, entropy, val

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor.get_distribution(state)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        val = self.critic(state)
        
        return log_prob, entropy, val
