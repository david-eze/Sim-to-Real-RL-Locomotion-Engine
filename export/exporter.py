import os
import torch
import torch.nn as nn
from typing import Optional

from models.actor_critic import ActorCriticPPO, ActorNetwork


class PolicyExporter:
    def __init__(self, policy: ActorCriticPPO, state_dim: int = 14, action_dim: int = 4):
        self.policy = policy
        self.actor = policy.actor
        self.state_dim = state_dim
        self.action_dim = action_dim

    def export_onnx(self, output_path: str = "./export_models/policy.onnx") -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.actor.eval()
        
        class ActorMeanWrapper(nn.Module):
            def __init__(self, actor_net: ActorNetwork):
                super().__init__()
                self.actor_net = actor_net
                
            def forward(self, state: torch.Tensor) -> torch.Tensor:
                mu, _ = self.actor_net(state)
                return mu
                
        wrapper = ActorMeanWrapper(self.actor)
        wrapper.eval()
        dummy_input = torch.zeros(1, self.state_dim, dtype=torch.float32)
        
        try:
            torch.onnx.export(
                wrapper,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["observation"],
                output_names=["action_torque"],
                dynamic_axes={"observation": {0: "batch_size"}, "action_torque": {0: "batch_
