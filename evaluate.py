import os
import time
import argparse
import numpy as np
import torch

from config import EnvConfig, PPOConfig
from envs import BipedalWalkerCustomEnv
from models import ActorCriticPPO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Robotics Policy")
    parser.add_argument("--model-path", type=str, default="./checkpoints/ppo_biped_final.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation test episodes")
    parser.add_argument("--render", action="store_true", help="Render real-time simulation delay")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_config = EnvConfig()
    ppo_config = PPOConfig()
    
    env = BipedalWalkerCustomEnv(config=env_config)
    policy = ActorCriticPPO(state_dim=env_config.state_dim, action_dim=env_config.action_dim, config=ppo_config)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["policy_state_dict"])
        else:
            policy.load_state_dict(checkpoint)
        print(f"[Evaluate] Successfully loaded checkpoint: {args.model_path}")
    else:
        print(f"[Evaluate Warning] Checkpoint {args.model_path} not found. Running with un-trained policy for sanity test.")
        
    policy.to(device)
    policy.eval()
    
    print("\n" + "=" * 70)
    print("        REAL-TIME ROBOTICS POLICY EVALUATION & TELEMETRY")
    print("=" * 70)
    
    for ep in range(1, args.episodes + 1):
        state, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        
        vel_x_history = []
        hull_angle_history = []
        torque_energy_history = []
        
        start_time = time.time()
        
        while not done:
            step += 1
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, _, _, _ = policy.step(state_tensor)
                
            action_np = action.squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, step_info = env.step(action_np)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
            vel_x_history.append(step_info["vel_x"])
            hull_angle_history.append(step_info["hull_angle"])
            torque_energy_history.append(step_info["energy_cost"])
            
            if args.render:
                time.sleep(0.016)
                
        duration = time.time() - start_time
        avg_vx = float(np.mean(vel_x_history))
        avg_angle = float(np.mean(np.abs(hull_angle_history)))
        avg_energy = float(np.mean(torque_energy_history))
        
        print(f"Episode {ep:2d} | Steps: {step:4d} | Total Return: {total_reward:7.2f} | Avg Speed: {avg_vx:5.2f} m/s | Torso Drift: {avg_angle:5.3f} rad | Energy/Step: {avg_energy:5.3f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate()
