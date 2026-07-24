"""Main Training Pipeline for Autonomous Robotics Reinforcement Learning Agent."""

import os
import time
import argparse
import numpy as np
import torch

from config import EnvConfig, PPOConfig, CurriculumConfig, LoggingConfig
from envs import BipedalWalkerCustomEnv, CurriculumManager
from models import ActorCriticPPO, PPOTrainer
from utils import MetricsLogger
from export import PolicyExporter


def parse_args():
    """Command line argument parser for controlling training settings."""
    parser = argparse.ArgumentParser(description="Autonomous Robotics Continuous Control RL Trainer")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total environment practice steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable experiments")
    parser.add_argument("--log-dir", type=str, default="./logs/tb_logs", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Model checkpoint directory")
    parser.add_argument("--export-dir", type=str, default="./export_models", help="Model export directory")
    return parser.parse_args()


def evaluate_policy(env: BipedalWalkerCustomEnv, policy: ActorCriticPPO, device: torch.device, num_episodes: int = 5):
    """Run 5 test games without exploration noise to evaluate true robot walking skill."""
    policy.eval() # Put neural network in evaluation mode
    returns = []
    lengths = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = policy.step(state_tensor) # Ask AI for best motor action
            action_np = action.squeeze(0).cpu().numpy()
            
            state, reward, terminated, truncated, _ = env.step(action_np) # Step physics simulator
            done = terminated or truncated
            ep_return += reward
            ep_len += 1
            
        returns.append(ep_return)
        lengths.append(ep_len)
        
    policy.train() # Return neural network back to training mode
    return float(np.mean(returns)), float(np.mean(lengths))


def main():
    """Main training loop."""
    args = parse_args()
    
    # Set random seeds for consistent performance across runs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Detect GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using compute device: {device}")
    
    # Load configuration settings
    env_config = EnvConfig()
    ppo_config = PPOConfig(total_timesteps=args.timesteps)
    curriculum_config = CurriculumConfig()
    logging_config = LoggingConfig(log_dir=args.log_dir, save_dir=args.save_dir, export_dir=args.export_dir)
    
    # Instantiate Robot Environment & Teacher (Curriculum Manager)
    env = BipedalWalkerCustomEnv(config=env_config)
    eval_env = BipedalWalkerCustomEnv(config=env_config)
    curriculum_mgr = CurriculumManager(config=curriculum_config)
    curriculum_mgr.apply_stage_to_env(env)
    curriculum_mgr.apply_stage_to_env(eval_env)
    
    # Instantiate Neural Network Policy, PPO Trainer Engine, and TensorBoard Logger
    policy = ActorCriticPPO(state_dim=env_config.state_dim, action_dim=env_config.action_dim, config=ppo_config)
    trainer = PPOTrainer(policy=policy, env_config=env_config, ppo_config=ppo_config, device=device)
    logger = MetricsLogger(log_dir=logging_config.log_dir, use_tensorboard=logging_config.tensorboard)
    
    os.makedirs(logging_config.save_dir, exist_ok=True)
    os.makedirs(logging_config.export_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("      RL ROBOTICS CONTINUOUS CONTROL TRAINING INITIALIZED")
    print(f"      Total Timesteps: {ppo_config.total_timesteps:,}")
    print(f"      Curriculum Stages: {len(curriculum_config.stages)}")
    print("=" * 70 + "\n")
    
    state, _ = env.reset()
    episode_return = 0.0
    episode_length = 0
    episodes_completed = 0
    
    start_time = time.time()
    
    # --- MAIN PRACTICE LOOP ---
    for step in range(1, ppo_config.total_timesteps + 1):
        # 1. Ask neural network for continuous motor torque actions
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, val = policy.step(state_tensor)
            
        action_np = action.squeeze(0).cpu().numpy()
        val_scalar = val.item()
        log_prob_scalar = log_prob.item()
        
        # 2. Execute motor action step in physics simulation
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        
        # 3. Store step in PPO memory buffer
        trainer.buffer.add(
            state=state,
            action=action_np,
            reward=reward,
            value=val_scalar,
            log_prob=log_prob_scalar,
            done=done
        )
        
        state = next_state
        episode_return += reward
        episode_length += 1
        
        # If game ends (fall or timeout), reset robot back to standing posture
        if done:
            episodes_completed += 1
            state, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
            
        # 4. Trigger PPO Neural Network Update every 2048 memory steps
        if step % ppo_config.n_steps == 0:
            with torch.no_grad():
                last_state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                last_val = policy.critic(last_state_tensor).item()
                
            trainer.buffer.compute_gae(
                last_val=last_val,
                done=done,
                gamma=ppo_config.gamma,
                gae_lambda=ppo_config.gae_lambda
            )
            
            # Decay learning rate linearly
            progress_remaining = 1.0 - (step / ppo_config.total_timesteps)
            trainer.update_learning_rate(progress_remaining)
            
            # Train neural network weights
            train_metrics = trainer.train_step()
            logger.log_scalars("PPO_Losses", train_metrics, step)
            
        # 5. Evaluate Robot Performance & Check Level Progression every 10,000 steps
        if step % logging_config.eval_freq_steps == 0:
            mean_return, mean_len = evaluate_policy(eval_env, policy, device, num_episodes=logging_config.eval_episodes)
            eval_metrics = {
                "mean_return": mean_return,
                "mean_length": mean_len,
                "curriculum_stage": curriculum_mgr.current_stage_idx
            }
            logger.log_scalars("Evaluation", eval_metrics, step)
            
            # Promote environment difficulty if score beat target threshold
            advanced = curriculum_mgr.update_curriculum(mean_return, env)
            if advanced:
                curriculum_mgr.apply_stage_to_env(eval_env)
                
        # 6. Save periodic model checkpoint (.pt file) every 50,000 steps
        if step % logging_config.save_freq_steps == 0:
            ckpt_path = os.path.join(logging_config.save_dir, f"ppo_biped_step_{step}.pt")
            torch.save({
                "step": step,
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "curriculum_stage": curriculum_mgr.current_stage_idx
            }, ckpt_path)
            print(f"[Checkpoint Saved] -> {ckpt_path}")
            
    total_time = time.time() - start_time
    print(f"\n[Training Complete] Finished in {total_time:.2f}s ({ppo_config.total_timesteps / total_time:.1f} FPS)")
    
    # Save Final Trained Policy Checkpoint
    final_path = os.path.join(logging_config.save_dir, "ppo_biped_final.pt")
    torch.save(policy.state_dict(), final_path)
    print(f"[Final Policy Saved] -> {final_path}")
    
    # Export trained policy to ONNX format and C++ header file for Sim-to-Real hardware deployment!
    exporter = PolicyExporter(policy=policy, state_dim=env_config.state_dim, action_dim=env_config.action_dim)
    exporter.export_onnx(os.path.join(logging_config.export_dir, "policy.onnx"))
    exporter.export_cpp_header(os.path.join(logging_config.export_dir, "embedded_policy.h"))
    
    logger.close()


if __name__ == "__main__":
    main()
