"""Configuration parameters and settings dataclasses for the RL Robotics pipeline."""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class EnvConfig:
    """Robot & Environment Settings (Sensors, Motors, and Point System Weights)."""
    env_id: str = "BipedalWalkerCustom-v0"   # Name identifier of our custom robotics environment
    state_dim: int = 14                       # 14 body sensors (tilt, speed, joint angles, foot contact)
    action_dim: int = 4                       # 4 motor torque outputs (Hip 1, Knee 1, Hip 2, Knee 2)
    max_episode_steps: int = 1600             # Max time ticks per game episode (~26 seconds at 60 FPS)
    render_mode: str = "rgb_array"            # How graphics are drawn ('human' window or 'rgb_array' frames)
    
    # Physics settings
    gravity: float = -9.81                    # Earth downward gravity force in m/s^2
    time_step: float = 1.0 / 60.0             # Physics updates 60 times per second (0.0166s per tick)
    action_bounds: Tuple[float, float] = (-1.0, 1.0) # Motor torque command limits [-1.0 max back, +1.0 max forward]
    
    # Multi-Objective Reward Weights (The Point System)
    w_forward: float = 2.5                    # Reward multiplier for running forward fast
    w_ctrl_cost: float = 0.005                # Penalty for wasting motor battery power
    w_smoothness: float = 0.01                # Penalty for jerking or shaking leg joints
    w_posture: float = 0.5                    # Penalty for leaning torso sideways
    w_impact: float = 0.02                    # Penalty for slamming feet too hard onto ground
    r_alive: float = 0.2                      # Small survival reward given every tick the robot stays standing
    fall_penalty: float = -100.0              # Massive 100-point penalty if robot falls over


@dataclass
class CurriculumStage:
    """Settings for a single level in the game (Level 0, Level 1, Level 2)."""
    stage_id: int                             # Level number (0, 1, 2)
    name: str                                 # Friendly descriptive name of the stage
    target_return: float                      # Score needed to graduate to the next level
    push_force_std: float                     # How hard random wind gusts push the robot torso
    terrain_roughness: float                  # Bumps and roughness on the floor
    slope_angle_deg: float                    # Hill incline angle in degrees
    reward_scale_penalty: float               # Penalty multiplier (ramped up on higher levels)


@dataclass
class CurriculumConfig:
    """List of all difficulty levels the robot must beat."""
    stages: List[CurriculumStage] = field(default_factory=lambda: [
        # Level 0: Easy flat ground to learn basic balance
        CurriculumStage(
            stage_id=0,
            name="Flat Ground Walking (Basic Balance)",
            target_return=150.0,              # Beat 150 points to level up
            push_force_std=0.0,               # No wind pushes
            terrain_roughness=0.0,            # Completely smooth floor
            slope_angle_deg=0.0,              # 0 degree slope
            reward_scale_penalty=0.5,
        ),
        # Level 1: Slightly bumpy ground and small pushes
        CurriculumStage(
            stage_id=1,
            name="Rough Terrain & Smooth Control",
            target_return=230.0,              # Beat 230 points to level up
            push_force_std=2.0,               # Light 2N wind pushes
            terrain_roughness=0.05,           # Small bumps
            slope_angle_deg=2.0,              # 2 degree hill
            reward_scale_penalty=0.8,
        ),
        # Level 2: Steep hill and heavy wind gusts
        CurriculumStage(
            stage_id=2,
            name="Robust Locomotion with External Disturbances",
            target_return=300.0,              # Final mastery score
            push_force_std=5.0,               # Strong 5N wind gusts
            terrain_roughness=0.12,           # Heavy bumps
            slope_angle_deg=5.0,              # 5 degree steep incline
            reward_scale_penalty=1.0,
        ),
    ])


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO Neural Network Learning Engine."""
    # Neural Network Layout
    hidden_dims: Tuple[int, ...] = (256, 256) # 2 hidden layers with 256 virtual neurons each
    activation: str = "tanh"                  # Activation function shape
    use_layer_norm: bool = True               # Normalizes numbers so brain signals stay stable
    ortho_init: bool = True                   # Clean weight initialization to start learning safely
    
    # Training Parameters
    learning_rate: float = 3e-4               # How fast neural network weights change (0.0003)
    lr_schedule: str = "linear"               # Gradually slow down learning rate as training nears end
    gamma: float = 0.99                       # Discount factor: how much future rewards matter vs immediate
    gae_lambda: float = 0.95                  # GAE advantage smoothing parameter
    clip_range: float = 0.2                   # Keeps PPO brain updates locked between 80% and 120%
    clip_range_vf: float = 0.2                # Value function clipping limit
    ent_coef: float = 0.005                   # Encourages exploration (trying creative motor moves)
    vf_coef: float = 0.5                      # Weight of Judge (Critic) loss vs Muscle (Actor) loss
    max_grad_norm: float = 0.5                # Max limit on gradient updates to prevent training crashes
    
    # Batch sizes & updates
    total_timesteps: int = 1_000_000          # Total practice steps (1 million steps default)
    n_steps: int = 2048                       # Memory rollout size before triggering a brain update
    batch_size: int = 64                      # Mini-batch size for training updates
    n_epochs: int = 10                        # How many times to re-examine memory buffer per update
    target_kl: float = 0.015                  # Safety limit for how much policy can shift in 1 update


@dataclass
class LoggingConfig:
    """Settings for saving model checkpoints and TensorBoard score charts."""
    log_dir: str = "./logs/tb_logs"           # Folder for TensorBoard live graphs
    save_dir: str = "./checkpoints"           # Folder for saved PyTorch model files (.pt)
    export_dir: str = "./export_models"       # Folder for ONNX and C++ embedded header files
    save_freq_steps: int = 50_000             # Save a checkpoint every 50,000 steps
    eval_freq_steps: int = 10_000             # Evaluate robot skill every 10,000 steps
    eval_episodes: int = 5                    # Test robot over 5 test games during evaluation
    tensorboard: bool = True                  # Enable live TensorBoard graph recording
