from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class EnvConfig:
    env_id: str = "BipedalWalkerCustom-v0"
    state_dim: int = 14
    action_dim: int = 4
    max_episode_steps: int = 1600
    render_mode: str = "rgb_array"
    
    gravity: float = -9.81
    time_step: float = 1.0 / 60.0
    action_bounds: Tuple[float, float] = (-1.0, 1.0)
    
    w_forward: float = 2.5
    w_ctrl_cost: float = 0.005
    w_smoothness: float = 0.01
    w_posture: float = 0.5
    w_impact: float = 0.02
    r_alive: float = 0.2
    fall_penalty: float = -100.0


@dataclass
class CurriculumStage:
    stage_id: int
    name: str
    target_return: float
    push_force_std: float
    terrain_roughness: float
    slope_angle_deg: float
    reward_scale_penalty: float


@dataclass
class CurriculumConfig:
    stages: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage(
            stage_id=0,
            name="Flat Ground Walking (Basic Balance)",
            target_return=150.0,
            push_force_std=0.0,
            terrain_roughness=0.0,
            slope_angle_deg=0.0,
            reward_scale_penalty=0.5,
        ),
        CurriculumStage(
            stage_id=1,
            name="Rough Terrain & Smooth Control",
            target_return=230.0,
            push_force_std=2.0,
            terrain_roughness=0.05,
            slope_angle_deg=2.0,
            reward_scale_penalty=0.8,
        ),
        CurriculumStage(
            stage_id=2,
            name="Robust Locomotion with External Disturbances",
            target_return=300.0,
            push_force_std=5.0,
            terrain_roughness=0.12,
            slope_angle_deg=5.0,
            reward_scale_penalty=1.0,
        ),
    ])


@dataclass
class PPOConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    use_layer_norm: bool = True
    ortho_init: bool = True
    
    learning_rate: float = 3e-4
    lr_schedule: str = "linear"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    total_timesteps: int = 1_000_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    target_kl: float = 0.015


@dataclass
class LoggingConfig:
    log_dir: str = "./logs/tb_logs"
    save_dir: str = "./checkpoints"
    export_dir: str = "./export_models"
    save_freq_steps: int = 50_000
    eval_freq_steps: int = 10_000
    eval_episodes: int = 5
    tensorboard: bool = True
