import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from config import EnvConfig


class BipedalWalkerCustomEnv(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.config.state_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=self.config.action_bounds[0],
            high=self.config.action_bounds[1],
            shape=(self.config.action_dim,),
            dtype=np.float32
        )
        
        self._state = np.zeros(self.config.state_dim, dtype=np.float32)
        self.prev_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self.step_count = 0
        
        self.push_force_std = 0.0
        self.terrain_roughness = 0.0
        self.slope_angle_rad = 0.0
        self.penalty_scaler = 1.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        hull_angle = np.random.uniform(-0.02, 0.02)     
        hull_ang_vel = np.random.uniform(-0.01, 0.01)   
        vel_x = 0.0                                     
        vel_y = 0.0                                     
        
        hip1 = np.random.uniform(-0.05, 0.05)
        hip1_v = 0.0
        knee1 = np.random.uniform(0.1, 0.2)
        knee1_v = 0.0
        
        hip2 = np.random.uniform(-0.05, 0.05)
        hip2_v = 0.0
        knee2 = np.random.uniform(0.1, 0.2)
        knee2_v = 0.0
        
        foot1 = 1.0  
        foot2 = 1.0
        
        self._state = np.array([
            hull_angle, hull_ang_vel, vel_x, vel_y,
            hip1, hip1_v, knee1, knee1_v,
            hip2, hip2_v, knee2, knee2_v,
            foot1, foot2
        ], dtype=np.float32)
        
        self.prev_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self.step_count = 0
        
        info = {
            "curriculum_push_std": self.push_force_std,
            "terrain_roughness": self.terrain_roughness,
            "slope_angle_deg": np.degrees(self.slope_angle_rad)
        }
        return self._state.copy(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Run 1 physics step (1/60th of a second) using motor commands from the AI."""
        self.step_count += 1
        action = np.clip(action, self.config.action_bounds[0], self.config.action_bounds[1])
        
        dt = self.config.time_step  
        
        
        hull_angle, hull_ang_vel, vel_x, vel_y, hip1, hip1_v, knee1, knee1_v, hip2, hip2_v, knee2, knee2_v, foot1, foot2 = self._state
        
        external_push = 0.0
        if self.push_force_std > 0.0 and np.random.rand() < 0.1:
            external_push = np.random.normal(0.0, self.push_force_std)
            
        acc_x = (
            1.8 * (action[0] * np.sin(hip1) + action[2] * np.sin(hip2))
            - 0.1 * vel_x
            + external_push
            - 9.81 * np.sin(self.slope_angle_rad)
        )
        
        acc_y = 1.2 * (action[1] + action[3]) + self.config.gravity + 0.15 * (foot1 + foot2) * 9.81
        
        acc_hull = -2.5 * hull_angle - 0.8 * hull_ang_vel + 0.5 * (action[0] - action[2]) + 0.1 * external_push
        
        vel_x = np.clip(vel_x + acc_x * dt, -5.0, 10.0)
        vel_y = np.clip(vel_y + acc_y * dt, -5.0, 5.0)
        hull_ang_vel = np.clip(hull_ang_vel + acc_hull * dt, -10.0, 10.0)
        hull_angle += hull_ang_vel * dt
        
        hip1_v = np.clip(hip1_v + (action[0] * 15.0 - hip1 * 5.0) * dt, -8.0, 8.0)
        hip1 += hip1_v * dt
        knee1_v = np.clip(knee1_v + (action[1] * 20.0 - knee1 * 8.0) * dt, -10.0, 10.0)
        knee1 += knee1_v * dt
        
        hip2_v = np.clip(hip2_v + (action[2] * 15.0 - hip2 * 5.0) * dt, -8.0, 8.0)
        hip2 += hip2_v * dt
        knee2_v = np.clip(knee2_v + (action[3] * 20.0 - knee2 * 8.0) * dt, -10.0, 10.0)
        knee2 += knee2_v * dt
        
        foot1 = 1.0 if np.sin(self.step_count * 0.1) > -0.2 else 0.0
        foot2 = 1.0 if np.sin(self.step_count * 0.1 + np.pi) > -0.2 else 0.0
        
        self._state = np.array([
            hull_angle, hull_ang_vel, vel_x, vel_y,
            hip1, hip1_v, knee1, knee1_v,
            hip2, hip2_v, knee2, knee2_v,
            foot1, foot2
        ], dtype=np.float32)
        
        terminated = False
        if abs(hull_angle) > 0.8:
            terminated = True
            
        truncated = self.step_count >= self.config.max_episode_steps
        
        reward = self._compute_reward(action, terminated)
        
        self.prev_action = action.copy()
        
        info = {
            "vel_x": float(vel_x),
            "hull_angle": float(hull_angle),
            "energy_cost": float(np.sum(np.square(action))),
            "smoothness_cost": float(np.sum(np.square(action - self.prev_action))),
            "terminated": terminated
        }
        
        return self._state.copy(), reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray, terminated: bool) -> float:

        if terminated:
            return float(self.config.fall_penalty) 
            
        vel_x = self._state[2]
        hull_angle = self._state[0]
        
        r_fwd = self.config.w_forward * vel_x
        
        ctrl_cost = self.config.w_ctrl_cost * float(np.sum(np.square(action))) * self.penalty_scaler
        smoothness_cost = self.config.w_smoothness * float(np.sum(np.square(action - self.prev_action))) * self.penalty_scaler
        posture_cost = self.config.w_posture * float(hull_angle ** 2)
        
        reward = r_fwd - ctrl_cost - smoothness_cost - posture_cost + self.config.r_alive
        return float(reward)

    def set_curriculum_params(
        self,
        push_force_std: float,
        terrain_roughness: float,
        slope_angle_deg: float,
        penalty_scaler: float
    ) -> None:
        self.push_force_std = push_force_std
        self.terrain_roughness = terrain_roughness
        self.slope_angle_rad = np.radians(slope_angle_deg)
        self.penalty_scaler = penalty_scaler
