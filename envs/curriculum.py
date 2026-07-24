"""Dynamic Curriculum Learning Manager for Reinforcement Learning Robotics Training."""

import logging
from typing import Optional

from config import CurriculumConfig, CurriculumStage
from envs.bipedal_walker import BipedalWalkerCustomEnv

logger = logging.getLogger(__name__)


class CurriculumManager:
    """
    The Robot Teacher:
    Monitors average scores and promotes the robot to harder levels (bumps, wind, hills)
    so the AI doesn't get stuck early on.
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self.current_stage_idx = 0            # Start at Level 0 (Flat Ground)
        self.stages = self.config.stages       # Load all level definitions
        
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current level settings (Level 0, Level 1, or Level 2)."""
        return self.stages[self.current_stage_idx]
        
    def apply_stage_to_env(self, env: BipedalWalkerCustomEnv) -> None:
        """Apply the current level's floor roughness, wind pushes, and slope to the environment."""
        stage = self.current_stage
        env.set_curriculum_params(
            push_force_std=stage.push_force_std,
            terrain_roughness=stage.terrain_roughness,
            slope_angle_deg=stage.slope_angle_deg,
            penalty_scaler=stage.reward_scale_penalty
        )
        logger.info(
            f"[CurriculumManager] Stage {stage.stage_id}: '{stage.name}' applied to environment. "
            f"(PushStd: {stage.push_force_std}, Roughness: {stage.terrain_roughness}, Slope: {stage.slope_angle_deg}°)"
        )

    def update_curriculum(self, mean_episodic_return: float, env: BipedalWalkerCustomEnv) -> bool:
        """
        Check test scores: If average score beat target threshold, advance to the next level!
        
        Args:
            mean_episodic_return (float): Average score earned over test games.
            env (BipedalWalkerCustomEnv): Environment to apply new level settings to.
            
        Returns:
            bool: True if robot leveled up, False if staying on current level.
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            # Already at maximum level (Level 2)
            return False
            
        target_return = self.current_stage.target_return
        # If average score is higher than required level score -> LEVEL UP!
        if mean_episodic_return >= target_return:
            self.current_stage_idx += 1        # Move to next stage index
            new_stage = self.current_stage
            logger.info(
                f"[Curriculum Advancement!] Mean Return {mean_episodic_return:.2f} >= Target {target_return:.2f}. "
                f"Promoting to Stage {new_stage.stage_id}: '{new_stage.name}'"
            )
            self.apply_stage_to_env(env)       # Update environment physics with new level rules
            return True
            
        return False
