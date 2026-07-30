import logging
from typing import Optional

from config import CurriculumConfig, CurriculumStage
from envs.bipedal_walker import BipedalWalkerCustomEnv

logger = logging.getLogger(__name__)


class CurriculumManager:
    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self.current_stage_idx = 0
        self.stages = self.config.stages

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    def apply_stage_to_env(self, env: BipedalWalkerCustomEnv) -> None:
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

    def set_stage(self, stage_idx: int) -> None:
        """Select a curriculum stage explicitly for evaluation or visualization."""
        if not 0 <= stage_idx < len(self.stages):
            raise ValueError(f"stage_idx must be in [0, {len(self.stages) - 1}], got {stage_idx}")
        self.current_stage_idx = stage_idx

    def update_curriculum(self, mean_episodic_return: float, env: BipedalWalkerCustomEnv) -> bool:
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        target_return = self.current_stage.target_return
        if mean_episodic_return >= target_return:
            self.current_stage_idx += 1
            new_stage = self.current_stage
            logger.info(
                f"[Curriculum Advancement!] Mean Return {mean_episodic_return:.2f} >= Target {target_return:.2f}. "
                f"Promoting to Stage {new_stage.stage_id}: '{new_stage.name}'"
            )
            self.apply_stage_to_env(env)
            return True

        return False

