"""
Environments package init file.

When you write `from envs import BipedalWalkerCustomEnv`, Python looks here first.
This file tells Python: "These are the things you can import from the 'envs' folder."
"""

from envs.bipedal_walker import BipedalWalkerCustomEnv  # The robot's physics simulation world
from envs.curriculum import CurriculumManager            # The teacher that ramps up difficulty

# __all__ controls exactly what gets exported when someone writes `from envs import *`
__all__ = ["BipedalWalkerCustomEnv", "CurriculumManager"]
