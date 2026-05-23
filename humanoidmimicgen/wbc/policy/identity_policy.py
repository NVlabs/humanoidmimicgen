from copy import deepcopy
from typing import Optional

import gymnasium as gym
import numpy as np

from humanoidmimicgen.wbc.base.policy import Policy


class IdentityPolicy(Policy):
    def __init__(self, init_values: dict[str, np.ndarray]):
        self.goal = init_values

    def get_action(self, time: Optional[float] = None) -> dict[str, any]:
        return self.goal

    def set_goal(self, goal: dict[str, any]) -> None:
        _goal = deepcopy(goal)
        _goal.pop("interpolation_garbage_collection_time", None)
        _goal.pop("target_time", None)
        self.goal.update(_goal)

    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict()

    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict()
