from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pinocchio import rpy

from humanoidmimicgen.wbc.base.policy import Policy
from humanoidmimicgen.wbc_constants import DEFAULT_BASE_HEIGHT

WBC_VERSIONS = [
    "homie_v2",
    "homie_v2_grav_comp",
    "homie_v2_grav_comp_tuned",
    "homie_v2_grav_comp_tuned_legs",
]


class IdentityPolicy(Policy):
    def __init__(self, init_values: dict[str, np.ndarray]):
        self.goal = init_values

    def get_action(self, time: Optional[float] = None) -> dict[str, Any]:
        return self.goal

    def set_goal(self, goal: dict[str, Any]) -> None:
        _goal = deepcopy(goal)
        _goal.pop("interpolation_garbage_collection_time", None)
        _goal.pop("target_time", None)
        self.goal.update(_goal)


class G1DecoupledWholeBodyPolicy(Policy):
    def __init__(
        self,
        robot_model,
        lower_body_policy: Policy,
        upper_body_policy: Policy,
    ):
        self.robot_model = robot_model
        self.lower_body_policy = lower_body_policy
        self.upper_body_policy = upper_body_policy

    def set_observation(self, observation):
        self.lower_body_policy.set_observation(observation)

    def set_goal(self, goal):
        upper_body_goal = {}
        lower_body_goal = {}

        for key in (
            "target_upper_body_pose",
            "base_height_command",
            "target_time",
            "interpolation_garbage_collection_time",
        ):
            if key in goal:
                upper_body_goal[key] = goal[key]

        for key in (
            "toggle_stand_command",
            "toggle_policy_action",
            "navigate_cmd",
        ):
            if key in goal:
                lower_body_goal[key] = goal[key]

        self.upper_body_policy.set_goal(upper_body_goal)
        self.lower_body_policy.set_goal(lower_body_goal)

    def get_action(self, time: Optional[float] = None):
        lower_body_indices = self.robot_model.get_joint_group_indices("lower_body")
        upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")

        q = np.zeros(self.robot_model.num_dofs)

        upper_body_action = self.upper_body_policy.get_action(time)
        q[upper_body_indices] = upper_body_action["target_upper_body_pose"]
        q_arms = q[self.robot_model.get_joint_group_indices("arms")]
        base_height_command = upper_body_action.get("base_height_command", None)

        self.robot_model.cache_forward_kinematics(q, auto_clip=False)
        torso_orientation = self.robot_model.frame_placement("torso_link").rotation
        waist_orientation = self.robot_model.frame_placement("pelvis").rotation
        waist_yaw = np.arctan2(waist_orientation[1, 0], waist_orientation[0, 0])
        waist_yaw_only_rotation = rpy.rpyToMatrix(0, 0, waist_yaw)
        yaw_only_waist_from_torso = waist_yaw_only_rotation.T @ torso_orientation
        torso_orientation_rpy = rpy.matrixToRpy(yaw_only_waist_from_torso)

        lower_body_action = self.lower_body_policy.get_action(
            time, q_arms, base_height_command, torso_orientation_rpy
        )
        q[lower_body_indices] = lower_body_action["body_action"][0][
            : len(lower_body_indices)
        ]
        self.last_action = {"q": q}
        return {"q": q}

    def handle_keyboard_button(self, key):
        try:
            self.lower_body_policy.locomotion_policy.handle_keyboard_button(key)
        except AttributeError:
            self.lower_body_policy.handle_keyboard_button(key)

    def activate_policy(self):
        self.handle_keyboard_button("]")


def get_wbc_policy(
    robot_type,
    robot_model,
    wbc_config,
    init_time=None,
):
    if robot_type != "g1":
        raise ValueError(f"Unsupported robot type for local replay: {robot_type}")

    lower_body_policy_type = wbc_config.get("VERSION", "default")
    if lower_body_policy_type not in WBC_VERSIONS:
        raise ValueError(f"Unsupported WBC version for local replay: {lower_body_policy_type}")

    upper_body_policy = IdentityPolicy(
        init_values={
            "target_upper_body_pose": robot_model.get_initial_upper_body_pose(),
            "base_height_command": np.array([DEFAULT_BASE_HEIGHT]),
        }
    )
    wbc_root = Path(__file__).resolve().parents[1]
    from .g1_homie_policy import G1HomiePolicyV2

    lower_body_policy = G1HomiePolicyV2(
        robot_model=robot_model,
        config=str(wbc_root / wbc_config["HOMIE_CONFIG"]),
        model_path=wbc_config["model_path"],
    )
    return G1DecoupledWholeBodyPolicy(
        robot_model=robot_model,
        upper_body_policy=upper_body_policy,
        lower_body_policy=lower_body_policy,
    )
