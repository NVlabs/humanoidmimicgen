from pathlib import Path
import time

import numpy as np

from humanoidmimicgen.wbc.main.constants import DEFAULT_BASE_HEIGHT
from humanoidmimicgen.wbc.policy.g1_homie_policy import G1HomiePolicy, G1HomiePolicyV2
from humanoidmimicgen.wbc.policy.identity_policy import IdentityPolicy
from humanoidmimicgen.wbc.policy.interpolation_policy import InterpolationPolicy

from .g1_decoupled_whole_body_policy import G1DecoupledWholeBodyPolicy

WBC_VERSIONS = ["v1", "v2", "homie", "homie_v2", "homie_v2_grav_comp", "homie_v2_grav_comp_tuned", "homie_v2_grav_comp_tuned_legs", "locomotion_z", "local_tracking"]


def get_wbc_policy(
    robot_type,
    robot_model,
    wbc_config,
    init_time=time.monotonic(),
):
    current_upper_body_pose = robot_model.get_initial_upper_body_pose()

    if robot_type == "g1":
        upper_body_policy_type = wbc_config.get("upper_body_policy_type", "interpolation")
        if upper_body_policy_type == "identity":
            upper_body_policy = IdentityPolicy(
                init_values={
                    "target_upper_body_pose": current_upper_body_pose,
                    "base_height_command": np.array([DEFAULT_BASE_HEIGHT]),
                }
            )
        else:
            upper_body_policy = InterpolationPolicy(
                init_time=init_time,
                init_values={
                    "target_upper_body_pose": current_upper_body_pose,
                    "base_height_command": np.array([DEFAULT_BASE_HEIGHT]),
                },
                max_change_rate=wbc_config["upper_body_max_joint_speed"],
            )

        lower_body_policy_type = wbc_config.get("VERSION", "default")
        if lower_body_policy_type in ["homie", "homie_v2", "homie_v2_grav_comp", "homie_v2_grav_comp_tuned", "homie_v2_grav_comp_tuned_legs"]:
            wbc_root = Path(__file__).resolve().parents[1]
            homie_config = str(wbc_root / wbc_config["HOMIE_CONFIG"])
            if lower_body_policy_type == "homie":
                lower_body_policy = G1HomiePolicy(
                    robot_model=robot_model,
                    config=homie_config,
                    model_path=wbc_config["model_path"],
                )
            elif lower_body_policy_type in ["homie_v2", "homie_v2_grav_comp", "homie_v2_grav_comp_tuned", "homie_v2_grav_comp_tuned_legs"]:
                lower_body_policy = G1HomiePolicyV2(
                    robot_model=robot_model,
                    config=homie_config,
                    model_path=wbc_config["model_path"],
                )
        else:
            raise ValueError(f"Unsupported WBC version for local replay: {lower_body_policy_type}")

        wbc_policy = G1DecoupledWholeBodyPolicy(
            robot_model=robot_model,
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
        )
    else:
        raise ValueError(f"Unsupported robot type for local replay: {robot_type}")
    return wbc_policy
