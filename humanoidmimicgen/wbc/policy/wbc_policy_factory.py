from pathlib import Path

import numpy as np

from humanoidmimicgen.wbc.main.constants import DEFAULT_BASE_HEIGHT
from humanoidmimicgen.wbc.policy.g1_homie_policy import G1HomiePolicyV2
from humanoidmimicgen.wbc.policy.identity_policy import IdentityPolicy

from .g1_decoupled_whole_body_policy import G1DecoupledWholeBodyPolicy

WBC_VERSIONS = [
    "homie_v2",
    "homie_v2_grav_comp",
    "homie_v2_grav_comp_tuned",
    "homie_v2_grav_comp_tuned_legs",
]


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
