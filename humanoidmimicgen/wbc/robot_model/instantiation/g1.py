import os
from pathlib import Path
from typing import Literal

from humanoidmimicgen.wbc.robot_model.robot_model import RobotModel
from humanoidmimicgen.wbc.robot_model.supplemental_info.g1.g1_supplemental_info import (
    G1SupplementalInfo,
    G1SupplementalInfoWaistLowerAndUpperBody,
    G1SupplementalInfoWaistUpperBody,
)


def instantiate_g1_robot_model(
    waist_location: Literal["lower_body", "upper_body"] = "lower_body",
):
    """
    Instantiate a G1 robot model with configurable waist location.

    Args:
        waist_location: Whether to put waist in "lower_body" (default G1 behavior),
                        "upper_body" (waist controlled with arms/manipulation via IK),
                        or "lower_and_upper_body" (waist reference from arms/manipulation
                        via IK then passed to lower body policy)

    Returns:
        RobotModel: Configured G1 robot model
    """
    wbc_root = Path(__file__).resolve().parents[2]
    robot_model_config = {
        "asset_path": os.path.join(wbc_root, "robot_model/model_data/g1"),
        "urdf_path": os.path.join(
            wbc_root, "robot_model/model_data/g1/g1_29dof_with_hand.urdf"
        ),
    }
    assert waist_location in [
        "lower_body",
        "upper_body",
        "lower_and_upper_body",
    ], f"Invalid waist_location: {waist_location}. Must be 'lower_body' or 'upper_body' or 'lower_and_upper_body'"
    # Choose supplemental info based on waist location preference
    if waist_location == "lower_body":
        robot_model_supplemental_info = G1SupplementalInfo()
    elif waist_location == "upper_body":
        robot_model_supplemental_info = G1SupplementalInfoWaistUpperBody()
    elif waist_location == "lower_and_upper_body":
        robot_model_supplemental_info = G1SupplementalInfoWaistLowerAndUpperBody()

    robot_model = RobotModel(
        robot_model_config["urdf_path"],
        robot_model_config["asset_path"],
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model
