# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .g1 import instantiate_g1_robot_model


def get_robot_type_and_model(robot: str, enable_waist_ik: bool = False):
    """Get the robot type from the robot name."""
    if robot.lower().startswith("g1"):
        if "FixedLowerBody" in robot or "FloatingBody" in robot:
            waist_location = "upper_body"
        elif enable_waist_ik:
            waist_location = "lower_and_upper_body"
        else:
            waist_location = "lower_body"
        return "g1", instantiate_g1_robot_model(waist_location=waist_location)
    else:
        raise ValueError(f"Unsupported robot name for HumanoidMimicGen replay: {robot}")
