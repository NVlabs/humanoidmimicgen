# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from robosuite.models.grippers import PandaGripper

from humanoidmimicgen.locomanipulation.models.grippers import G1ThreeFingerLeftHand, G1ThreeFingerRightHand
from humanoidmimicgen.locomanipulation.models.robots.manipulators import (
    G1,
    G1ArmsOnly,
    G1ArmsOnlyFloating,
    G1FixedBase,
    G1FixedLowerBody,
    G1FloatingBody,
    G1FloatingBodyWithVertical,
)


def unformat_gripper_space(gripper, formatted_action):
    """Return gripper actions in the actuator order used by the XML model."""
    formatted_action = np.asarray(formatted_action)
    if isinstance(gripper, (G1ThreeFingerLeftHand, G1ThreeFingerRightHand, PandaGripper)):
        return formatted_action
    raise TypeError(f"Unsupported gripper type for local replay: {type(gripper).__name__}")


GROOT_LOCOMANIP_ENVS_G1 = {
    "G1": "g1",
    "G1ArmsOnly": "g1",
    "G1ArmsOnlyFloating": "g1",
    "G1FixedLowerBody": "g1",
    "G1FixedBase": "g1",
    "G1FloatingBody": "g1",
    "G1FloatingBodyWithVertical": "g1",
}

GROOT2_ENVS_ROBOTS = {**GROOT_LOCOMANIP_ENVS_G1}

__all__ = [
    "G1",
    "G1ArmsOnly",
    "G1ArmsOnlyFloating",
    "G1FixedBase",
    "G1FixedLowerBody",
    "G1FloatingBody",
    "G1FloatingBodyWithVertical",
    "GROOT2_ENVS_ROBOTS",
    "unformat_gripper_space",
]
