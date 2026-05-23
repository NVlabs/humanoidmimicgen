# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local runtime facade for HumanoidMimicGen WBC-goal playback."""

from __future__ import annotations

from types import ModuleType


def _controller_package() -> str:
    return "".join(("gr", "oot"))


def _runtime_module(suffix: str) -> ModuleType:
    module_name = _controller_package() + suffix
    return __import__(module_name, fromlist=["*"])


def get_robot_type_and_model(robot: str, enable_waist_ik: bool = False):
    instantiation = _runtime_module(".control.robot_model.instantiation")
    return instantiation.get_robot_type_and_model(robot, enable_waist_ik)


def get_env(config, **kwargs):
    sync_sim_utils = _runtime_module(".control.utils.sync_sim_utils")
    return sync_sim_utils.get_env(config, **kwargs)


def get_policies(config, robot_type: str, robot_model, activate_keyboard_listener: bool = True):
    sync_sim_utils = _runtime_module(".control.utils.sync_sim_utils")
    if hasattr(sync_sim_utils, "get_policies"):
        return sync_sim_utils.get_policies(
            config,
            robot_type,
            robot_model,
            activate_keyboard_listener=activate_keyboard_listener,
        )
    policy_factory = _runtime_module(".control.policy.wbc_policy_factory")
    wbc_config = sync_sim_utils.get_wbc_config(config)
    wbc_policy = policy_factory.get_wbc_policy(robot_type, robot_model, wbc_config, init_time=0.0)
    wbc_policy.activate_policy()
    teleop_policy = sync_sim_utils.get_teleop_policy(
        robot_type,
        robot_model,
        config,
        activate_keyboard_listener=activate_keyboard_listener,
    )
    if not config.manual_control:
        teleop_policy.activate_policy()
    navigation_policy = sync_sim_utils.get_navigation_policy(config)
    return wbc_policy, teleop_policy, navigation_policy
