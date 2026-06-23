# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local WBC runtime helpers for HumanoidMimicGen WBC-goal playback."""

from __future__ import annotations

import os


def get_robot_type_and_model(robot: str, enable_waist_ik: bool = False):
    from humanoidmimicgen.wbc.robot_model.instantiation import get_robot_type_and_model

    return get_robot_type_and_model(robot, enable_waist_ik)


def get_env(config, **kwargs):
    sim_timestep = 1.0 / float(config.sim_frequency)
    os.environ["HMG_SIMULATION_TIMESTEP"] = str(sim_timestep)

    from humanoidmimicgen.wbc.envs.robocasa.sync_env import G1SyncEnv
    from humanoidmimicgen.wbc.envs.robocasa.utils import robocasa_env
    from humanoidmimicgen.wbc.envs.robocasa.utils.controller_utils import (
        update_robosuite_controller_configs,
    )

    robocasa_env.macros.SIMULATION_TIMESTEP = sim_timestep

    robot_type, _ = get_robot_type_and_model(config.robot, enable_waist_ik=config.enable_waist)
    env_name = f"groot2_{robot_type}/{config.task_name}_{config.robot}_Env"
    print("Instantiating environment:", env_name)
    controller_configs = update_robosuite_controller_configs(
        robot=config.robot,
        wbc_version=config.wbc_version,
    )
    kwargs.update(
        {
            "ik_indicator": config.ik_indicator,
            "control_freq": config.control_frequency,
            "renderer": config.renderer,
            "controller_configs": controller_configs,
            "enable_waist": config.enable_waist,
        }
    )
    if robot_type != "g1":
        raise ValueError(f"Unsupported robot type for HumanoidMimicGen replay: {robot_type}")
    return G1SyncEnv(
        env_name=env_name,
        **kwargs,
    )


def get_policies(config, robot_type: str, robot_model, activate_keyboard_listener: bool = True):
    from humanoidmimicgen.wbc.policy import get_wbc_policy

    wbc_config = config.load_wbc_yaml()
    wbc_policy = get_wbc_policy(robot_type, robot_model, wbc_config, init_time=0.0)
    wbc_policy.activate_policy()
    return wbc_policy, None, None
