# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Robot actuator-control playback helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


class MujocoControlPlaybackEnv:
    """Env-like adapter for stepping recorded MuJoCo actuator commands."""

    def __init__(self, sim: Any, control_indices: list[int], control_freq: float):
        self.sim = sim
        self.control_indices = np.asarray(control_indices, dtype=np.int64)
        self.action_dim = len(control_indices)
        self.steps_per_action = max(1, round(1.0 / (control_freq * self.sim.model.opt.timestep)))

    def step(self, action: np.ndarray) -> tuple[None, float, bool, dict[str, Any]]:
        action = np.asarray(action).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.size}.")

        self.sim.data.ctrl[:] = 0
        self.sim.data.ctrl[self.control_indices] = action
        for _ in range(self.steps_per_action):
            self.sim.step()
        return None, 0.0, False, {}

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def make_mujoco_control_playback_env(
    sim: Any,
    actions: np.ndarray,
    action_names: list[str] | None,
    control_freq: float,
) -> MujocoControlPlaybackEnv:
    control_indices = build_mujoco_control_indices(sim, actions, action_names)
    return MujocoControlPlaybackEnv(sim, control_indices, control_freq)


def build_mujoco_control_indices(
    sim: Any,
    actions: np.ndarray,
    action_names: list[str] | None,
) -> list[int]:
    action_dim = actions.shape[1] if actions.ndim == 2 else 1
    actuator_names = set(sim.model.actuator_names)
    if action_names is None:
        if action_dim != sim.model.nu:
            raise ValueError(
                "Cannot infer MuJoCo control mapping: action metadata is missing and "
                f"action dim {action_dim} != model.nu {sim.model.nu}."
            )
        return list(range(sim.model.nu))

    if len(action_names) != action_dim:
        raise ValueError(
            f"Action metadata has {len(action_names)} names but actions have dim {action_dim}."
        )

    control_indices = []
    missing = []
    for action_name in action_names:
        actuator_name = next(
            (candidate for candidate in action_name_candidates(action_name) if candidate in actuator_names),
            None,
        )
        if actuator_name is None:
            missing.append(action_name)
            continue
        control_indices.append(sim.model.actuator_name2id(actuator_name))
    if missing:
        raise ValueError(f"Could not map action names to MuJoCo actuators: {missing}")
    return control_indices


def action_name_candidates(action_name: str) -> list[str]:
    candidates = [action_name, f"robot0_{action_name}"]
    if action_name.startswith("left_hand_"):
        candidates.append(f"gripper0_left_{action_name}")
    if action_name.startswith("right_hand_"):
        candidates.append(f"gripper0_right_{action_name}")
    return candidates
