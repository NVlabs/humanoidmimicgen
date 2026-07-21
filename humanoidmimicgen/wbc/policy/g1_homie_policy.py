# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
import torch

from humanoidmimicgen.wbc.base.policy import Policy
from humanoidmimicgen.wbc.utils.homie_utils import get_gravity_orientation, load_config


class G1HomiePolicyV2(Policy):
    """G1 lower-body policy backed by the Homie stand/walk ONNX pair."""

    def __init__(self, robot_model, config: str, model_path: str):
        self.config, _ = load_config(config)
        self.robot_model = robot_model
        self.use_teleop_policy_cmd = False

        wbc_root = Path(__file__).resolve().parents[1]
        sim2mujoco_root = wbc_root / "external_dependencies" / "sim2mujoco"
        stand_path, walk_path = model_path.split(",")
        model_root = sim2mujoco_root / "resources" / "robots" / "g1"
        self.stand_policy = self.load_onnx_policy(str(model_root / stand_path))
        self.walk_policy = self.load_onnx_policy(str(model_root / walk_path))

        self.observation = None
        self.obs_tensor = None
        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
        self.obs_buffer = np.zeros(self.config["num_obs"], dtype=np.float32)

        self.use_policy_action = False
        self.action = np.zeros(self.config["num_actions"], dtype=np.float32)
        self.cmd = self.config["cmd_init"].copy()
        self.height_cmd = self.config["height_cmd"]
        self.freq_cmd = self.config["freq_cmd"]
        self.roll_cmd = self.config["rpy_cmd"][0]
        self.pitch_cmd = self.config["rpy_cmd"][1]
        self.yaw_cmd = self.config["rpy_cmd"][2]

    def load_onnx_policy(self, model_path: str):
        print(f"Loading ONNX policy from {model_path}")
        model = ort.InferenceSession(model_path)

        def run_inference(input_tensor):
            ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device="cpu")

        print(f"Successfully loaded ONNX policy from {model_path}")
        return run_inference

    def compute_observation(self, observation: dict[str, Any]) -> tuple[np.ndarray, int]:
        body_indices = self.robot_model.get_joint_group_indices("body")
        n_joints = len(body_indices)

        qj = observation["q"][body_indices].copy()
        dqj = observation["dq"][body_indices].copy()
        quat = observation["floating_base_pose"][3:7].copy()
        omega = observation["floating_base_vel"][3:6].copy()

        if len(self.config["default_angles"]) < n_joints:
            default_angles = np.zeros(n_joints, dtype=np.float32)
            default_angles[: len(self.config["default_angles"])] = self.config["default_angles"]
        else:
            default_angles = self.config["default_angles"][:n_joints]

        qj_scaled = (qj - default_angles) * self.config["dof_pos_scale"]
        dqj_scaled = dqj * self.config["dof_vel_scale"]
        omega_scaled = omega * self.config["ang_vel_scale"]
        gravity_orientation = get_gravity_orientation(quat)

        single_obs_dim = 13 + 2 * n_joints + self.config["num_actions"]
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)
        single_obs[0:3] = self.cmd[:3] * self.config["cmd_scale"]
        single_obs[3:4] = np.array([self.height_cmd])
        single_obs[4:7] = np.array([self.roll_cmd, self.pitch_cmd, self.yaw_cmd])
        single_obs[7:10] = omega_scaled
        single_obs[10:13] = gravity_orientation
        single_obs[13 : 13 + n_joints] = qj_scaled
        single_obs[13 + n_joints : 13 + 2 * n_joints] = dqj_scaled
        single_obs[13 + 2 * n_joints : 13 + 2 * n_joints + len(self.action)] = self.action
        return single_obs, single_obs_dim

    def set_observation(self, observation: dict[str, Any]):
        self.observation = observation
        single_obs, _ = self.compute_observation(observation)
        self.obs_history.append(single_obs)
        while len(self.obs_history) < self.config["obs_history_len"]:
            self.obs_history.appendleft(np.zeros_like(single_obs))

        single_obs_dim = len(single_obs)
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * single_obs_dim
            self.obs_buffer[start_idx : start_idx + single_obs_dim] = hist_obs

        self.obs_tensor = torch.from_numpy(self.obs_buffer).unsqueeze(0)
        assert self.obs_tensor.shape[1] == self.config["num_obs"]

    def set_goal(self, goal: dict[str, Any]):
        if goal:
            self.use_teleop_policy_cmd = True

        if "navigate_cmd" in goal:
            nav_cmd = goal["navigate_cmd"]
            if isinstance(nav_cmd, list):
                nav_cmd = np.array(nav_cmd)
            self.cmd = nav_cmd[0] if nav_cmd.ndim > 1 else nav_cmd

        if goal.get("toggle_policy_action"):
            self.use_policy_action = not self.use_policy_action

    def get_action(
        self,
        time: Optional[float] = None,
        arms_target_pose: Optional[np.ndarray] = None,
        base_height_command: Optional[np.ndarray] = None,
        torso_orientation_rpy: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        if self.obs_tensor is None:
            raise ValueError("No observation set. Call set_observation() first.")

        if base_height_command is not None and self.use_teleop_policy_cmd:
            self.height_cmd = (
                base_height_command[0]
                if isinstance(base_height_command, list)
                else base_height_command
            )

        if torso_orientation_rpy is not None and self.use_teleop_policy_cmd:
            self.roll_cmd = torso_orientation_rpy[0]
            self.pitch_cmd = torso_orientation_rpy[1]
            self.yaw_cmd = torso_orientation_rpy[2]

        with torch.no_grad():
            policy = self.stand_policy if np.linalg.norm(self.cmd) < 0.05 else self.walk_policy
            self.action = policy(self.obs_tensor).detach().numpy().squeeze()

        if self.use_policy_action:
            cmd_q = self.action * self.config["action_scale"] + self.config["default_angles"]
        else:
            cmd_q = self.observation["q"][self.robot_model.get_joint_group_indices("lower_body")]

        cmd_dq = np.zeros(self.config["num_actions"])
        cmd_tau = np.zeros(self.config["num_actions"])
        return {"body_action": (cmd_q, cmd_dq, cmd_tau)}

    def handle_keyboard_button(self, key):
        if key == "]":
            self.use_policy_action = True
        elif key == "o":
            self.use_policy_action = False
        elif key == "w":
            self.cmd[0] += 0.2
        elif key == "s":
            self.cmd[0] -= 0.2
        elif key == "a":
            self.cmd[1] += 0.2
        elif key == "d":
            self.cmd[1] -= 0.2
        elif key == "q":
            self.cmd[2] += 0.2
        elif key == "e":
            self.cmd[2] -= 0.2
        elif key == "z":
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0
        elif key == "1":
            self.height_cmd += 0.1
        elif key == "2":
            self.height_cmd -= 0.1
        elif key == "n":
            self.freq_cmd = max(1.0, self.freq_cmd - 0.1)
        elif key == "m":
            self.freq_cmd = min(2.0, self.freq_cmd + 0.1)
        elif key == "3":
            self.roll_cmd -= np.deg2rad(10)
        elif key == "4":
            self.roll_cmd += np.deg2rad(10)
        elif key == "5":
            self.pitch_cmd -= np.deg2rad(10)
        elif key == "6":
            self.pitch_cmd += np.deg2rad(10)
        elif key == "7":
            self.yaw_cmd -= np.deg2rad(10)
        elif key == "8":
            self.yaw_cmd += np.deg2rad(10)

        if key:
            print("--------------------------------")
            print(f"Linear velocity command: {self.cmd}")
            print(f"Base height command: {self.height_cmd}")
            print(f"Use policy action: {self.use_policy_action}")
            print(f"roll deg angle: {np.rad2deg(self.roll_cmd)}")
            print(f"pitch deg angle: {np.rad2deg(self.pitch_cmd)}")
            print(f"yaw deg angle: {np.rad2deg(self.yaw_cmd)}")
            print(f"Gait frequency: {self.freq_cmd}")
