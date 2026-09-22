# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from copy import deepcopy
import os
from typing import Type
import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import robosuite
from robosuite.environments.base import EnvMeta, REGISTERED_ENVS
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, find_elements
from robosuite.utils.observables import Observable, sensor

import humanoidmimicgen.locomanipulation as locomanipulation
from humanoidmimicgen.locomanipulation.models.scenes import GroundArena
import humanoidmimicgen.locomanipulation.utils.camera_utils as CamUtils


RETAINED_LOCOMANIPULATION_ENV_NAMES = (
    "LMBoxLiftFloor",
    "LMPushButton",
    "LMBoxLift",
    "LMPushShelfForward",
    "LMDrillLift",
    "LMDrillPnP90",
    "LMBoxTableToShelfStaticIndustrial",
    "LMPickDrillFromHolder",
    "LMDrillLiftObstacle",
)

REGISTERED_LOCOMANIPULATION_ENVS = {}


def register_locomanipulation_env(target_class):
    if target_class.__name__ not in RETAINED_LOCOMANIPULATION_ENV_NAMES:
        return
    REGISTERED_LOCOMANIPULATION_ENVS[target_class.__name__] = target_class


class LocoManipulationEnvMeta(EnvMeta):
    """Metaclass for registering the retained loco-manipulation paper tasks."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        if name not in RETAINED_LOCOMANIPULATION_ENV_NAMES and REGISTERED_ENVS.get(name) is cls:
            del REGISTERED_ENVS[name]
        register_locomanipulation_env(cls)
        return cls


class RobotPoseRandomizer:
    @staticmethod
    def set_pose(
        env: "LocoManipulationEnv",
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        yaw_range: tuple[float, float],
    ):
        new_x = env.rng.uniform(*x_range)
        new_y = env.rng.uniform(*y_range)
        new_yaw = env.rng.uniform(*yaw_range)

        if env.robots[0].name == "G1":
            base_offset = env.ROBOT_POS_OFFSETS[env.robots[0].robot_model.__class__.__name__]
            target_pos = np.array([new_x, new_y, base_offset[2]], dtype=float)
            quat = np.zeros(4, dtype=float)
            mujoco.mju_euler2Quat(quat, np.array([0.0, 0.0, new_yaw]), "xyz")
            base_freejoint = f"{env.robots[0].robot_model.naming_prefix}base"
            if base_freejoint in env.sim.model.joint_names:
                env.sim.data.set_joint_qpos(base_freejoint, np.concatenate([target_pos, quat]))
            else:
                warnings.warn(f"Base joint {base_freejoint} not found in the model.")
            return

        base_joint_pos = np.array([new_x, new_y, new_yaw])
        base_joint_names = [
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
            "mobilebase0_joint_mobile_yaw",
        ]
        for i, base_joint_name in enumerate(base_joint_names):
            if base_joint_name not in env.sim.model.joint_names:
                warnings.warn(
                    f"Base joint {base_joint_name} not found in the model. "
                    f"Skipping randomization of {base_joint_name}."
                )
            else:
                env.sim.data.set_joint_qpos(base_joint_name, base_joint_pos[i])

    @staticmethod
    def set_arm(env: ManipulationEnv, elbow_qpos: float, shoulder_pitch_qpos: float):
        """Reinitialize the G1 arm configuration."""
        robot = env.robots[0]
        if "G1" not in robot.name:
            return

        for joint_name, pos_idx in zip(robot.robot_joints, robot._ref_joint_pos_indexes):
            if "elbow" in joint_name:
                env.sim.data.qpos[pos_idx] = elbow_qpos
            elif "shoulder_pitch" in joint_name:
                env.sim.data.qpos[pos_idx] = shoulder_pitch_qpos


class LocoManipulationEnv(ManipulationEnv, metaclass=LocoManipulationEnvMeta):
    """Base RoboSuite manipulation env used by the retained paper tasks."""

    MUJOCO_ARENA_CLS: Type[Arena] = GroundArena

    ROBOT_POS_OFFSETS: dict[str, list[float]] = {
        "PandaOmron": [0, 0, 0],
        "GR1FloatingBody": [0, 0, 0.97],
        "GR1": [0, 0, 0.97],
        "GR1FixedLowerBody": [0, 0, 0.97],
        "GR1FixedLowerBodyInspireHands": [0, 0, 0.97],
        "GR1FixedLowerBodyFourierHands": [0, 0, 0.97],
        "GR1ArmsOnly": [0, 0, 0.97],
        "GR1ArmsOnlyInspireHands": [0, 0, 0.97],
        "GR1ArmsOnlyFourierHands": [0, 0, 0.97],
        "GR1ArmsAndWaistFourierHands": [0, 0, 0.97],
        "G1": [0, 0, 0.793],
        "G1FixedBase": [0, 0, 0.750],
        "G1FixedLowerBody": [0, 0, 0.750],
        "G1ArmsOnly": [0, 0, 0.750],
        "G1ArmsOnlyFloating": [0, 0, 0.750],
        "G1FloatingBody": [0, 0, 0.750],
        "G1FloatingBodyWithVertical": [0, 0, 0.750],
    }

    def __init__(
        self, translucent_robot: bool = False, use_object_obs: bool = False, *args, **kwargs
    ):
        self.mujoco_objects = []
        super().__init__(*args, **kwargs)
        self.translucent_robot = translucent_robot

    def _reset_internal(self):
        """Match the seeded robot-reset contract used to produce recorded demos.

        Public RoboSuite 1.5.1 samples robot initialization noise from NumPy's
        process-global RNG. The data-generation environment sampled from the
        env-owned Generator instead, before scene placement. Preserve that ordering
        locally so a seed fixes both the robot configuration and subsequent object
        poses.
        """
        sampled_qpos = []
        if not self.deterministic_reset:
            for robot in self.robots:
                qpos = np.asarray(robot.init_qpos, dtype=float).copy()
                noise = robot.initialization_noise
                magnitude = noise["magnitude"]
                if noise["type"] == "gaussian":
                    qpos += self.rng.normal(size=qpos.shape) * magnitude
                elif noise["type"] == "uniform":
                    qpos += self.rng.uniform(-1.0, 1.0, qpos.shape) * magnitude
                else:
                    raise ValueError(f"Unsupported robot initialization noise: {noise['type']}")
                sampled_qpos.append(qpos)

        # Public RoboSuite also draws initialization noise from NumPy's global
        # RNG, even at zero magnitude. Our env-owned samples above replace that
        # noise, so those discarded draws must not perturb the caller's RNG.
        global_rng_state = np.random.get_state()
        try:
            super()._reset_internal()
        finally:
            np.random.set_state(global_rng_state)
        for robot, qpos in zip(self.robots, sampled_qpos):
            self.sim.data.qpos[robot._ref_joint_pos_indexes] = qpos

    def _load_model(self):
        super()._load_model()

        self.mujoco_arena = self.MUJOCO_ARENA_CLS()
        self.mujoco_arena.set_origin([0, 0, 0])
        self.set_cameras()

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.mujoco_objects,
        )

        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(self.ROBOT_POS_OFFSETS[robot_model.__class__.__name__])

    def set_cameras(self):
        self._cam_configs = deepcopy(CamUtils.CAM_CONFIGS)

        for robot in self.robots:
            if hasattr(robot.robot_model, "get_camera_configs"):
                self._cam_configs.update(robot.robot_model.get_camera_configs())

        for cam_name, cam_cfg in self._cam_configs.items():
            if cam_cfg.get("parent_body", None) is not None:
                continue
            self.mujoco_arena.set_camera(
                camera_name=cam_name,
                pos=cam_cfg["pos"],
                quat=cam_cfg["quat"],
                camera_attribs=cam_cfg.get("camera_attribs", None),
            )

        self.mujoco_arena.set_camera(
            camera_name="egoview",
            pos=[0.078, 0, 1.308],
            quat=[0.66491268, 0.24112495, -0.24112507, -0.66453637],
            camera_attribs=dict(fovy="90"),
        )

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

        visual_geom_names = []
        for robot in self.robots:
            visual_geom_names += robot.robot_model.visual_geoms

        for name in visual_geom_names:
            rgba = self.sim.model.geom_rgba[self.sim.model.geom_name2id(name)]
            rgba[-1] = 0.10 if self.translucent_robot else 1.0

    def reward(self, action=None):
        return 1.0 if self._check_success() else 0

    def _check_success(self):
        return False

    def edit_model_xml(self, xml_str):
        xml_str = super().edit_model_xml(xml_str)

        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        actuator = root.find("actuator")
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        robosuite_assets_root = os.path.join(os.path.split(robosuite.__file__)[0], "models", "assets")
        locomanip_assets_root = locomanipulation.models.assets_root
        try:
            import robosuite_models

            robosuite_models_assets_root = os.path.join(
                os.path.dirname(robosuite_models.__file__), "assets"
            )
        except ImportError:
            robosuite_models_assets_root = None

        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None or "models/assets" not in old_path:
                continue

            old_path_split = old_path.split("/")
            if "/robosuite/" in old_path:
                check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite"]
                ind = max(check_lst)
                new_path = "/".join(
                    os.path.split(robosuite.__file__)[0].split("/") + old_path_split[ind + 1 :]
                )
            elif "robocasa" in old_path_split:
                suffix = old_path.split("models/assets", maxsplit=1)[1].lstrip("/")
                new_path = os.path.join(locomanip_assets_root, suffix)
            else:
                suffix = old_path.split("models/assets", maxsplit=1)[1].lstrip("/")
                locomanip_candidate = os.path.join(locomanip_assets_root, suffix)
                robosuite_candidate = os.path.join(robosuite_assets_root, suffix)
                new_path = (
                    locomanip_candidate
                    if os.path.exists(locomanip_candidate)
                    else robosuite_candidate
                )

            if (
                robosuite_models_assets_root is not None
                and not os.path.exists(new_path)
                and "models/assets" in new_path
            ):
                suffix = new_path.split("models/assets", maxsplit=1)[1].lstrip("/")
                robosuite_models_candidate = os.path.join(robosuite_models_assets_root, suffix)
                if os.path.exists(robosuite_models_candidate):
                    new_path = robosuite_models_candidate

            elem.set("file", new_path)

        for cam_name, cam_config in self._cam_configs.items():
            parent_body = cam_config.get("parent_body", None)
            cam_root = worldbody
            if parent_body is not None:
                cam_root = find_elements(root=worldbody, tags="body", attribs={"name": parent_body})
                if cam_root is None:
                    continue

            cam = find_elements(root=cam_root, tags="camera", attribs={"name": cam_name})
            if cam is None:
                old_cam = find_elements(root=worldbody, tags="camera", attribs={"name": cam_name})
                if old_cam is not None:
                    continue
                cam = ET.Element("camera")
                cam.set("mode", "fixed")
                cam.set("name", cam_name)
                cam_root.append(cam)

            cam.set("pos", array_to_string(cam_config["pos"]))
            cam.set("quat", array_to_string(cam_config["quat"]))
            for key, value in cam_config.get("camera_attribs", {}).items():
                cam.set(key, value)

        if find_elements(root=worldbody, tags="site", attribs={"name": "robot0_imu"}) is None:
            parent_map = {child: parent for parent in root.iter() for child in parent}
            for site_name in ("robot0_imu_in_torso", "robot0_imu_in_pelvis"):
                site = find_elements(root=worldbody, tags="site", attribs={"name": site_name})
                if site is None:
                    continue
                site_alias = deepcopy(site)
                site_alias.set("name", "robot0_imu")
                parent_map[site].append(site_alias)
                break

        self._add_site_aliases(root, worldbody)
        self._rename_legacy_mobile_base_elements(worldbody, actuator)
        return ET.tostring(root).decode("utf8")

    def _add_site_aliases(self, root, worldbody):
        site_aliases = {
            "mobilebase0_center": ("nullbase0_center",),
            "robot0_head": ("robot0_imu_in_torso", "robot0_imu"),
        }
        parent_map = {child: parent for parent in root.iter() for child in parent}
        for alias_name, source_names in site_aliases.items():
            if find_elements(root=worldbody, tags="site", attribs={"name": alias_name}) is not None:
                continue
            for source_name in source_names:
                site = find_elements(root=worldbody, tags="site", attribs={"name": source_name})
                if site is None:
                    continue
                site_alias = deepcopy(site)
                site_alias.set("name", alias_name)
                parent_map[site].append(site_alias)
                break

        body_site_aliases = {
            "robot0_l_foot": "robot0_left_ankle_roll_link",
            "robot0_r_foot": "robot0_right_ankle_roll_link",
        }
        for alias_name, body_name in body_site_aliases.items():
            if find_elements(root=worldbody, tags="site", attribs={"name": alias_name}) is not None:
                continue
            body = find_elements(root=worldbody, tags="body", attribs={"name": body_name})
            if body is not None:
                body.append(ET.Element("site", attrib={"name": alias_name, "pos": "0 0 0"}))

    @staticmethod
    def _rename_legacy_mobile_base_elements(worldbody, actuator):
        for elem in find_elements(
            root=worldbody, tags=["geom", "site", "body", "joint"], return_first=False
        ):
            name = elem.get("name")
            if name and name.startswith("base0_"):
                elem.set("name", "mobilebase0_" + name[6:])

        for elem in find_elements(
            root=actuator,
            tags=["velocity", "position", "motor", "general"],
            return_first=False,
        ):
            name = elem.get("name")
            joint = elem.get("joint")
            if name and name.startswith("base0_"):
                elem.set("name", "mobilebase0_" + name[6:])
            if joint and joint.startswith("base0_"):
                elem.set("joint", "mobilebase0_" + joint[6:])

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {}

    def _reset_observables(self):
        if self.hard_reset:
            self._observables = self._setup_observables()

        disabled_sensors = [
            "base_to_left_eef_pos",
            "base_to_left_eef_quat",
            "base_to_left_eef_quat_site",
            "base_to_right_eef_pos",
            "base_to_right_eef_quat",
            "base_to_right_eef_quat_site",
        ]
        for name in disabled_sensors:
            for robot in self.robots:
                robot_name_prefix = robot.robot_model.naming_prefix
                obs_name = f"{robot_name_prefix}{name}"
                if obs_name in self._observables:
                    self._observables[obs_name].set_enabled(False)
                    self._observables[obs_name].set_active(False)

    def get_state(self):
        return {"states": self.sim.get_state().flatten()}


class GroundOnly(LocoManipulationEnv):
    pass


@sensor(modality="object")
def world_pose_in_gripper(obs_cache):
    return np.eye(4)


def robot_base_to_eef_pos(robot_prefix: str, gripper: str, eef_site_name: str):
    @sensor(modality=robot_prefix + "proprio")
    def base_to_eef_pos(obs_cache):
        return obs_cache[f"{robot_prefix}base_to_{gripper}_eef_quat"][:3, 3]

    return base_to_eef_pos


def robot_base_to_eef_quat(robot_prefix: str, gripper: str, eef_site_name: str):
    @sensor(modality=robot_prefix + "proprio")
    def base_to_eef_quat(obs_cache):
        return obs_cache[f"{robot_prefix}base_to_{gripper}_eef_quat"][:3, :3].ravel()

    return base_to_eef_quat


def robot_base_to_eef_quat_site(robot_prefix: str, gripper: str, eef_site_name: str):
    @sensor(modality=robot_prefix + "proprio")
    def base_to_eef_quat_site(obs_cache):
        return obs_cache[f"{robot_prefix}base_to_{gripper}_eef_quat"][:3, :3].ravel()

    return base_to_eef_quat_site


def make_robot_base_to_eef_observables(robot_prefix: str, gripper: str, eef_site_name: str):
    return {
        f"{robot_prefix}base_to_{gripper}_eef_pos": Observable(
            name=f"{robot_prefix}base_to_{gripper}_eef_pos",
            sensor=robot_base_to_eef_pos(robot_prefix, gripper, eef_site_name),
        ),
        f"{robot_prefix}base_to_{gripper}_eef_quat": Observable(
            name=f"{robot_prefix}base_to_{gripper}_eef_quat",
            sensor=robot_base_to_eef_quat(robot_prefix, gripper, eef_site_name),
        ),
        f"{robot_prefix}base_to_{gripper}_eef_quat_site": Observable(
            name=f"{robot_prefix}base_to_{gripper}_eef_quat_site",
            sensor=robot_base_to_eef_quat_site(robot_prefix, gripper, eef_site_name),
        ),
    }
