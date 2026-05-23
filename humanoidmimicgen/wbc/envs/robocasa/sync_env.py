import sys
from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from robocasa.models.robots import GROOT2_ENVS_ROBOTS
from robocasa.utils.gym_utils.gymnasium_basic import REGISTERED_ENVS
from scipy.spatial.transform import Rotation as R

from humanoidmimicgen.wbc.envs.robocasa.utils.controller_utils import update_robosuite_controller_configs
from humanoidmimicgen.wbc.envs.robocasa.utils.robocasa_env import Groot2RoboCasaEnv  # noqa: F401
from humanoidmimicgen.wbc.policy.wbc_policy_factory import WBC_VERSIONS
from humanoidmimicgen.wbc.robot_model.instantiation import get_robot_type_and_model
from humanoidmimicgen.wbc.teleop.util import prepare_gym_space_for_eval, prepare_observation_for_eval
from humanoidmimicgen.wbc.data.constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
from robosuite.environments.robot_env import RobotEnv
from humanoidmimicgen.wbc.envs.robocasa.utils.randomization import randomize_appearances



class SyncEnv(gym.Env):
    MAX_MUJOCO_STATE_LEN = 800

    def __init__(self, env_name, **kwargs):
        env_kwargs = {
            "onscreen": kwargs.get("onscreen", True),
            "offscreen": kwargs.get("offscreen", False),
            "renderer": kwargs.get("renderer", "mjviewer"),
            "camera_names": kwargs.get("camera_names", ["frontview"]),
            "camera_heights": kwargs.get("camera_heights", None),
            "camera_widths": kwargs.get("camera_widths", None),
            "render_camera": kwargs.get("render_camera", "frontview"),
            "control_freq": kwargs.get("control_freq", 50),
            "translucent_robot": kwargs.get("translucent_robot", True),
            "ik_indicator": kwargs.get("ik_indicator", False),
            "controller_configs": kwargs.get("controller_configs", None),
        }
        self.onscreen = env_kwargs["onscreen"]
        self.env_name = env_name
        env_name, robot_name = env_name.split("/")[1].split("_")[:2]
        _, self.robot_model = get_robot_type_and_model(
            robot_name, enable_waist_ik=kwargs.get("enable_waist", False)
        )
        self.env = Groot2RoboCasaEnv(
            env_name, robot_name, robot_model=self.robot_model, **env_kwargs
        )
        self.init_cache()

        self.reset()

    @property
    def base_env(self) -> RobotEnv:
        return self.env.env

    def overwrite_floating_base_action(
        self, navigate_cmd: np.ndarray, base_height_command: float, use_pos_nav: bool = False
    ):
        if self.base_env.robots[0].robot_model.default_base in [
            "FloatingLeggedBase",
            "FloatingLeggedBaseWithVertical",
        ]:
            from robosuite.controllers.parts.mobile_base.joint_vel import (
                MobileBaseJointVelocityAndPositionController,
            )

            base_controller = self.base_env.robots[0].composite_controller.part_controllers["base"]
            if base_controller.control_dim == 3:
                self.env.unwrapped.overridden_floating_base_action = navigate_cmd
                return
            assert isinstance(
                base_controller, MobileBaseJointVelocityAndPositionController
            ), "Only MobileBaseJointVelocityAndPositionController is supported for xyztheta control for now"
            # MobileBaseJointVelocityontroller also works, but that's a coincidence

            if use_pos_nav:
                current_xy_yaw = np.array(
                    [
                        self.base_env.sim.data.joint("mobilebase0_joint_mobile_forward").qpos[0],
                        self.base_env.sim.data.joint("mobilebase0_joint_mobile_side").qpos[0],
                        self.base_env.sim.data.joint("mobilebase0_joint_mobile_yaw").qpos[0],
                    ]
                )
                target_xy_yaw = navigate_cmd[:3]
                navigate_cmd = self.get_base_vel_from_pos_target(
                    current_xy_yaw, target_xy_yaw, dt=0.05
                )

            self.env.unwrapped.overridden_floating_base_action = np.concatenate(
                [navigate_cmd, [base_height_command]]
            )

    def get_base_vel_from_pos_target(
        self, current_xy_yaw: np.ndarray, target_xy_yaw: np.ndarray, dt: float, max_vel: float = 1
    ) -> np.ndarray:
        dx, dy, dyaw = target_xy_yaw - current_xy_yaw
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))  # wrap_circular
        navigate_cmd_vel = np.clip(np.array([dx, dy, dyaw]) * 1 / dt, -max_vel, max_vel)
        return navigate_cmd_vel

    def get_base_pos_from_vel_target(
        self, current_xy_yaw: np.ndarray, navigate_cmd_vel: np.ndarray, dt: float
    ) -> np.ndarray:
        dx, dy, dyaw = navigate_cmd_vel
        target_xy_yaw = current_xy_yaw + np.array([dx, dy, dyaw]) * dt
        return target_xy_yaw

    def get_base_vel_from_pd_control(
        self,
        current_xy_yaw: np.ndarray,
        target_xy_yaw: np.ndarray,
        current_vel_xy_yaw: np.ndarray,
        target_vel_xy_yaw: np.ndarray = None,
        kp: np.ndarray = None,
        kd: np.ndarray = None,
        max_vel: float = 1.0,
        max_accel: float = 2.0,
        dt: float = 0.02,
    ) -> np.ndarray:
        """
        Compute navigation command velocity using PD control.
        Note that the current_xy_yaw, target_xy_yaw, current_vel_xy_yaw are all in the world frame,
        while the target_vel_xy_yaw is in the robot base frame. The output navigate_cmd_vel is in the robot base frame.

        Args:
            current_xy_yaw: Current position [x, y, yaw] in meters and radians in the world frame
            target_xy_yaw: Target position [x, y, yaw] in meters and radians in the world frame
            current_vel_xy_yaw: Current velocity [vx, vy, vyaw] in the world frame
            target_vel_xy_yaw: Target velocity [vx, vy, vyaw] in the robot base frame (optional, defaults to zero)
            kp: Proportional gains [kp_x, kp_y, kp_yaw] (optional, defaults to [2.0, 2.0, 2.0])
            kd: Derivative gains [kd_x, kd_y, kd_yaw] (optional, defaults to [0.5, 0.5, 0.5])
            max_vel: Maximum velocity magnitude
            max_accel: Maximum acceleration magnitude
            dt: Time step

        Returns:
            navigate_cmd_vel: Command velocity [vx, vy, vyaw] in the robot base frame
        """
        # Default PD gains if not provided
        if kp is None:
            kp = np.array([2.0, 2.0, 0.5])
        if kd is None:
            kd = np.array([0.2, 0.2, 0.05]) * 0.0

        # Default velocities to zero if not provided
        if target_vel_xy_yaw is None:
            target_vel_xy_yaw = np.zeros(3)

        # Extract current yaw angle
        theta = current_xy_yaw[2]
        c = np.cos(theta)
        s = np.sin(theta)

        # World to body rotation matrix
        # Rwb = [[ cos(θ),  sin(θ)],
        #        [-sin(θ),  cos(θ)]]

        # --- Transform position error from world to body frame ---
        pos_error_world = target_xy_yaw - current_xy_yaw
        pos_error_world[2] = np.arctan2(
            np.sin(pos_error_world[2]), np.cos(pos_error_world[2])
        )  # Wrap angle

        ex_world = pos_error_world[0]
        ey_world = pos_error_world[1]
        ex_body = c * ex_world + s * ey_world
        ey_body = -s * ex_world + c * ey_world
        eth = pos_error_world[2]  # Angular error is frame-invariant

        # --- Transform current velocity from world to body frame ---
        vx_world = current_vel_xy_yaw[0]
        vy_world = current_vel_xy_yaw[1]
        vx_body = c * vx_world + s * vy_world
        vy_body = -s * vx_world + c * vy_world
        vyaw_body = current_vel_xy_yaw[2]  # Angular velocity is frame-invariant

        # --- Target velocity is already in body frame ---
        vxd_body = target_vel_xy_yaw[0]
        vyd_body = target_vel_xy_yaw[1]
        vyawd_body = target_vel_xy_yaw[2]

        # --- PD control in body frame (with feedforward) ---
        vx_cmd = vxd_body + kp[0] * ex_body + kd[0] * (vxd_body - vx_body)
        vy_cmd = vyd_body + kp[1] * ey_body + kd[1] * (vyd_body - vy_body)
        vyaw_cmd = vyawd_body + kp[2] * eth + kd[2] * (vyawd_body - vyaw_body)

        navigate_cmd_vel = np.array([vx_cmd, vy_cmd, vyaw_cmd])

        # Apply acceleration limits (change in velocity)
        # if max_accel > 0:
        #     # Current velocity in body frame for acceleration limiting
        #     current_vel_body = np.array([vx_body, vy_body, vyaw_body])
        #     vel_change = navigate_cmd_vel - current_vel_body
        #     vel_change_norm = np.linalg.norm(vel_change)
        #     if vel_change_norm > max_accel * dt:
        #         vel_change = vel_change * (max_accel * dt / vel_change_norm)
        #     navigate_cmd_vel = current_vel_body + vel_change

        # Apply velocity limits
        navigate_cmd_vel = np.clip(navigate_cmd_vel, -max_vel, max_vel)

        return navigate_cmd_vel

    def get_mujoco_state_info(self):
        mujoco_state = self.base_env.sim.get_state().flatten()
        assert len(mujoco_state) < SyncEnv.MAX_MUJOCO_STATE_LEN
        padding_width = SyncEnv.MAX_MUJOCO_STATE_LEN - len(mujoco_state)
        padded_mujoco_state = np.pad(
            mujoco_state, (0, padding_width), mode="constant", constant_values=0
        )
        max_mujoco_state_len = SyncEnv.MAX_MUJOCO_STATE_LEN
        mujoco_state_len = len(mujoco_state)
        mujoco_state = padded_mujoco_state.copy()
        return max_mujoco_state_len, mujoco_state_len, mujoco_state

    def reset_to(self, state: Dict[str, Any], do_visual_domain_randomization: bool = False) -> Dict[str, Any] | None:
        if hasattr(self.base_env, "reset_to"):
            result = self.base_env.reset_to(state)
        else:
            # todo: maybe update robosuite to have reset_to()
            env = self.base_env
            if "model_file" in state:
                xml = env.edit_model_xml(state["model_file"])
                env.reset_from_xml_string(xml)
                env.sim.reset()
                if do_visual_domain_randomization:
                    randomize_appearances(
                        env,
                        seed=None,                  # or pass a seed for reproducibility
                        geom_include=None,          # or a subset: ["mug", "table", ...]
                        color_jitter=0.25,
                        light_pos_jitter=0.2,
                        light_rgb_jitter=0.25,
                        brightness=0.20,
                        contrast=0.20,
                        noise=0.05,
                    )
                    env.sim.forward()
            if "states" in state:
                env.sim.set_state_from_flattened(state["states"])
                if do_visual_domain_randomization:
                    randomize_appearances(
                        env,
                        seed=None,                  # or pass a seed for reproducibility
                        geom_include=None,          # or a subset: ["mug", "table", ...]
                        color_jitter=0.25,
                        light_pos_jitter=0.2,
                        light_rgb_jitter=0.25,
                        brightness=0.20,
                        contrast=0.20,
                        noise=0.05,
                    )
                env.sim.forward()
            result = None

        # Follow the same observation processing pipeline as step_only_kinematics
        # Note that this will make playback diverge
        obs = self.env.force_update_observation()
        self.cache["obs"] = obs

        return result

    def get_state(self) -> Dict[str, Any]:
        return self.base_env.get_state()

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        # First, try to use the base environment's is_success method if it exists
        if hasattr(self.base_env, "is_success"):
            return self.base_env.is_success()

        # Fall back to using _check_success if available
        elif hasattr(self.base_env, "_check_success"):
            succ = self.base_env._check_success()
            if isinstance(succ, dict):
                assert "task" in succ
                return succ
            return {"task": succ}

        # If neither method exists, return failure
        else:
            return {"task": False}

    def init_cache(self):
        self.cache = {
            "obs": None,
            "reward": None,
            "terminated": None,
            "truncated": None,
            "info": None,
        }

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, any], Dict[str, any]]:
        self.init_cache()
        obs, info = self.env.reset(seed=seed, options=options)
        self.cache["obs"] = obs
        self.cache["reward"] = 0
        self.cache["terminated"] = False
        self.cache["truncated"] = False
        self.cache["info"] = info
        return self.observe(), info

    def observe(self) -> Dict[str, any]:
        # Get observations from body and hands
        assert (
            self.cache["obs"] is not None
        ), "Observation cache is not initialized, please reset the environment first"
        raw_obs = self.cache["obs"]

        # Body and hand joint measurements come in actuator order, so we need to convert them to joint order
        whole_q = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=raw_obs["body_q"],
            left_hand_actuated_joint_values=raw_obs["left_hand_q"],
            right_hand_actuated_joint_values=raw_obs["right_hand_q"],
        )
        whole_dq = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=raw_obs["body_dq"],
            left_hand_actuated_joint_values=raw_obs["left_hand_dq"],
            right_hand_actuated_joint_values=raw_obs["right_hand_dq"],
        )
        whole_ddq = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=raw_obs["body_ddq"],
            left_hand_actuated_joint_values=raw_obs["left_hand_ddq"],
            right_hand_actuated_joint_values=raw_obs["right_hand_ddq"],
        )
        whole_tau_est = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=raw_obs["body_tau_est"],
            left_hand_actuated_joint_values=raw_obs["left_hand_tau_est"],
            right_hand_actuated_joint_values=raw_obs["right_hand_tau_est"],
        )
        eef_obs = self.get_eef_obs(whole_q)

        obs = {
            "q": whole_q,
            "dq": whole_dq,
            "ddq": whole_ddq,
            "tau_est": whole_tau_est,
            "floating_base_pose": raw_obs["floating_base_pose"],
            "floating_base_vel": raw_obs["floating_base_vel"],
            "floating_base_acc": raw_obs["floating_base_acc"],
            "wrist_pose": np.concatenate([eef_obs["left_wrist_pose"], eef_obs["right_wrist_pose"]]),
        }

        # Add state keys for model input
        obs = prepare_observation_for_eval(self.robot_model, obs)

        if hasattr(self.base_env, "get_privileged_obs_keys"):
            for key in self.base_env.get_privileged_obs_keys():
                obs[key] = raw_obs[key]

        for key in raw_obs.keys():
            if key.endswith("_image"):
                obs[key] = raw_obs[key]
                # TODO: add video.key without _image suffix for evaluation, remove later
                obs[f"video.{key.replace('_image', '')}"] = raw_obs[key]
        return obs

    def step(
        self, action: Dict[str, any]
    ) -> Tuple[Dict[str, any], float, bool, bool, Dict[str, any]]:
        self.queue_action(action)
        return self.get_step_info()

    def get_observation(self):
        return self.base_env._get_observations()  # assumes base env is robosuite

    def get_step_info(self) -> Dict[str, any]:
        return (
            self.observe(),
            self.cache["reward"],
            self.cache["terminated"],
            self.cache["truncated"],
            self.cache["info"],
        )

    def convert_q_to_actuated_joint_order(self, q: np.ndarray) -> np.ndarray:
        body_q = self.robot_model.get_body_actuated_joints(q)
        left_hand_q = self.robot_model.get_hand_actuated_joints(q, side="left")
        right_hand_q = self.robot_model.get_hand_actuated_joints(q, side="right")

        whole_q = np.zeros_like(q)
        whole_q[self.robot_model.get_joint_group_indices("body")] = body_q
        whole_q[self.robot_model.get_joint_group_indices("left_hand")] = left_hand_q
        whole_q[self.robot_model.get_joint_group_indices("right_hand")] = right_hand_q

        return whole_q

    def set_ik_indicator(self, teleop_cmd):
        """Set the IK indicators for the simulator"""
        if "left_wrist" in teleop_cmd and "right_wrist" in teleop_cmd:
            left_wrist_input_pose = teleop_cmd["left_wrist"]
            right_wrist_input_pose = teleop_cmd["right_wrist"]
            ik_wrapper = self.base_env
            ik_wrapper.set_target_poses_outside_env([left_wrist_input_pose, right_wrist_input_pose])

    def render(self):
        if self.base_env.viewer is not None:
            self.base_env.viewer.update()
        if self.onscreen:
            self.base_env.render()

    def queue_action(self, action: Dict[str, any]):
        # action is in pinocchio joint order, we need to convert it to actuator order
        action_q = self.convert_q_to_actuated_joint_order(action["q"])
        obs, reward, terminated, truncated, info = self.env.step({"q": action_q})
        self.cache["obs"] = obs
        self.cache["reward"] = reward
        self.cache["terminated"] = terminated
        self.cache["truncated"] = truncated
        self.cache["info"] = info

    def queue_state(self, state: Dict[str, any]):
        # This function is for debugging or cross-playback between sim and real only.
        state_q = self.convert_q_to_actuated_joint_order(state["q"])
        obs, reward, terminated, truncated, info = self.env.unwrapped.step_only_kinematics(
            {"q": state_q}
        )
        self.cache["obs"] = obs
        self.cache["reward"] = reward
        self.cache["terminated"] = terminated
        self.cache["truncated"] = truncated
        self.cache["info"] = info

    @property
    def observation_space(self) -> gym.Space:
        # @todo: check if the low and high bounds are correct for body_obs.
        q_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        dq_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        ddq_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        tau_est_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        floating_base_pose_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        floating_base_vel_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        floating_base_acc_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        wrist_pose_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 7,))

        obs_space = gym.spaces.Dict(
            {
                "floating_base_pose": floating_base_pose_space,
                "floating_base_vel": floating_base_vel_space,
                "floating_base_acc": floating_base_acc_space,
                "q": q_space,
                "dq": dq_space,
                "ddq": ddq_space,
                "tau_est": tau_est_space,
                "wrist_pose": wrist_pose_space,
            }
        )

        obs_space = prepare_gym_space_for_eval(self.robot_model, obs_space)

        if hasattr(self.base_env, "get_privileged_obs_keys"):
            for key, shape in self.base_env.get_privileged_obs_keys().items():
                space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)
                obs_space[key] = space

        robocasa_obs_space = self.env.observation_space
        for key in robocasa_obs_space.keys():
            if key.endswith("_image"):
                space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=robocasa_obs_space[key].shape
                )
                obs_space[key] = space
                # TODO: add video.key without _image suffix for evaluation, remove later
                space_uint = gym.spaces.Box(low=0, high=255, shape=space.shape, dtype=np.uint8)
                obs_space[f"video.{key.replace('_image', '')}"] = space_uint

        return obs_space

    def reset_obj_pos(self):
        # For Tairan's goal-reaching task, a hacky way to reset the object position is needed.
        if hasattr(self.base_env, "reset_obj_pos"):
            self.base_env.reset_obj_pos()

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    def close(self):
        self.env.close()

    def __repr__(self):
        return (
            f"SyncEnv(env_name={self.env_name}, \n"
            f"            observation_space={self.observation_space}, \n"
            f"            action_space={self.action_space})"
        )

    def get_joint_gains(self):
        controller = self.base_env.robots[0].composite_controller

        gains = {}
        key_mapping = {
            "left": "left_arm",
            "right": "right_arm",
            "legs": "legs",
            "torso": "waist",
            "head": "neck",
        }
        for k in controller.part_controllers.keys():
            if hasattr(controller.part_controllers[k], "kp"):
                if k in key_mapping:
                    gains[key_mapping[k]] = controller.part_controllers[k].kp
                else:
                    gains[k] = controller.part_controllers[k].kp
        gains.update(
            {
                "left_hand": self.base_env.sim.model.actuator_gainprm[
                    self.base_env.robots[0]._ref_actuators_indexes_dict["left_gripper"], 0
                ],
                "right_hand": self.base_env.sim.model.actuator_gainprm[
                    self.base_env.robots[0]._ref_actuators_indexes_dict["right_gripper"], 0
                ],
            }
        )
        joint_gains = np.zeros(self.robot_model.num_dofs)
        for k in gains.keys():
            joint_gains[self.robot_model.get_joint_group_indices(k)] = gains[k]
        return joint_gains

    def get_joint_damping(self):
        controller = self.base_env.robots[0].composite_controller
        damping = {}
        key_mapping = {
            "left": "left_arm",
            "right": "right_arm",
            "legs": "legs",
            "torso": "waist",
            "head": "neck",
        }
        for k in controller.part_controllers.keys():
            if hasattr(controller.part_controllers[k], "kd"):
                if k in key_mapping:
                    damping[key_mapping[k]] = controller.part_controllers[k].kd
                else:
                    damping[k] = controller.part_controllers[k].kd
        damping.update(
            {
                "left_hand": -self.base_env.sim.model.actuator_biasprm[
                    self.base_env.robots[0]._ref_actuators_indexes_dict["left_gripper"], 2
                ],
                "right_hand": -self.base_env.sim.model.actuator_biasprm[
                    self.base_env.robots[0]._ref_actuators_indexes_dict["right_gripper"], 2
                ],
            }
        )
        joint_damping = np.zeros(self.robot_model.num_dofs)
        for k in damping.keys():
            joint_damping[self.robot_model.get_joint_group_indices(k)] = damping[k]
        return joint_damping

    def get_eef_obs(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        self.robot_model.cache_forward_kinematics(q)
        eef_obs = {}
        for side in ["left", "right"]:
            wrist_placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            wrist_pos, wrist_quat = wrist_placement.translation[:3], R.from_matrix(
                wrist_placement.rotation
            ).as_quat(scalar_first=True)
            eef_obs[f"{side}_wrist_pose"] = np.concatenate([wrist_pos, wrist_quat])

        return eef_obs


class G1SyncEnv(SyncEnv):
    def __init__(
        self,
        env_name,
        **kwargs,
    ):
        renderer = kwargs.get("renderer", "mjviewer")
        if renderer == "mjviewer":
            default_render_camera = ["robot0_oak_egoview"]
        elif renderer in ["mujoco", "rerun"]:
            default_render_camera = [
                "robot0_oak_egoview",
                "robot0_oak_left_monoview",
                "robot0_oak_right_monoview",
            ]
        else:
            raise NotImplementedError
        default_camera_names = [
            "robot0_oak_egoview",
            "robot0_oak_left_monoview",
            "robot0_oak_right_monoview",
            "robot0_left_eef_view",
            "robot0_right_eef_view",
        ]
        default_camera_heights = [
            RS_VIEW_CAMERA_HEIGHT,
            RS_VIEW_CAMERA_HEIGHT,
            RS_VIEW_CAMERA_HEIGHT,
            RS_VIEW_CAMERA_HEIGHT,
            RS_VIEW_CAMERA_HEIGHT,
        ]
        default_camera_widths = [
            RS_VIEW_CAMERA_WIDTH,
            RS_VIEW_CAMERA_WIDTH,
            RS_VIEW_CAMERA_WIDTH,
            RS_VIEW_CAMERA_WIDTH,
            RS_VIEW_CAMERA_WIDTH,
        ]

        env_kwargs = {
            "onscreen": kwargs.get("onscreen", True),
            "offscreen": kwargs.get("offscreen", False),
            "renderer": kwargs.get("renderer", "mjviewer"),
            "render_camera": kwargs.get("render_camera", default_render_camera),
            "camera_names": kwargs.get("camera_names", default_camera_names),
            "camera_heights": kwargs.get("camera_heights", default_camera_heights),
            "camera_widths": kwargs.get("camera_widths", default_camera_widths),
            "controller_configs": kwargs[
                "controller_configs"
            ],  # must be provided by calling get_env()
            "control_freq": kwargs.get("control_freq", 50),
            "translucent_robot": kwargs.get("translucent_robot", True),
            "ik_indicator": kwargs.get("ik_indicator", False),
        }
        super().__init__(env_name=env_name, **env_kwargs)

    @property
    def observation_space(self):
        obs_space = super().observation_space
        obs_space["torso_quat"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        obs_space["torso_ang_vel"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        return obs_space

    def observe(self):
        obs = super().observe()
        obs["torso_quat"] = self.cache["obs"]["secondary_imu_quat"]
        obs["torso_ang_vel"] = self.cache["obs"]["secondary_imu_vel"][3:6]
        return obs


class GR1SyncEnv(SyncEnv):
    def __init__(
        self,
        env_name,
        **kwargs,
    ):
        env_kwargs = {
            "onscreen": kwargs.get("onscreen", True),
            "offscreen": kwargs.get("offscreen", False),
            "renderer": kwargs.get("renderer", "mjviewer"),
            "render_camera": kwargs.get("render_camera", "egoview"),
            "camera_names": kwargs.get("camera_names", ["egoview"]),
            "camera_heights": kwargs.get("camera_heights", [RS_VIEW_CAMERA_HEIGHT]),
            "camera_widths": kwargs.get("camera_widths", [RS_VIEW_CAMERA_WIDTH]),
            "control_freq": kwargs.get("control_freq", 50),
            "controller_configs": kwargs[
                "controller_configs"
            ],  # must be provided by calling get_env()
            "translucent_robot": kwargs.get("translucent_robot", True),
            "ik_indicator": kwargs.get("ik_indicator", False),
        }
        super().__init__(env_name=env_name, **env_kwargs)


def create_gym_sync_env_class(env, robot, robot_alias, wbc_version):
    class_name = f"{env}_{robot}_{wbc_version}"
    id_name = f"groot2_{robot_alias}/{class_name}"

    if robot_alias.startswith("g1"):
        env_class_type = G1SyncEnv
    elif robot_alias.startswith("gr1"):
        env_class_type = GR1SyncEnv
    else:
        env_class_type = SyncEnv

    controller_configs = update_robosuite_controller_configs(
        robot=robot,
        wbc_version=wbc_version,
    )

    env_class_type = type(
        class_name,
        (env_class_type,),
        {
            "__init__": lambda self, **kwargs: super(self.__class__, self).__init__(
                env_name=id_name,
                controller_configs=controller_configs,
                **kwargs,
            )
        },
    )

    current_module = sys.modules["humanoidmimicgen.wbc.envs.robocasa.sync_env"]
    setattr(current_module, class_name, env_class_type)
    register(
        id=id_name,  # Unique ID for the environment
        entry_point=f"humanoidmimicgen.wbc.envs.robocasa.sync_env:{class_name}",
    )

    if robot_alias.startswith("gr1"):
        id_name = f"groot2_gr1/{class_name}"
        register(
            id=id_name,  # Unique ID for the environment
            entry_point=f"humanoidmimicgen.wbc.envs.robocasa.sync_env:{class_name}",
        )


for ENV in REGISTERED_ENVS:
    for ROBOT, ROBOT_ALIAS in GROOT2_ENVS_ROBOTS.items():
        for WBC_VERSION in WBC_VERSIONS:
            create_gym_sync_env_class(ENV, ROBOT, ROBOT_ALIAS, WBC_VERSION)


if __name__ == "__main__":

    env = SyncEnv(
        env_name="groot2_g1/PnPBottle_G1_homie",
        camera_names=["egoview"],
        camera_heights=[600],
        camera_widths=[600],
        render_camera="egoview",
        onscreen=False,
        offscreen=True,
        # env_name="groot2_gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
        # camera_names=["egoview"],
        # camera_heights=[600],
        # camera_widths=[600],
        # render_camera="egoview",
        # onscreen=False,
        # offscreen=True,
    )
