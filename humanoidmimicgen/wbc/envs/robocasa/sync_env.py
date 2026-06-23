from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from humanoidmimicgen.wbc.envs.robocasa.utils.robocasa_env import Groot2RoboCasaEnv  # noqa: F401
from humanoidmimicgen.wbc.robot_model.instantiation import get_robot_type_and_model
from humanoidmimicgen.wbc_constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
from robosuite.environments.robot_env import RobotEnv


def add_eval_observation_keys(robot_model, obs: dict) -> dict:
    whole_q = obs["q"]
    obs["state.left_arm"] = whole_q[..., robot_model.get_joint_group_indices("left_arm")]
    obs["state.right_arm"] = whole_q[..., robot_model.get_joint_group_indices("right_arm")]
    obs["state.waist"] = whole_q[..., robot_model.get_joint_group_indices("waist")]
    obs["state.left_leg"] = whole_q[..., robot_model.get_joint_group_indices("left_leg")]
    obs["state.right_leg"] = whole_q[..., robot_model.get_joint_group_indices("right_leg")]
    obs["state.left_hand"] = whole_q[..., robot_model.get_joint_group_indices("left_hand")]
    obs["state.right_hand"] = whole_q[..., robot_model.get_joint_group_indices("right_hand")]
    return obs


def add_eval_space_keys(robot_model, obs_space: gym.spaces.Dict) -> gym.spaces.Dict:
    for name in (
        "left_arm",
        "right_arm",
        "waist",
        "left_leg",
        "right_leg",
        "left_hand",
        "right_hand",
    ):
        obs_space[f"state.{name}"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(robot_model.get_joint_group_indices(name)),),
        )
    return obs_space



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

    def reset_to(self, state: Dict[str, Any]) -> Dict[str, Any] | None:
        if hasattr(self.base_env, "reset_to"):
            result = self.base_env.reset_to(state)
        else:
            # todo: maybe update robosuite to have reset_to()
            env = self.base_env
            if "model_file" in state:
                xml = env.edit_model_xml(state["model_file"])
                env.reset_from_xml_string(xml)
                env.sim.reset()
            if "states" in state:
                try:
                    env.sim.set_state_from_flattened(state["states"])
                except ValueError:
                    legacy_state = np.asarray(state["states"])
                    nq = env.sim.model.nq
                    nv = env.sim.model.nv
                    full_state_len = 1 + nq + nv
                    if legacy_state.shape[0] < 1 + nq or legacy_state.shape[0] > full_state_len:
                        raise
                    padded_state = np.zeros(full_state_len, dtype=legacy_state.dtype)
                    padded_state[: legacy_state.shape[0]] = legacy_state
                    env.sim.set_state_from_flattened(padded_state)
                env.sim.forward()
            result = None

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
        obs = add_eval_observation_keys(self.robot_model, obs)

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

        obs_space = add_eval_space_keys(self.robot_model, obs_space)

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
