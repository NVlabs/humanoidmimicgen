# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay LeRobot G1 episodes through stored WBC goals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Literal

import numpy as np
from tqdm import tqdm
import yaml

CONTROL_NODE_NAME = "humanoidmimicgen_playback_node"
GREEN_BOLD = "\033[1;32m"
RED_BOLD = "\033[1;31m"
RESET = "\033[0m"


def override_wbc_config(
    wbc_config: dict, config: "SyncSimPlaybackConfig", missed_keys_only: bool = False
) -> dict:
    """Override WBC YAML values with local playback config values."""
    key_to_value = {
        "INTERFACE": config.interface,
        "ENV_TYPE": config.env_type,
        "VERSION": config.wbc_version,
        "SIMULATOR": config.simulator,
        "SIMULATE_DT": 1 / float(config.sim_frequency),
        "ENABLE_OFFSCREEN": config.enable_offscreen,
        "ENABLE_ONSCREEN": config.enable_onscreen,
        "model_path": config.wbc_model_path,
        "enable_waist": config.enable_waist,
        "with_hands": config.with_hands,
        "verbose": config.verbose,
        "verbose_timing": config.verbose_timing,
        "upper_body_max_joint_speed": config.upper_body_joint_speed,
        "keyboard_dispatcher_type": config.keyboard_dispatcher_type,
        "enable_gravity_compensation": config.enable_gravity_compensation,
        "gravity_compensation_joints": config.gravity_compensation_joints,
        "high_elbow_pose": config.high_elbow_pose,
        "joint_safety_mode": config.joint_safety_mode,
        "arm_velocity_limit": config.arm_velocity_limit,
        "hand_velocity_limit": config.hand_velocity_limit,
        "lower_body_velocity_limit": config.lower_body_velocity_limit,
        "waist_pitch_limit": config.waist_pitch_limit,
        "hand_torque_limit": config.hand_torque_limit,
        "enable_natural_walk": config.enable_natural_walk,
    }

    for key, value in key_to_value.items():
        if not missed_keys_only or key not in wbc_config:
            wbc_config[key] = value

    if config.env_type == "real":
        wbc_config["MOTOR_KD"][14] = wbc_config["MOTOR_KD"][14] - 10

    return wbc_config


@dataclass
class SyncSimPlaybackConfig:
    """Configuration for WBC-goal replay.

    The fields mirror the subset of sync-sim playback config needed by the
    local runtime facade and the WBC policy factory.
    """

    dataset_version: str = "v1"
    wbc_version: str = "homie_v2"
    wbc_model_path: str = "policy/stand.onnx,policy/walk.onnx"
    wbc_policy_class: str = "G1DecoupledWholeBodyPolicy"
    interface: str = "sim"
    env_type: str = "sim"
    simulator: str = "mujoco"
    sim_sync_mode: bool = False
    control_frequency: int = 50
    sim_frequency: int = 200
    enable_waist: bool = True
    with_hands: bool = True
    high_elbow_pose: bool = False
    verbose: bool = True
    enable_offscreen: bool = False
    enable_onscreen: bool = True
    enable_teleop_evaluator: bool = False
    upper_body_joint_speed: float = 1000
    env_name: str = "default"
    ik_indicator: bool = False
    verbose_timing: bool = False
    keyboard_dispatcher_type: str = "raw"
    enable_gravity_compensation: bool = False
    use_dual_wbc_env: bool = False
    controller_initial_base_height: float = 0.74
    controller_min_base_height: float = 0.3
    controller_max_base_height: float = 1.1
    use_raised_arm_pose: bool = False
    gravity_compensation_joints: list[str] | None = None
    joint_safety_mode: Literal["kill", "freeze"] = "kill"
    arm_velocity_limit: float = 25.0
    hand_velocity_limit: float = 1000.0
    lower_body_velocity_limit: float = 20.0
    waist_pitch_limit: float = 15.0
    hand_torque_limit: float = 0.1
    enable_natural_walk: bool = False
    body_control_device: str = "dummy"
    hand_control_device: str | None = "dummy"
    hand_type: Literal["dex3", "gripper"] = "dex3"
    body_streamer_ip: str = "10.112.210.229"
    body_streamer_keyword: str = "knee"
    enable_visualization: bool = False
    enable_real_device: bool = False
    teleop_frequency: int = 20
    teleop_replay_path: str | None = None
    robot_ip: str = "192.168.123.164"
    data_collection: bool = True
    data_collection_frequency: int = 20
    root_output_dir: str = "outputs"
    offline_dc: bool = False
    enable_upper_body_operation: bool = True
    upper_body_operation_mode: Literal["teleop", "inference"] = "teleop"
    inference_host: str = "localhost"
    inference_port: int = 5550
    inference_on_osmo: bool = False
    inference_prompt: str = "Pick up apple from table to plate"
    inference_action_horizon: int = 16
    inference_control_freq: int = 20
    inference_rate: float = 2.5
    state_delay: float = 0.0
    proprio_hist_budget: int = 0
    inference_plot_rerun: bool = False
    inference_push_evals: bool = True
    inference_publish_single_action: bool = False
    commit_id: str = ""
    enable_mode_switch: bool = False
    initial_mode: str = "idle"
    robot: str = "G1"
    task_name: str = "GroundOnly"
    remove_existing_dir: bool = False
    hardcode_teleop_cmd: bool = False
    save_img_obs: bool = False
    success_hold_steps: int = 50
    renderer: Literal["mjviewer", "mujoco", "rerun"] = "mjviewer"
    replay_data_path: str | None = None
    replay_speed: float = 2.5
    debug: bool = False
    manual_control: bool = False
    binary_hand_ik: bool = True
    dataset: str | None = None
    save_video: bool = True
    video_path: str | None = None
    num_episodes: int | None = None

    def __post_init__(self) -> None:
        if self.gravity_compensation_joints is None:
            self.gravity_compensation_joints = ["arms"]
        if self.interface in {"sim", "real"}:
            self.env_type = self.interface
        elif self.interface.startswith("sim"):
            self.interface, self.env_type = "sim", "sim"
        elif self.interface.startswith("real"):
            self.interface, self.env_type = "real", "real"
        else:
            self.env_type = self.interface
        try:
            self.commit_id = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            self.commit_id = ""

    def update(
        self,
        config_dict: dict,
        strict: bool = False,
        skip_keys: list[str] | None = None,
        allowed_keys: list[str] | None = None,
    ) -> None:
        skip_keys = skip_keys or []
        for key, value in config_dict.items():
            if key in skip_keys:
                continue
            if allowed_keys is not None and key not in allowed_keys:
                continue
            if strict and not hasattr(self, key):
                raise ValueError(f"Config {key} not found in {self.__class__.__name__}")
            if not strict and not hasattr(self, key):
                continue
            setattr(self, key, value)

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
        strict: bool = False,
        skip_keys: list[str] | None = None,
        allowed_keys: list[str] | None = None,
    ) -> "SyncSimPlaybackConfig":
        instance = cls()
        instance.update(config_dict, strict=strict, skip_keys=skip_keys, allowed_keys=allowed_keys)
        return instance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if hasattr(self, key) else default

    def load_wbc_yaml(self) -> dict:
        config_root = Path(__file__).resolve().parent / "configs" / "wbc"
        if self.wbc_version == "v1":
            config_path = config_root / "g1_43dof_hist.yaml"
        elif self.wbc_version == "v2":
            config_path = config_root / "g1_29dof_waist_stand_height_history_fixed3dex.yaml"
        elif self.wbc_version == "homie":
            config_path = config_root / "g1_29dof_homie.yaml"
        elif self.wbc_version == "homie_v2" or self.wbc_version.startswith("homie_v2_"):
            config_path = config_root / "g1_29dof_homie_v2.yaml"
        elif self.wbc_version == "locomotion_z":
            config_path = config_root / "g1_29dof_locomotion_z.yaml"
        elif self.wbc_version == "local_tracking":
            config_path = config_root / "g1_29dof_local_tracking.yaml"
        else:
            raise ValueError(f"Invalid wbc_version: {self.wbc_version}")

        with config_path.open() as file:
            wbc_config = yaml.load(file, Loader=yaml.FullLoader)
        return override_wbc_config(wbc_config, self)


def load_lerobot_dataset(root_path: str | os.PathLike[str], max_episodes: int | None = None):
    from humanoidmimicgen.lerobot_dataset import TypedLeRobotDataset

    task_name = None
    episodes = []
    start_index = 0
    with (Path(root_path) / "meta/episodes.jsonl").open() as file:
        for line in file:
            episode = json.loads(line)
            episode["start_index"] = start_index
            start_index += episode["length"]
            assert (
                task_name is None or task_name == episode["tasks"][0]
            ), "All episodes should have the same task name"
            task_name = episode["tasks"][0]
            episodes.append(episode)

    dataset = TypedLeRobotDataset(repo_id="tmp/test", root=root_path, load_video=False)
    script_config = dataset.meta.info["script_config"]
    assert len(dataset) == start_index, "Dataset length does not match expected length"

    if max_episodes is not None:
        episodes = episodes[:max_episodes]
        print(
            f"Loading only first {len(episodes)} episodes (limited by max_episodes={max_episodes})"
        )

    frames = {}
    seeds = []
    for ep in tqdm(range(len(episodes))):
        seed = None
        frames[f"data/demo_{ep + 1}/states"] = []
        frames[f"data/demo_{ep + 1}/wbc_goal"] = []
        start_index = episodes[ep]["start_index"]
        end_index = start_index + episodes[ep]["length"]
        for i in tqdm(range(start_index, end_index)):
            frame = dataset[i]
            assert seed is None or seed == np.array(frame["observation.sim.seed"]).item()
            seed = np.array(frame["observation.sim.seed"]).item()

            mujoco_state_len = frame["observation.sim.mujoco_state_len"]
            mujoco_state = frame["observation.sim.mujoco_state"]
            frames[f"data/demo_{ep + 1}/states"].append(
                np.array(mujoco_state[:mujoco_state_len])
            )
            frames[f"data/demo_{ep + 1}/wbc_goal"].append(
                {
                    "wrist_pose": np.array(frame["action.eef"]),
                    "target_upper_body_pose": np.array(
                        frame["observation.sim.target_upper_body_pose"]
                    ),
                    "navigate_cmd": np.array(frame["teleop.navigate_command"]),
                    "base_height_command": np.array(frame["teleop.base_height_command"]),
                }
            )
        seeds.append(seed)

    return seeds, frames, script_config


def validate_state(recorded_state, playback_state, ep, step, tolerance=1e-5) -> bool:
    if recorded_state.shape != playback_state.shape:
        print(
            f"[warning] state shape changed from {recorded_state.shape} to "
            f"{playback_state.shape} for ep {ep} at step {step}"
        )
        return False
    if not np.allclose(recorded_state, playback_state, atol=tolerance):
        err = np.linalg.norm(recorded_state - playback_state)
        print(f"[warning] state diverged by {err:.12f} for ep {ep} at step {step}")
        return False
    return True


def write_video_frame(env, video_writer) -> None:
    from humanoidmimicgen.wbc_constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH

    import cv2

    img = env.sim.render(
        width=RS_VIEW_CAMERA_WIDTH,
        height=RS_VIEW_CAMERA_HEIGHT,
        camera_name=env.render_camera[0],
    )
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = np.flipud(img_bgr)
    video_writer.write(img_bgr)


def get_task_success(sync_env) -> bool:
    """Return the robocasa-style task success bit for the current replay state."""
    success = sync_env.is_success()
    if isinstance(success, dict):
        return bool(success.get("task", False))
    return bool(success)


def get_video_fps(config: SyncSimPlaybackConfig) -> float:
    # Match GR00T add-skillgen playback: videos are encoded at the dataset
    # collection frequency, which is 20 Hz for the released G1 demos.
    return float(config.data_collection_frequency)


class WBCGoalEnv:
    """Env facade that steps recorded WBC goals through a low-level sync env."""

    def __init__(self, sync_env, wbc_policy) -> None:
        self.sync_env = sync_env
        self.wbc_policy = wbc_policy

    def __getattr__(self, name: str):
        return getattr(self.sync_env, name)

    def step(self, wbc_goal: dict[str, Any]):
        obs = self.sync_env.observe()
        self.wbc_policy.set_observation(obs)
        self.wbc_policy.set_goal(wbc_goal)
        self.sync_env.overwrite_floating_base_action(
            wbc_goal.get("navigate_cmd", np.zeros(3)),
            wbc_goal.get("base_height_command", 0.0),
        )
        return self.sync_env.step(self.wbc_policy.get_action())


def format_success_summary(ep: str, stats: dict[str, int | bool | None]) -> str:
    first_step = stats["first_task_success_step"]
    first_step_str = "never" if first_step is None else str(first_step)
    return (
        f"Episode {ep} task success: {stats['task_success']} "
        f"(success_steps={stats['task_success_steps']}/{stats['checked_steps']}, "
        f"first_success_step={first_step_str}, "
        f"final_success={stats['final_task_success']})"
    )


def aggregate_success_stats(stats_by_episode: dict[str, dict[str, int | bool | None]]) -> dict:
    episodes = len(stats_by_episode)
    successes = sum(1 for stats in stats_by_episode.values() if stats["task_success"])
    final_successes = sum(1 for stats in stats_by_episode.values() if stats["final_task_success"])
    return {
        "episodes": episodes,
        "task_successes": successes,
        "final_task_successes": final_successes,
        "task_success_rate": successes / episodes if episodes else 0.0,
        "final_task_success_rate": final_successes / episodes if episodes else 0.0,
        "episodes_detail": stats_by_episode,
    }


def playback_wbc_goal_dataset(config: SyncSimPlaybackConfig) -> bool:
    from humanoidmimicgen.wbc_constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
    from humanoidmimicgen.wbc_runtime import get_env, get_policies, get_robot_type_and_model

    ret = True
    start_time = time.time()
    np.set_printoptions(precision=5, suppress=True, linewidth=120)

    assert config.dataset is not None, "Dataset must be specified for playback"
    seeds, frames, script_config = load_lerobot_dataset(config.dataset, config.num_episodes)

    config.update(
        script_config,
        allowed_keys=[
            "wbc_version",
            "wbc_model_path",
            "wbc_policy_class",
            "control_frequency",
            "enable_waist",
            "with_hands",
            "env_name",
            "robot",
            "task_name",
            "teleop_frequency",
            "data_collection_frequency",
            "enable_gravity_compensation",
            "gravity_compensation_joints",
        ],
    )
    robot_type, robot_model = get_robot_type_and_model(config.robot, config.enable_waist)
    onscreen = False if config.save_video else config.enable_onscreen
    offscreen = True if config.save_video else config.enable_offscreen

    if config.save_video and config.video_path is None:
        video_folder = Path(config.dataset)
        video_folder.mkdir(parents=True, exist_ok=True)
        config.video_path = str(video_folder / "playback_video.mp4")
        print(f"Video recording enabled. Output: {config.video_path}")

    sync_env = get_env(config, onscreen=onscreen, offscreen=offscreen)
    wbc_policy, _, _ = get_policies(
        config, robot_type, robot_model, activate_keyboard_listener=False
    )
    env = WBCGoalEnv(sync_env, wbc_policy)

    video_writer = None
    if config.save_video:
        import cv2

        video_writer = cv2.VideoWriter(
            config.video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            get_video_fps(config),
            (RS_VIEW_CAMERA_WIDTH, RS_VIEW_CAMERA_HEIGHT),
        )

    demos = [f"demo_{i + 1}" for i in range(len(seeds))]
    print(f"Loaded {len(demos)} episodes from {config.dataset}")
    print("seeds:", seeds)
    print("demos:", demos, "\n\n")

    task_success_by_episode = {}
    for episode_index, ep in enumerate(demos):
        print(f"Playing back episode: {ep}")
        seed = seeds[episode_index]
        env.reset(seed=seed)
        states = frames[f"data/{ep}/states"]
        wbc_goals = frames[f"data/{ep}/wbc_goal"]
        env.reset_to({"states": states[0]})
        num_wbc_goals = min(20, len(wbc_goals)) if config.debug else len(wbc_goals)
        task_success_steps = 0
        first_task_success_step = None
        last_task_success = False

        for jj in range(num_wbc_goals):
            env.step(wbc_goals[jj])
            if video_writer is not None:
                write_video_frame(env, video_writer)
            elif onscreen:
                env.render()

            task_success = get_task_success(env)
            if task_success:
                task_success_steps += 1
                if first_task_success_step is None:
                    first_task_success_step = jj
            last_task_success = task_success

            if jj < len(states) - 1:
                state_playback = env.sim.get_state().flatten()
                if not validate_state(states[jj + 1], state_playback, ep, jj):
                    ret = False

        task_success_stats = {
            "task_success": task_success_steps > 0,
            "task_success_steps": task_success_steps,
            "first_task_success_step": first_task_success_step,
            "final_task_success": last_task_success,
            "checked_steps": num_wbc_goals,
        }
        task_success_by_episode[ep] = task_success_stats
        print(format_success_summary(ep, task_success_stats))
        print(f"Episode {ep} playback finished.\n\n")

    env.close()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {config.video_path}")

    task_success_summary = aggregate_success_stats(task_success_by_episode)
    print("Task success summary:")
    print(json.dumps(task_success_summary, indent=2))

    elapsed_time = time.time() - start_time
    print(
        f"{GREEN_BOLD}Playback with WBC version: {config.wbc_version}, {config.wbc_model_path}, "
        f"{config.wbc_policy_class}{RESET}"
    )
    if ret:
        print(f"{GREEN_BOLD}Playback completed successfully in {elapsed_time:.2f} seconds!{RESET}")
    else:
        print(
            f"{RED_BOLD}Playback completed with state divergence in "
            f"{elapsed_time:.2f} seconds!{RESET}"
        )
    return ret
