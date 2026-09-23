# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay HMG datasets from recorded actions, WBC goals, or states."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
from tqdm import tqdm
import yaml

from humanoidmimicgen.wbc_constants import DEFAULT_BASE_HEIGHT

GREEN_BOLD = "\033[1;32m"
RED_BOLD = "\033[1;31m"
RESET = "\033[0m"


def override_wbc_config(wbc_config: dict, config: "PlaybackConfig") -> dict:
    """Apply local playback overrides used by the WBC policy factory."""
    wbc_config["VERSION"] = config.wbc_version
    wbc_config["model_path"] = config.wbc_model_path
    return wbc_config


@dataclass
class PlaybackConfig:
    """Configuration for dataset playback.

    The fields mirror the subset of sync-sim playback config needed by the
    local runtime facade and the WBC policy factory.
    """

    wbc_version: str = "homie_v2_grav_comp_tuned"
    wbc_model_path: str = "policy/stand.onnx,policy/walk.onnx"
    wbc_policy_class: str = "G1DecoupledWholeBodyPolicy"
    control_frequency: int = 50
    sim_frequency: int = 200
    enable_waist: bool = True
    enable_offscreen: bool = False
    enable_onscreen: bool = True
    ik_indicator: bool = False
    controller_initial_base_height: float = DEFAULT_BASE_HEIGHT
    data_collection_frequency: int = 20
    robot: str = "G1"
    task_name: str = "GroundOnly"
    renderer: Literal["mjviewer", "mujoco", "rerun"] = "mjviewer"
    debug: bool = False
    dataset: str | None = None
    save_video: bool = True
    video_path: str | None = None
    num_episodes: int | None = None
    state_tolerance: float = 1e-5
    action_source: Literal["recorded", "wbc-goal", "state"] = "recorded"
    require_task_success: bool = False
    allow_state_divergence: bool = False

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
    ) -> "PlaybackConfig":
        instance = cls()
        instance.update(config_dict, strict=strict, skip_keys=skip_keys, allowed_keys=allowed_keys)
        return instance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if hasattr(self, key) else default

    def load_wbc_yaml(self) -> dict:
        config_root = Path(__file__).resolve().parent / "configs" / "wbc"
        if self.wbc_version == "homie_v2_grav_comp_tuned":
            config_path = config_root / "g1_29dof_homie_v2.yaml"
        else:
            raise ValueError(f"Invalid wbc_version: {self.wbc_version}")

        with config_path.open() as file:
            wbc_config = yaml.load(file, Loader=yaml.FullLoader)
        return override_wbc_config(wbc_config, self)


def load_lerobot_dataset(
    root_path: str | os.PathLike[str],
    max_episodes: int | None = None,
    action_source: Literal["recorded", "wbc-goal", "state"] = "recorded",
):
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

    if action_source == "wbc-goal":
        features = set(dataset.meta.info.get("features", {}))
        required_features = {
            "action.eef",
            "observation.sim.target_upper_body_pose",
        }
        missing_features = sorted(required_features - features)
        if missing_features:
            missing = ", ".join(missing_features)
            raise ValueError(
                "WBC-goal playback is unavailable for this LeRobot dataset; "
                f"missing required features: {missing}. The published 1K replay "
                "datasets contain recorded low-level actions but not WBC pose "
                "goals. Use --action-source recorded, or use a human source-demo "
                "HDF5 file for --action-source wbc-goal."
            )

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
            action = {
                "navigate_cmd": np.array(frame["teleop.navigate_command"]),
                "base_height_command": np.array(frame["teleop.base_height_command"]),
            }
            if action_source == "recorded":
                action["recorded_action"] = np.array(frame["action"])
            elif action_source == "wbc-goal":
                action["wrist_pose"] = np.array(frame["action.eef"])
                action["target_upper_body_pose"] = np.array(
                    frame["observation.sim.target_upper_body_pose"]
                )
            elif action_source == "state":
                pass
            else:
                raise ValueError(f"Invalid action_source: {action_source}")
            frames[f"data/demo_{ep + 1}/wbc_goal"].append(action)
        seeds.append(seed)

    return seeds, frames, script_config


def _decode_hdf5_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected an HDF5 string attribute, got {type(value).__name__}")


def load_hdf5_dataset(
    dataset_path: str | os.PathLike[str],
    max_episodes: int | None = None,
    action_source: Literal["recorded", "wbc-goal", "state"] = "recorded",
):
    """Load source-demo playback arrays, including each episode's scene XML."""
    import h5py

    path = Path(dataset_path)
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"Missing HDF5 group 'data': {path}")
        data = handle["data"]
        if "script_config" not in data.attrs or "env_info" not in data.attrs:
            raise ValueError(f"Missing script_config or env_info metadata: {path}")
        script_config = json.loads(_decode_hdf5_attr(data.attrs["script_config"]))
        env_info = json.loads(_decode_hdf5_attr(data.attrs["env_info"]))
        recorded_seeds = env_info.get("seeds", [])
        demo_names = sorted(
            (name for name in data if name.startswith("demo_")),
            key=lambda value: int(value.split("_")[-1]),
        )
        if max_episodes is not None:
            demo_names = demo_names[:max_episodes]

        frames = {}
        seeds = []
        for output_index, demo_name in enumerate(demo_names, start=1):
            source_index = int(demo_name.split("_")[-1]) - 1
            demo = data[demo_name]
            if "model_file" not in demo.attrs:
                raise ValueError(f"{demo_name} is missing embedded model_file XML")
            if source_index >= len(recorded_seeds):
                raise ValueError(f"{demo_name} has no corresponding seed in env_info")

            states = np.asarray(demo["states"])
            recorded_actions = np.asarray(demo["actions"])
            navigate_commands = np.asarray(demo["wbc_goal/navigate_cmd"])
            base_height_commands = np.asarray(demo["teleop_cmd/base_height_command"])
            action_count = len(recorded_actions)
            if len(states) != action_count + 1:
                raise ValueError(
                    f"{demo_name} has {len(states)} states for {action_count} actions"
                )
            if len(navigate_commands) != action_count or len(base_height_commands) != action_count:
                raise ValueError(f"{demo_name} command lengths do not match its actions")

            output_name = f"demo_{output_index}"
            frames[f"data/{output_name}/states"] = list(states)
            frames[f"data/{output_name}/model_file"] = _decode_hdf5_attr(
                demo.attrs["model_file"]
            )
            episode_actions = []
            if action_source == "wbc-goal":
                wrist_poses = np.asarray(demo["wbc_goal/wrist_pose"])
                upper_body_poses = np.asarray(demo["wbc_goal/target_upper_body_pose"])
                if len(wrist_poses) != action_count or len(upper_body_poses) != action_count:
                    raise ValueError(f"{demo_name} WBC-goal lengths do not match its actions")
            elif action_source not in {"recorded", "state"}:
                raise ValueError(f"Invalid action_source: {action_source}")

            for index in range(action_count):
                action = {
                    "navigate_cmd": navigate_commands[index],
                    "base_height_command": base_height_commands[index],
                }
                if action_source == "recorded":
                    action["recorded_action"] = recorded_actions[index]
                elif action_source == "wbc-goal":
                    action["wrist_pose"] = wrist_poses[index]
                    action["target_upper_body_pose"] = upper_body_poses[index]
                episode_actions.append(action)
            frames[f"data/{output_name}/wbc_goal"] = episode_actions
            seeds.append(recorded_seeds[source_index])

    return seeds, frames, script_config


def load_playback_dataset(
    dataset_path: str | os.PathLike[str],
    max_episodes: int | None = None,
    action_source: Literal["recorded", "wbc-goal", "state"] = "recorded",
):
    path = Path(dataset_path)
    if path.is_file() and path.suffix in {".h5", ".hdf5"}:
        return load_hdf5_dataset(path, max_episodes, action_source)
    return load_lerobot_dataset(path, max_episodes, action_source)


def validate_state(recorded_state, playback_state, ep, step, tolerance=1e-5) -> bool:
    if recorded_state.shape != playback_state.shape:
        print(
            f"[warning] state shape changed from {recorded_state.shape} to "
            f"{playback_state.shape} for ep {ep} at step {step}"
        )
        return False
    if not np.allclose(recorded_state, playback_state, atol=tolerance, rtol=0.0):
        err = np.linalg.norm(recorded_state - playback_state)
        print(f"[warning] state diverged by {err:.12f} for ep {ep} at step {step}")
        return False
    return True


def write_video_frame(env, video_writer) -> None:
    from humanoidmimicgen.wbc_constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH

    import cv2

    base_env = env.base_env if hasattr(env, "base_env") else env
    render_camera = getattr(base_env, "render_camera", ["frontview"])
    camera_name = render_camera[0] if isinstance(render_camera, list) else render_camera
    img = base_env.sim.render(
        width=RS_VIEW_CAMERA_WIDTH,
        height=RS_VIEW_CAMERA_HEIGHT,
        camera_name=camera_name,
    )
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = np.flipud(img_bgr)
    video_writer.write(img_bgr)


def get_task_success(sync_env) -> bool:
    """Return the task success bit for the current replay state."""
    success = sync_env.is_success()
    if isinstance(success, dict):
        return bool(success.get("task", False))
    return bool(success)


def get_video_fps(config: PlaybackConfig) -> float:
    # Encode videos at the dataset collection frequency used by the G1 demos.
    return float(config.data_collection_frequency)


class ActionEnv:
    """Env facade for recorded low-level actions and regenerated WBC actions."""

    def __init__(
        self,
        sync_env,
        wbc_policy,
        default_base_height: float = DEFAULT_BASE_HEIGHT,
    ) -> None:
        self.sync_env = sync_env
        self.wbc_policy = wbc_policy
        self.default_base_height = default_base_height

    def __getattr__(self, name: str):
        return getattr(self.sync_env, name)

    def _apply_base_command(self, action: dict[str, Any]) -> None:
        self.sync_env.overwrite_floating_base_action(
            action.get("navigate_cmd", np.zeros(3)),
            action.get("base_height_command", self.default_base_height),
        )

    def step(self, action: dict[str, Any]):
        """Regenerate a low-level action from the recorded WBC goal."""
        obs = self.sync_env.observe()
        self.wbc_policy.set_observation(obs)
        self.wbc_policy.set_goal(action)
        self._apply_base_command(action)
        return self.sync_env.step(self.wbc_policy.get_action())

    def step_recorded(self, action: dict[str, Any]):
        """Apply the recorded low-level joint action without rerunning the WBC."""
        self._apply_base_command(action)
        return self.sync_env.step({"q": action["recorded_action"]})


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


def playback_result(
    states_match: bool,
    task_success_summary: dict,
    *,
    allow_state_divergence: bool,
    require_task_success: bool,
) -> bool:
    fidelity_ok = states_match or allow_state_divergence
    task_success_ok = (
        not require_task_success
        or task_success_summary["task_successes"] == task_success_summary["episodes"]
    )
    return fidelity_ok and task_success_ok


def playback_dataset(config: PlaybackConfig) -> bool:
    from humanoidmimicgen.wbc_constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
    from humanoidmimicgen.wbc_runtime import get_env, get_policies, get_robot_type_and_model

    if config.state_tolerance < 0:
        raise ValueError("state_tolerance must be non-negative")

    ret = True
    start_time = time.time()
    np.set_printoptions(precision=5, suppress=True, linewidth=120)

    assert config.dataset is not None, "Dataset must be specified for playback"
    seeds, frames, script_config = load_playback_dataset(
        config.dataset,
        config.num_episodes,
        action_source=config.action_source,
    )

    config.update(
        script_config,
        allowed_keys=[
            "wbc_version",
            "wbc_model_path",
            "wbc_policy_class",
            "control_frequency",
            "enable_waist",
            "robot",
            "task_name",
            "data_collection_frequency",
        ],
    )
    onscreen = False if config.save_video else config.enable_onscreen
    offscreen = True if config.save_video else config.enable_offscreen

    if config.save_video and config.video_path is None:
        video_folder = Path(config.dataset)
        video_folder.mkdir(parents=True, exist_ok=True)
        config.video_path = str(video_folder / "playback_video.mp4")
        print(f"Video recording enabled. Output: {config.video_path}")

    sync_env = get_env(config, onscreen=onscreen, offscreen=offscreen)
    wbc_policy = None
    if config.action_source == "wbc-goal":
        robot_type, robot_model = get_robot_type_and_model(
            config.robot, config.enable_waist
        )
        wbc_policy, _, _ = get_policies(
            config, robot_type, robot_model, activate_keyboard_listener=False
        )
    elif config.action_source not in {"recorded", "state"}:
        raise ValueError(f"Invalid action_source: {config.action_source}")
    env = ActionEnv(
        sync_env,
        wbc_policy,
        default_base_height=config.controller_initial_base_height,
    )

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
        actions = frames[f"data/{ep}/wbc_goal"]
        reset_payload = {"states": states[0]}
        model_file_key = f"data/{ep}/model_file"
        if model_file_key in frames:
            reset_payload["model_file"] = frames[model_file_key]
        env.reset_to(reset_payload)
        num_actions = min(len(actions), len(states) - 1)
        if config.debug:
            num_actions = min(20, num_actions)
        task_success_steps = 0
        first_task_success_step = None
        last_task_success = False

        for jj in range(num_actions):
            if config.action_source == "recorded":
                env.step_recorded(actions[jj])
            elif config.action_source == "state":
                env.reset_to({"states": states[jj + 1]})
            else:
                env.step(actions[jj])
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

            if config.action_source != "state" and jj < len(states) - 1:
                state_playback = env.base_env.sim.get_state().flatten()
                if not validate_state(
                    states[jj + 1],
                    state_playback,
                    ep,
                    jj,
                    tolerance=config.state_tolerance,
                ):
                    ret = False

        task_success_stats = {
            "task_success": task_success_steps > 0,
            "task_success_steps": task_success_steps,
            "first_task_success_step": first_task_success_step,
            "final_task_success": last_task_success,
            "checked_steps": num_actions,
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
    required_task_success_missing = (
        config.require_task_success
        and task_success_summary["task_successes"] != task_success_summary["episodes"]
    )
    if required_task_success_missing:
        print(
            f"{RED_BOLD}Required task success was not reached in every episode "
            f"({task_success_summary['task_successes']}/"
            f"{task_success_summary['episodes']}).{RESET}"
        )
    ret = playback_result(
        ret,
        task_success_summary,
        allow_state_divergence=config.allow_state_divergence,
        require_task_success=config.require_task_success,
    )

    elapsed_time = time.time() - start_time
    if config.action_source == "recorded":
        print(f"{GREEN_BOLD}Playback action source: recorded low-level actions{RESET}")
    elif config.action_source == "wbc-goal":
        print(
            f"{GREEN_BOLD}Playback action source: regenerated WBC actions "
            f"({config.wbc_version}, {config.wbc_model_path}, {config.wbc_policy_class}){RESET}"
        )
    else:
        print(
            f"{GREEN_BOLD}Playback source: recorded simulator states{RESET}\n"
            "State playback injects every stored state for faithful visualization; "
            "it does not validate action or physics fidelity."
        )
    if ret:
        print(f"{GREEN_BOLD}Playback completed successfully in {elapsed_time:.2f} seconds!{RESET}")
    else:
        print(
            f"{RED_BOLD}Playback did not satisfy the requested validation gates in "
            f"{elapsed_time:.2f} seconds!{RESET}"
        )
    return ret
