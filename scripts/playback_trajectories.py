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

"""Play back robosuite / RoboCasa trajectory datasets stored as HDF5."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.utils.robot_control_playback import make_mujoco_control_playback_env


INTERNAL_METADATA_KEYS = {
    "env_lang",
    "groot_commit_id",
    "mujoco_version",
    "robocasa_version",
    "robosuite_version",
    "seeds",
}

CAMERA_ALIASES = {
    "3pv": "frontview",
    "ego": "egoview",
    "ego_view": "egoview",
    "ego-view": "egoview",
    "third_person": "frontview",
    "third-person": "frontview",
    "thirdperson": "frontview",
    "tpv": "frontview",
}

REPLAY_ENV_ALIASES = {
    "LMDrillLift": "LMDrillLiftBi",
    "LMDrillLiftObstacle": "LMDrillLiftObstacleBi",
    "LMDrillPnP90": "LMDrillPnP90Bi",
    "LMPickDrillFromHolder": "LMPickDrillFromHolderStandingEasyFar",
}


def import_sim_stack() -> Any:
    # Import robocasa first so the trimmed package registers its environments
    # with robosuite before robosuite.make is called.
    import robocasa
    import robosuite

    return robosuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play back trajectories from a robosuite / RoboCasa HDF5 dataset. "
            "The input can be a demo.hdf5 file or a folder containing demo.hdf5."
        )
    )
    parser.add_argument("dataset", type=Path, help="Path to a demo HDF5 file or folder.")
    parser.add_argument(
        "--mode",
        choices=("state", "action", "auto"),
        default="action",
        help="Playback mode. Action playback is the default; state playback remains available for inspection.",
    )
    parser.add_argument(
        "--action-backend",
        choices=("auto", "env", "mujoco-control"),
        default="auto",
        help=(
            "Action execution backend. 'env' uses robosuite env.step; "
            "'mujoco-control' applies recorded whole-body controls to the saved MuJoCo XML."
        ),
    )
    parser.add_argument(
        "--episodes",
        help="Comma-separated episodes to play, e.g. demo_0,demo_5 or 0,5.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to play when --episodes and --all are not set.",
    )
    parser.add_argument("--all", action="store_true", help="Play every episode in the dataset.")
    parser.add_argument("--random", action="store_true", help="Sample episodes randomly.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --random episode sampling.")
    parser.add_argument("--start-step", type=int, default=0, help="First state / action index to play.")
    parser.add_argument("--max-steps", type=int, help="Maximum number of states / actions to play per episode.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="State playback stride. In action mode, every action is stepped and stride only throttles rendering.",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Live viewer playback rate.")
    parser.add_argument("--env-name", help="Override the environment name stored in the dataset.")
    parser.add_argument(
        "--robots",
        help="Override robot name(s). Use a comma-separated value for multi-robot envs.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        help="Optional RoboCasa assets root when assets are outside this checkout.",
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        help="Optional JSON file used as robosuite controller_configs.",
    )
    parser.add_argument(
        "--camera",
        default="egoview",
        help="Camera name for video export. Aliases such as '3pv' map to a third-person view.",
    )
    parser.add_argument(
        "--show-sites",
        action="store_true",
        help="Render MuJoCo site markers such as gripper / IK debug indicators.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Video frame width.")
    parser.add_argument("--height", type=int, default=720, help="Video frame height.")
    parser.add_argument(
        "--video-path",
        type=Path,
        help="Optional output video path. Requires the optional imageio dependency.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        help="Video frame rate. Defaults to --fps.",
    )
    parser.add_argument("--no-render", action="store_true", help="Run playback without rendering.")
    parser.add_argument(
        "--viewer-camera-id",
        type=int,
        default=0,
        help="Viewer camera id for live robosuite rendering.",
    )
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="In action mode, compare simulated states against recorded states.",
    )
    return parser.parse_args()


def resolve_dataset(path: Path) -> Path:
    dataset = path.expanduser()
    if dataset.is_dir():
        dataset = dataset / "demo.hdf5"
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    return dataset


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return value


def load_json_attr(attrs: h5py.AttributeManager, name: str) -> dict[str, Any] | None:
    if name not in attrs:
        return None
    value = decode_attr(attrs[name])
    if not value:
        return None
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Expected HDF5 attr {name!r} to be a JSON string, got {type(value).__name__}")


def metadata_to_env_kwargs(data_group: h5py.Group) -> dict[str, Any]:
    env_args = load_json_attr(data_group.attrs, "env_args")
    if env_args:
        kwargs = dict(env_args.get("env_kwargs", {}))
        if "env_name" not in kwargs and env_args.get("env_name"):
            kwargs["env_name"] = env_args["env_name"]
        return kwargs

    env_info = load_json_attr(data_group.attrs, "env_info")
    kwargs = dict(env_info or {})
    if "env_name" not in kwargs and "env" in data_group.attrs:
        kwargs["env_name"] = decode_attr(data_group.attrs["env"])
    return kwargs


def normalize_env_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = {key: value for key, value in kwargs.items() if key not in INTERNAL_METADATA_KEYS}
    env_name = kwargs.get("env_name")
    if not isinstance(env_name, str) or "/" not in env_name or not env_name.startswith("groot"):
        return kwargs

    task_spec = env_name.split("/", maxsplit=1)[1]
    parts = task_spec.split("_")
    if parts and parts[0]:
        kwargs["env_name"] = parts[0]
    if len(parts) > 1 and parts[1] and "robots" not in kwargs:
        kwargs["robots"] = parts[1]
    kwargs["env_name"] = REPLAY_ENV_ALIASES.get(kwargs["env_name"], kwargs["env_name"])
    return kwargs


def parse_robots(value: str) -> str | list[str]:
    robots = [robot.strip() for robot in value.split(",") if robot.strip()]
    if len(robots) == 1:
        return robots[0]
    return robots


def resolve_camera_name(camera: str) -> str:
    return CAMERA_ALIASES.get(camera, camera)


def configure_assets_root(args: argparse.Namespace) -> None:
    if not args.assets_root:
        return
    assets_root = args.assets_root.expanduser().resolve()
    if not assets_root.exists():
        raise FileNotFoundError(f"Assets root not found: {assets_root}")
    import robocasa.models

    robocasa.models.assets_root = str(assets_root)


def build_env_kwargs(data_group: h5py.Group, args: argparse.Namespace) -> dict[str, Any]:
    kwargs = normalize_env_kwargs(metadata_to_env_kwargs(data_group))
    if args.env_name:
        kwargs["env_name"] = args.env_name
    elif "env_name" in kwargs:
        kwargs["env_name"] = REPLAY_ENV_ALIASES.get(kwargs["env_name"], kwargs["env_name"])
    if args.robots:
        kwargs["robots"] = parse_robots(args.robots)
    if args.controller_config:
        with args.controller_config.expanduser().open("r", encoding="utf-8") as f:
            kwargs["controller_configs"] = json.load(f)
    if "env_name" not in kwargs:
        raise ValueError(
            "Could not infer an env name from dataset metadata. Pass --env-name and --robots explicitly."
        )

    live_render = not args.no_render and args.video_path is None
    kwargs["has_renderer"] = live_render
    kwargs["has_offscreen_renderer"] = args.video_path is not None
    kwargs["ignore_done"] = True
    kwargs["use_camera_obs"] = False
    kwargs.setdefault("control_freq", 20)
    if args.video_path:
        kwargs.setdefault("camera_names", [resolve_camera_name(args.camera)])
        kwargs.setdefault("camera_widths", args.width)
        kwargs.setdefault("camera_heights", args.height)
    return kwargs


def load_script_config(data_group: h5py.Group) -> dict[str, Any]:
    return load_json_attr(data_group.attrs, "script_config") or {}


def infer_action_control_freq(data_group: h5py.Group, env_kwargs: dict[str, Any]) -> float:
    script_config = load_script_config(data_group)
    for key in ("data_collection_frequency", "control_frequency"):
        value = script_config.get(key)
        if value:
            return float(value)
    return float(env_kwargs.get("control_freq", 20))


def load_lerobot_action_names(dataset_path: Path) -> list[str] | None:
    info_path = dataset_path.parent / "meta" / "info.json"
    if not info_path.exists():
        return None
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    names = info.get("features", {}).get("action", {}).get("names")
    return names if isinstance(names, list) else None


def natural_episode_key(name: str) -> tuple[str, int]:
    prefix, _, suffix = name.rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return name, -1


def select_episodes(data_group: h5py.Group, args: argparse.Namespace) -> list[str]:
    episodes = sorted(data_group.keys(), key=natural_episode_key)
    if args.episodes:
        selected = []
        for item in args.episodes.split(","):
            item = item.strip()
            episode = f"demo_{item}" if item.isdigit() else item
            if episode not in data_group:
                raise KeyError(f"Episode {episode!r} not found. Available examples: {episodes[:5]}")
            selected.append(episode)
        return selected

    if args.all:
        selected = episodes
    else:
        selected = episodes[: max(args.num_episodes, 0)]
    if args.random:
        rng = random.Random(args.seed)
        selected = rng.sample(episodes, k=min(len(episodes), len(selected)))
    return selected


def episode_dataset(episode_group: h5py.Group, key: str) -> np.ndarray | None:
    if key not in episode_group:
        return None
    return np.asarray(episode_group[key][()])


def configure_visual_options(vopt: Any, args: argparse.Namespace) -> None:
    if hasattr(vopt, "geomgroup"):
        vopt.geomgroup[0] = 0
        vopt.geomgroup[1] = 1
    if not args.show_sites and hasattr(vopt, "sitegroup"):
        vopt.sitegroup[:] = 0


def configure_render_context(env: Any, args: argparse.Namespace) -> None:
    context = getattr(env.sim, "_render_context_offscreen", None)
    if context is not None:
        configure_visual_options(context.vopt, args)
    viewer = getattr(env, "viewer", None)
    if viewer is not None and hasattr(viewer, "vopt"):
        configure_visual_options(viewer.vopt, args)


def read_model_xml(dataset_path: Path, episode_group: h5py.Group) -> str | None:
    if "model_file" not in episode_group.attrs:
        return None
    model_file = decode_attr(episode_group.attrs["model_file"])
    if not isinstance(model_file, str):
        return None
    if model_file.lstrip().startswith("<"):
        return model_file

    for candidate in (dataset_path.parent / model_file, dataset_path.parent / "models" / model_file):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return model_file


def reset_env_from_episode(
    env: Any,
    dataset_path: Path,
    episode_group: h5py.Group,
    args: argparse.Namespace,
    mode: str,
) -> None:
    env.reset()
    model_xml = read_model_xml(dataset_path, episode_group)
    if model_xml:
        if hasattr(env, "edit_model_xml"):
            model_xml = env.edit_model_xml(model_xml)
        if mode == "state":
            from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

            env.sim = MjSim.from_xml_string(model_xml)
            env.sim.reset()
            if args.video_path:
                MjRenderContextOffscreen(
                    env.sim,
                    device_id=-1,
                    max_width=max(args.width, 640),
                    max_height=max(args.height, 480),
                )
                configure_render_context(env, args)
        else:
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            configure_render_context(env, args)
    if not args.no_render and args.video_path is None and getattr(env, "viewer", None) is not None:
        env.viewer.set_camera(args.viewer_camera_id)
        configure_render_context(env, args)


def resolve_mode(args: argparse.Namespace, states: np.ndarray | None, actions: np.ndarray | None) -> str:
    if args.mode == "auto":
        return "action" if actions is not None else "state"
    if args.mode == "state" and states is None:
        raise ValueError("State playback requested, but the episode has no 'states' dataset.")
    if args.mode == "action" and actions is None:
        raise ValueError("Action playback requested, but the episode has no 'actions' dataset.")
    return args.mode


def open_video_writer(args: argparse.Namespace) -> Any:
    if not args.video_path:
        return None
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("Video export requires imageio. Install with: python -m pip install '.[video]'") from exc

    args.video_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(args.video_path, fps=args.video_fps or args.fps)


def prepare_mujoco_control_env(
    env: Any,
    dataset_path: Path,
    episode_group: h5py.Group,
    actions: np.ndarray,
    action_names: list[str] | None,
    control_freq: float,
    args: argparse.Namespace,
) -> Any:
    model_xml = read_model_xml(dataset_path, episode_group)
    if model_xml is None:
        raise ValueError("MuJoCo-control action playback requires episode model XML.")
    if hasattr(env, "edit_model_xml"):
        model_xml = env.edit_model_xml(model_xml)

    from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

    sim = MjSim.from_xml_string(model_xml)
    sim.reset()
    action_env = make_mujoco_control_playback_env(sim, actions, action_names, control_freq)
    if args.video_path:
        MjRenderContextOffscreen(
            sim,
            device_id=-1,
            max_width=max(args.width, 640),
            max_height=max(args.height, 480),
        )
        configure_render_context(action_env, args)
    return action_env


def render_frame(env: Any, args: argparse.Namespace) -> np.ndarray:
    configure_render_context(env, args)
    frame = env.sim.render(width=args.width, height=args.height, camera_name=resolve_camera_name(args.camera))
    if isinstance(frame, tuple):
        frame = frame[0]
    return np.asarray(frame)[::-1]


def maybe_render(
    env: Any,
    args: argparse.Namespace,
    writer: Any,
    frame_index: int,
    render_stride: int = 1,
) -> None:
    if render_stride > 1 and frame_index % render_stride != 0:
        return
    if writer is not None:
        writer.append_data(render_frame(env, args))
        return
    if args.no_render:
        return
    if getattr(env, "renderer", None) == "mjviewer" and getattr(env, "viewer", None) is not None:
        env.viewer.update()
    env.render()
    if args.fps > 0:
        time.sleep(1.0 / args.fps)


def playback_states(env: Any, states: np.ndarray, args: argparse.Namespace, writer: Any) -> int:
    start = max(args.start_step, 0)
    stop = len(states) if args.max_steps is None else min(len(states), start + args.max_steps)
    played = 0
    for frame_index, state_index in enumerate(range(start, stop, max(args.stride, 1))):
        env.sim.set_state_from_flattened(states[state_index])
        env.sim.forward()
        maybe_render(env, args, writer, frame_index)
        played += 1
    return played


def playback_actions(
    env: Any,
    states: np.ndarray | None,
    actions: np.ndarray,
    args: argparse.Namespace,
    writer: Any,
    episode: str,
) -> int:
    start = max(args.start_step, 0)
    stop = len(actions) if args.max_steps is None else min(len(actions), start + args.max_steps)
    if states is not None and start < len(states):
        env.sim.set_state_from_flattened(states[start])
        env.sim.forward()
    played = 0
    for frame_index, action_index in enumerate(range(start, stop)):
        env.step(actions[action_index])
        maybe_render(env, args, writer, frame_index, render_stride=args.stride)
        if args.check_drift and states is not None and action_index + 1 < len(states):
            state_playback = env.sim.get_state().flatten()
            err = np.linalg.norm(states[action_index + 1] - state_playback)
            if err > 1e-8:
                print(f"[warning] {episode} action step {action_index} drifted by {err:.6g}")
        played += 1
    return played


def resolve_action_backend(
    args: argparse.Namespace,
    env: Any,
    dataset_path: Path,
    episode_group: h5py.Group,
    actions: np.ndarray,
) -> str:
    action_dim = actions.shape[1] if actions.ndim == 2 else 1
    env_action_dim = getattr(env, "action_dim", None)
    if args.action_backend == "env" and env_action_dim is not None and action_dim != env_action_dim:
        raise ValueError(
            f"Episode actions have dim {action_dim}, but the robosuite env action_dim is {env_action_dim}. "
            "Pass a matching --controller-config or use --action-backend mujoco-control."
        )
    if args.action_backend != "auto":
        return args.action_backend
    if action_dim == env_action_dim:
        return "env"
    if read_model_xml(dataset_path, episode_group) is not None:
        return "mujoco-control"
    return "env"


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    dataset_path = resolve_dataset(args.dataset)

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{dataset_path} does not contain a top-level 'data' group.")
        data_group = f["data"]
        env_kwargs = build_env_kwargs(data_group, args)
        control_freq = infer_action_control_freq(data_group, env_kwargs)
        action_names = load_lerobot_action_names(dataset_path)
        episodes = select_episodes(data_group, args)
        print(f"Creating env: {env_kwargs['env_name']}")
        robosuite_module = import_sim_stack()
        configure_assets_root(args)
        env = robosuite_module.make(**env_kwargs)
        writer = open_video_writer(args)
        try:
            for episode in episodes:
                episode_group = data_group[episode]
                states = episode_dataset(episode_group, "states")
                actions = episode_dataset(episode_group, "actions")
                mode = resolve_mode(args, states, actions)
                if mode == "state":
                    reset_env_from_episode(env, dataset_path, episode_group, args, mode)
                    count = playback_states(env, states, args, writer)
                else:
                    backend = resolve_action_backend(args, env, dataset_path, episode_group, actions)
                    print(f"Using action backend for {episode}: {backend}")
                    if backend == "env":
                        reset_env_from_episode(env, dataset_path, episode_group, args, mode)
                        count = playback_actions(env, states, actions, args, writer, episode)
                    else:
                        action_env = prepare_mujoco_control_env(
                            env,
                            dataset_path,
                            episode_group,
                            actions,
                            action_names,
                            control_freq,
                            args,
                        )
                        count = playback_actions(action_env, states, actions, args, writer, episode)
                        action_env.close()
                print(f"Played {episode}: {count} {mode} steps")
        finally:
            if writer is not None:
                writer.close()
            if hasattr(env, "close"):
                env.close()


if __name__ == "__main__":
    main()
