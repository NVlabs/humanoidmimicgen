#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate an HMG 64/50 Diffusion Policy with native WBC.

Install the exact tested runtime first::

    pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEROBOT_REPOSITORY = "https://github.com/huggingface/lerobot.git"
LEROBOT_VERSION = "0.4.4"
LEROBOT_COMMIT = "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
STATE_KEY = "observation.state"
IMAGE_KEY = "observation.images.ego_view"
ACTION_KEY = "action"
STATE_DIM = 43
UPPER_BODY_DIM = 31
NAVIGATION_DIM = 3
ACTION_DIM = 35
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
TRAINING_STATE_FILES = (
    "optimizer_param_groups.json",
    "optimizer_state.safetensors",
    "rng_state.safetensors",
    "scheduler_state.json",
    "training_step.json",
)
TASKS = {
    "01_box_lift_floor": "LMBoxLiftFloor",
    "02_push_button": "LMPushButton",
    "03_box_lift": "LMBoxLift",
    "04_push_shelf_forward": "LMPushShelfForward",
    "05_drill_lift": "LMDrillLift",
    "06_drill_pnp": "LMDrillPnP90",
    "07_box_table_to_shelf": "LMBoxTableToShelf",
    "08_pick_drill_from_holder": "LMPickDrillFromHolder",
    "09_obstacle_aware_pick_drill": "LMDrillLiftObstacle",
}


def require_lerobot_runtime() -> dict[str, str]:
    """Fail unless LeRobot came from the exact tested source commit."""
    try:
        distribution = metadata.distribution("lerobot")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Install the pinned LeRobot runtime shown in --help") from exc
    direct_url = distribution.read_text("direct_url.json")
    if distribution.version != LEROBOT_VERSION or not direct_url:
        raise RuntimeError(
            f"LeRobot must be {LEROBOT_VERSION} installed from {LEROBOT_COMMIT}"
        )
    payload = json.loads(direct_url)
    commit = (payload.get("vcs_info") or {}).get("commit_id")
    if payload.get("url") != LEROBOT_REPOSITORY or commit != LEROBOT_COMMIT:
        raise RuntimeError(
            f"LeRobot must come from {LEROBOT_REPOSITORY}@{LEROBOT_COMMIT}"
        )
    return {
        "repository": LEROBOT_REPOSITORY,
        "version": LEROBOT_VERSION,
        "commit": LEROBOT_COMMIT,
    }


def _safetensor_count(path: Path) -> int:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        count = len(handle.keys())
    if count < 1:
        raise ValueError(f"Safetensors file is empty: {path}")
    return count


def validate_checkpoint_artifacts(checkpoint_dir: Path) -> tuple[Path, dict[str, int]]:
    """Open every inference file and, for full checkpoints, every resume state."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    model_dir = checkpoint_dir / "pretrained_model"
    full_checkpoint = model_dir.is_dir()
    if not full_checkpoint:
        model_dir = checkpoint_dir

    missing = [str(model_dir / name) for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing policy checkpoint files: " + ", ".join(missing))
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        validate_policy_config(json.load(handle))
    summary = {"model_tensors": _safetensor_count(model_dir / "model.safetensors")}

    if not full_checkpoint:
        return model_dir, summary

    training_state = checkpoint_dir / "training_state"
    required = [model_dir / "train_config.json"] + [
        training_state / name for name in TRAINING_STATE_FILES
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing resumable checkpoint files: " + ", ".join(missing))

    with (training_state / "training_step.json").open(encoding="utf-8") as handle:
        step = json.load(handle).get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError(f"Invalid training step: {step!r}")
    if checkpoint_dir.name.isdigit() and int(checkpoint_dir.name) != step:
        raise ValueError(
            f"Checkpoint directory {checkpoint_dir.name} disagrees with training step {step}"
        )

    with (training_state / "scheduler_state.json").open(encoding="utf-8") as handle:
        scheduler_state = json.load(handle)
    if scheduler_state.get("last_epoch") != step:
        raise ValueError(
            "Scheduler state disagrees with training step: "
            f"{scheduler_state.get('last_epoch')!r} != {step}"
        )

    with (training_state / "optimizer_param_groups.json").open(encoding="utf-8") as handle:
        optimizer_groups = json.load(handle)
    if not isinstance(optimizer_groups, list) or not optimizer_groups:
        raise ValueError("Optimizer parameter groups must be a non-empty list")
    optimizer_parameters = sum(
        len(group.get("params", []))
        for group in optimizer_groups
        if isinstance(group, Mapping)
    )
    if optimizer_parameters < 1:
        raise ValueError("Optimizer parameter groups contain no parameters")

    # These files encode different objects, so their tensor counts are validated
    # independently rather than compared with the model tensor count.
    summary.update(
        training_step=step,
        optimizer_parameters=optimizer_parameters,
        optimizer_tensors=_safetensor_count(training_state / "optimizer_state.safetensors"),
        rng_tensors=_safetensor_count(training_state / "rng_state.safetensors"),
    )
    return model_dir, summary


def _shape(feature: Mapping[str, Any], key: str) -> tuple[int, ...]:
    raw = feature.get("shape")
    if not isinstance(raw, (list, tuple)) or not all(isinstance(value, int) for value in raw):
        raise ValueError(f"Feature {key!r} has invalid shape: {raw!r}")
    return tuple(raw)


def validate_policy_config(config: Mapping[str, Any]) -> None:
    """Fail before a checkpoint with a mismatched data/action contract can run."""
    inputs = config.get("input_features")
    outputs = config.get("output_features")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("Checkpoint must define input_features and output_features")
    if set(inputs) != {STATE_KEY, IMAGE_KEY} or set(outputs) != {ACTION_KEY}:
        raise ValueError("Expected only 43D state + ego image -> 35D action")
    if _shape(inputs[STATE_KEY], STATE_KEY) != (STATE_DIM,):
        raise ValueError(f"{STATE_KEY} must have shape [{STATE_DIM}]")
    image_shape = _shape(inputs[IMAGE_KEY], IMAGE_KEY)
    if len(image_shape) != 3 or 3 not in (image_shape[0], image_shape[-1]):
        raise ValueError(f"{IMAGE_KEY} must be a three-channel image")
    if _shape(outputs[ACTION_KEY], ACTION_KEY) != (ACTION_DIM,):
        raise ValueError(f"{ACTION_KEY} must have shape [{ACTION_DIM}]")
    expected = {
        "n_obs_steps": 1,
        "horizon": 64,
        "n_action_steps": 50,
        "drop_n_last_frames": 49,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value}, got {config.get(key)!r}")
    if tuple(config.get("resize_shape") or ()) != (256, 256):
        raise ValueError("resize_shape must be [256, 256]")
    mapping = {
        str(key).upper(): str(value).upper()
        for key, value in (config.get("normalization_mapping") or {}).items()
    }
    for key, value in {"ACTION": "MIN_MAX", "STATE": "MIN_MAX", "VISUAL": "MEAN_STD"}.items():
        if mapping.get(key) != value:
            raise ValueError(f"Normalization for {key} must be {value}")


def policy_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    state = np.asarray(observation["q"], dtype=np.float32)
    image = np.asarray(observation["video.ego_view"])
    if state.shape != (STATE_DIM,):
        raise ValueError(f"HMG q must have shape [{STATE_DIM}], got {state.shape}")
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC ego image, got {image.shape}")
    if image.shape[-1] == 3:
        image = np.moveaxis(image, -1, 0)
    elif image.shape[0] != 3:
        raise ValueError(f"Expected a three-channel ego image, got {image.shape}")
    image = image.astype(np.float32)
    if np.asarray(observation["video.ego_view"]).dtype == np.uint8:
        image /= 255.0
    return {STATE_KEY: state, IMAGE_KEY: image}


def action_to_wbc_goal(action: Any) -> dict[str, np.ndarray]:
    target = np.asarray(action, dtype=np.float64)
    if target.shape != (ACTION_DIM,):
        raise ValueError(f"DP action must have shape [{ACTION_DIM}], got {target.shape}")
    return {
        "target_upper_body_pose": target[:UPPER_BODY_DIM],
        "navigate_cmd": target[UPPER_BODY_DIM : UPPER_BODY_DIM + NAVIGATION_DIM],
        "base_height_command": target[-1:],
    }


class LeRobotDPPolicy:
    """Thin inference adapter around upstream LeRobot's saved processors/model."""

    def __init__(self, checkpoint_dir: Path, device: str) -> None:
        model_dir, self.checkpoint_validation = validate_checkpoint_artifacts(checkpoint_dir)
        with (model_dir / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        normalization = config["normalization_mapping"]
        self.min_max_keys = tuple(
            key
            for key, feature in config["input_features"].items()
            if normalization[feature["type"]] == "MIN_MAX"
        )

        import torch
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        self.model = DiffusionPolicy.from_pretrained(model_dir).to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.model.config, pretrained_path=str(model_dir)
        )

    def reset(self) -> None:
        self.model.reset()

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        batch = self.preprocessor(
            {
                key: self.torch.as_tensor(value)
                for key, value in policy_observation(observation).items()
            }
        )
        for key in self.min_max_keys:
            batch[key] = batch[key].clip(-1.0, 1.0)
        with self.torch.inference_mode():
            action = self.model.select_action(batch)
        return self.postprocessor(action).detach().cpu().numpy().reshape(-1)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--wbc-policy-dir",
        type=Path,
        help="directory containing stand.onnx and walk.onnx from download_wbc_policies",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminate-on-success", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.num_episodes < 1 or args.max_episode_steps < 1:
        raise ValueError("episode counts and step limits must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("WANDB_MODE", "disabled")
    provenance = require_lerobot_runtime()

    import humanoidmimicgen.locomanipulation  # noqa: F401
    from humanoidmimicgen.dataset_playback import ActionEnv, PlaybackConfig, get_task_success
    from humanoidmimicgen.wbc_runtime import (
        get_env,
        get_policies,
        get_robot_type_and_model,
    )

    wbc_model_path = PlaybackConfig.wbc_model_path
    if args.wbc_policy_dir is not None:
        policy_dir = args.wbc_policy_dir.expanduser().resolve()
        stand_path = policy_dir / "stand.onnx"
        walk_path = policy_dir / "walk.onnx"
        missing = [str(path) for path in (stand_path, walk_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing WBC policies: {', '.join(missing)}")
        wbc_model_path = f"{stand_path},{walk_path}"
    config = PlaybackConfig(
        task_name=TASKS[args.task],
        enable_offscreen=True,
        enable_onscreen=False,
        wbc_model_path=wbc_model_path,
    )
    robot_type, robot_model = get_robot_type_and_model(
        config.robot, config.enable_waist
    )
    sync_env = get_env(
        config,
        onscreen=False,
        offscreen=True,
        camera_names=["robot0_oak_egoview"],
        camera_heights=[480],
        camera_widths=[640],
    )
    policy = LeRobotDPPolicy(args.checkpoint_dir, args.device)
    print(
        "HMG_CHECKPOINT_VALIDATED "
        + json.dumps(policy.checkpoint_validation, sort_keys=True),
        flush=True,
    )
    successes, episode_steps = [], []
    try:
        for episode in range(args.num_episodes):
            policy.reset()
            observation, _ = sync_env.reset(seed=args.seed + episode)
            wbc_policy, _, _ = get_policies(
                config, robot_type, robot_model, activate_keyboard_listener=False
            )
            env = ActionEnv(sync_env, wbc_policy)
            success = False
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < args.max_episode_steps:
                action = action_to_wbc_goal(policy.predict(observation))
                observation, _, terminated, truncated, _ = env.step(action)
                success = success or get_task_success(sync_env)
                steps += 1
                if success and args.terminate_on_success:
                    terminated = True
            successes.append(success)
            episode_steps.append(steps)
            print(
                f"[{args.task}] episode {episode + 1}/{args.num_episodes} "
                f"success={success} steps={steps}",
                flush=True,
            )
    finally:
        sync_env.close()

    result = {
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "task_preset": args.task,
        "task_name": config.task_name,
        "policy_type": "lerobot_diffusion_hmg_release_v1",
        "checkpoint_validation": policy.checkpoint_validation,
        "lerobot_upstream": provenance,
        "num_episodes": args.num_episodes,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
        "episode_successes": successes,
        "episode_steps": episode_steps,
        "success_rate": sum(successes) / len(successes),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
