# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run random actions in a HumanoidMimicGen WBC RoboCasa env."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from humanoidmimicgen.wbc.main.constants import DEFAULT_BASE_HEIGHT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random upper-body actions with explicit base navigation velocities."
    )
    parser.add_argument(
        "--task",
        "--env",
        dest="task",
        default="LMPushButton",
        help="HumanoidMimicGen task name, e.g. LMPushButton.",
    )
    parser.add_argument(
        "--robot",
        default="G1",
        help="Active-base robot name.",
    )
    parser.add_argument(
        "--wbc-version",
        "--wbc_version",
        dest="wbc_version",
        choices=(
            "homie_v2",
            "homie_v2_grav_comp",
            "homie_v2_grav_comp_tuned",
            "homie_v2_grav_comp_tuned_legs",
        ),
        help="Optional WBC/controller preset override. Defaults to SyncSimPlaybackConfig.",
    )
    parser.add_argument(
        "--wbc-model-path",
        "--wbc_model_path",
        dest="wbc_model_path",
        help="Optional lower-body ONNX path override for the selected WBC version.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="Render on screen.")
    parser.add_argument(
        "--video-path",
        "--video_path",
        dest="video_path",
        type=Path,
        help="Optional MP4 path for offscreen render.",
    )
    parser.add_argument("--camera", help="Render camera. Defaults to the WBC env camera.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--arm-scale",
        "--action-scale",
        dest="arm_scale",
        type=float,
        default=0.06,
        help="Uniform random arm target perturbation in radians.",
    )
    parser.add_argument(
        "--joint-group",
        default="arms",
        help="Upper-body robot-model joint group to perturb.",
    )
    parser.add_argument(
        "--arm-mode",
        choices=("random", "hold", "zero"),
        default="random",
        help="Arm target mode: random perturbation, hold reset pose, or absolute zero joints.",
    )
    parser.add_argument(
        "--navigate-cmd",
        "--nav-cmd",
        dest="navigate_cmd",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "VYAW"),
        help="Base velocity command [vx vy vyaw].",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=5,
        help="Hold each sampled action for this many env steps.",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=DEFAULT_BASE_HEIGHT,
        help="Standing base-height command for the WBC lower body.",
    )
    parser.add_argument(
        "--lower-body-mode",
        choices=("hold", "policy"),
        default="policy",
        help="Use 'policy' for Homie stand/walk or 'hold' to reuse current lower-body joints.",
    )
    parser.add_argument(
        "--control-freq",
        "--control_freq",
        dest="control_freq",
        type=int,
        default=50,
        help="WBC control frequency. Keep at 50Hz to match playback/control scripts.",
    )
    parser.add_argument(
        "--mujoco-gl",
        "--mujoco_gl",
        dest="mujoco_gl",
        default="egl",
        help="MUJOCO_GL backend.",
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.action_repeat < 1:
        parser.error("--action-repeat must be at least 1")
    if args.video_path is not None:
        args.video_path = args.video_path.expanduser().resolve()
    return args


def camera_name(env, requested: str | None) -> str:
    if requested is not None:
        return requested
    render_camera = env.base_env.render_camera
    return render_camera[0] if isinstance(render_camera, (list, tuple)) else render_camera


def write_video_frame(env, writer, args: argparse.Namespace) -> None:
    import cv2

    frame = env.base_env.sim.render(
        width=args.width,
        height=args.height,
        camera_name=camera_name(env, args.camera),
    )
    writer.write(cv2.cvtColor(np.flipud(frame), cv2.COLOR_RGB2BGR))


def make_action(
    reset_upper_body_pose: np.ndarray,
    robot_model,
    joint_group: str,
    rng: np.random.Generator,
    arm_mode: str,
    scale: float,
    navigate_cmd: np.ndarray,
    base_height: float,
    control_freq: int,
):
    target = reset_upper_body_pose.copy()
    target_joint_ids = set(robot_model.get_joint_group_indices(joint_group))
    target_indices = [
        target_index
        for target_index, joint_id in enumerate(
            robot_model.get_joint_group_indices("upper_body")
        )
        if joint_id in target_joint_ids
    ]
    if len(target_indices) == 0:
        raise ValueError(f"{joint_group!r} has no joints in the upper-body action")
    if arm_mode == "random":
        target[target_indices] += rng.uniform(-scale, scale, size=len(target_indices))
    elif arm_mode == "zero":
        target[target_indices] = 0.0
    target_time = time.monotonic()
    return {
        "target_time": target_time,
        "interpolation_garbage_collection_time": target_time - 2.0 / control_freq,
        "target_upper_body_pose": target,
        "navigate_cmd": navigate_cmd.copy(),
        "base_height_command": base_height,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    from humanoidmimicgen.wbc_goal_playback import (
        ActionEnv,
        SyncSimPlaybackConfig,
    )
    from humanoidmimicgen.wbc_runtime import get_env, get_policies, get_robot_type_and_model

    rng = np.random.default_rng(args.seed)
    navigate_cmd = np.asarray(args.navigate_cmd, dtype=np.float32)
    video_enabled = args.video_path is not None
    config = SyncSimPlaybackConfig(
        task_name=args.task,
        robot=args.robot,
        control_frequency=args.control_freq,
        enable_offscreen=video_enabled,
        enable_onscreen=args.render,
    )
    if args.wbc_version is not None:
        config.wbc_version = args.wbc_version
    if args.wbc_model_path is not None:
        config.wbc_model_path = args.wbc_model_path
    env_kwargs = {
        "onscreen": args.render,
        "offscreen": video_enabled,
    }
    if args.camera is not None:
        env_kwargs["render_camera"] = [args.camera]
    robot_type, robot_model = get_robot_type_and_model(config.robot, config.enable_waist)
    sync_env = get_env(config, **env_kwargs)
    wbc_policy, _, _ = get_policies(
        config, robot_type, robot_model, activate_keyboard_listener=False
    )
    lower_body_policy = getattr(
        wbc_policy.lower_body_policy, "locomotion_policy", wbc_policy.lower_body_policy
    )
    lower_body_policy.use_policy_action = args.lower_body_mode == "policy"
    print("WBC version:", config.wbc_version)
    print(
        "Lower-body mode:",
        args.lower_body_mode,
        f"(use_policy_action={lower_body_policy.use_policy_action})",
    )
    print("Navigation command [vx, vy, vyaw]:", navigate_cmd)
    print("Arm mode:", args.arm_mode)
    print("Base height command:", args.base_height)
    env = ActionEnv(sync_env, wbc_policy, default_base_height=args.base_height)

    writer = None
    start_base_xyz = None
    end_base_xyz = None
    try:
        if args.video_path is not None:
            import cv2

            args.video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(args.video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.control_freq),
                (args.width, args.height),
            )

        obs, _ = env.reset(seed=args.seed)
        start_base_xyz = obs["floating_base_pose"][:3].copy()
        end_base_xyz = start_base_xyz.copy()
        reset_upper_body_pose = obs["q"][
            robot_model.get_joint_group_indices("upper_body")
        ].copy()

        for step in range(args.steps):
            start = time.time()
            if step % args.action_repeat == 0:
                action = make_action(
                    reset_upper_body_pose,
                    robot_model,
                    args.joint_group,
                    rng,
                    args.arm_mode,
                    args.arm_scale,
                    navigate_cmd,
                    args.base_height,
                    args.control_freq,
                )
            obs, _reward, terminated, truncated, _info = env.step(action)
            end_base_xyz = obs["floating_base_pose"][:3].copy()
            done = terminated or truncated
            if args.render:
                env.render()
            if writer is not None:
                write_video_frame(env, writer, args)
            if args.render:
                elapsed = time.time() - start
                time.sleep(max(0.0, 1.0 / args.control_freq - elapsed))
            if done:
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"saved video to {args.video_path}")
        if start_base_xyz is not None and end_base_xyz is not None:
            displacement = end_base_xyz - start_base_xyz
            print("Base displacement xyz:", displacement)
            print("Base displacement norm:", float(np.linalg.norm(displacement[:2])))
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
