#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay an HMG dataset using recorded actions, WBC goals, or states."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="LeRobot dataset root or source-demo HDF5 file.",
    )
    parser.add_argument(
        "--action-source",
        choices=("recorded", "wbc-goal", "state"),
        default="recorded",
        help=(
            "Use recorded low-level actions (default), regenerate actions from "
            "WBC goals, or inject every stored state for visualization. State "
            "playback does not validate action or physics fidelity."
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        help="Number of complete episodes to replay. Defaults to all episodes.",
    )
    parser.add_argument(
        "--state-atol",
        type=float,
        default=1e-5,
        help="Absolute state-comparison tolerance with zero relative tolerance.",
    )
    parser.add_argument(
        "--allow-state-divergence",
        action="store_true",
        help="Return success despite state divergence. Intended only for inspection renders.",
    )
    parser.add_argument(
        "--require-task-success",
        action="store_true",
        help="Return failure unless every selected episode reaches the task predicate.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        help="Optional MP4 path. Rendering is disabled when omitted.",
    )
    parser.add_argument(
        "--lowres-video-path",
        type=Path,
        help="Optional downscaled copy of --video-path, written with ffmpeg.",
    )
    parser.add_argument(
        "--lowres-width",
        type=int,
        default=320,
        help="Width of --lowres-video-path.",
    )
    parser.add_argument(
        "--sim-frequency",
        type=int,
        default=200,
        help="MuJoCo/WBC simulation frequency in Hz.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Replay only the first 20 actions of each selected episode.",
    )
    parser.add_argument(
        "--mujoco-gl",
        default="egl",
        help="MUJOCO_GL backend used when rendering.",
    )
    args = parser.parse_args(argv)
    if args.num_episodes is not None and args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.state_atol < 0:
        parser.error("--state-atol must be non-negative")
    if args.sim_frequency <= 0:
        parser.error("--sim-frequency must be positive")
    if args.lowres_width <= 0:
        parser.error("--lowres-width must be positive")
    if args.lowres_video_path is not None and args.video_path is None:
        parser.error("--lowres-video-path requires --video-path")
    return args


def resolve_dataset(path: Path) -> Path:
    dataset = path.expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset}")
    if dataset.is_file():
        if dataset.suffix not in {".h5", ".hdf5"}:
            raise ValueError(f"Unsupported playback file: {dataset}")
        return dataset
    if not (dataset / "meta" / "episodes.jsonl").is_file():
        raise FileNotFoundError(
            f"{dataset} does not look like a LeRobot dataset root; "
            "missing meta/episodes.jsonl"
        )
    return dataset


def maybe_downscale(source: Path, destination: Path | None, width: int) -> None:
    if destination is None:
        return
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale={width}:-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    os.environ["HMG_SIMULATION_TIMESTEP"] = str(1.0 / float(args.sim_frequency))
    dataset = resolve_dataset(args.dataset)

    try:
        from humanoidmimicgen.dataset_playback import (
            PlaybackConfig,
            playback_dataset,
        )
    except ImportError as exc:
        raise ImportError(
            "Dataset playback requires HumanoidMimicGen with the wbc-replay extra. "
            'Install with: python -m pip install -e ".[wbc-replay]"'
        ) from exc

    video_path = None
    if args.video_path is not None:
        video_path = args.video_path.expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)

    config = PlaybackConfig(
        dataset=str(dataset),
        save_video=video_path is not None,
        video_path=None if video_path is None else str(video_path),
        num_episodes=args.num_episodes,
        debug=args.debug,
        sim_frequency=args.sim_frequency,
        state_tolerance=args.state_atol,
        action_source=args.action_source,
        require_task_success=args.require_task_success,
        allow_state_divergence=args.allow_state_divergence,
        enable_onscreen=False,
        enable_offscreen=video_path is not None,
    )
    states_match = bool(playback_dataset(config))
    if video_path is not None:
        maybe_downscale(video_path, args.lowres_video_path, args.lowres_width)
    if not states_match:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
