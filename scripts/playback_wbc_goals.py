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

"""Replay G1 loco-manipulation LeRobot episodes through stored WBC goals."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a LeRobot dataset with HumanoidMimicGen's stored WBC-goal playback path. "
            "Pass a LeRobot dataset root; if demo.hdf5 is passed, its parent is used."
        )
    )
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root, or a demo.hdf5 inside it.")
    parser.add_argument("--video-path", type=Path, required=True, help="Raw replay MP4 output path.")
    parser.add_argument(
        "--lowres-video-path",
        type=Path,
        help="Optional downscaled MP4 written with ffmpeg after raw replay completes.",
    )
    parser.add_argument("--lowres-width", type=int, default=320, help="Width for --lowres-video-path.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        help="Optional number of complete episodes to replay. Defaults to all episodes.",
    )
    parser.add_argument("--ci-test", action="store_true", help="Run only the first 20 steps.")
    parser.add_argument("--mujoco-gl", default="egl", help="MUJOCO_GL backend. Use egl for headless Linux.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when state replay diverges. Videos can still be valid when strict replay fails.",
    )
    return parser.parse_args()


def resolve_dataset(path: Path) -> Path:
    dataset = path.expanduser().resolve()
    if dataset.name == "demo.hdf5":
        dataset = dataset.parent
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset}")
    if not (dataset / "meta" / "episodes.jsonl").exists():
        raise FileNotFoundError(
            f"{dataset} does not look like a LeRobot dataset root; missing meta/episodes.jsonl"
        )
    return dataset


def maybe_downscale(video_path: Path, lowres_video_path: Path | None, width: int) -> None:
    if lowres_video_path is None:
        return
    lowres_video_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"scale={width}:-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(lowres_video_path),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    dataset = resolve_dataset(args.dataset)
    args.video_path = args.video_path.expanduser().resolve()
    args.video_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from humanoidmimicgen.wbc_goal_playback import (
            SyncSimPlaybackConfig,
            playback_wbc_goal_dataset,
        )
    except ImportError as exc:
        raise ImportError(
            "WBC-goal playback requires HumanoidMimicGen and the WBC controller runtime "
            "on PYTHONPATH. Install this repo plus the controller runtime checkout."
        ) from exc

    config = SyncSimPlaybackConfig()
    config.dataset = str(dataset)
    config.use_actions = True
    config.use_wbc_goals = True
    config.use_teleop_cmd = False
    config.save_video = True
    config.video_path = str(args.video_path)
    config.enable_offscreen = True
    config.enable_onscreen = False
    config.ci_test = args.ci_test
    config.num_episodes = args.num_episodes
    ok = bool(playback_wbc_goal_dataset(config))

    maybe_downscale(args.video_path, args.lowres_video_path, args.lowres_width)
    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
