# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "playback_dataset.py"
    spec = importlib.util.spec_from_file_location("playback_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_playback_is_strict_and_render_free_by_default():
    args = load_script().parse_args(["/tmp/dataset"])

    assert not args.allow_state_divergence
    assert args.state_atol == 1e-5
    assert args.video_path is None
    assert args.action_source == "recorded"
    assert not args.require_task_success


def test_wbc_goal_action_source_is_explicit():
    args = load_script().parse_args(
        ["/tmp/dataset", "--action-source", "wbc-goal"]
    )

    assert args.action_source == "wbc-goal"


def test_state_action_source_is_explicit():
    args = load_script().parse_args(
        ["/tmp/dataset", "--action-source", "state"]
    )

    assert args.action_source == "state"


def test_task_success_gate_is_explicit():
    args = load_script().parse_args(
        ["/tmp/dataset", "--require-task-success"]
    )

    assert args.require_task_success


def test_video_output_is_opt_in():
    args = load_script().parse_args(
        [
            "/tmp/dataset",
            "--state-atol",
            "0",
            "--video-path",
            "/tmp/replay.mp4",
            "--lowres-video-path",
            "/tmp/replay-small.mp4",
        ]
    )

    assert args.state_atol == 0
    assert args.video_path == Path("/tmp/replay.mp4")


def test_lowres_video_requires_primary_video():
    with pytest.raises(SystemExit):
        load_script().parse_args(
            ["/tmp/dataset", "--lowres-video-path", "/tmp/replay-small.mp4"]
        )


def test_resolve_dataset_accepts_source_demo_hdf5(tmp_path):
    source_demo = tmp_path / "demo.hdf5"
    source_demo.touch()

    assert load_script().resolve_dataset(source_demo) == source_demo.resolve()
