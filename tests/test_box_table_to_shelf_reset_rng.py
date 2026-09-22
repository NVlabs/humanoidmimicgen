# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task 07 uses the dataset-matched static start beside the box."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
pytest.importorskip("pinocchio")

from humanoidmimicgen.locomanipulation.envs.base import (
    LocoManipulationEnv,
    RobotPoseRandomizer,
)
from humanoidmimicgen.locomanipulation.envs.locomanip_basic import (
    LMBoxTableToShelf,
)
from humanoidmimicgen.locomanipulation.envs.locomanip_pnp import (
    LMBoxLift,
    LMBoxLiftFloor,
    LMDrillLift,
    LMDrillLiftObstacle,
    LMDrillPnP90,
)
from humanoidmimicgen.locomanipulation.envs.locomanip_push import LMPushShelfForward
from humanoidmimicgen.locomanipulation.envs.locomanip_simple import LMPushButton


class CountingRNG:
    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)
        self.ranges = []

    def uniform(self, low, high):
        self.ranges.append((low, high))
        return self.generator.uniform(low, high)


def make_env(monkeypatch, cls, *, seed=0, deterministic=False):
    events, poses = [], []
    env = object.__new__(cls)
    env.rng = CountingRNG(seed)
    env.deterministic_reset = deterministic
    # Isolate task pose draws from robot initialization and scene placement.
    monkeypatch.setattr(
        LocoManipulationEnv, "_reset_internal", lambda self: events.append("parent")
    )
    env.scene = SimpleNamespace(reset=lambda: events.append("scene"))
    model = type("G1", (), {"naming_prefix": "robot0_"})()
    env.robots = [SimpleNamespace(
        name="G1", robot_model=model, robot_joints=[], _ref_joint_pos_indexes=[]
    )]
    qpos = np.arange(7, dtype=float)

    def set_joint_qpos(joint, pose):
        assert joint == "robot0_base"
        qpos[:] = pose
        poses.append(pose.copy())

    env.sim = SimpleNamespace(
        model=SimpleNamespace(joint_names=["robot0_base"]),
        data=SimpleNamespace(qpos=qpos, set_joint_qpos=set_joint_qpos),
        forward=lambda: events.append("forward"),
    )
    original = RobotPoseRandomizer.set_pose

    def record_pose(self, x_range, y_range, yaw_range):
        events.append((x_range, y_range, yaw_range))
        original(self, x_range, y_range, yaw_range)

    monkeypatch.setattr(RobotPoseRandomizer, "set_pose", staticmethod(record_pose))
    return env, events, poses


@pytest.mark.parametrize("seed", [0, 17])
def test_nondeterministic_reset_uses_static_pose_beside_box(monkeypatch, seed):
    env, events, poses = make_env(
        monkeypatch, LMBoxTableToShelf, seed=seed
    )
    fixed_start = ((0.73, 0.73), (-0.06, 0.06), (0.0, 0.0))
    reference = np.random.default_rng(seed)
    fixed_pose = [reference.uniform(*bounds) for bounds in fixed_start]

    env._reset_internal()

    assert events == ["parent", "scene", fixed_start, "forward"]
    assert env.rng.ranges == [*fixed_start]
    assert len(env.rng.ranges) == 3
    assert env.rng.generator.bit_generator.state == reference.bit_generator.state
    assert len(poses) == 1
    np.testing.assert_array_equal(
        env.sim.data.qpos, [*fixed_pose[:2], 0.793, 1, 0, 0, 0]
    )


def test_deterministic_reset_consumes_no_pose_draws(monkeypatch):
    env, events, poses = make_env(
        monkeypatch, LMBoxTableToShelf, deterministic=True
    )
    initial_state = env.rng.generator.bit_generator.state
    initial_qpos = env.sim.data.qpos.copy()

    env._reset_internal()

    assert events == ["parent", "forward"]
    assert not poses and not env.rng.ranges
    assert env.rng.generator.bit_generator.state == initial_state
    np.testing.assert_array_equal(env.sim.data.qpos, initial_qpos)


@pytest.mark.parametrize("cls", [
    LMBoxLiftFloor, LMBoxLift, LMPushShelfForward, LMDrillLift,
    LMDrillPnP90, LMDrillLiftObstacle, LMPushButton,
])
def test_other_mapped_tasks_do_not_gain_compatibility_pose_draws(monkeypatch, cls):
    env, events, poses = make_env(monkeypatch, cls)
    initial_state = env.rng.generator.bit_generator.state
    initial_qpos = env.sim.data.qpos.copy()

    env._reset_internal()

    assert events == ["parent", "scene"]
    assert not poses and not env.rng.ranges
    assert env.rng.generator.bit_generator.state == initial_state
    np.testing.assert_array_equal(env.sim.data.qpos, initial_qpos)
