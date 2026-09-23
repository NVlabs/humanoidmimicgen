# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reset RNG regressions found by the seed-0 PushButton physics audit."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
pytest.importorskip("pinocchio")

from humanoidmimicgen.locomanipulation.envs.base import LocoManipulationEnv
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv


@pytest.fixture(autouse=True)
def restore_global_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def make_env(noise_type="gaussian", magnitude=0.0, deterministic=False):
    env = object.__new__(LocoManipulationEnv)
    env.rng = np.random.default_rng(17)
    env.deterministic_reset = deterministic
    env.robots = [
        SimpleNamespace(
            init_qpos=np.array([0.1, -0.2, 0.3]),
            initialization_noise={"type": noise_type, "magnitude": magnitude},
            _ref_joint_pos_indexes=[0, 1, 2],
        )
    ]
    env.sim = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(3)))
    return env


@pytest.mark.parametrize("noise_type", ["gaussian", "uniform"])
@pytest.mark.parametrize("magnitude", [0.0, 0.02])
def test_seeded_reset_preserves_global_rng_and_scene_draw_order(
    monkeypatch, noise_type, magnitude
):
    env = make_env(noise_type, magnitude)
    reference = np.random.default_rng(17)
    expected_noise = (
        reference.normal(size=3)
        if noise_type == "gaussian"
        else reference.uniform(-1.0, 1.0, 3)
    )
    expected_qpos = env.robots[0].init_qpos + magnitude * expected_noise
    expected_scene_draws = reference.uniform(size=4)

    def parent_reset(self):
        # RoboSuite 1.5.1 consumes these draws even at zero magnitude. The
        # env-owned sample replaces the resulting robot qpos afterward.
        self.sim.data.qpos[:] = np.random.normal(size=3)

    monkeypatch.setattr(ManipulationEnv, "_reset_internal", parent_reset)
    np.random.seed(23)
    expected_global_draws = np.random.RandomState(23).normal(size=8)
    env._reset_internal()

    np.testing.assert_array_equal(env.sim.data.qpos, expected_qpos)
    np.testing.assert_array_equal(env.rng.uniform(size=4), expected_scene_draws)
    np.testing.assert_array_equal(np.random.normal(size=8), expected_global_draws)


def test_deterministic_reset_does_not_sample_or_overwrite_robot(monkeypatch):
    env = make_env(deterministic=True)
    initial_env_rng = env.rng.bit_generator.state

    def parent_reset(self):
        self.sim.data.qpos[:] = [1.0, 2.0, 3.0]

    monkeypatch.setattr(ManipulationEnv, "_reset_internal", parent_reset)
    env._reset_internal()
    assert env.rng.bit_generator.state == initial_env_rng
    np.testing.assert_array_equal(env.sim.data.qpos, [1.0, 2.0, 3.0])
    assert env.deterministic_reset is True


def test_parent_reset_failure_does_not_leak_global_rng_draws(monkeypatch):
    env = make_env()

    def failed_reset(self):
        np.random.normal(size=3)
        raise RuntimeError("reset failed")

    monkeypatch.setattr(ManipulationEnv, "_reset_internal", failed_reset)
    np.random.seed(29)
    expected = np.random.RandomState(29).uniform(size=8)
    with pytest.raises(RuntimeError, match="reset failed"):
        env._reset_internal()
    np.testing.assert_array_equal(np.random.uniform(size=8), expected)
