# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from humanoidmimicgen.dataset_playback import (
    ActionEnv,
    load_hdf5_dataset,
    load_lerobot_dataset,
    playback_result,
    validate_state,
)


class FakeSyncEnv:
    def __init__(self):
        self.observation = {"q": np.array([0.0])}
        self.base_commands = []
        self.actions = []

    def observe(self):
        return self.observation

    def overwrite_floating_base_action(self, navigate_cmd, base_height_command):
        self.base_commands.append((navigate_cmd, base_height_command))

    def step(self, action):
        self.actions.append(action)
        return "obs", 1.0, False, False, {"ok": True}

    def is_success(self):
        return {"task": True}


class FakePolicy:
    def __init__(self):
        self.observations = []
        self.goals = []

    def set_observation(self, observation):
        self.observations.append(observation)

    def set_goal(self, goal):
        self.goals.append(goal)

    def get_action(self):
        return {"q": np.array([1.0, 2.0])}


def test_action_env_steps_action_through_policy_and_sync_env():
    sync_env = FakeSyncEnv()
    policy = FakePolicy()
    action_env = ActionEnv(sync_env, policy)
    action = {
        "navigate_cmd": np.array([0.1, 0.2, 0.3]),
        "base_height_command": 0.74,
    }

    result = action_env.step(action)

    assert result == ("obs", 1.0, False, False, {"ok": True})
    assert policy.observations == [sync_env.observation]
    assert policy.goals == [action]
    assert len(sync_env.actions) == 1
    np.testing.assert_allclose(sync_env.actions[0]["q"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(sync_env.base_commands[0][0], action["navigate_cmd"])
    assert sync_env.base_commands[0][1] == action["base_height_command"]


def test_action_env_steps_recorded_low_level_action_directly():
    sync_env = FakeSyncEnv()
    policy = FakePolicy()
    action_env = ActionEnv(sync_env, policy)
    action = {
        "recorded_action": np.array([3.0, 4.0]),
        "navigate_cmd": np.array([0.1, 0.2, 0.3]),
        "base_height_command": 0.76,
    }

    result = action_env.step_recorded(action)

    assert result == ("obs", 1.0, False, False, {"ok": True})
    assert policy.observations == []
    assert policy.goals == []
    np.testing.assert_allclose(sync_env.actions[0]["q"], action["recorded_action"])
    np.testing.assert_allclose(sync_env.base_commands[0][0], action["navigate_cmd"])
    assert sync_env.base_commands[0][1] == action["base_height_command"]


def test_action_env_defaults_base_command_when_action_omits_it():
    sync_env = FakeSyncEnv()
    policy = FakePolicy()
    action_env = ActionEnv(sync_env, policy)

    action_env.step({})

    np.testing.assert_allclose(sync_env.base_commands[0][0], np.zeros(3))
    assert sync_env.base_commands[0][1] == 0.74
    assert action_env.is_success() == {"task": True}


def test_action_env_uses_configured_default_base_height():
    sync_env = FakeSyncEnv()
    policy = FakePolicy()
    action_env = ActionEnv(sync_env, policy, default_base_height=0.8)

    action_env.step({})

    assert sync_env.base_commands[0][1] == 0.8


def test_validate_state_uses_absolute_tolerance_only():
    recorded = np.array([1_000_000.0])
    playback = np.array([1_000_001.0])

    assert not validate_state(recorded, playback, "demo_1", 0, tolerance=1e-5)
    assert validate_state(recorded, recorded + 1e-6, "demo_1", 0, tolerance=1e-5)


def test_load_hdf5_dataset_preserves_scene_xml(tmp_path):
    path = tmp_path / "demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_info"] = json.dumps({"seeds": [123]})
        data.attrs["script_config"] = json.dumps({"task_name": "LMPushButton"})
        demo = data.create_group("demo_1")
        demo.attrs["model_file"] = "<mujoco model='source-scene'/>"
        demo.create_dataset("states", data=np.zeros((3, 4)))
        demo.create_dataset("actions", data=np.ones((2, 3)))
        demo.create_dataset("wbc_goal/navigate_cmd", data=np.zeros((2, 3)))
        demo.create_dataset("teleop_cmd/base_height_command", data=np.full(2, 0.74))

    seeds, frames, config = load_hdf5_dataset(path)

    assert seeds == [123]
    assert config["task_name"] == "LMPushButton"
    assert frames["data/demo_1/model_file"] == "<mujoco model='source-scene'/>"
    np.testing.assert_array_equal(frames["data/demo_1/states"], np.zeros((3, 4)))
    np.testing.assert_array_equal(
        frames["data/demo_1/wbc_goal"][0]["recorded_action"], np.ones(3)
    )


def test_load_hdf5_dataset_state_mode_needs_no_action_payload(tmp_path):
    path = tmp_path / "demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_info"] = json.dumps({"seeds": [123]})
        data.attrs["script_config"] = json.dumps({"task_name": "LMPushButton"})
        demo = data.create_group("demo_1")
        demo.attrs["model_file"] = "<mujoco model='source-scene'/>"
        demo.create_dataset("states", data=np.zeros((3, 4)))
        demo.create_dataset("actions", data=np.ones((2, 3)))
        demo.create_dataset("wbc_goal/navigate_cmd", data=np.zeros((2, 3)))
        demo.create_dataset("teleop_cmd/base_height_command", data=np.full(2, 0.74))

    _, frames, _ = load_hdf5_dataset(path, action_source="state")

    actions = frames["data/demo_1/wbc_goal"]
    assert len(actions) == 2
    assert set(actions[0]) == {"navigate_cmd", "base_height_command"}
    np.testing.assert_array_equal(actions[0]["navigate_cmd"], np.zeros(3))
    assert actions[0]["base_height_command"] == 0.74


def test_load_hdf5_dataset_wbc_goal_mode_reads_goal_payload(tmp_path):
    path = tmp_path / "demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_info"] = json.dumps({"seeds": [123]})
        data.attrs["script_config"] = json.dumps({"task_name": "LMPushButton"})
        demo = data.create_group("demo_1")
        demo.attrs["model_file"] = "<mujoco model='source-scene'/>"
        demo.create_dataset("states", data=np.zeros((3, 4)))
        demo.create_dataset("actions", data=np.ones((2, 3)))
        demo.create_dataset("wbc_goal/navigate_cmd", data=np.zeros((2, 3)))
        demo.create_dataset("wbc_goal/wrist_pose", data=np.full((2, 14), 0.5))
        demo.create_dataset(
            "wbc_goal/target_upper_body_pose", data=np.full((2, 31), 0.25)
        )
        demo.create_dataset("teleop_cmd/base_height_command", data=np.full(2, 0.74))

    _, frames, _ = load_hdf5_dataset(path, action_source="wbc-goal")

    action = frames["data/demo_1/wbc_goal"][0]
    assert "recorded_action" not in action
    np.testing.assert_array_equal(action["wrist_pose"], np.full(14, 0.5))
    np.testing.assert_array_equal(
        action["target_upper_body_pose"], np.full(31, 0.25)
    )


def test_lerobot_wbc_goal_mode_explains_missing_goal_fields(tmp_path, monkeypatch):
    metadata = tmp_path / "meta"
    metadata.mkdir()
    (metadata / "episodes.jsonl").write_text(
        json.dumps({"length": 1, "tasks": ["LMPushButton"]}) + "\n"
    )

    class FakeLeRobotDataset:
        def __init__(self, **_kwargs):
            self.meta = SimpleNamespace(
                info={
                    "script_config": {"task_name": "LMPushButton"},
                    "features": {
                        "action": {"shape": [43]},
                        "teleop.navigate_command": {"shape": [3]},
                    },
                }
            )

        def __len__(self):
            return 1

    monkeypatch.setitem(
        sys.modules,
        "humanoidmimicgen.lerobot_dataset",
        SimpleNamespace(TypedLeRobotDataset=FakeLeRobotDataset),
    )

    with pytest.raises(ValueError, match="published 1K replay datasets") as error:
        load_lerobot_dataset(tmp_path, action_source="wbc-goal")

    assert "action.eef" in str(error.value)
    assert "observation.sim.target_upper_body_pose" in str(error.value)
    assert "--action-source recorded" in str(error.value)


def test_allow_state_divergence_does_not_override_required_task_success():
    summary = {"episodes": 1, "task_successes": 0}

    assert not playback_result(
        states_match=False,
        task_success_summary=summary,
        allow_state_divergence=True,
        require_task_success=True,
    )


def test_allow_state_divergence_accepts_successful_goal_replay():
    summary = {"episodes": 1, "task_successes": 1}

    assert playback_result(
        states_match=False,
        task_success_summary=summary,
        allow_state_divergence=True,
        require_task_success=True,
    )
