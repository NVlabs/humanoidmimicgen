# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from humanoidmimicgen.wbc_goal_playback import ActionEnv


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
