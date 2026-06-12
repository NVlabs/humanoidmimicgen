import numpy as np

from humanoidmimicgen.wbc_goal_playback import WBCGoalEnv


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


def test_wbc_goal_env_steps_goal_through_policy_and_sync_env():
    sync_env = FakeSyncEnv()
    policy = FakePolicy()
    wbc_env = WBCGoalEnv(sync_env, policy)
    goal = {
        "navigate_cmd": np.array([0.1, 0.2, 0.3]),
        "base_height_command": 0.74,
    }

    result = wbc_env.step(goal)

    assert result == ("obs", 1.0, False, False, {"ok": True})
    assert policy.observations == [sync_env.observation]
    assert policy.goals == [goal]
    assert len(sync_env.actions) == 1
    np.testing.assert_allclose(sync_env.actions[0]["q"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(sync_env.base_commands[0][0], goal["navigate_cmd"])
    assert sync_env.base_commands[0][1] == goal["base_height_command"]


def test_wbc_goal_env_defaults_base_command_when_goal_omits_it():
    sync_env = FakeSyncEnv()
    policy = FakePolicy()
    wbc_env = WBCGoalEnv(sync_env, policy)

    wbc_env.step({})

    np.testing.assert_allclose(sync_env.base_commands[0][0], np.zeros(3))
    assert sync_env.base_commands[0][1] == 0.0
    assert wbc_env.is_success() == {"task": True}
