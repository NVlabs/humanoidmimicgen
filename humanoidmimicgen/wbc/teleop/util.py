import gymnasium as gym
import numpy as np

from humanoidmimicgen.wbc.robot_model.robot_model import RobotModel


def prepare_observation_for_eval(robot_model: RobotModel, obs: dict) -> dict:
    """
    Prepare observation for evaluation.
    This function is used in both real and sim evaluation.
    """
    assert "q" in obs, "q is not in the observation"

    whole_q = obs["q"]
    assert whole_q.shape[-1] == robot_model.num_joints, "q has wrong shape"

    left_arm_q = whole_q[..., robot_model.get_joint_group_indices("left_arm")]
    right_arm_q = whole_q[..., robot_model.get_joint_group_indices("right_arm")]
    waist_q = whole_q[..., robot_model.get_joint_group_indices("waist")]
    left_leg_q = whole_q[..., robot_model.get_joint_group_indices("left_leg")]
    right_leg_q = whole_q[..., robot_model.get_joint_group_indices("right_leg")]
    left_hand_q = whole_q[..., robot_model.get_joint_group_indices("left_hand")]
    right_hand_q = whole_q[..., robot_model.get_joint_group_indices("right_hand")]

    obs["state.left_arm"] = left_arm_q
    obs["state.right_arm"] = right_arm_q
    obs["state.waist"] = waist_q
    obs["state.left_leg"] = left_leg_q
    obs["state.right_leg"] = right_leg_q
    obs["state.left_hand"] = left_hand_q
    obs["state.right_hand"] = right_hand_q

    return obs


def prepare_gym_space_for_eval(
    robot_model: RobotModel, gym_space: gym.spaces.Dict
) -> gym.spaces.Dict:
    """
    Prepare gym space for evaluation.
    This function is used only in sim evaluation.
    """
    left_arm_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("left_arm")),),
    )
    right_arm_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("right_arm")),),
    )
    waist_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("waist")),),
    )
    left_leg_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("left_leg")),),
    )
    right_leg_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("right_leg")),),
    )
    left_hand_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("left_hand")),),
    )
    right_hand_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("right_hand")),),
    )

    gym_space["state.left_arm"] = left_arm_space
    gym_space["state.right_arm"] = right_arm_space
    gym_space["state.waist"] = waist_space
    gym_space["state.left_leg"] = left_leg_space
    gym_space["state.right_leg"] = right_leg_space
    gym_space["state.left_hand"] = left_hand_space
    gym_space["state.right_hand"] = right_hand_space

    return gym_space
