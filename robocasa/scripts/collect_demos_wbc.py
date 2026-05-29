"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import yaml
import argparse
from copy import deepcopy
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import imageio
import mujoco
import numpy as np
import robosuite

# from robosuite import load_controller_config
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from termcolor import colored

import robocasa
from robocasa.environments.tabletop.tabletop import Tabletop
from robocasa.environments.locomanipulation.base import LocoManipulationEnv
import robocasa.macros as macros
from robocasa.wrappers.ik_wrapper import IKWrapper
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures import FixtureType
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format
from robocasa.scripts.collect_demos import (
    init_device,
    init_env,
    is_empty_input_spacemouse,
    gather_demonstrations_as_hdf5,
)


def collect_human_trajectory(
    env,
    teleop_device,
    unitree_bridge,
    arm,
    env_configuration,
    mirror_actions,
    render=True,
    max_fr=None,
    print_info=True,
    enable_profile=False,
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        teleop_device (Device): to receive upper-body teleoperation controls from the device
        rl_policy (Policy): to receive lower-body RL controls from the policy
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
        enable_profile (bool): whether to enable profiling using cProfile
    """

    env.reset()

    unitree_bridge.update_model(env.sim.model._model, env.sim.data._data)

    ep_meta = env.get_ep_meta()
    # print(json.dumps(ep_meta, indent=4))
    lang = ep_meta.get("lang", None)
    if print_info and lang is not None:
        print(colored(f"Instruction: {lang}", "green"))

    # degugging: code block here to quickly test and close env
    # env.close()
    # return None, True

    if render:
        # ID = 2 always corresponds to agentview
        env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    teleop_device.start_control()

    nonzero_ac_seen = False

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    zero_action = np.zeros(env.action_dim)
    for _ in range(1):
        # do a dummy step thru base env to initalize things, but don't record the step
        if isinstance(env, DataCollectionWrapper):
            env.env.step(zero_action)
        else:
            env.step(zero_action)

    discard_traj = False

    if enable_profile:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    try:
        # Loop until we get a reset from the input or the task completes
        sim_cnt = 0
        while True:
            sim_cnt += 1
            start = time.time()

            # publish the current state
            # print('qpos', env.sim.data.qpos.shape, env.sim.data.qpos)
            unitree_bridge.PublishLowState()

            # Set active robot
            active_robot = env.robots[teleop_device.active_robot]
            active_arm = teleop_device.active_arm

            # Get the newest action
            input_ac_dict = teleop_device.input2action(mirror_actions=mirror_actions)

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                discard_traj = True
                break

            action_dict = deepcopy(input_ac_dict)

            # set arm actions
            for arm in active_robot.arms:
                controller_input_type = active_robot.part_controllers[arm].input_type
                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            if is_empty_input_spacemouse(action_dict):
                if not nonzero_ac_seen:
                    if render:
                        env.render()
                    continue
            else:
                nonzero_ac_seen = True

            # set lower-body actions
            if unitree_bridge.low_cmd:
                # torques = np.zeros(unitree_bridge.num_motor)
                # for i in range(unitree_bridge.num_motor):
                #     torques[i] = (
                #         unitree_bridge.low_cmd.motor_cmd[i].tau
                #         + unitree_bridge.low_cmd.motor_cmd[i].kp
                #         * (unitree_bridge.low_cmd.motor_cmd[i].q - env.sim.data.qpos[7+unitree_bridge.joint_index_in_sim_model[i]])
                #         + unitree_bridge.low_cmd.motor_cmd[i].kd
                #         * (
                #             unitree_bridge.low_cmd.motor_cmd[i].dq
                #             - env.sim.data.qvel[6+unitree_bridge.joint_index_in_sim_model[i]]
                #         )
                #     )
                # # Set the torque limit
                # torques = np.clip(torques,
                #                 -unitree_bridge.torque_limit,
                #                 unitree_bridge.torque_limit)
                # leg_actions = torques[:12]
                leg_actions = np.array(
                    [
                        unitree_bridge.low_cmd.motor_cmd[i].q
                        for i in env.robots[0]._ref_actuators_indexes_dict["legs"]
                    ]
                )
                # # print('leg_actions', leg_actions)
                # print("target qpos", [unitree_bridge.low_cmd.motor_cmd[i].q for i in range(12, 29)])
                # print("qpos", [env.sim.data.qpos[7+i] for i in unitree_bridge.joint_index_in_sim_model[12:]])
                # print("qvel", [env.sim.data.qvel[6+i] for i in unitree_bridge.joint_index_in_sim_model[12:]])
                # print('computed torques', torques[12:])
                # print("kp", [unitree_bridge.low_cmd.motor_cmd[i].kp for i in range(12, 29)])
                # print("kd", [unitree_bridge.low_cmd.motor_cmd[i].kd for i in range(12, 29)])
            else:
                leg_actions = np.array(
                    [0.0] * len(env.robots[0]._ref_actuators_indexes_dict["legs"])
                )
            action_dict["legs"] = leg_actions

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [
                robot.create_action_vector(all_prev_gripper_actions[i])
                for i, robot in enumerate(env.robots)
            ]

            env_action[teleop_device.active_robot] = active_robot.create_action_vector(action_dict)

            # env_action = [
            #     np.concatenate([torques, np.zeros(14)])
            #     for i, robot in enumerate(env.robots)
            # ]
            env_action = np.concatenate(env_action)
            # eef_action, leg_action, gripper_action
            # print(env_action.shape)

            # Run environment step
            obs, _, _, _ = env.step(env_action)

            # env.sim.data.ctrl = env_action
            # mujoco.mj_step(env.sim.model._model, env.sim.data._data)
            # if sim_cnt % 10 == 0:
            #     env.viewer.update()

            if render:
                env.render()

            # Also break if we complete the task
            if task_completion_hold_count == 0:
                break

            # state machine to check for having a success for 10 consecutive timesteps
            if env._check_success():
                if task_completion_hold_count > 0:
                    task_completion_hold_count -= 1  # latched state, decrement count
                else:
                    task_completion_hold_count = 15  # reset count on first success timestep
            else:
                task_completion_hold_count = -1  # null the counter if there's no success

            # limit frame rate if necessary
            if max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)
                else:
                    print(
                        f"Warning: current frame rate is {1/elapsed} Hz, lower than max_fr {max_fr} Hz"
                    )

    finally:
        if enable_profile:
            pr.disable()
            pr.dump_stats("collect_human_trajectory.prof")
            print(f"Profiling results saved to collect_human_trajectory.prof")

    if nonzero_ac_seen and hasattr(env, "ep_directory"):
        ep_directory = env.ep_directory
    else:
        ep_directory = None

    # cleanup for end of data collection episodes
    env.close()
    return ep_directory, discard_traj


def get_args():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Kitchen")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="PandaOmron",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )

    parser.add_argument(
        "--camera",
        type=str,
        nargs="+",
        default=None,
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="spacemouse",
        choices=[
            "keyboard",
            "keyboardmobile",
            "spacemouse",
            "leapmotion",
            "leapmotion_with_fingers",
            "leapmotion_without_fingers",
            "manuswithvive",
            "dummy",
        ],
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale rotation user inputs",
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"])
    parser.add_argument("--max_fr", default=30, type=int, help="If specified, limit the frame rate")

    parser.add_argument("--layout", type=int, nargs="+", default=-1)
    parser.add_argument("--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    parser.add_argument("--generative_textures", action="store_true")
    parser.add_argument(
        "--obj_registries",
        type=str,
        nargs="+",
        default=None,
        help="(optional) object registries. choose among [objaverse, aigen, infinigen, sketchfab, lightwheel]",
    )
    parser.add_argument(
        "--prefer_leading_registry",
        action="store_true",
        help="Prefer the first registry in obj_registries when sampling objects",
    )
    parser.add_argument(
        "--ik_indicator",
        action="store_true",
        help="Render the indicator that receives input from the input device as input to the inverse kinetic solver",
    )
    parser.add_argument(
        "--store_collected_demo",
        action="store_true",
        help="Store the demo into the parent directory",
    )

    parser.add_argument(
        "--control_freq",
        type=int,
        default=20,
        help="Control frequency",
    )

    # for policy
    parser.add_argument(
        "--policy_config",
        type=str,
        default="../minimal-g1-deployment/sim2real/config/robocasa_g1_29dof_hist.yaml",
        help="G1 config file",
    )

    parser.add_argument(
        "--enable_profile",
        action="store_true",
        default=False,
        help="Enable profiling of the collect_human_trajectory function",
    )

    args = parser.parse_args()

    return args


def init_policy(args, env):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from robocasa.scripts.unitree_sdk2py_bridge import UnitreeSdk2Bridge

    with open(args.policy_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # locomotion_policy = MotionTrackingDecLocoHeightPolicy(
    #     config=config,
    #     node=node,
    #     loco_model_path=args.loco_model_path,
    #     mimic_model_paths=args.mimic_model_paths,
    #     use_jit=args.use_jit,
    #     rl_rate=50,
    #     decimation=4)

    # return locomotion_policy

    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])

    motor_index = []  # exclude the gripper motors
    for i in range(env.sim.model.nu):
        if "gripper" not in env.sim.model.actuator(i).name:
            motor_index.append(i)

    # hack here, actually we should use the mapping from actuator to joint.
    joint_index = []
    for i in range(env.sim.model.njnt):
        name = env.sim.model.joint(i).name
        if "gripper" not in name and "base" not in name:
            joint_index.append(
                i - 1
            )  # -1 for the base free joint, which is considered in the unitree_bridge

    unitree_bridge = UnitreeSdk2Bridge(
        env.sim.model._model, env.sim.data._data, config, motor_index, joint_index
    )

    return unitree_bridge


if __name__ == "__main__":

    args = get_args()

    env, env_info, mirror_actions = init_env(args)

    # for upper-body teleoperation
    teleop_device = init_device(args, env)

    # for lower-body RL
    unitree_bridge = init_policy(args, env)

    # make a new timestamped directory
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y-%m-%d-%H-%M-%S")

    if not args.debug:
        # wrap the environment with data collection wrapper
        tmp_directory = "/tmp/{}".format(time_str)
        env = DataCollectionWrapper(env, tmp_directory)

    new_dir = os.path.join(args.directory, time_str)
    os.makedirs(new_dir)

    excluded_eps = []

    # collect demonstrations
    while True:
        print()
        ep_directory, discard_traj = collect_human_trajectory(
            env,
            teleop_device,
            unitree_bridge,
            args.arm,
            args.config,
            mirror_actions,
            render=(args.renderer != "mjviewer"),
            max_fr=args.max_fr,
            enable_profile=args.enable_profile,
        )

        print("Keep traj?", not discard_traj)

        if not args.debug:
            if discard_traj and ep_directory is not None:
                excluded_eps.append(ep_directory.split("/")[-1])
            hdf5_path = gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, excluded_episodes=excluded_eps
            )
            # hdf5 file might be empty if no successful demos were collected
            if hdf5_path is not None:
                convert_to_robomimic_format(hdf5_path)
                if args.store_collected_demo:
                    final_path = os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "collected_demo",
                        f"{args.robots[0]}_{args.environment}.hdf5",
                    )
                    shutil.copy(hdf5_path, final_path)
                    print("Saving hdf5 to", final_path)
