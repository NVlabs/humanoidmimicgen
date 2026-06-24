"""Small HumanoidMimicGen compatibility patches for public robosuite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_INSTALLED = False


def _install_composite_helpers():
    from robosuite.controllers.composite.composite_controller import CompositeController, WholeBody
    from robosuite.controllers.parts.generic.joint_pos import JointPositionController
    from robosuite.controllers.parts.generic.joint_vel import JointVelocityController
    from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

    if not hasattr(CompositeController, "create_action_vector"):

        def create_action_vector(self, action_dict):
            full_action_vector = np.zeros(self.action_limits[0].shape)
            for part_name, action_vector in action_dict.items():
                if part_name not in self._action_split_indexes:
                    ROBOSUITE_DEFAULT_LOGGER.debug(
                        f"{part_name} is not specified in the action space"
                    )
                    continue
                start_idx, end_idx = self._action_split_indexes[part_name]
                if end_idx - start_idx == 0:
                    continue
                assert len(action_vector) == end_idx - start_idx, ROBOSUITE_DEFAULT_LOGGER.error(
                    f"Action vector for {part_name} is not the correct size. "
                    f"Expected {end_idx - start_idx} for {part_name}, got {len(action_vector)}"
                )
                full_action_vector[start_idx:end_idx] = action_vector
            return full_action_vector

        CompositeController.create_action_vector = create_action_vector

    if not hasattr(CompositeController, "get_action_info"):

        def get_action_info(self):
            action_index_info = []
            action_dim_info = []
            for part_name, (start_idx, end_idx) in self._action_split_indexes.items():
                action_dim_info.append(f"{part_name}: {end_idx - start_idx} dim")
                action_index_info.append(f"{part_name}: {start_idx}:{end_idx}")
            return action_index_info, action_dim_info

        CompositeController.get_action_info = get_action_info

    if not hasattr(CompositeController, "print_action_info"):

        def print_action_info(self):
            action_index_info, action_dim_info = self.get_action_info()
            ROBOSUITE_DEFAULT_LOGGER.info(f"Action Dimensions: [{', '.join(action_dim_info)}]")
            ROBOSUITE_DEFAULT_LOGGER.info(f"Action Indices: [{', '.join(action_index_info)}]")

        CompositeController.print_action_info = print_action_info

    if not hasattr(CompositeController, "get_action_info_dict"):

        def get_action_info_dict(self):
            return {
                "Action Dimension": self.action_limits[0].shape,
                **dict(self._action_split_indexes),
            }

        CompositeController.get_action_info_dict = get_action_info_dict

    if not hasattr(CompositeController, "print_action_info_dict"):

        def print_action_info_dict(self, name: str = ""):
            info_dict = self.get_action_info_dict()
            info_dict_str = f"\nAction Info for {name}:\n\n{json.dumps(dict(info_dict), indent=4)}"
            ROBOSUITE_DEFAULT_LOGGER.info(info_dict_str)

        CompositeController.print_action_info_dict = print_action_info_dict

    if not hasattr(CompositeController, "get_hold_configuration_action_dict"):

        def get_hold_configuration_action_dict(self):
            action = {}
            for part_name, controller in self.part_controllers.items():
                if part_name in self.grippers:
                    action[part_name] = np.array([-1.0] * self.grippers[part_name].dof)
                elif isinstance(controller, JointVelocityController):
                    action[part_name] = np.zeros(controller.control_dim)
                elif isinstance(controller, JointPositionController):
                    controller.update(force=True)
                    action[part_name] = controller.initial_joint.copy()
                elif hasattr(controller, "get_hold_configuration_action"):
                    action[part_name] = controller.get_hold_configuration_action()
                else:
                    action[part_name] = np.zeros(controller.control_dim)
            return action

        CompositeController.get_hold_configuration_action_dict = get_hold_configuration_action_dict

    if not hasattr(CompositeController, "get_hold_configuration_action"):

        def get_hold_configuration_action(self):
            return self.create_action_vector(self.get_hold_configuration_action_dict())

        CompositeController.get_hold_configuration_action = get_hold_configuration_action

    if not getattr(WholeBody, "_hmg_skip_wbc_patch", False):
        original_set_goal = WholeBody.set_goal
        original_action_limits = WholeBody.action_limits.fget
        original_create_action_vector = WholeBody.create_action_vector
        original_get_action_info = WholeBody.get_action_info
        original_get_action_info_dict = WholeBody.get_action_info_dict

        def set_goal(self, all_action):
            if self.composite_controller_specific_config.get("skip_wbc_action", False):
                return CompositeController.set_goal(self, all_action)
            return original_set_goal(self, all_action)

        @property
        def action_limits(self):
            if self.composite_controller_specific_config.get("skip_wbc_action", False):
                return CompositeController.action_limits.fget(self)
            return original_action_limits(self)

        def create_action_vector(self, action_dict):
            if self.composite_controller_specific_config.get("skip_wbc_action", False):
                return CompositeController.create_action_vector(self, action_dict)
            return original_create_action_vector(self, action_dict)

        def get_action_info(self):
            if self.composite_controller_specific_config.get("skip_wbc_action", False):
                return CompositeController.get_action_info(self)
            return original_get_action_info(self)

        def get_action_info_dict(self):
            if self.composite_controller_specific_config.get("skip_wbc_action", False):
                return CompositeController.get_action_info_dict(self)
            return original_get_action_info_dict(self)

        def get_hold_configuration_action_dict(self):
            if self.composite_controller_specific_config.get("skip_wbc_action", False):
                return CompositeController.get_hold_configuration_action_dict(self)
            raise NotImplementedError(
                "Hold configuration action not implemented for WholeBody controller"
            )

        WholeBody.set_goal = set_goal
        WholeBody.action_limits = action_limits
        WholeBody.create_action_vector = create_action_vector
        WholeBody.get_action_info = get_action_info
        WholeBody.get_action_info_dict = get_action_info_dict
        WholeBody.get_hold_configuration_action_dict = get_hold_configuration_action_dict
        WholeBody.get_hold_configuration_action = CompositeController.get_hold_configuration_action
        WholeBody._hmg_skip_wbc_patch = True


def _install_mobile_base_controller():
    import robosuite.controllers as controllers
    from robosuite.controllers.parts import controller_factory as part_factory
    from robosuite.controllers.parts import mobile_base as mobile_base_controllers
    from robosuite.controllers.parts.mobile_base import joint_vel as joint_vel_module
    from robosuite.controllers.parts.mobile_base.joint_vel import MobileBaseJointVelocityController
    import robosuite.utils.transform_utils as T

    def set_goal(self, action, set_qpos=None):
        self.update()

        jnt_dim = len(self.qpos_index)
        if self.impedance_mode == "variable":
            damping_ratio, kp, delta = (
                action[:jnt_dim],
                action[jnt_dim : 2 * jnt_dim],
                action[2 * jnt_dim :],
            )
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = (
                2
                * np.sqrt(self.kp)
                * np.clip(damping_ratio, self.damping_ratio_min, self.damping_ratio_max)
            )
        elif self.impedance_mode == "variable_kp":
            kp, delta = action[:jnt_dim], action[jnt_dim:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp)
        else:
            delta = action

        assert len(delta) == jnt_dim, "Delta qpos must be equal to the robot's joint dimension space!"
        _ = self.scale_action(delta) if delta is not None else None

        curr_pos, curr_ori = self.get_base_pose()
        del curr_pos
        init_theta = T.mat2euler(self.init_ori)[2]
        curr_theta = T.mat2euler(curr_ori)[2]
        theta = curr_theta - init_theta

        base_action = np.array(action, dtype=float, copy=True)
        x, y = base_action[0:2]
        base_action[0] = x * np.cos(theta) - y * np.sin(theta)
        base_action[1] = x * np.sin(theta) + y * np.cos(theta)

        self.goal_qvel = base_action
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qvel)

    MobileBaseJointVelocityController.set_goal = set_goal

    class MobileBaseJointVelocityAndPositionController(MobileBaseJointVelocityController):
        def __init__(
            self,
            sim,
            joint_indexes,
            actuator_range,
            velocity_indices=None,
            position_indices=None,
            input_max=1,
            input_min=-1,
            output_max=1,
            output_min=-1,
            kp=50,
            damping_ratio=1,
            impedance_mode="fixed",
            kp_limits=(0, 300),
            damping_ratio_limits=(0, 100),
            policy_freq=20,
            qpos_limits=None,
            interpolator=None,
            **kwargs,
        ):
            super().__init__(
                sim=sim,
                joint_indexes=joint_indexes,
                actuator_range=actuator_range,
                input_max=input_max,
                input_min=input_min,
                output_max=output_max,
                output_min=output_min,
                kp=kp,
                damping_ratio=damping_ratio,
                impedance_mode=impedance_mode,
                kp_limits=kp_limits,
                damping_ratio_limits=damping_ratio_limits,
                policy_freq=policy_freq,
                qpos_limits=qpos_limits,
                interpolator=interpolator,
                **kwargs,
            )

            jnt_dim = len(joint_indexes["joints"])
            if velocity_indices is None and position_indices is None:
                self.velocity_indices = list(range(min(3, jnt_dim)))
                self.position_indices = list(range(3, jnt_dim))
            else:
                self.velocity_indices = velocity_indices if velocity_indices is not None else []
                self.position_indices = position_indices if position_indices is not None else []

            all_indices = set(self.velocity_indices + self.position_indices)
            expected_indices = set(range(jnt_dim))
            assert all_indices == expected_indices, (
                "velocity_indices + position_indices must cover all joint indices. "
                f"Expected: {expected_indices}, Got: {all_indices}"
            )
            self.jnt_dim = jnt_dim

        def run_controller(self):
            if self.goal_qvel is None:
                self.set_goal(np.zeros(self.control_dim))

            self.update()
            if self.interpolator is not None and self.interpolator.order == 1:
                desired_commands = self.interpolator.get_interpolated_goal()
            else:
                desired_commands = np.array(self.goal_qvel)

            output_commands = np.zeros_like(desired_commands)
            if self.velocity_indices:
                vel_indices = np.array(self.velocity_indices)
                vel_commands = desired_commands[vel_indices]
                ctrl_range = np.stack(
                    [self.actuator_min[vel_indices], self.actuator_max[vel_indices]], axis=-1
                )
                bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
                weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
                output_commands[vel_indices] = bias + weight * vel_commands

            if self.position_indices:
                pos_indices = np.array(self.position_indices)
                output_commands[pos_indices] = np.clip(
                    desired_commands[pos_indices],
                    self.actuator_min[pos_indices],
                    self.actuator_max[pos_indices],
                )

            self.vels = output_commands
            super().run_controller()
            return self.vels

        @property
        def name(self):
            return "JOINT_VELOCITY_AND_POSITION"

    joint_vel_module.MobileBaseJointVelocityAndPositionController = (
        MobileBaseJointVelocityAndPositionController
    )
    mobile_base_controllers.MobileBaseJointVelocityAndPositionController = (
        MobileBaseJointVelocityAndPositionController
    )

    original_mobile_base_factory = part_factory.mobile_base_controller_factory

    def mobile_base_controller_factory(name, params):
        if name == "JOINT_VELOCITY_AND_POSITION":
            return MobileBaseJointVelocityAndPositionController(interpolator=None, **params)
        return original_mobile_base_factory(name, params)

    part_factory.mobile_base_controller_factory = mobile_base_controller_factory
    controllers.PART_CONTROLLER_INFO.setdefault(
        "JOINT_VELOCITY_AND_POSITION", "Joint Velocity and Position"
    )


def _install_joint_position_controller():
    from robosuite.controllers.parts.generic.joint_pos import JointPositionController

    if getattr(JointPositionController, "_hmg_torque_compensation_patch", False):
        return

    original_init = JointPositionController.__init__
    original_run_controller = JointPositionController.run_controller

    def __init__(self, *args, **kwargs):
        use_torque_compensation = kwargs.get("use_torque_compensation", True)
        original_init(self, *args, **kwargs)
        self.use_torque_compensation = use_torque_compensation

    def run_controller(self):
        if getattr(self, "use_torque_compensation", True):
            return original_run_controller(self)

        if self.goal_qpos is None:
            self.set_goal(np.zeros(self.control_dim))

        self.update()
        if self.interpolator is not None and self.interpolator.order == 1:
            desired_qpos = self.interpolator.get_interpolated_goal()
        else:
            desired_qpos = np.array(self.goal_qpos)

        position_error = desired_qpos - self.joint_pos
        vel_pos_error = -self.joint_vel
        self.torques = np.multiply(np.array(position_error), np.array(self.kp)) + np.multiply(
            vel_pos_error, self.kd
        )

        super(JointPositionController, self).run_controller()
        return self.torques

    JointPositionController.__init__ = __init__
    JointPositionController.run_controller = run_controller
    JointPositionController._hmg_torque_compensation_patch = True


def _install_joint_torque_controller():
    from robosuite.controllers.parts.generic.joint_tor import JointTorqueController

    if getattr(JointTorqueController, "_hmg_torque_compensation_patch", False):
        return

    original_init = JointTorqueController.__init__
    original_run_controller = JointTorqueController.run_controller

    def __init__(self, *args, **kwargs):
        use_torque_compensation = kwargs.get("use_torque_compensation", True)
        original_init(self, *args, **kwargs)
        self.use_torque_compensation = use_torque_compensation

    def run_controller(self):
        if getattr(self, "use_torque_compensation", True):
            return original_run_controller(self)

        if self.goal_torque is None:
            self.set_goal(np.zeros(self.control_dim))

        self.update()
        if self.interpolator is not None and self.interpolator.order == 1:
            self.current_torque = self.interpolator.get_interpolated_goal()
        else:
            self.current_torque = np.array(self.goal_torque)

        self.torques = self.current_torque
        super(JointTorqueController, self).run_controller()
        return self.torques

    JointTorqueController.__init__ = __init__
    JointTorqueController.run_controller = run_controller
    JointTorqueController._hmg_torque_compensation_patch = True


def _install_joint_velocity_controller():
    from robosuite.controllers.parts.generic.joint_vel import JointVelocityController

    if getattr(JointVelocityController, "_hmg_torque_compensation_patch", False):
        return

    original_init = JointVelocityController.__init__
    original_run_controller = JointVelocityController.run_controller

    def __init__(self, *args, **kwargs):
        use_torque_compensation = kwargs.get("use_torque_compensation", True)
        original_init(self, *args, **kwargs)
        self.use_torque_compensation = use_torque_compensation

    def run_controller(self):
        if getattr(self, "use_torque_compensation", True):
            return original_run_controller(self)

        if self.goal_vel is None:
            self.set_goal(np.zeros(self.joint_dim))

        self.update()
        if self.interpolator is not None and self.interpolator.order == 1:
            self.current_vel = self.interpolator.get_interpolated_goal()
        else:
            self.current_vel = np.array(self.goal_vel)

        err = self.current_vel - self.joint_vel
        derr = err - self.last_err
        self.last_err = err
        self.derr_buf.push(derr)
        if not self.saturated:
            self.summed_err += err

        torques = self.kp * err + self.ki * self.summed_err + self.kd * self.derr_buf.average
        self.torques = self.clip_torques(torques)
        self.saturated = False if np.sum(np.abs(self.torques - torques)) == 0 else True

        super(JointVelocityController, self).run_controller()
        return self.torques

    JointVelocityController.__init__ = __init__
    JointVelocityController.run_controller = run_controller
    JointVelocityController._hmg_torque_compensation_patch = True


def _install_legged_model_cleanup():
    from robosuite.models.robots.manipulators.legged_manipulator_model import LeggedManipulatorModel
    from robosuite.utils.mjcf_utils import find_parent

    if getattr(LeggedManipulatorModel, "_hmg_sensor_cleanup_patch", False):
        return

    original_remove_joint_actuation = LeggedManipulatorModel._remove_joint_actuation

    def _remove_joint_actuation(self, part_name):
        original_remove_joint_actuation(self, part_name)
        for sensor_tag in ("jointpos", "jointvel", "jointactuatorfrc"):
            for sensor in self.root.findall(f".//{sensor_tag}"):
                if part_name in sensor.get("joint"):
                    find_parent(self.root, sensor).remove(sensor)

    LeggedManipulatorModel._remove_joint_actuation = _remove_joint_actuation
    LeggedManipulatorModel._hmg_sensor_cleanup_patch = True


def _install_null_base():
    import numpy as np
    from robosuite.models.bases import BASE_MAPPING
    from robosuite.models.bases.robot_base_model import RobotBaseModel
    from robosuite.models.robots.robot_model import RobotModel

    if "NullBase" not in BASE_MAPPING:

        class NullBase(RobotBaseModel):
            def __init__(self, idn=0):
                asset = Path(__file__).resolve().parents[1] / "models/assets/bases/null_base.xml"
                super().__init__(str(asset), idn=idn)

            @property
            def naming_prefix(self):
                return f"nullbase{self.idn}_"

            @property
            def top_offset(self):
                return np.array((0, 0, 0))

            @property
            def horizontal_radius(self):
                return 0

        BASE_MAPPING["NullBase"] = NullBase

    if getattr(RobotModel, "_hmg_null_base_patch", False):
        return

    original_add_base = RobotModel.add_base

    def add_null_base(self, base):
        if self.base is not None:
            raise ValueError("Mobile base already added for this robot!")

        self.base = base
        for body in base.worldbody:
            site = body.find("site")
            if site is not None:
                self.worldbody.append(site)

        self.cameras = self.get_element_names(self.worldbody, "camera")

    def add_base(self, base):
        if base.__class__.__name__ == "NullBase":
            return self.add_null_base(base)
        return original_add_base(self, base)

    RobotModel.add_null_base = add_null_base
    RobotModel.add_base = add_base
    RobotModel._hmg_null_base_patch = True


def install_robosuite_compat():
    global _INSTALLED
    if _INSTALLED:
        return

    _install_composite_helpers()
    _install_mobile_base_controller()
    _install_joint_position_controller()
    _install_joint_torque_controller()
    _install_joint_velocity_controller()
    _install_legged_model_cleanup()
    _install_null_base()

    # Importing this module registers HYBRID_WHOLE_BODY_MINK_IK and WHOLE_BODY_MINK_IK.
    from robocasa.examples.third_party_controller import mink_controller  # noqa: F401

    _INSTALLED = True
