"""Small HumanoidMimicGen compatibility patches for public robosuite."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


_INSTALLED = False


def _assert_public_robosuite_runtime():
    import robosuite

    path = Path(robosuite.__file__).resolve()
    allowed_root = os.environ.get("HMG_PUBLIC_ROBOSUITE_ROOT")
    if allowed_root and path.is_relative_to(Path(allowed_root).expanduser().resolve()):
        return
    if getattr(robosuite, "__version__", "").startswith("1.5.") and "grootrobosuite" not in path.parts:
        return

    raise ImportError(
        "Expected public robosuite==1.5.1, but imported "
        f"{path}. Install the public package or set HMG_PUBLIC_ROBOSUITE_ROOT "
        "to an explicit public robosuite target."
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

    _assert_public_robosuite_runtime()
    _install_joint_position_controller()
    _install_legged_model_cleanup()
    _install_null_base()

    # Importing this module registers HYBRID_WHOLE_BODY_MINK_IK and WHOLE_BODY_MINK_IK.
    from humanoidmimicgen.locomanipulation.controllers import mink  # noqa: F401

    _INSTALLED = True
