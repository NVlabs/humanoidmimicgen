from typing import List

import numpy as np

from robosuite.models.robots.manipulators.legged_manipulator_model import (
    LeggedManipulatorModel,
)
from robosuite.robots import register_robot_class
from robosuite.utils.mjcf_utils import find_parent, xml_path_completion

import humanoidmimicgen.locomanipulation as locomanipulation


@register_robot_class("LeggedRobot")
class Neo(LeggedManipulatorModel):
    """
    Neo is a mobile manipulator robot created by 1x Technology.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right", "left"]

    def __init__(self, idn: int = 0):
        super().__init__(
            xml_path_completion(
                "robots/1x_neo/robot.xml",
                root=locomanipulation.models.assets_root,
            ),
            idn=idn,
        )

    def _print_joint_info(self):
        """Print formatted information about joints"""
        # ANSI color codes
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"

        print(f"\n{BOLD}{BLUE}================ joints info ================{ENDC}")

        # Print joints by category
        categories = {
            "Arms": self._arms_joints,
            "Base": self._base_joints,
            "Torso": self._torso_joints,
            "Head": self._head_joints,
            "Legs": self._legs_joints,
        }

        for category, joints in categories.items():
            if joints:
                print(f"{GREEN}{category} ({len(joints)}){ENDC}: {', '.join(joints)}")
            else:
                print(f"{RED}{category} (disabled){ENDC}")
        print()

    def _print_actuator_info(self):
        """Print formatted information about actuators"""
        # ANSI color codes
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"

        print(f"\n{BOLD}{BLUE}================ actuators info ================{ENDC}")

        # Print actuators by category
        categories = {
            "Arms": self._arms_actuators,
            "Base": self._base_actuators,
            "Torso": self._torso_actuators,
            "Head": self._head_actuators,
            "Legs": self._legs_actuators,
        }

        for category, actuators in categories.items():
            if actuators:
                print(f"{GREEN}{category} ({len(actuators)}){ENDC}: {', '.join(actuators)}")
            else:
                print(f"{RED}{category} (disabled){ENDC}")
        print()

    def update_joints(self):
        """internal function to update joint lists"""
        for joint in self.all_joints:
            if "spine" in joint:
                self.torso_joints.append(joint)
            elif "mobile" in joint:
                self.base_joints.append(joint)
            elif "neck" in joint:
                self.head_joints.append(joint)
            elif "knee" in joint or "hip" in joint or "ankle" in joint:
                self.legs_joints.append(joint)

        for joint in self.all_joints:
            if (
                joint not in self._base_joints
                and joint not in self._torso_joints
                and joint not in self._head_joints
                and joint not in self._legs_joints
            ):
                self._arms_joints.append(joint)

        self._print_joint_info()

    def update_actuators(self):
        """internal function to update actuator lists"""
        for actuator in self.all_actuators:
            if "spine" in actuator:
                self.torso_actuators.append(actuator)
            elif "mobile" in actuator:
                self.base_actuators.append(actuator)
            elif "neck" in actuator:
                self.head_actuators.append(actuator)
            elif "knee" in actuator or "hip" in actuator or "ankle" in actuator:
                self.legs_actuators.append(actuator)

        for actuator in self.all_actuators:
            if (
                actuator not in self._base_actuators
                and actuator not in self._torso_actuators
                and actuator not in self._head_actuators
                and actuator not in self._legs_actuators
            ):
                self._arms_actuators.append(actuator)

        self._print_actuator_info()

    @property
    def default_base(self):
        return "NoActuationBase"

    @property
    def default_gripper(self):
        return {"right": "FourierRightHand", "left": "FourierLeftHand"}

    @property
    def init_qpos(self):
        joints = self.worldbody.findall(".//joint")
        init_qpos = np.array([0.0] * len(joints))
        for joint_id, joint in enumerate(joints):
            if "elbow_y" in joint.get("name"):
                init_qpos[joint_id] = -1.2
            elif "r_elbow_z" in joint.get("name"):
                init_qpos[joint_id] = -1
            elif "l_elbow_z" in joint.get("name"):
                init_qpos[joint_id] = 1
            elif "shoulder_y" in joint.get("name"):
                init_qpos[joint_id] = -0.9
            elif "l_shoulder_z" in joint.get("name"):
                init_qpos[joint_id] = -0.3
            elif "r_shoulder_z" in joint.get("name"):
                init_qpos[joint_id] = 0.3
            elif "r_shoulder_x" in joint.get("name"):
                init_qpos[joint_id] = -0.5
            elif "l_shoulder_x" in joint.get("name"):
                init_qpos[joint_id] = 0.5
        return init_qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.95),
            "empty": (-0.29, 0, 0.95),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.95),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {"right": "right_eef", "left": "left_eef"}


@register_robot_class("LeggedRobot")
class NeoFixedLowerBody(Neo):
    def __init__(self, idn: int = 0):
        super().__init__(idn=idn)

        # Remove lower body actuation
        self._remove_joint_actuation("knee")
        self._remove_joint_actuation("hip")
        self._remove_joint_actuation("ankle")

        self._remove_free_joint()


@register_robot_class("LeggedRobot")
class NeoArmsOnly(Neo):
    def __init__(self, idn: int = 0):
        super().__init__(idn=idn)

        # Remove lower body actuation
        self._remove_joint_actuation("knee")
        self._remove_joint_actuation("hip")
        self._remove_joint_actuation("ankle")

        # Remove spine & head actuation
        self._remove_joint_actuation("spine")
        self._remove_joint_actuation("neck")

        self._remove_free_joint()
