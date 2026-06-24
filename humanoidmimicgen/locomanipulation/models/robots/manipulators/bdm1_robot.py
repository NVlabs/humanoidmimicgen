import numpy as np

from robosuite.models.robots.manipulators.legged_manipulator_model import (
    LeggedManipulatorModel,
)
from robosuite.robots import register_robot_class
from robosuite.utils.mjcf_utils import find_parent, xml_path_completion

import humanoidmimicgen.locomanipulation as locomanipulation


@register_robot_class("LeggedRobot")
class BDM1(LeggedManipulatorModel):
    """
    M1 is a mobile manipulator robot created by Boston Dynamics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion(
                "robots/bd_m1/assets/eatlas/atlas_nub_hand_pelvis_base.xml",
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
            if "back" in joint:
                self.torso_joints.append(joint)
            elif "mobile" in joint:
                self.base_joints.append(joint)
            elif "head" in joint:
                self.head_joints.append(joint)
            elif "leg" in joint:
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
            if "back" in actuator:
                self.torso_actuators.append(actuator)
            elif "mobile" in actuator:
                self.base_actuators.append(actuator)
            elif "head" in actuator:
                self.head_actuators.append(actuator)
            elif "leg" in actuator:
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
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "GR2RightHand", "left": "GR2LeftHand"}

    @property
    def init_qpos(self):
        """
        Since this is bimanual robot, returns [right, left] array corresponding to respective values

        Note that this is a pose such that the arms are half extended

        Returns:
            np.array: default initial qpos for the right, left arms
        """
        joints = self.worldbody.findall(".//joint")
        init_qpos = np.array([0.0] * len(joints))
        for joint_id, joint in enumerate(joints):
            if "arm.el" in joint.get("name"):
                init_qpos[joint_id] = 2.0
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
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific eef names
        """
        return {"right": "right_eef", "left": "left_eef"}

    def _update_joint_properties(self):
        self.motor_name2gear = {}
        for motor in self.actuator.findall(".//motor"):
            self.motor_name2gear[motor.get("name")] = float(motor.get("gear").split(" ")[0])

        for joint in self.worldbody.findall(".//joint"):
            joint.set("solimplimit", "0.998 0.9999 0.00001 0.1 1")
            joint.set("solreflimit", "0.00001 1")
            joint.set("armature", "0.019997166")
            joint.set("frictionloss", "1.0")
            joint.set("damping", "1.0")

        for geom in self.worldbody.findall(".//geom"):
            geom.set("friction", "0.8 0.005 0.0001")
            geom.set("solimp", "0.95 1.0 0.001 0.5 2")
            geom.set("solref", "0.005 1.0")

        for equality in self.worldbody.findall(".//equality"):
            equality.set("solimp", "0.998 0.9999 0.00001 0.1 1")
            equality.set("solref", "0.00001 1")


@register_robot_class("LeggedRobot")
class BDM1FixedLowerBody(BDM1):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

        # fix lower body
        self._remove_joint_actuation("leg")

        # TODO: due to the base joint in current model is `torso',
        # TODO: the back joint will actually cause the movement of the lower-body part
        # TODO: better to change the base joint to `pelvis' to enable the movement of waist.
        # self._remove_joint_actuation("back")
        self._align_actuators_with_joints()

        self._remove_free_joint()

        self._update_joint_properties()

    def _align_actuators_with_joints(self, use_dummy_actuator=False):
        # Fixed joint-actuator mismatch in the Robosuite codebase.
        # commit id: 45df0d87e25613e9edcc87739963539a93d390a9
        joint_name2actuator = {}
        for actuator in self.actuator:
            joint_name2actuator[actuator.get("joint")] = actuator
        new_actuators = []
        for joint in self.worldbody.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name in joint_name2actuator:
                new_actuators.append(joint_name2actuator[joint_name])
            elif use_dummy_actuator:
                import xml.etree.ElementTree as ET

                dummy_actuator = ET.Element(
                    "motor",
                    {
                        "name": joint_name,
                        "joint": joint_name,
                        "gear": "1 0 0 0 0 0",
                        "ctrllimited": "true",
                        "ctrlrange": "-1e-12 1e-12",
                    },
                )
                new_actuators.append(dummy_actuator)
            else:
                parent_body = find_parent(self.worldbody, joint)
                parent_body.remove(joint)
                self._joints.remove(joint.get("name").replace(self.naming_prefix, ""))
        self.actuator[:] = new_actuators
        self._actuators = [m.get("name").replace(self.naming_prefix, "") for m in self.actuator]


@register_robot_class("LeggedRobot")
class BDM1ArmsOnly(BDM1FixedLowerBody):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

        self._remove_joint_actuation("leg")

        self._remove_joint_actuation("back")
        self._remove_joint_actuation("head")

        self._align_actuators_with_joints()

        self._remove_free_joint()

        self._update_joint_properties()
