from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple
from xml.etree import ElementTree as ET
import mujoco
import mujoco.viewer


from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.binding_utils import MjSim
from robosuite.controllers.composite.composite_controller import (
    register_composite_controller,
)
from robosuite.utils.mjcf_utils import find_parent

from robocasa.examples.third_party_controller.mink_solver import IKSolverMink, WholeBodyMinkIK


@register_composite_controller
class HybridWholeBodyMinkIK(WholeBodyMinkIK):
    name = "HYBRID_WHOLE_BODY_MINK_IK"

    def __init__(self, sim: MjSim, robot_model: RobotModel, grippers: Dict[str, GripperModel]):
        super().__init__(sim, robot_model, grippers)

        # TODO: try add the CoM task later

    def _init_joint_action_policy(self):
        joint_names: str = []
        # the order of the joint names is important! Should be consistent with self._action_split_indexes
        for part_name in self.composite_controller_specific_config["actuation_part_names"]:
            if part_name in self.part_controllers:
                joint_names += self.part_controllers[part_name].joint_names

        # These joints are controlled by other controllers, not this IK controller
        self.others_controlled_joint_names: str = []
        if "external_part_names" in self.composite_controller_specific_config:
            for part_name in self.composite_controller_specific_config["external_part_names"]:
                if part_name in self.part_controllers:
                    self.others_controlled_joint_names += self.part_controllers[
                        part_name
                    ].joint_names

        default_site_names: List[str] = []
        for arm in ["right", "left"]:
            if arm in self.part_controller_config:
                default_site_names.append(self.part_controller_config[arm]["ref_name"])

        # remove the joints in robot model that are controlled by this IK controller
        # env_xml_str = self.sim.model.get_xml()
        xml_str = deepcopy(self.robot_model.get_xml())
        root = ET.fromstring(xml_str)
        # remove base joint
        for joint in root.findall(".//freejoint"):
            find_parent(root, joint).remove(joint)
        for joint in root.findall(".//joint"):
            if joint.get("name") in self.others_controlled_joint_names:
                find_parent(root, joint).remove(joint)
        for motor in root.findall(".//motor"):
            if motor.get("joint") in self.others_controlled_joint_names:
                find_parent(root, motor).remove(motor)
        for sensor in root.findall(".//jointpos"):
            if sensor.get("joint") in self.others_controlled_joint_names:
                find_parent(root, sensor).remove(sensor)
        for sensor in root.findall(".//jointvel"):
            if sensor.get("joint") in self.others_controlled_joint_names:
                find_parent(root, sensor).remove(sensor)
        for sensor in root.findall(".//jointactuatorfrc"):
            if sensor.get("joint") in self.others_controlled_joint_names:
                find_parent(root, sensor).remove(sensor)
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="utf-8").decode("utf-8"))
        robot_model = spec.compile()
        # robot_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="utf-8").decode("utf-8"))
        # mujoco.MjModel.from_xml_string(env_xml_str)

        self.joint_action_policy = IKSolverMink(
            model=self.sim.model._model,
            data=self.sim.data._data,
            site_names=(
                self.composite_controller_specific_config["ref_name"]
                if "ref_name" in self.composite_controller_specific_config
                else default_site_names
            ),
            robot_model=robot_model,
            robot_joint_names=joint_names,
            input_type=self.composite_controller_specific_config.get("ik_input_type", "absolute"),
            input_ref_frame=self.composite_controller_specific_config.get(
                "ik_input_ref_frame", "world"
            ),
            input_rotation_repr=self.composite_controller_specific_config.get(
                "ik_input_rotation_repr", "axis_angle"
            ),
            solve_freq=self.composite_controller_specific_config.get("ik_solve_freq", 20),
            posture_weights=self.composite_controller_specific_config.get("ik_posture_weights", {}),
            hand_pos_cost=self.composite_controller_specific_config.get("ik_hand_pos_cost", 1.0),
            hand_ori_cost=self.composite_controller_specific_config.get("ik_hand_ori_cost", 0.5),
            verbose=self.composite_controller_specific_config.get("verbose", False),
        )
