from typing import Dict, List, Literal, Tuple

import mujoco
import numpy as np
from robosuite.controllers.composite.composite_controller import (
    WholeBody,
    register_composite_controller,
)
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.binding_utils import MjSim
import robosuite.utils.transform_utils as T


class IKSolverExternal:
    def __init__(
        self,
        robot_type: str,
        body_active_joint_groups: List[str],
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_names: List[str],
        robot_joint_names: List[str],
        input_ref_frame: Literal["world", "base", "eef"] = "world",
        input_rotation_repr: Literal["quat_wxyz", "axis_angle"] = "axis_angle",
    ):
        raise NotImplementedError(
            "The third-party external IK controller is not included in the local "
            "HumanoidMimicGen runtime. Use the bundled WBC-goal replay path instead."
        )

        if robot_type == "g1":
            self.robot_model = instantiate_g1_robot_model()
            # hand frame names in urdf
            self.left_hand_frame = self.robot_model.supplemental_info.hand_frame_names["left"]
            self.right_hand_frame = self.robot_model.supplemental_info.hand_frame_names["right"]
            # NOTE: the correction should be applied right before feeding into the ik solver
            self.hand_frame_rotation_correction = {
                "left": np.matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                "right": np.matrix([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
            }
        elif robot_type == "gr1":
            self.robot_model = instantiate_gr1_robot_model()
            # hand frame names in urdf
            self.left_hand_frame = self.robot_model.supplemental_info.hand_frame_names["left"]
            self.right_hand_frame = self.robot_model.supplemental_info.hand_frame_names["right"]
            # NOTE: the correction should be applied right before feeding into the ik solver
            self.hand_frame_rotation_correction = {
                "left": np.matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                "right": np.matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            }
        else:
            assert False, f"Unsupported robot type: {robot_type}"

        self.body = ReducedRobotModel.from_active_groups(self.robot_model, body_active_joint_groups)
        self.full_robot = self.body.full_robot
        body_ik_solver_settings = BodyIKSolverSettings()
        self.body_ik_solver = BodyIKSolver(body_ik_solver_settings)
        self.body_ik_solver.register_robot(self.body)
        self.in_warmup = True

        # Imported from robosuite
        self.full_model: mujoco.MjModel = model
        self.full_model_data: mujoco.MjData = data

        self.controlled_robot_qpos_indexes: List[int] = []
        for joint_name in robot_joint_names:
            joint_name = joint_name.replace("robot0_", "")
            if joint_name.startswith("l_"):
                joint_name = "left_" + joint_name[2:]
            if joint_name.startswith("r_"):
                joint_name = "right_" + joint_name[2:]
            if joint_name.startswith("torso_"):
                joint_name = joint_name.replace("torso_", "")
            if not joint_name.endswith("_joint"):
                joint_name += "_joint"
            self.controlled_robot_qpos_indexes.append(self.robot_model.dof_index(joint_name))

        self.site_names = site_names

        self.input_ref_frame = input_ref_frame
        self.input_rotation_repr = input_rotation_repr
        ROTATION_REPRESENTATION_DIMS: Dict[str, int] = {"quat_wxyz": 4, "axis_angle": 3}
        self.rot_dim = ROTATION_REPRESENTATION_DIMS[input_rotation_repr]
        self.pos_dim = 3
        self.control_dim = len(self.site_names) * (self.pos_dim + self.rot_dim)
        # hardcoded control limits for now
        self.control_limits = np.array([-np.inf] * self.control_dim), np.array(
            [np.inf] * self.control_dim
        )

    def reset(self):
        self.body.reset_forward_kinematics()  # self.body is the same one as self.body_ik_solver.robot
        self.full_robot.reset_forward_kinematics()
        self.body_ik_solver.initialize()
        # If in the future, the hand IK solver has initialize method, call it
        self.in_warmup = True

    def action_split_indexes(self) -> Dict[str, Tuple[int, int]]:
        action_split_indexes: Dict[str, Tuple[int, int]] = {}
        previous_idx = 0

        for site_name in self.site_names:
            total_dim = self.pos_dim + self.rot_dim
            last_idx = previous_idx + total_dim
            simplified_site_name = (
                "left" if "left" in site_name else "right"
            )  # hack to simplify site names
            # goal is to specify the end effector actions as "left" or "right" instead of the actual site name
            # we assume that the site names for the ik solver are unique and contain "left" or "right" in them
            action_split_indexes[simplified_site_name] = (previous_idx, last_idx)
            previous_idx = last_idx

        return action_split_indexes

    def transform_pose(
        self,
        src_frame_pose: np.ndarray,
        src_frame: Literal["world", "base"],
        dst_frame: Literal["world", "base"],
    ) -> np.ndarray:
        """
        Transforms src_frame_pose from src_frame to dst_frame.
        """
        if src_frame == dst_frame:
            return src_frame_pose

        X_src_frame_pose = src_frame_pose
        # convert src frame pose to world frame pose
        if src_frame != "world":
            X_W_src_frame = T.make_pose(
                translation=self.full_model.body(src_frame).pos,
                rotation=T.quat2mat(np.roll(self.full_model.body(src_frame).quat, shift=-1)),
            )
            X_W_pose = X_W_src_frame @ X_src_frame_pose
        else:
            X_W_pose = src_frame_pose

        # now convert to destination frame
        if dst_frame == "world":
            X_dst_frame_pose = X_W_pose
        elif dst_frame == "base":
            X_dst_frame_W = np.linalg.inv(
                T.make_pose(
                    translation=self.full_model.body("robot0_base").pos,
                    rotation=T.quat2mat(
                        np.roll(self.full_model.body("robot0_base").quat, shift=-1)
                    ),
                )
            )  # hardcode name of base
            X_dst_frame_pose = X_dst_frame_W.dot(X_W_pose)

        return X_dst_frame_pose

    def solve(self, input_action: np.ndarray) -> np.ndarray:
        input_action = input_action.reshape(len(self.site_names), -1)
        input_pos = input_action[:, : self.pos_dim]
        input_ori = input_action[:, self.pos_dim :]

        input_quat_wxyz = None
        if self.input_rotation_repr == "axis_angle":
            input_quat_wxyz = np.array(
                [np.roll(T.axisangle2quat(input_ori[i]), 1) for i in range(len(input_ori))]
            )
        elif self.input_rotation_repr == "mat":
            input_quat_wxyz = np.array(
                [np.roll(T.mat2quat(input_ori[i])) for i in range(len(input_ori))]
            )
        elif self.input_rotation_repr == "quat_wxyz":
            input_quat_wxyz = input_ori

        right_hand_pose = T.make_pose(
            input_pos[0], T.quat2mat(T.convert_quat(input_quat_wxyz[0], "xyzw"))
        )
        left_hand_pose = T.make_pose(
            input_pos[1], T.quat2mat(T.convert_quat(input_quat_wxyz[1], "xyzw"))
        )

        target_pose = {
            self.right_hand_frame: right_hand_pose,
            self.left_hand_frame: left_hand_pose,
        }
        if self.in_warmup:
            for _ in range(50):
                body_q = self.body.reduced_to_full_configuration(self.body_ik_solver(target_pose))
            self.in_warmup = False
        else:
            body_q = self.body.reduced_to_full_configuration(self.body_ik_solver(target_pose))
        return body_q[self.controlled_robot_qpos_indexes]


@register_composite_controller
class WholeBodyExternalIK(WholeBody):
    name = "WHOLE_BODY_EXTERNAL_IK"

    def __init__(self, sim: MjSim, robot_model: RobotModel, grippers: Dict[str, GripperModel]):
        super().__init__(sim, robot_model, grippers)

    def _validate_composite_controller_specific_config(self) -> None:
        from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

        # Check that all actuation_part_names exist in part_controllers
        original_ik_controlled_parts = self.composite_controller_specific_config[
            "actuation_part_names"
        ]
        self.valid_ik_controlled_parts = []
        valid_ref_names = []

        assert (
            "ref_name" in self.composite_controller_specific_config
        ), "The 'ref_name' key is missing from composite_controller_specific_config."

        for part in original_ik_controlled_parts:
            if part in self.part_controllers:
                self.valid_ik_controlled_parts.append(part)
            else:
                ROBOSUITE_DEFAULT_LOGGER.warning(
                    f"Part '{part}' specified in 'actuation_part_names' "
                    "does not exist in part_controllers. Removing ..."
                )

        # Update the configuration with only the valid parts
        self.composite_controller_specific_config["actuation_part_names"] = (
            self.valid_ik_controlled_parts
        )

        # Loop through ref_names and validate against mujoco model
        original_ref_names = self.composite_controller_specific_config.get("ref_name", [])
        for ref_name in original_ref_names:
            if (
                ref_name in self.sim.model.site_names
            ):  # Check if the site exists in the mujoco model
                valid_ref_names.append(ref_name)
            else:
                ROBOSUITE_DEFAULT_LOGGER.warning(
                    f"Reference name '{ref_name}' specified in configuration"
                    " does not exist in the mujoco model. Removing ..."
                )

        # Update the configuration with only the valid reference names
        self.composite_controller_specific_config["ref_name"] = valid_ref_names

    def _init_joint_action_policy(self):
        robot_name = self.robot_model.__class__.__name__

        if robot_name == "G1ArmsOnly":
            robot_type, body_active_joint_groups = "g1", ["arms"]
        elif robot_name in ["G1FixedLowerBody", "G1FloatingBody", "G1FloatingBodyWithVertical"]:
            robot_type, body_active_joint_groups = "g1", ["arms", "waist"]
        elif robot_name in ["GR1ArmsOnlyFourierHands", "GR1ArmsOnlyInspireHands", "GR1ArmsOnly"]:
            robot_type, body_active_joint_groups = "gr1", ["arms"]
        elif robot_name in [
            "GR1ArmsAndWaistFourierHands",
            "GR1ArmsAndWaistInspireHands",
            "GR1ArmsAndWaist",
        ]:
            robot_type, body_active_joint_groups = "gr1", ["arms", "waist"]
        else:
            assert False, f"Unsupported robot type: {robot_name}"

        joint_names: str = []
        for part_name in self.composite_controller_specific_config["actuation_part_names"]:
            if part_name in self.part_controllers:
                joint_names += self.part_controllers[part_name].joint_names

        default_site_names: List[str] = []
        for arm in ["right", "left"]:
            if arm in self.part_controller_config:
                default_site_names.append(self.part_controller_config[arm]["ref_name"])

        self.joint_action_policy = IKSolverExternal(
            robot_type=robot_type,
            body_active_joint_groups=body_active_joint_groups,
            model=self.sim.model._model,
            data=self.sim.data._data,
            site_names=(
                self.composite_controller_specific_config["ref_name"]
                if "ref_name" in self.composite_controller_specific_config
                else default_site_names
            ),
            robot_joint_names=joint_names,
            input_ref_frame=self.composite_controller_specific_config.get(
                "ik_input_ref_frame", "world"
            ),
            input_rotation_repr=self.composite_controller_specific_config.get(
                "ik_input_rotation_repr", "axis_angle"
            ),
        )

    def reset(self):
        super().reset()
        self.joint_action_policy.reset()
