from robosuite.devices import Device
import robosuite.utils.transform_utils as T

from robosuite.models.grippers import GripperModel

from pynput.keyboard import Listener

import hydra
import leap
import time
import threading
import numpy as np
from typing import Optional, Dict
from scipy.spatial.transform import Rotation as R


class DistanceSolver:
    def __init__(self):
        self.clip_min = np.array([36.28062662, 48.75686028, 46.34104357, 42.4643279, 35.96989512])
        self.clip_max = np.array([72.8848676, 72.91646019, 78.1876032, 73.09950146, 64.39725001])

    def calculate_x_plane(self, x_plane_points):
        centroid = np.mean(x_plane_points, axis=0)
        # Translate points to the origin
        translated_points = x_plane_points - centroid
        # Apply SVD
        _, _, Vt = np.linalg.svd(translated_points)
        # The normal vector of the plane is the last column of V
        return Vt[-1, :]

    def angle_between(self, v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(cos_theta)
        if theta > np.pi / 2:
            theta = np.pi - theta
        return theta

    def calculate_thumb_rot(self, target_points, x_plane_normal, z_plane_normal):
        # use the index finger plane as the new z_plane
        mid_f_normal = z_plane_normal
        y_plane_normal = np.cross(x_plane_normal, mid_f_normal)

        # value check
        y_plane_norm = np.linalg.norm(y_plane_normal)
        if np.isnan(y_plane_norm) or np.isinf(y_plane_norm) or y_plane_norm < 1e-6:
            self.value_error = True

        # calculate 1,2 shadow on y_plane
        p1, p2 = target_points[1], target_points[2]
        v = p2 - p1
        v_proj = (
            v
            - (np.dot(v, y_plane_normal) / np.linalg.norm(y_plane_normal) ** 2 + 1e-3)
            * y_plane_normal
        )

        # calculate angle between z_plane_normal and v_proj
        theta = self.angle_between(mid_f_normal, v_proj)

        # modify delte theta into quadratic
        if theta > np.pi / 4:
            delta_theta = np.pi * 1 / 2 - theta
            theta = np.pi * 1 / 2 - delta_theta**2

        return theta

    #  https://github.com/Improbable-AI/VisionProTeleop
    #  4 9 14 19 24
    #  | | |   |  |
    #  2 6 11 16 21
    #       0
    def __call__(self, finger_data, id, verbose=False):
        target_points = finger_data["position"][[0, 2, 4, 6, 9, 11, 14, 16, 19, 21, 24], :3, 3]
        finger_tip_position = target_points[[2, 4, 6, 8, 10]]
        finger_base_position = target_points[[5, 0, 0, 0, 0]]

        finger_distance = np.linalg.norm(finger_tip_position - finger_base_position, axis=1)

        # Clip finger distances based on individual clip parameters
        finger_clipped_distances = (
            np.clip(finger_distance, self.clip_min, self.clip_max) - self.clip_min
        )
        finger_angles = np.arccos(finger_clipped_distances / (self.clip_max - self.clip_min))

        # Find the plane of the hand
        x_plane_points = target_points[[0, 3, 5, 7, 9], :]
        x_plane_normal = self.calculate_x_plane(x_plane_points)
        z_plane_normal = np.cross(x_plane_normal, target_points[5] - target_points[0])

        thumb_rot_theta = self.calculate_thumb_rot(target_points, x_plane_normal, z_plane_normal)

        joint_actuators = np.concatenate((finger_angles[::-1], np.array([thumb_rot_theta])))
        if verbose:
            print(joint_actuators)

        return joint_actuators


class FingerIKSolver:
    def __init__(self, grippers: list[GripperModel]):
        with hydra.initialize(
            config_path="../../../groot/groot/teleop/configs/hand", version_base=None
        ):
            solver_config = hydra.compose(config_name="fourier")
        # TODO: clean this up
        # Implementation see groot/groot/teleop/solver/hand_ik_solver.py
        solver_config.robot.solver.thumb_pinky_dis = 0.07
        solver_config.robot.solver.thumb_bend_range = [0.06, 0.09]
        self.left_hand_solver = hydra.utils.instantiate(solver_config.robot.solver, side="left")
        self.right_hand_solver = hydra.utils.instantiate(solver_config.robot.solver, side="right")

        # Instantiate robots
        left_robot = hydra.utils.instantiate(solver_config.robot, side="left")
        right_robot = hydra.utils.instantiate(solver_config.robot, side="right")

        # Register the solvers to the robot instances
        self.left_hand_solver.register_robot(left_robot)
        self.right_hand_solver.register_robot(right_robot)

    def __call__(self, finger_data, id, verbose=False):
        finger_order = [4, 6, 2, 0, 9, 8]
        finger_signs = [-1, -1, -1, -1, 1, -1]
        if id == 1:
            left_data_robot = self.left_hand_solver(finger_data)
            processed_left = [
                left_data_robot[i] * finger_signs[idx] for idx, i in enumerate(finger_order)
            ]
            return processed_left
        else:
            right_data_robot = self.right_hand_solver(finger_data)
            processed_right = [
                right_data_robot[i] * finger_signs[idx] for idx, i in enumerate(finger_order)
            ]
            return processed_right


class PinchStrengthSolver:
    def __init__(self, grippers: list[str]):
        self.grippers = grippers

    def __call__(self, pinch_strength, id):
        state = -1 if pinch_strength == 0 else 1
        if hasattr(self.grippers[id], "grasp_qpos"):
            return getattr(self.grippers[id], "grasp_qpos")[state]
        else:
            return np.ones(self.grippers[id].dof) * state


class LeapMotionListener(leap.Listener):
    class DummyDevice(Device):
        def __init__(self):
            self._control = [None, None]
            self._gripper = [
                np.array([0 for i in range(6)]),
                np.array([0 for i in range(6)]),
            ]
            self._reset_state = 0

        def start_control(self):
            pass

        def get_controller_state(self):
            pass

    def __init__(
        self,
        pos_sensitivity,
        left_hand_pos_offset,
        right_hand_pos_offset,
        device: Device = DummyDevice(),
        grippers: list[GripperModel] = [],
        finger_tracking: bool = True,
    ):
        super().__init__()
        self.device = device
        self.pos_sensitivity = pos_sensitivity
        self.finger_tracking = finger_tracking
        if self.finger_tracking:
            self.solver = FingerIKSolver(grippers)
        else:
            self.solver = PinchStrengthSolver(grippers)
        self.hand_pos_offset = [right_hand_pos_offset, left_hand_pos_offset]

    def on_connection_event(self, event):
        print("Connected")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        if self.device._reset_state:
            return

        for hand in event.hands:
            id = 1 if str(hand.type) == "HandType.Left" else 0
            # hand pose
            if self.finger_tracking:

                def xyz2np_array(position):
                    return np.array([position.x, position.y, position.z])

                def quat2np_array(quaternion):
                    return np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

                target_points = np.array([np.eye(4) for _ in range(25)])
                target_points[0, :3, :3] = T.quat2mat(quat2np_array(hand.palm.orientation))
                target_points[0, :3, 3] = xyz2np_array(hand.arm.next_joint)
                target_points[1, :3, 3] = xyz2np_array(hand.thumb.bones[0].next_joint)
                target_points[2, :3, 3] = xyz2np_array(hand.thumb.bones[1].next_joint)
                target_points[3, :3, 3] = xyz2np_array(hand.thumb.bones[2].next_joint)
                target_points[4, :3, 3] = xyz2np_array(hand.thumb.bones[3].next_joint)
                target_points[5, :3, 3] = xyz2np_array(hand.index.bones[0].prev_joint)
                target_points[6, :3, 3] = xyz2np_array(hand.index.bones[0].next_joint)
                target_points[7, :3, 3] = xyz2np_array(hand.index.bones[1].next_joint)
                target_points[8, :3, 3] = xyz2np_array(hand.index.bones[2].next_joint)
                target_points[9, :3, 3] = xyz2np_array(hand.index.bones[3].next_joint)
                target_points[10, :3, 3] = xyz2np_array(hand.middle.bones[0].prev_joint)
                target_points[11, :3, 3] = xyz2np_array(hand.middle.bones[0].next_joint)
                target_points[12, :3, 3] = xyz2np_array(hand.middle.bones[1].next_joint)
                target_points[13, :3, 3] = xyz2np_array(hand.middle.bones[2].next_joint)
                target_points[14, :3, 3] = xyz2np_array(hand.middle.bones[3].next_joint)
                target_points[15, :3, 3] = xyz2np_array(hand.ring.bones[0].prev_joint)
                target_points[16, :3, 3] = xyz2np_array(hand.ring.bones[0].next_joint)
                target_points[17, :3, 3] = xyz2np_array(hand.ring.bones[1].next_joint)
                target_points[18, :3, 3] = xyz2np_array(hand.ring.bones[2].next_joint)
                target_points[19, :3, 3] = xyz2np_array(hand.ring.bones[3].next_joint)
                target_points[20, :3, 3] = xyz2np_array(hand.pinky.bones[0].prev_joint)
                target_points[21, :3, 3] = xyz2np_array(hand.pinky.bones[0].next_joint)
                target_points[22, :3, 3] = xyz2np_array(hand.pinky.bones[1].next_joint)
                target_points[23, :3, 3] = xyz2np_array(hand.pinky.bones[2].next_joint)
                target_points[24, :3, 3] = xyz2np_array(hand.pinky.bones[3].next_joint)

                # TODO: clean this up
                # import pinocchio as pin
                # for solver in [self.solver.left_hand_solver, self.solver.right_hand_solver]:
                #     # Initialize joint configuration
                #     q = pin.neutral(solver.robot.model)  # Or provide a custom configuration
                #     # Compute forward kinematics
                #     pin.forwardKinematics(solver.robot.model, solver.robot.data, q)
                #     pin.updateFramePlacements(solver.robot.model, solver.robot.data)
                #     if solver.side == "L" and id == 1:
                #         palm = solver.robot.data.oMf[solver.robot.model.getFrameId("l_hand_base_link")].translation
                #         L_thumb_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("L_thumb_tip")].translation
                #         L_pinky_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("L_pinky_tip")].translation
                #         L_middle_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("L_middle_tip")].translation
                #         L_index_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("L_index_tip")].translation
                #         L_ring_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("L_ring_tip")].translation

                #     elif solver.side == "R" and id == 0:
                #         palm = solver.robot.data.oMf[solver.robot.model.getFrameId("r_hand_base_link")].translation
                #         R_thumb_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("R_thumb_tip")].translation
                #         R_pinky_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("R_pinky_tip")].translation
                #         R_middle_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("R_middle_tip")].translation
                #         R_index_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("R_index_tip")].translation
                #         R_ring_tip = solver.robot.data.oMf[solver.robot.model.getFrameId("R_ring_tip")].translation

                pose_palm_root = target_points[0, :3, 3].copy()
                rot_palm = target_points[0, :3, :3].copy()

                if id == 0:  # Right hand
                    rot_leap2base = R.from_euler("z", [180], degrees=True).as_matrix()
                    rot_reverse = np.array(
                        [[[0, 0, 1], [0, -1, 0], [1, 0, 0]]]
                    )  # due to the target_base_rotation in hand IK solver
                else:
                    rot_leap2base = np.eye(3)
                    rot_reverse = np.array([[[0, 0, -1], [0, -1, 0], [-1, 0, 0]]])
                offset = (target_points[:, :3, 3] - pose_palm_root).copy() / 1000.0  # mm to m
                offset = offset @ rot_palm @ rot_leap2base @ rot_reverse

                target_points[:, :3, 3] = offset

                with self.device._gripper_lock:
                    self.device._gripper[id] = self.solver({"position": target_points}, id)
            else:
                with self.device._gripper_lock:
                    self.device._gripper[id] = self.solver(hand.pinch_strength, id)

            # wrist pose
            with self.device._control_lock:
                if self.device._control[id] is None:
                    self.device._control[id] = np.array([0.0] * 6)
                self.device._control[id][0] = (
                    -hand.palm.position.z * self.pos_sensitivity + self.hand_pos_offset[id][0]
                )
                self.device._control[id][1] = (
                    -hand.palm.position.x * self.pos_sensitivity + self.hand_pos_offset[id][1]
                )
                self.device._control[id][2] = (
                    hand.palm.position.y * self.pos_sensitivity + self.hand_pos_offset[id][2]
                )
            palm_normal = hand.palm.normal
            direction = hand.palm.direction
            palm_normal_np = np.array([-palm_normal.z, -palm_normal.x, palm_normal.y])
            direction_np = np.array([-direction.z, -direction.x, direction.y])
            if id == 1:
                original_matrix = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
                lh_thumb_np = -np.cross(direction_np, palm_normal_np)
                rotation_matrix = np.array([direction_np, -palm_normal_np, lh_thumb_np]).T
            else:
                original_matrix = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
                rh_thumb_np = np.cross(direction_np, palm_normal_np)
                rotation_matrix = np.array([direction_np, palm_normal_np, rh_thumb_np]).T
            rotation_axisangle = T.quat2axisangle(
                T.mat2quat(np.dot(rotation_matrix, original_matrix))
            )

            with self.device._control_lock:
                self.device._control[id][3] = rotation_axisangle[0]
                self.device._control[id][4] = rotation_axisangle[1]
                self.device._control[id][5] = rotation_axisangle[2]


class LeapMotion(Device):
    def __init__(
        self,
        env,
        finger_tracking,
        pos_sensitivity=1.0 / 300.0,
        left_hand_pos_offset=[0.0, -0.3, -0.5],
        right_hand_pos_offset=[0.0, 0.3, -0.5],
    ):
        super().__init__(env)

        leap_motion_listener = LeapMotionListener(
            pos_sensitivity,
            left_hand_pos_offset,
            right_hand_pos_offset,
            device=self,
            grippers=[env.robots[0].gripper["right"], env.robots[0].gripper["left"]],
            finger_tracking=finger_tracking,
        )

        self.gripper_dof = env.robots[0].gripper["right"].dof

        self.connection = leap.Connection()
        self.connection.add_listener(leap_motion_listener)

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        # also add a keyboard for aux controls
        keyboard_listener = Listener(on_press=self.on_press, on_release=self.on_release)
        keyboard_listener.start()

        self._control = [None, None]
        self._gripper = [
            np.array([0 for i in range(self.gripper_dof)]),  # Right
            np.array([0 for i in range(self.gripper_dof)]),  # Left
        ]
        self._reset_state = 0

        self._gripper_lock = threading.Lock()
        self._control_lock = threading.Lock()

    def run(self):
        with self.connection.open():
            self.connection.set_tracking_mode(leap.TrackingMode.Desktop)
            while True:
                time.sleep(1)

    def _reset_internal_state(self):
        super()._reset_internal_state()

        self._control = [None, None]
        with self._gripper_lock:
            self._gripper = [
                np.array([0 for i in range(self.gripper_dof)]),
                np.array([0 for i in range(self.gripper_dof)]),
            ]

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0

    def get_controller_state(self):
        raise NotImplementedError

    def input2action(self, mirror_actions=False) -> Optional[Dict]:
        if self._reset_state:
            return None

        robot = self.env.robots[self.active_robot]

        ac_dict = {"base_mode": -1}
        for arm in robot.arms:
            arm_action = self.get_arm_action(
                robot,
                arm,
                norm_delta=np.zeros(6),
            )
            ac_dict[f"{arm}_abs"] = arm_action["abs"]
            ac_dict[f"{arm}_delta"] = arm_action["delta"]
            ac_dict[f"{arm}_gripper"] = np.array([0 for i in range(self.gripper_dof)])

        if robot.is_mobile:
            ac_dict["base"] = np.zeros(3)
            ac_dict["base_mode"] = np.array([-1])

        for self.active_arm_index in [0, 1]:
            if self._control[self.active_arm_index] is None:
                continue
            active_arm = self.active_arm
            arm_norm_delta = self._control[self.active_arm_index] - ac_dict[f"{active_arm}_abs"]
            arm_norm_delta = np.clip(arm_norm_delta, -1, 1)
            arm_action = self.get_arm_action(
                robot,
                active_arm,
                norm_delta=arm_norm_delta,
            )
            ac_dict[f"{active_arm}_abs"] = arm_action["abs"]
            ac_dict[f"{active_arm}_delta"] = arm_action["delta"]
            with self._gripper_lock:
                ac_dict[f"{active_arm}_gripper"] = self._gripper[self.active_arm_index]

            assert (
                self.env.robots[self.active_robot].part_controllers[active_arm].input_type
                == "absolute"
            )
            with self._control_lock:
                # Overwrite the rotation part of the absolute action with the one from control
                ac_dict[f"{active_arm}_abs"][3:6] = self._control[self.active_arm_index][3:6]

            # This is a temporary solution to correct the hand frame discrepancy between mjcf and urdf files
            if robot.composite_controller_config["type"] == "WHOLE_BODY_EXTERNAL_IK":
                quat_wxyz = np.roll(T.axisangle2quat(ac_dict[f"{active_arm}_abs"][3:6]), 1)
                mat = T.quat2mat(T.convert_quat(quat_wxyz, "xyzw"))
                hand_frame_rotation_correction = self.env.robots[
                    0
                ].composite_controller.joint_action_policy.hand_frame_rotation_correction
                corrected_mat = mat @ hand_frame_rotation_correction[active_arm]
                ac_dict[f"{active_arm}_abs"][3:6] = T.quat2axisangle(T.mat2quat(corrected_mat))

        return ac_dict

    def on_press(self, key):
        pass

    def on_release(self, key):
        try:
            # controls for mobile base (only applicable if mobile base present)
            if key.char == "b":
                self.base_modes[self.active_robot] = not self.base_modes[
                    self.active_robot
                ]  # toggle mobile base
            elif key.char == "r":
                self._reset_state = 1
            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots

        except AttributeError as e:
            pass


if __name__ == "__main__":
    pos_sensitivity = 1.0 / 300.0
    left_hand_pos_offset = [0.0, -0.3, -0.5]
    right_hand_pos_offset = [0.0, 0.3, -0.5]
    listener = LeapMotionListener(pos_sensitivity, left_hand_pos_offset, right_hand_pos_offset)
    connection = leap.Connection()
    connection.add_listener(listener)

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        while True:
            print("[Right Wrist]", listener.device._control[0])
            print("[Left Wrist]", listener.device._control[1])
            print("[Right Gripper]", listener.device._gripper[0])
            print("[Left Gripper]", listener.device._gripper[1])
            print("")
            time.sleep(0.1)
