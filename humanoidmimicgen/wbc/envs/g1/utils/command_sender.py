from typing import Dict

from loguru import logger
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
from unitree_sdk2py.utils.crc import CRC

from humanoidmimicgen.wbc.envs.g1.utils.robot import Robot


class BodyCommandSender:
    def __init__(self, config: Dict):
        self.config = config
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif (
            self.config["ROBOT_TYPE"] == "g1_29dof"
            or self.config["ROBOT_TYPE"] == "h1-2_21dof"
            or self.config["ROBOT_TYPE"] == "h1-2_27dof"
        ):
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        else:
            raise NotImplementedError(
                f"Robot type {self.config['ROBOT_TYPE']} is not supported yet"
            )
        # init robot and kp kd
        self.robot = Robot(self.config)
        self.kp_level = 1.0
        self.waist_kp_level = 1.0
        self.robot_default_kp = np.zeros(self.robot.NUM_MOTORS)
        self.robot_default_kd = np.zeros(self.robot.NUM_MOTORS)
        # set kp level
        for i in range(len(self.robot.MOTOR_KP)):
            self.robot_default_kp[i] = self.robot.MOTOR_KP[i]
        for i in range(len(self.robot.MOTOR_KD)):
            self.robot_default_kd[i] = self.robot.MOTOR_KD[i]
        self.weak_motor_joint_index = []
        for _, value in self.robot.WeakMotorJointIndex.items():
            self.weak_motor_joint_index.append(value)
        # init low cmd publisher
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.InitLowCmd()
        self.low_state = None
        self.crc = CRC()

    def InitLowCmd(self):
        # h1/go2:
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
        else:
            pass

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(self.robot.NUM_MOTORS):
            if self.is_weak_motor(i):
                self.low_cmd.motor_cmd[i].mode = 0x01
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q = self.robot.UNITREE_LEGGED_CONST["PosStopF"]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = self.robot.UNITREE_LEGGED_CONST["VelStopF"]
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
            if (
                self.config["ROBOT_TYPE"] == "g1_29dof"
                or self.config["ROBOT_TYPE"] == "h1-2_21dof"
                or self.config["ROBOT_TYPE"] == "h1-2_27dof"
            ):
                self.low_cmd.mode_machine = self.config["UNITREE_LEGGED_CONST"]["MODE_MACHINE"]
                self.low_cmd.mode_pr = self.config["UNITREE_LEGGED_CONST"]["MODE_PR"]
            else:
                pass

    def is_weak_motor(self, motor_index: int) -> bool:
        return motor_index in self.weak_motor_joint_index

    def send_command(self, cmd_q: np.ndarray, cmd_dq: np.ndarray, cmd_tau: np.ndarray):
        for i in range(self.robot.NUM_MOTORS):
            motor_index = self.robot.JOINT2MOTOR[i]
            joint_index = self.robot.MOTOR2JOINT[i]
            # print(f"motor_index: {motor_index}, joint_index: {joint_index}")
            if joint_index == -1:
                # send default joint position command
                self.low_cmd.motor_cmd[motor_index].q = self.robot.DEFAULT_MOTOR_ANGLES[motor_index]
                self.low_cmd.motor_cmd[motor_index].dq = 0.0
                self.low_cmd.motor_cmd[motor_index].tau = 0.0
            else:
                self.low_cmd.motor_cmd[motor_index].q = cmd_q[joint_index]
                self.low_cmd.motor_cmd[motor_index].dq = cmd_dq[joint_index]
                self.low_cmd.motor_cmd[motor_index].tau = cmd_tau[joint_index]
            # kp kd
            self.low_cmd.motor_cmd[motor_index].kp = (
                self.robot_default_kp[motor_index] * self.kp_level
            )
            self.low_cmd.motor_cmd[motor_index].kd = self.robot_default_kd[motor_index]

        # print("kp", [self.low_cmd.motor_cmd[i].kp for i in range(self.robot.NUM_MOTORS)])
        # print("kd", [self.low_cmd.motor_cmd[i].kd for i in range(self.robot.NUM_MOTORS)])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def handle_keyboard_button(self, key):
        if key == "1":
            self.kp_level -= 0.1
        elif key == "2":
            self.kp_level -= 0.01
        elif key == "3":
            self.kp_level += 0.01
        elif key == "4":
            self.kp_level += 0.1
        elif key == "0":
            self.kp_level = 1.0

        logger.info("kp_level: {}", self.kp_level)
        logger.info("current kp: {}", self.robot_default_kp * self.kp_level)
        logger.info("current kd: {}", self.robot_default_kd * self.kp_level)


def make_hand_mode(motor_index: int) -> int:
    status = 0x01
    timeout = 0x01
    mode = motor_index & 0x0F
    mode |= status << 4  # bits [4..6]
    mode |= timeout << 7  # bit 7
    return mode


class HandCommandSender:
    def __init__(self, is_left: bool = True):
        self.is_left = is_left
        if self.is_left:
            self.cmd_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        else:
            self.cmd_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)

        self.cmd_pub.Init()
        self.cmd = unitree_hg_msg_dds__HandCmd_()

        self.hand_dof = 7

        self.kp = [1.0] * self.hand_dof
        self.kd = [0.2] * self.hand_dof
        self.kp[0] = 2.0
        self.kd[0] = 0.5

    def send_command(self, cmd: np.ndarray):
        for i in range(self.hand_dof):
            # Build the bitfield mode (see your C++ example)
            mode_val = make_hand_mode(i)
            self.cmd.motor_cmd[i].mode = mode_val
            self.cmd.motor_cmd[i].q = cmd[i]
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
            self.cmd.motor_cmd[i].kp = self.kp[i]
            self.cmd.motor_cmd[i].kd = self.kd[i]

        self.cmd_pub.Write(self.cmd)
