import numpy as np

from robosuite.robots import register_robot_class
from robosuite.models.robots import GR1
from robosuite.models.robots.manipulators.gr1_robot import (
    GR1FixedLowerBody,
    GR1ArmsOnly,
)
from robosuite.utils.mjcf_utils import find_elements


def get_camera_configs(self):
    _cam_config = {}

    # stereo cameras
    stereo_mount_body = find_elements(
        root=self.root,
        tags="body",
        attribs={"name": f"{self.naming_prefix}head_pitch"},
    )

    if stereo_mount_body is not None:
        base_pos = np.array(
            [
                2.650 - 2.65017178 + 0.23,
                -1.944 + 2.174 - 0.23,
                1.538 - 1.4475,
            ]
        )
        fovy = 70  # Max. 110°(H) x 70°(V) x 120°(D)
        _cam_config[f"{self.naming_prefix}stereo_leftview"] = dict(
            pos=(base_pos + [0, 0.06, 0.0]).tolist(),
            quat=[0.676, 0.205, -0.205, -0.676],
            camera_attribs=dict(fovy=f"{fovy}"),
            parent_body=f"{self.naming_prefix}head_pitch",
        )
        _cam_config[f"{self.naming_prefix}stereo_rightview"] = dict(
            pos=(base_pos - [0, 0.06, 0.0]).tolist(),
            quat=[0.676, 0.205, -0.205, -0.676],
            camera_attribs=dict(fovy=f"{fovy}"),
            parent_body=f"{self.naming_prefix}head_pitch",
        )

    return _cam_config


GR1.get_camera_configs = get_camera_configs


@register_robot_class("LeggedRobot")
class GR1FixedLowerBodyInspireHands(GR1FixedLowerBody):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand", "left": "InspireLeftHand"}


@register_robot_class("LeggedRobot")
class GR1FixedLowerBodyFourierHands(GR1FixedLowerBody):
    @property
    def default_gripper(self):
        return {"right": "FourierRightHand", "left": "FourierLeftHand"}


@register_robot_class("LeggedRobot")
class GR1ArmsOnlyInspireHands(GR1ArmsOnly):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand", "left": "InspireLeftHand"}


@register_robot_class("LeggedRobot")
class GR1ArmsOnlyFourierHands(GR1ArmsOnly):
    @property
    def default_gripper(self):
        return {"right": "FourierRightHand", "left": "FourierLeftHand"}


@register_robot_class("LeggedRobot")
class GR1ArmsAndWaist(GR1):
    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._remove_joint_actuation("leg")
        self._remove_joint_actuation("head")
        self._remove_free_joint()

    @property
    def init_qpos(self):
        init_qpos = np.array([0.0] * 17)
        right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        left_arm_init = np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[3:10] = right_arm_init
        init_qpos[10:17] = left_arm_init
        return init_qpos


@register_robot_class("LeggedRobot")
class GR1ArmsAndWaistFourierHands(GR1ArmsAndWaist):
    pass
