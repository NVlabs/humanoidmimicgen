"""
Dexterous hands for GR1 robot.
"""

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import find_parent, xml_path_completion
from robosuite.models.grippers import register_gripper

import robocasa.models


@register_gripper
class GR2LeftHand(GripperModel):
    """
    Dexterous left hand of GR1 robot
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion(
                "robots/bd_m1/assets/eatlas/left_floating_gr2.xml",
                root=robocasa.models.assets_root,
            ),
            idn=idn,
        )

    def format_action(self, action):
        grasp = action[-1]
        action = [0] * 7
        action[0] = action[2] = action[5] = grasp
        action[1] = action[3] = action[6] = 1
        action[4] = -1
        return action

    @property
    def init_qpos(self):
        return np.array([-1.5708, 2.00713, -1.5708, -2.00713, -3.14159, -1.5708, -2.00713])

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 6  # 12

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [
                "l_f0prox_link_col",
                "l_f0dist_link_col",
                "l_f1prox_link_col",
                "l_f1dist_link_col",
                "l_palm_rot_link_col",
                "l_f2prox_link_col",
                "l_f2dist_link_col",
            ],
            "right_fingerpad": [
                "l_f0prox_link_col",
                "l_f0dist_link_col",
                "l_f1prox_link_col",
                "l_f1dist_link_col",
                "l_palm_rot_link_col",
                "l_f2prox_link_col",
                "l_f2dist_link_col",
            ],
        }


@register_gripper
class GR2RightHand(GripperModel):
    """
    Dexterous right hand of GR1 robot
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion(
                "robots/bd_m1/assets/eatlas/right_floating_gr2.xml",
                root=robocasa.models.assets_root,
            ),
            idn=idn,
        )

    def format_action(self, action):
        grasp = action[-1]
        action = [0] * 7
        action[0] = action[2] = action[5] = grasp
        action[1] = action[3] = action[6] = 1
        action[4] = -1
        return action

    @property
    def init_qpos(self):
        return np.array([-1.5708, 2.00713, -1.5708, -2.00713, -3.14159, -1.5708, -2.00713])

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 6  # 12

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [
                "r_f0prox_link_col",
                "r_f0dist_link_col",
                "r_f1prox_link_col",
                "r_f1dist_link_col",
                "r_palm_rot_link_col",
                "r_f2prox_link_col",
                "r_f2dist_link_col",
            ],
            "right_fingerpad": [
                "r_f0prox_link_col",
                "r_f0dist_link_col",
                "r_f1prox_link_col",
                "r_f1dist_link_col",
                "r_palm_rot_link_col",
                "r_f2prox_link_col",
                "r_f2dist_link_col",
            ],
        }
