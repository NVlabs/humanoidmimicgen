# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
D1 variants of locomanipulation environments with increased randomization.
These variants have 4-5x larger randomization ranges for x_range, y_range, and rotation
compared to the original environments, intended for more diverse data collection.
"""

import numpy as np
from dataclasses import replace

from robocasa.environments.locomanipulation.base import RobotPoseRandomizer
from robocasa.environments.locomanipulation.locomanip_pnp import (
    LMBoxLift,
    LMBoxLiftFloor,
    LMDrillLift,
    LMDrillLiftObstacle,
    LMDrillLiftObstacleBi,
    LMDrillPnP90,
    LMDrillPnP90Bi,
    LMDrillPnPCloser,
    LMNavDrillPnPCloser,
    LMPickDrillFromHolder,
)
from robocasa.environments.locomanipulation.locomanip_simple import LMPushButton
from robocasa.environments.locomanipulation.locomanip_push import LMPushShelfForward
from robocasa.environments.locomanipulation.locomanip_basic import (
    LMBoxTableToShelfStaticIndustrial,
)
from robocasa.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
)
from robocasa.utils.scene.scene import SceneObject


# ==================== Base Environments D1 Variants ====================

class LMBoxLiftD1(LMBoxLift):
    """BoxLift with 8x increased randomization ranges"""

    def _get_objects(self):
        objects = super()._get_objects()
        # Original: x_range=[-0.02, 0.02], y_range=[-0.02, 0.02], rotation=[pi*0.98, pi*1.02]
        # D1: 8x increase
        self.table.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.16, 0.16]),  # 8x
                y_range=np.array([-0.16, 0.16]),  # 8x
                reference_pos=np.array([1.0, 0, 0]),
                rotation=np.array([np.pi * 0.84, np.pi * 1.16]),  # 8x range (0.16 vs 0.02)
            )
        )
        # Original box: x_range=[-0.26, -0.24], y_range=[-0.02, 0.02], rotation=[pi/2*0.98, pi/2*1.02]
        self.box.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.41, -0.09]),  # 8x (0.16 vs 0.02, centered at -0.25)
                y_range=np.array([-0.16, 0.16]),   # 8x
                rotation=np.array([np.pi / 2 * 0.84, np.pi / 2 * 1.16]),  # 8x
                reference=ReferenceConfig(obj=self.table),
            )
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


class LMBoxLiftFloorD1(LMBoxLiftFloor):
    """BoxLiftFloor with 8x increased randomization ranges"""

    def _get_objects(self):
        objects = super()._get_objects()
        # Original: x_range=[-0.02, 0.02], y_range=[-0.02, 0.02], rotation=[pi*0.98, pi*1.02]
        # D1: 8x increase
        self.box.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.16, 0.16]),  # 8x
                y_range=np.array([-0.16, 0.16]),  # 8x
                reference_pos=np.array([1.0, 0, 0]),
                rotation=np.array([np.pi * 0.84, np.pi * 1.16]),  # 8x
                z_offset=0.05,
            )
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


class LMDrillLiftD1(LMDrillLift):
    """DrillLift with 8x increased randomization ranges"""

    def _get_objects(self):
        objects = super()._get_objects()
        # Original table: x_range=[-0.02, 0.02], y_range=[-0.02, 0.02]
        self.table.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.16, 0.16]),  # 8x
                y_range=np.array([-0.16, 0.16]),  # 8x
                reference_pos=np.array([1.0, 0, 0]),
                rotation=np.array([np.pi * 0.84, np.pi * 1.16]),  # 8x
            )
        )
        # Original drill: x_range=[-0.28, -0.24], y_range=[-0.05, 0.05], rotation=[-pi*0.05, pi*0.05]
        self.bottle.update_cfg(
            scale=1.0,
            mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
            sampler_config=SamplingConfig(
                x_range=np.array([-0.42, -0.10]),  # 8x (0.32 vs 0.04, centered at -0.26)
                y_range=np.array([-0.40, 0.40]),   # 8x (0.80 vs 0.10)
                rotation=np.array([-np.pi * 0.40, np.pi * 0.40]),  # 8x
                reference=ReferenceConfig(obj=self.table),
            ),
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


class LMDrillLiftObstacleD1(LMDrillLiftObstacle):
    """DrillLiftObstacle with 6x increased randomization ranges"""

    def _get_objects(self):
        # Build from parent's parent to have full control
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.12, 0.12]),  # 6x
                    y_range=np.array([-0.12, 0.12]),  # 6x
                    reference_pos=np.array([1.0, 0, 0]),
                    rotation=np.array([np.pi * 0.88, np.pi * 1.12]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.38, -0.14]),  # 6x
                    y_range=np.array([0.28, 0.67]),    # 6x increase in range (0.39 vs ~0.065, centered at ~0.475)
                    rotation=np.array([-np.pi * 0.30, np.pi * 0.30]),  # 6x
                    reference=ReferenceConfig(obj=self.table),
                ),
            )
        )
        # Original obstacle: x_range=[-0.02, 0.02], y_range=[-0.1, 0.1]
        self.obstacle = SceneObject(
            ObjectConfig(
                name="obstacle",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.12, 0.12]),  # 6x
                    y_range=np.array([-0.60, 0.60]),  # 6x
                    reference_pos=np.array([0.35, 0, 0]),
                    rotation=np.array([np.pi * 0.88, np.pi * 1.12]),
                ),
                scale=0.7
            )
        )
        return [self.table, self.bottle, self.obstacle]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


class LMDrillPnP90D1(LMDrillPnP90):
    """DrillPnP90 with 6x increased randomization ranges"""

    def _get_objects(self):
        objects = super()._get_objects()
        # Original: x_range=[-0.02, 0.02], y_range=[-0.02, 0.02]
        # D1: 6x increase
        self.table_origin.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.12, 0.12]),  # 6x
                y_range=np.array([-0.12, 0.12]),  # 6x
                reference_pos=np.array([1.2, 0, 0]),
                rotation=np.array([-np.pi, -np.pi]),
            ),
        )
        self.table_target.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.12, 0.12]),  # 6x
                y_range=np.array([-0.12, 0.12]),  # 6x
                reference_pos=np.array([0, 1.2, 0]),
                rotation=np.array([-np.pi / 2, -np.pi / 2]),
            ),
        )
        # Increase bottle randomization
        self.bottle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.38, -0.14]),  # 6x (0.24 vs 0.04, centered at -0.26)
                y_range=np.array([-0.30, 0.30]),   # 6x (0.60 vs 0.10)
                rotation=np.array([-np.pi * 0.30, np.pi * 0.30]),  # 6x
                reference=ReferenceConfig(obj=self.table_origin),
            ),
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


class LMDrillPnPCloserD1(LMDrillPnPCloser):
    """DrillPnPCloser with 6x increased randomization ranges"""

    def _get_objects(self):
        objects = super()._get_objects()
        # Original: x_range=[-0.28, -0.24], y_range=[0.4, 0.45]
        # D1: 6x increase
        self.bottle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.38, -0.14]),  # 6x (0.24 vs 0.04, centered at -0.26)
                y_range=np.array([0.275, 0.575]),   # 6x (0.30 vs 0.05, centered at 0.425)
                rotation=np.array([-np.pi * 0.30, np.pi * 0.30]),  # 6x
                reference=ReferenceConfig(obj=self.table_origin),
            ),
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


class LMPickDrillFromHolderD1(LMPickDrillFromHolder):
    """PickDrillFromHolder with 6x increased randomization ranges"""

    def _get_objects(self):
        # Original shelf: x_range=[-0.02, 0.02], y_range=[-0.02, 0.02]
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.12, 0.12]),  # 6x
                    y_range=np.array([-0.12, 0.12]),  # 6x
                    reference_pos=np.array([1.35, 0, 0]),
                    rotation=np.array([np.pi / 2 * 0.88, np.pi / 2 * 1.12]),  # 6x
                ),
            )
        )
        # no change since holder is part of shelf
        self.holder = SceneObject(
            ObjectConfig(
                name="holder",
                mjcf_path="objects/omniverse/locomanip/powerdrill_holder/model.xml",
                static=True,
                friction=(0.0001, 0.005, 0.0001),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.75, -0.75]),  # 6x variation (0.30 range)
                    y_range=np.array([-0.02, 0.02]),   # 6x
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                    z_offset=-0.1,
                    reference=ReferenceConfig(self.shelf, spawn_id=1),
                ),
            )
        )
        # Drill position is relative to holder: x_range=[-0.1, -0.1], y_range=[-0.055, -0.055]
        self.drill = SceneObject(
            ObjectConfig(
                name="drill",
                mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
                static=False,
                friction=(1, 0.005, 0.0001),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.12, -0.08]),  # 4x variation (0.04 range, centered at -0.1)
                    y_range=np.array([-0.075, -0.035]),  # 4x variation (0.04 range, centered at -0.055)
                    z_offset=-0.145,
                    reference=ReferenceConfig(obj=self.holder, on_top=False),
                ),
            )
        )
        return [self.drill, self.shelf, self.holder]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )

class LMPushButtonD1(LMPushButton):
    """PushButton with 4x increased randomization ranges"""

    def _get_objects(self):
        # Original: x_range=[-1, -0.5], y_range=[-1, -0.5] (range of 0.5)
        # D1: 4x increase in range
        self.control_box = SceneObject(
            ObjectConfig(
                name="control_box",
                mjcf_path="objects/omniverse/locomanip/control_box/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-1, -0.2]),  # a bit further than D0
                    y_range=np.array([-1.2, -0.2]),  # more left-right range than D0
                    reference_pos=np.array([1, 1, 0]),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                ),
            )
        )
        # Panel position is relative to box: y_range=[-0.42, -0.38] (range of 0.04, centered at -0.40)
        self.control_panel = SceneObject(
            ObjectConfig(
                name="control_panel",
                mjcf_path="objects/omniverse/locomanip/control_conveyorbelt_a08/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.3, -0.3]),
                    y_range=np.array([-0.48, -0.38]),  # 4x (0.16 vs 0.04, centered at -0.40)
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                    z_offset=1,
                    reference=ReferenceConfig(obj=self.control_box, on_top=False),
                ),
            )
        )
        return [self.control_box, self.control_panel]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )

# old D1 too large randomization ranges
# class LMPushButtonD1(LMPushButton):
#     """PushButton with 4x increased randomization ranges"""

#     def _get_objects(self):
#         # Original: x_range=[-1, -0.5], y_range=[-1, -0.5] (range of 0.5)
#         # D1: 4x increase in range
#         self.control_box = SceneObject(
#             ObjectConfig(
#                 name="control_box",
#                 mjcf_path="objects/omniverse/locomanip/control_box/model.xml",
#                 static=True,
#                 sampler_config=SamplingConfig(
#                     x_range=np.array([-2.0, 0.0]),    # 4x range (2.0 vs 0.5)
#                     y_range=np.array([-2.0, 0.0]),    # 4x range
#                     reference_pos=np.array([1, 1, 0]),
#                     rotation=np.array([np.pi / 2, np.pi / 2]),
#                 ),
#             )
#         )
#         # Panel position is relative to box: y_range=[-0.42, -0.38] (range of 0.04, centered at -0.40)
#         self.control_panel = SceneObject(
#             ObjectConfig(
#                 name="control_panel",
#                 mjcf_path="objects/omniverse/locomanip/control_conveyorbelt_a08/model.xml",
#                 static=True,
#                 sampler_config=SamplingConfig(
#                     x_range=np.array([-0.3, -0.3]),
#                     y_range=np.array([-0.48, -0.32]),  # 4x (0.16 vs 0.04, centered at -0.40)
#                     rotation=np.array([np.pi / 2, np.pi / 2]),
#                     z_offset=1,
#                     reference=ReferenceConfig(obj=self.control_box, on_top=False),
#                 ),
#             )
#         )
#         return [self.control_box, self.control_panel]

#     def _reset_internal(self):
#         super()._reset_internal()
#         if not self.deterministic_reset:
#             RobotPoseRandomizer.set_pose(
#                 self,
#                 (-0.05, 0.05),
#                 (-0.05, 0.05),
#                 (-np.pi / 18, np.pi / 18)
#             )


class LMPushShelfForwardD1(LMPushShelfForward):
    """PushShelfForward with 3x shelf randomization (decreased from 6x)"""

    def _get_objects(self):
        # Get objects from parent (including box and target)
        super()._get_objects()
        # Only override shelf configuration with increased randomization (3x)
        self.shelf.cfg = replace(
            self.shelf.cfg,
            sampler_config=SamplingConfig(
                x_range=np.array([-0.30, 0.30]),  # 3x (0.60 vs 0.20 from parent)
                y_range=np.array([-0.30, 0.30]),  # 3x
                rotation=np.array([-np.pi * 1.06, -np.pi * 0.94]),  # 3x rotation variance
                reference_pos=np.array([1.5, 0, 0]),
            ),
        )
        # box and target remain unchanged from parent
        return [self.shelf, self.box, self.target]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


class LMBoxTableToShelfStaticIndustrialD1(LMBoxTableToShelfStaticIndustrial):
    """BoxTableToShelfStaticIndustrial with 6x increased randomization for the box and shelf"""

    def _get_objects(self):
        objects = super()._get_objects()

        # The parent environment has very static placement
        # For D1, add significant randomization to the box on table and shelf position

        # Randomize box position on table (6x)
        TABLE_CORNER = (-0.371296, 0.581634)
        BOX_OFFSET = (0.23, -0.56)
        BOX_CENTER = (TABLE_CORNER[0] + BOX_OFFSET[0], TABLE_CORNER[1] + BOX_OFFSET[1])

        self.box.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([BOX_CENTER[0] - 0.18, BOX_CENTER[0] + 0.18]),  # 6x variation
                y_range=np.array([BOX_CENTER[1] - 0.18, BOX_CENTER[1] + 0.18]),  # 6x variation
                rotation=np.array([np.pi - 0.6, np.pi + 0.6]),  # 6x rotation variation
                reference=ReferenceConfig(self.table, on_top=True),
            ),
        )

        # Randomize shelf cabinet position and rotation (6x)
        # Original: reference_pos=[0.55, -0.73, 0], rotation=[pi/2, pi/2]
        SHELF_CENTER = np.array([0.55, -0.73, 0])
        self.shelf_cabinet.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([SHELF_CENTER[0] - 0.18, SHELF_CENTER[0] + 0.18]),  # 6x variation
                y_range=np.array([SHELF_CENTER[1] - 0.18, SHELF_CENTER[1] + 0.18]),  # 6x variation
                rotation=np.array([np.pi * 0.5 - 0.36, np.pi * 0.5 + 0.36]),  # 6x rotation variation
            ),
        )

        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


# ==================== Nav Variants with D1 Randomization ====================

class LMNavBoxLiftD1(LMBoxLiftD1):
    """Nav version of BoxLift_D1"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Decreased rotation range
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.2, 0.2),
                (-np.pi / 12, np.pi / 12)
            )


class LMNavBoxLiftFloorD1(LMBoxLiftFloorD1):
    """Nav version of BoxLiftFloor_D1"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.2, 0.2),
                (-np.pi / 12, np.pi / 12)
            )


class LMNavDrillLiftD1(LMDrillLiftD1):
    """Nav version of DrillLift_D1"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.2, 0.2),
                (-np.pi / 12, np.pi / 12)
            )


class LMNavDrillLiftObstacleD1(LMDrillLiftObstacleD1):
    """Nav version of DrillLiftObstacle_D1"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.2, 0.2),
                (-np.pi / 12, np.pi / 12)
            )


class LMNavDrillPnP90D1(LMDrillPnP90D1):
    """Nav version of DrillPnP90_D1"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.2, 0.2),
                (-np.pi / 12, np.pi / 12)
            )


class LMNavDrillPnPCloserD1(LMDrillPnPCloserD1):
    """Nav version of DrillPnPCloser_D1"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.2, 0.2),
                (-np.pi / 12, np.pi / 12)
            )


class LMNavPickDrillFromHolderD1(LMPickDrillFromHolderD1):
    """Nav version of PickDrillFromHolder_D1"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.2, 0.2),
                (-np.pi / 12, np.pi / 12)
            )


class LMNavPushButtonD1(LMPushButtonD1):
    """Nav version of PushButton_D1"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.2, 0.2),
                (-np.pi / 12, np.pi / 12)
            )


class LMNavPushShelfForwardD1(LMPushShelfForwardD1):
    """Nav version of PushShelfForward_D1 with increased robot pose randomization"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Increased position randomization for robot pose
            RobotPoseRandomizer.set_pose(
                self,
                (-0.3, 0.3),  # Increased from (-0.15, 0.15)
                (-0.4, 0.4),  # Increased from (-0.2, 0.2)
                (-np.pi / 12, np.pi / 12)  # Decreased rotation
            )


class LMNavBoxTableToShelfStaticIndustrialD1(LMBoxTableToShelfStaticIndustrialD1):
    """Nav version of BoxTableToShelfStaticIndustrial_D1"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.2, 0.2),
                (-np.pi / 12, np.pi / 12)
            )


# ==================== D2 Variants (Even Larger Randomization) ====================

class LMBoxLiftD2(LMBoxLift):
    """BoxLift with 16x increased randomization ranges"""

    def _get_objects(self):
        objects = super()._get_objects()
        # D2: 16x increase (vs 8x in D1)
        self.table.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.72, 0.72]),  # 16x
                y_range=np.array([-0.92, 0.92]),  # 16x
                reference_pos=np.array([1.0, 0, 0]),
                rotation=np.array([np.pi * 0.68, np.pi * 1.32]),  # 16x range
            )
        )
        self.box.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.40, -0.26]),
                y_range=np.array([-0.48, 0.48]),
                rotation=np.array([np.pi / 2 * 0.76, np.pi / 2 * 1.24]),  # 12x
                reference=ReferenceConfig(obj=self.table),
            )
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.08, 0.08),
                (-0.08, 0.08),
                (-np.pi / 12, np.pi / 12)
            )


class LMBoxLiftFloorD2(LMBoxLiftFloor):
    """BoxLiftFloor with 16x increased randomization ranges"""

    def _get_objects(self):
        objects = super()._get_objects()
        # D2: 16x increase
        self.box.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.72, 0.72]),  # 16x
                y_range=np.array([-0.72, 0.72]),  # 16x
                reference_pos=np.array([1.0, 0, 0]),
                rotation=np.array([np.pi * 0.68, np.pi * 1.32]),  # 16x
                z_offset=0.05,
            )
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.08, 0.08),
                (-0.08, 0.08),
                (-np.pi / 12, np.pi / 12)
            )


# old D2 variants: flat scaling law
# class LMBoxLiftD2(LMBoxLift):
#     """BoxLift with 16x increased randomization ranges"""

#     def _get_objects(self):
#         objects = super()._get_objects()
#         # D2: 16x increase (vs 8x in D1)
#         self.table.update_cfg(
#             sampler_config=SamplingConfig(
#                 x_range=np.array([-0.32, 0.32]),  # 16x
#                 y_range=np.array([-0.32, 0.32]),  # 16x
#                 reference_pos=np.array([1.0, 0, 0]),
#                 rotation=np.array([np.pi * 0.68, np.pi * 1.32]),  # 16x range
#             )
#         )
#         self.box.update_cfg(
#             sampler_config=SamplingConfig(
#                 x_range=np.array([-0.57, -0.09]),  # 12x (centered at -0.33)
#                 y_range=np.array([-0.24, 0.24]),   # 12x
#                 rotation=np.array([np.pi / 2 * 0.76, np.pi / 2 * 1.24]),  # 12x
#                 reference=ReferenceConfig(obj=self.table),
#             )
#         )
#         return objects

#     def _reset_internal(self):
#         super()._reset_internal()
#         if not self.deterministic_reset:
#             RobotPoseRandomizer.set_pose(
#                 self,
#                 (-0.08, 0.08),
#                 (-0.08, 0.08),
#                 (-np.pi / 12, np.pi / 12)
#             )


# class LMBoxLiftFloorD2(LMBoxLiftFloor):
#     """BoxLiftFloor with 16x increased randomization ranges"""

#     def _get_objects(self):
#         objects = super()._get_objects()
#         # D2: 16x increase
#         self.box.update_cfg(
#             sampler_config=SamplingConfig(
#                 x_range=np.array([-0.32, 0.32]),  # 16x
#                 y_range=np.array([-0.32, 0.32]),  # 16x
#                 reference_pos=np.array([1.0, 0, 0]),
#                 rotation=np.array([np.pi * 0.68, np.pi * 1.32]),  # 16x
#                 z_offset=0.05,
#             )
#         )
#         return objects

#     def _reset_internal(self):
#         super()._reset_internal()
#         if not self.deterministic_reset:
#             RobotPoseRandomizer.set_pose(
#                 self,
#                 (-0.08, 0.08),
#                 (-0.08, 0.08),
#                 (-np.pi / 12, np.pi / 12)
#             )


class LMDrillLiftD2(LMDrillLift):
    """DrillLift with 16x increased randomization ranges"""

    def _get_objects(self):
        objects = super()._get_objects()
        # D2: 16x increase
        self.table.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.32, 0.32]),  # 16x
                y_range=np.array([-0.32, 0.32]),  # 16x
                reference_pos=np.array([1.0, 0, 0]),
                rotation=np.array([np.pi * 0.68, np.pi * 1.32]),  # 16x
            )
        )
        self.bottle.update_cfg(
            scale=1.0,
            mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
            sampler_config=SamplingConfig(
                x_range=np.array([-0.58, -0.10]),  # 12x (centered at -0.34)
                y_range=np.array([-0.60, 0.60]),   # 12x
                rotation=np.array([-np.pi * 0.60, np.pi * 0.60]),  # 12x
                reference=ReferenceConfig(obj=self.table),
            ),
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.08, 0.08),
                (-0.08, 0.08),
                (-np.pi / 12, np.pi / 12)
            )


class LMDrillPnP90D2(LMDrillPnP90):
    """DrillPnP90 with 12x increased randomization ranges"""

    def _get_objects(self):
        objects = super()._get_objects()
        # D2: 12x increase
        self.table_origin.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.32, 0.32]),  # 16x
                y_range=np.array([-0.32, 0.32]),  # 16x
                reference_pos=np.array([1.2, 0, 0]),
                rotation=np.array([-np.pi, -np.pi]),
            ),
        )
        self.table_target.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.32, 0.32]),  # 16x
                y_range=np.array([-0.32, 0.32]),  # 16x
                reference_pos=np.array([0, 1.2, 0]),
                rotation=np.array([-np.pi / 2, -np.pi / 2]),
            ),
        )
        self.bottle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.58, -0.10]),  # 12x
                y_range=np.array([-0.60, 0.60]),   # 12x
                rotation=np.array([-np.pi * 0.60, np.pi * 0.60]),  # 12x
                reference=ReferenceConfig(obj=self.table_origin),
            ),
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.08, 0.08),
                (-0.08, 0.08),
                (-np.pi / 12, np.pi / 12)
            )


# ==================== Nav D2 Variants ====================

class LMNavBoxLiftD2(LMBoxLiftD2):
    """Nav version of BoxLift_D2"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.2, 0.2),
                (-0.25, 0.25),
                (-np.pi / 9, np.pi / 9)
            )


class LMNavBoxLiftFloorD2(LMBoxLiftFloorD2):
    """Nav version of BoxLiftFloor_D2"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.2, 0.2),
                (-0.25, 0.25),
                (-np.pi / 9, np.pi / 9)
            )


class LMNavDrillLiftD2(LMDrillLiftD2):
    """Nav version of DrillLift_D2"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.2, 0.2),
                (-0.25, 0.25),
                (-np.pi / 9, np.pi / 9)
            )



class LMNavDrillPnP90D2(LMDrillPnP90D2):
    """Nav version of DrillPnP90_D2"""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.2, 0.2),
                (-0.25, 0.25),
                (-np.pi / 9, np.pi / 9)
            )


# ==================== D3 Variants (Extra-Large Randomization with Collision Avoidance) ====================

class LMDrillPnP90BiD3(LMDrillPnP90Bi):
    """DrillPnP90Bi with extra-large randomization and table collision avoidance.

    Larger than D2 (±0.40 vs ±0.32 table randomization).
    Uses rejection sampling to guarantee:
    - Tables do not collide (min 1.2m center-to-center distance)
    - table_origin stays in robot ego view (x > 0.6)
    Also adds slight table rotation variation and wider bottle/robot randomization.
    Target table side (left +y vs right -y) is randomized per episode.
    """

    # Minimum distance between table centers to prevent collision
    MIN_TABLE_DISTANCE = 1.2
    MAX_REJECTION_ATTEMPTS = 50

    def _get_objects(self):
        objects = super()._get_objects()
        # D3: ±0.40 table position randomization (20x base ±0.02)
        self.table_origin.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.40, 0.40]),
                y_range=np.array([-0.40, 0.40]),
                reference_pos=np.array([1.2, 0, 0]),
                rotation=np.array([-np.pi - 0.15, -np.pi + 0.15]),
            ),
        )
        self.table_target.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.40, 0.40]),
                y_range=np.array([-0.40, 0.40]),
                reference_pos=np.array([0, 1.2, 0]),
                rotation=np.array([-np.pi / 2 - 0.15, -np.pi / 2 + 0.15]),
            ),
        )
        # Drill randomization - constrained to stay on the table surface
        # Table is ~1.2m x 0.6m; keep drill well within bounds
        self.bottle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.45, -0.10]),
                y_range=np.array([-0.35, 0.35]),
                rotation=np.array([-np.pi * 0.50, np.pi * 0.50]),
                reference=ReferenceConfig(obj=self.table_origin),
            ),
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            # Rejection sampling: re-sample object positions until tables are valid
            origin_body_id = self.sim.model.body_name2id(self.table_origin.mj_obj.root_body)
            target_body_id = self.sim.model.body_name2id(self.table_target.mj_obj.root_body)

            # Randomize target table to the robot's left (+y) or right (-y) side
            target_on_right = self.rng.choice([False, True])

            def maybe_mirror_target():
                if target_on_right:
                    # Mirror y-position across the robot's sagittal plane
                    self.sim.model.body_pos[target_body_id][1] *= -1
                    # Mirror z-axis rotation: quat is [w, x, y, z]; negate z-component
                    self.sim.model.body_quat[target_body_id][3] *= -1

            # Apply mirror to the initial sample from super()._reset_internal()
            maybe_mirror_target()

            for _ in range(self.MAX_REJECTION_ATTEMPTS):
                origin_pos = self.sim.model.body_pos[origin_body_id][:2]
                target_pos = self.sim.model.body_pos[target_body_id][:2]
                distance = np.linalg.norm(origin_pos - target_pos)

                # Tables not colliding AND origin table in front of robot (ego view)
                if distance >= self.MIN_TABLE_DISTANCE and origin_pos[0] > 0.6:
                    break

                # Re-sample all object positions, then re-apply side mirror
                self.scene.reset()
                maybe_mirror_target()

            self.sim.forward()

            RobotPoseRandomizer.set_pose(
                self,
                (-0.12, 0.12),
                (-0.12, 0.12),
                (-np.pi / 9, np.pi / 9)
            )


class LMNavDrillPnP90BiD3(LMDrillPnP90BiD3):
    """Nav version of DrillPnP90Bi_D3 with larger robot pose randomization."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.25, 0.25),
                (-0.30, 0.30),
                (-np.pi / 6, np.pi / 6)
            )


# ==================== D1 Variants for DrillPnP90Bi ====================

class LMDrillPnP90BiD1(LMDrillPnP90Bi):
    """DrillPnP90Bi D1: 6x base table randomization (±0.12), no rejection sampling needed."""

    def _get_objects(self):
        objects = super()._get_objects()
        self.table_origin.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.12, 0.12]),
                y_range=np.array([-0.12, 0.12]),
                reference_pos=np.array([1.2, 0, 0]),
                rotation=np.array([-np.pi - 0.05, -np.pi + 0.05]),
            ),
        )
        self.table_target.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.12, 0.12]),
                y_range=np.array([-0.12, 0.12]),
                reference_pos=np.array([0, 1.2, 0]),
                rotation=np.array([-np.pi / 2 - 0.05, -np.pi / 2 + 0.05]),
            ),
        )
        # Drill randomization on the origin table surface (modest)
        self.bottle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.30, -0.18]),
                y_range=np.array([-0.12, 0.12]),
                rotation=np.array([-np.pi * 0.15, np.pi * 0.15]),
                reference=ReferenceConfig(obj=self.table_origin),
            ),
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


class LMNavDrillPnP90BiD1(LMDrillPnP90BiD1):
    """Nav variant of DrillPnP90Bi_D1 with wider robot pose randomization."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.15, 0.15),
                (-0.20, 0.20),
                (-np.pi / 6, np.pi / 6)
            )


class LMDrillLiftObstacleBiD1(LMDrillLiftObstacleBi):
    """DrillLiftObstacleBi with 6x increased randomization ranges (D1 variant)."""

    def _get_objects(self):
        objects = super()._get_objects()
        # 6x increase on table position
        self.table.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.12, 0.12]),
                y_range=np.array([-0.12, 0.12]),
                reference_pos=np.array([1.0, 0, 0]),
                rotation=np.array([np.pi * 0.88, np.pi * 1.12]),
            ),
        )
        # ~1/3 between original and 6x for bottle (drill) position/rotation
        self.bottle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.31, -0.21]),
                y_range=np.array([0.39, 0.55]),
                rotation=np.array([-np.pi * 0.13, np.pi * 0.13]),
                reference=ReferenceConfig(obj=self.table),
            ),
        )
        # 6x increase on obstacle position
        self.obstacle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.12, 0.12]),
                y_range=np.array([-0.60, 0.60]),
                reference_pos=np.array([0.35, 0, 0]),
                rotation=np.array([np.pi * 0.88, np.pi * 1.12]),
            ),
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(
                self,
                (-0.05, 0.05),
                (-0.05, 0.05),
                (-np.pi / 18, np.pi / 18)
            )


class LMNavDrillLiftObstacleBiD1(LMDrillLiftObstacleBiD1):
    """Nav variant of DrillLiftObstacleBiD1 with robot pose randomization."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))
