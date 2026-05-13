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

import numpy as np

from robocasa.environments.locomanipulation.base import RobotPoseRandomizer
from robocasa.environments.locomanipulation.locomanip import LMFactoryEnv
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
    SceneScaleConfig,
)
from robocasa.utils.scene.scene import SceneObject
from robocasa.utils.scene.success_criteria import (
    IsJointQposInRange,
    IsRobotInRange,
    SuccessCriteria,
)


class LMWalkToTarget(LMFactoryEnv):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(2, 0))

    def _get_objects(self) -> list[SceneObject]:
        self.target = SceneObject(
            ObjectConfig(
                name="target",
                mjcf_path="objects/omniverse/locomanip/target_zone/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.5, 0.5]),
                    y_range=np.array([-0.5, 0.5]),
                    reference_pos=np.array([1, 1, 0]),
                ),
            )
        )
        return [self.target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return IsRobotInRange(self.target, 0.5, True)

    def _get_instruction(self) -> str:
        return "Move to the marked area."


class LMPushButton(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(2, 1), vertical_scale=0.8)

    def _get_objects(self) -> list[SceneObject]:
        self.control_box = SceneObject(
            ObjectConfig(
                name="control_box",
                mjcf_path="objects/omniverse/locomanip/control_box/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-1, -0.5]),
                    y_range=np.array([-1, -0.5]),
                    reference_pos=np.array([1, 1, 0]),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                ),
            )
        )
        self.control_panel = SceneObject(
            ObjectConfig(
                name="control_panel",
                mjcf_path="objects/omniverse/locomanip/control_conveyorbelt_a08/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.3, -0.3]),
                    y_range=np.array([-0.42, -0.38]),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                    z_offset=1,
                    reference=ReferenceConfig(obj=self.control_box, on_top=False),
                ),
            )
        )
        return [self.control_box, self.control_panel]

    def _get_success_criteria(self) -> SuccessCriteria:
        return IsJointQposInRange(self.control_panel, 0, -1, -0.01)

    def _get_instruction(self) -> str:
        return "Press red button."

    def get_object(self):
        return dict(
            control_panel_button=dict(obj_name="control_panel_button", obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="control_panel_button",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()


class LMPushButtonStatic(LMPushButton):
    def _get_objects(self) -> list[SceneObject]:
        self.control_box = SceneObject(
            ObjectConfig(
                name="control_box",
                mjcf_path="objects/omniverse/locomanip/control_box/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-1.2, -1.2]),
                    y_range=np.array([-1, -1]),
                    reference_pos=np.array([1, 1, 0]),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                ),
            )
        )
        self.control_panel = SceneObject(
            ObjectConfig(
                name="control_panel",
                mjcf_path="objects/omniverse/locomanip/control_conveyorbelt_a08/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.3, -0.3]),
                    y_range=np.array([-0.38, -0.38]),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                    z_offset=1,
                    reference=ReferenceConfig(
                        obj=self.control_box,
                        on_top=False,
                    ),
                ),
            )
        )
        return [self.control_box, self.control_panel]


class LMPushButtonLow(LMPushButton):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(2, 1), vertical_scale=0.5)


class LMPushButtonHigh(LMPushButton):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(2, 1), vertical_scale=1.1)


class LMNavPushButton(LMPushButton):
    def _get_objects(self) -> list[SceneObject]:
        self.control_box = SceneObject(
            ObjectConfig(
                name="control_box",
                mjcf_path="objects/omniverse/locomanip/control_box/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([0, 0]),
                    y_range=np.array([0, 0]),
                    reference_pos=np.array([0.6, 0.3, 0]),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                ),
            )
        )
        self.control_panel = SceneObject(
            ObjectConfig(
                name="control_panel",
                mjcf_path="objects/omniverse/locomanip/control_conveyorbelt_a08/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.3, -0.3]),
                    y_range=np.array([-0.4, -0.4]),
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
            RobotPoseRandomizer.set_pose(self, (-0.4, -0.2), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))
