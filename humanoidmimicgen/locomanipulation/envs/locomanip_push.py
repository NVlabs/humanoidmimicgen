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

from humanoidmimicgen.locomanipulation.envs.locomanip import LMFactoryEnv
from humanoidmimicgen.locomanipulation.utils.hmg_config import HMGConfigHelper
from humanoidmimicgen.locomanipulation.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
    SceneHandedness,
    SceneScaleConfig,
)
from humanoidmimicgen.locomanipulation.utils.scene.scene import SceneObject
from humanoidmimicgen.locomanipulation.utils.scene.success_criteria import (
    AllCriteria,
    IsClose,
    IsInContact,
    IsUpright,
    SuccessCriteria,
)


def _subtask(object_ref):
    return dict(
        object_ref=object_ref,
        subtask_term_signal=None,
        subtask_term_offset_range=None,
        selection_strategy="random",
        selection_strategy_kwargs=None,
        action_noise=0.05,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )


class LMPushShelfForward(LMFactoryEnv, HMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0, handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/mobile_shelving_cart/model.xml",
                static=False,
                density=200,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1.5, 0, 0]),
                    rotation=np.array([-np.pi * 0.98, -np.pi * 1.02]),
                ),
            )
        )
        self.box = SceneObject(
            ObjectConfig(
                name="box",
                mjcf_path="objects/omniverse/locomanip/cardbox_a1/model.xml",
                static=False,
                scale=0.7,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.05, 0.05]),
                    y_range=np.array([-0.05, 0.05]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference=ReferenceConfig(obj=self.shelf, on_top=True),
                ),
            )
        )
        self.target = SceneObject(
            ObjectConfig(
                name="target",
                mjcf_path="objects/omniverse/locomanip/target_zone_trigger/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    rotation=np.array([np.pi * 0.95, np.pi * 1.05]),
                    reference_pos=np.array([4, 0, 0]),
                ),
            )
        )
        return [self.shelf, self.box, self.target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.shelf),
            IsClose(self.shelf, self.target, 0.3, True),
            IsInContact(self.shelf, self.box),
        )

    def _get_instruction(self) -> str:
        return "Push the shelf with the box to the marked area."

    def get_object(self):
        return dict(
            shelf=dict(obj_name=self.shelf.mj_obj.root_body, obj_type="body"),
            box=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            target=dict(obj_name=self.target.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        return dict()

    @staticmethod
    def task_config():
        task = HMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = _subtask("shelf")
        task.task_spec_1.subtask_1 = _subtask("shelf")
        task.task_spec_0.subtask_2 = _subtask("target")
        task.task_spec_1.subtask_2 = _subtask("target")
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        task_constraint = HMGConfigHelper.AttrDict()
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )
        task_constraint.constraint_2 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
        )
        return task_constraint.to_dict()
