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

from dataclasses import replace

import numpy as np

from humanoidmimicgen.locomanipulation.envs.base import RobotPoseRandomizer
from humanoidmimicgen.locomanipulation.envs.locomanip import LMFactoryEnv
from humanoidmimicgen.locomanipulation.utils.dexmg_utils import DexMGConfigHelper
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


class LMPushCart(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0, handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.cart = SceneObject(
            ObjectConfig(
                name="cart",
                mjcf_path="objects/omniverse/locomanip/workshop_trolley_a01/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1.5, -1.5, 0]),
                    rotation=np.array([-np.pi, np.pi]),
                ),
                density=200,
            ),
        )
        self.target = SceneObject(
            ObjectConfig(
                name="target",
                mjcf_path="objects/omniverse/locomanip/target_zone_trigger/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-1.0, 1.0]),
                    y_range=np.array([-1.0, 1.0]),
                    reference_pos=np.array([1.5, 2.0, 0]),
                ),
            )
        )
        return [self.cart, self.target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(IsUpright(self.cart), IsClose(self.cart, self.target, 0.3, True))

    def _get_instruction(self) -> str:
        return "Push cart to the marked area."

    def get_object(self):
        return dict(
            cart=dict(obj_name=self.cart.mj_obj.root_body, obj_type="body"),
            target=dict(obj_name=self.target.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is make contact with the cart for pushing
        task.task_spec_0.subtask_1 = dict(
            object_ref="cart",
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
            object_ref="cart",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # the main part of the task is to push the cart to the target area
        task.task_spec_0.subtask_2 = dict(
            object_ref="target",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_2 = dict(
            object_ref="target",
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

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()

        # Constraint for concurrent contact with the cart between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent push to the target area between both arms
        task_constraint.constraint_2 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
        )

        return task_constraint.to_dict()


class LMPushCartForward(LMPushCart):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.cart.cfg = replace(
            self.cart.cfg,
            sampler_config=SamplingConfig(
                x_range=np.array([-0.1, 0.1]),
                y_range=np.array([-0.1, 0.1]),
                rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                reference_pos=np.array([1.5, 0, 0]),
            ),
        )

        self.target.cfg = replace(
            self.target.cfg,
            sampler_config=SamplingConfig(
                x_range=np.array([-0.1, 0.1]),
                y_range=np.array([-0.1, 0.1]),
                rotation=np.array([np.pi * 0.95, np.pi * 1.05]),
                reference_pos=np.array([4, 0, 0]),
            ),
        )
        return [self.cart, self.target]


class LMPushCartOverRamp(LMFactoryEnv):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0, handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.cart = SceneObject(
            ObjectConfig(
                name="cart",
                mjcf_path="objects/omniverse/locomanip/workshop_trolley_a01/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.05, 0.05]),
                    y_range=np.array([-0.05, 0.05]),
                    reference_pos=np.array([1.5, -1.5, 0.1]),
                    rotation=np.array([-np.pi, np.pi]),
                ),
                density=200,
            )
        )
        self.pallet = SceneObject(
            ObjectConfig(
                name="pallet",
                mjcf_path="objects/omniverse/locomanip/pallet_c1/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1.5, 2, 0]),
                ),
            )
        )
        self.target_left = SceneObject(
            ObjectConfig(
                name="target",
                mjcf_path="objects/omniverse/locomanip/target_zone_trigger/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.05, 0.05]),
                    y_range=np.array([-0.05, 0.05]),
                    reference=ReferenceConfig(self.pallet),
                ),
            )
        )
        self.ramp = SceneObject(
            ObjectConfig(
                name="ramp",
                mjcf_path="objects/omniverse/locomanip/dock_board_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-1.7, -1.7]),
                    y_range=np.array([-0.01, 0.01]),
                    z_offset=-0.06,
                    reference=ReferenceConfig(self.pallet, on_top=False),
                ),
            )
        )
        return [self.cart, self.pallet, self.target_left, self.ramp]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(IsUpright(self.cart), IsClose(self.cart, self.target_left, 0.3, True))

    def _get_instruction(self) -> str:
        return "Push cart to the marked area."


class LMPushShelf(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0, handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/mobile_shelving_cart/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1.5, -1.5, 0]),
                    rotation=np.array([-np.pi, np.pi]),
                ),
                density=200,
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
                    x_range=np.array([-1.0, 1.0]),
                    y_range=np.array([-1.0, 1.0]),
                    reference_pos=np.array([1.5, 2.0, 0]),
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
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is make contact with the shelf for pushing
        task.task_spec_0.subtask_1 = dict(
            object_ref="shelf",
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
            object_ref="shelf",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # the main part of the task is to push the shelf to the target area
        task.task_spec_0.subtask_2 = dict(
            object_ref="target",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_2 = dict(
            object_ref="target",
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

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()

        # Constraint for concurrent contact with the shelf between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent push to the target area between both arms
        task_constraint.constraint_2 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
        )

        return task_constraint.to_dict()


class LMPushShelfForward(LMPushShelf):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.shelf.cfg = replace(
            self.shelf.cfg,
            sampler_config=SamplingConfig(
                x_range=np.array([-0.1, 0.1]),
                y_range=np.array([-0.1, 0.1]),
                rotation=np.array([-np.pi * 0.98, -np.pi * 1.02]),
                reference_pos=np.array([1.5, 0, 0]),
            ),
        )
        self.target.cfg = replace(
            self.target.cfg,
            sampler_config=SamplingConfig(
                x_range=np.array([-0.1, 0.1]),
                y_range=np.array([-0.1, 0.1]),
                rotation=np.array([np.pi * 0.95, np.pi * 1.05]),
                reference_pos=np.array([4, 0, 0]),
            ),
        )
        return [self.shelf, self.box, self.target]


class LMNavPushShelfForward(LMPushShelfForward):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))
