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
    IsGrasped,
    IsGripperFar,
    IsInContact,
    IsInContactWithRobot,
    IsPositionInRange,
    IsStatic,
    IsUpright,
    NotCriteria,
    SuccessCriteria,
)


class LMBottleLift(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=2.0)

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.0, 0, 0]),
                    rotation=np.array([np.pi * 0.98, np.pi * 1.02]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.28, -0.24]),
                    y_range=np.array([-0.05, 0.05]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(obj=self.table),
                ),
            )
        )
        return [self.table, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsInContact(self.bottle, self.table)),
            IsPositionInRange(self.bottle, 2, self.table.mj_obj.top_offset[2] + 0.1),
            IsGrasped(self.bottle),
        )

    def _get_instruction(self) -> str:
        return "Pick up the bottle."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # each arm's subtask is to pick up the object - just check task success
        return dict(obj_lifted=int(self._check_success()))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # right arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal="obj_lifted",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # dummy subtask for left arm
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

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()
        return task_constraint.to_dict()


class LMBottleLiftStatic(LMBottleLift):
    SCENE_SCALE = SceneScaleConfig(planar_scale=0.6)


class LMBottleLiftMiddleShelf(LMBottleLift):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 2.0))

    def _get_objects(self) -> list[SceneObject]:
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.0, 1.0, 0]),
                    rotation=np.array([-np.pi / 2 * 0.98, -np.pi / 2 * 1.02]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.24, -0.26]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(obj=self.shelf, spawn_id=1),
                ),
            )
        )
        return [self.shelf, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsInContact(self.bottle, self.shelf)),
            IsGrasped(self.bottle),
        )

    def _get_instruction(self) -> str:
        return "Pick up the bottle."


class LMBottleLiftLowShelf(LMBottleLiftMiddleShelf):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 2.0))

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.bottle = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.24, -0.26]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(obj=self.shelf, spawn_id=0),
                ),
            )
        )
        return [self.shelf, self.bottle]


class LMBottleLiftLowShelfStatic(LMBottleLiftLowShelf):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0.2, 0.6))


class LMBottleShelfLowToHighPnP(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 2.0))

    def _get_objects(self) -> list[SceneObject]:
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.0, 1.0, 0]),
                    rotation=np.array([-np.pi / 2 * 0.98, -np.pi / 2 * 1.02]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.24, -0.26]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(obj=self.shelf, spawn_id=0),
                ),
            )
        )
        return [self.shelf, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsInContact(self.bottle, self.shelf), IsPositionInRange(self.bottle, 2, 0.5, 1.0)
        )

    def _get_instruction(self) -> str:
        return "Pick up the bottle and put it on a higher shelf."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            shelf=dict(obj_name=self.shelf.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()

        # # right arm's subtasks is to pick up the object and then place it on the target table
        # task.task_spec_0.subtask_1 = dict(
        #     object_ref="bottle",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )
        # task.task_spec_0.subtask_2 = dict(
        #     object_ref="shelf",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )

        # # NOTE: version with 4 subtasks

        # # right arm squat down
        # task.task_spec_0.subtask_1 = dict(
        #     object_ref="bottle",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )
        # # right arm pick
        # task.task_spec_0.subtask_2 = dict(
        #     object_ref="bottle",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )
        # # right arm squat up
        # task.task_spec_0.subtask_3 = dict(
        #     object_ref="shelf",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )
        # # right arm place
        # task.task_spec_0.subtask_4 = dict(
        #     object_ref="shelf",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )

        # # dummy subtask for left arm
        # task.task_spec_1.subtask_1 = dict(
        #     object_ref=None,
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )

        # NOTE: version with 4 subtasks, both arms

        # right arm squat down
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # right arm pick
        task.task_spec_0.subtask_2 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # right arm squat up
        task.task_spec_0.subtask_3 = dict(
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
        # right arm place
        task.task_spec_0.subtask_4 = dict(
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

        # left arm squat down
        task.task_spec_1.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # left arm pick
        task.task_spec_1.subtask_2 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # left arm squat up
        task.task_spec_1.subtask_3 = dict(
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
        # left arm place
        task.task_spec_1.subtask_4 = dict(
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

        # # NOTE: version with 2 subtasks, both arms

        # # right arm's subtasks is to pick up the object and then place it on the target table
        # task.task_spec_0.subtask_1 = dict(
        #     object_ref="bottle",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )
        # task.task_spec_0.subtask_2 = dict(
        #     object_ref="shelf",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )

        # # left arm's subtasks are mirrored
        # task.task_spec_1.subtask_1 = dict(
        #     object_ref="bottle",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )
        # task.task_spec_1.subtask_2 = dict(
        #     object_ref="shelf",
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )

        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()
        return task_constraint.to_dict()


class LMBottleShelfLowToHighPnPStatic(LMBottleShelfLowToHighPnP):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0.2, 0.6))


class LMBottleLiftFloor(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)

    def _get_objects(self) -> list[SceneObject]:
        self.bottle = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference_pos=np.array([1.0, 0, 0]),
                    z_offset=0.05,
                ),
            )
        )
        return [self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsPositionInRange(self.bottle, 2, max_val=0.2)),
            IsGrasped(self.bottle),
        )

    def _get_instruction(self) -> str:
        return "Lift up the bottle."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # each arm's subtask is to pick up the object - just check task success
        return dict(obj_lifted=int(self._check_success()))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # right arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal="obj_lifted",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # dummy subtask for left arm
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

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()
        return task_constraint.to_dict()


class LMBottleLiftFloorStatic(LMBottleLiftFloor):
    SCENE_SCALE = SceneScaleConfig(planar_scale=0.4)


class LMBoxLift(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=2.0)

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.0, 0, 0]),
                    rotation=np.array([np.pi * 0.98, np.pi * 1.02]),
                ),
            )
        )
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1/model.xml",
                mjcf_path="objects/omniverse/locomanip/cardbox_a1_gripridge/model.xml",
                static=False,
                scale=0.7,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.26, -0.24]),
                    y_range=np.array([-0.02, 0.02]),
                    rotation=np.array([np.pi / 2 * 0.98, np.pi / 2 * 1.02]),
                    reference=ReferenceConfig(obj=self.table),
                ),
            )
        )
        return [self.table, self.box]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsInContact(self.box, self.table)),
            IsPositionInRange(self.box, 2, self.table.mj_obj.top_offset[2] + 0.1),
            NotCriteria(IsGripperFar(self.box)),
        )

    def _get_instruction(self) -> str:
        return "Lift up the box."

    def get_object(self):
        return dict(
            obj=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # each arm's subtask is to pick up the object - just check task success
        return dict(obj_lifted=int(self._check_success()))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="obj_lifted",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="obj_lifted",
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

        # Constraint for concurrent lift between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )

        return task_constraint.to_dict()


class LMBoxLiftStatic(LMBoxLift):
    SCENE_SCALE = SceneScaleConfig(planar_scale=0.65)


class LMBoxLiftFloor(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)

    def _get_objects(self) -> list[SceneObject]:
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/longbox_a08/model.xml",
                static=False,
                scale=1.8,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    rotation=np.array([np.pi * 0.98, np.pi * 1.02]),
                    reference_pos=np.array([1.0, 0, 0]),
                    z_offset=0.05,
                ),
            )
        )
        return [self.box]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsPositionInRange(self.box, 2, max_val=0.1)),
            IsUpright(self.box),
            # IsGrasped(self.box),
        )

    def _get_instruction(self) -> str:
        return "Lift up the box."

    def get_object(self):
        return dict(
            obj=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # each arm's subtask is to pick up the object - just check task success
        return dict(obj_lifted=int(self._check_success()))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="obj_lifted",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="obj_lifted",
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

        # Constraint for concurrent lift between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )

        return task_constraint.to_dict()


class LMBoxLiftFloorStatic(LMBoxLiftFloor):
    SCENE_SCALE = SceneScaleConfig(planar_scale=0.33)


class LMBoxFloorPnPTable(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=2.0)

    def _get_objects(self) -> list[SceneObject]:
        # Table in front of robot
        self.table = SceneObject(
            ObjectConfig(
                name="table_target",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.0, 0, 0]),
                    rotation=np.array([np.pi * 0.98, np.pi * 1.02]),
                ),
            )
        )
        # Box on floor
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/longbox_a08/model.xml",
                static=False,
                scale=1.8,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    rotation=np.array([np.pi * 0.98, np.pi * 1.02]),
                    reference_pos=np.array([0.5, 0, 0]),
                    z_offset=0.05,
                ),
            )
        )
        return [self.table, self.box]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsInContact(self.box, self.table),
            IsUpright(self.box),
        )

    def _get_instruction(self) -> str:
        return "Pick up the box from the floor and place it on the table."

    def get_object(self):
        return dict(
            obj=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            table_target=dict(obj_name=self.table.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # right arm picks and places
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="table_target",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # left arm picks and places
        task.task_spec_1.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_2 = dict(
            object_ref="table_target",
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
        task_constraint = DexMGConfigHelper.AttrDict()
        # Concurrent lift
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )
        # Concurrent place
        task_constraint.constraint_2 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
        )
        return task_constraint.to_dict()


class LMBottlePnP(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(1, 1), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.table_target = SceneObject(
            ObjectConfig(
                name="table_target",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.2, 0.8, 0]),
                    rotation=np.array([np.pi, np.pi]),
                ),
            )
        )
        self.table_origin = SceneObject(
            ObjectConfig(
                name="table_origin",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.2, -0.8, 0]),
                    rotation=np.array([np.pi, np.pi]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.4, -0.35]),
                    y_range=np.array([-0.1, 0.1]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(self.table_origin),
                ),
            )
        )
        return [self.table_origin, self.table_target, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(IsInContact(self.bottle, self.table_target), IsUpright(self.bottle))

    def _get_instruction(self) -> str:
        return "Pick up the bottle from one table and place it on the other."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            table_target=dict(obj_name=self.table_target.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # right arm's subtasks is to pick up the object and then place it on the target table
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="table_target",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # dummy subtask for left arm
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

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()
        return task_constraint.to_dict()


class LMBoxPnP(LMBottlePnP, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(1, 1), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles/model.xml",
                mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles_vertical/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                # scale=0.7,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.35, -0.3]),
                    y_range=np.array([-0.1, 0.1]),
                    # rotation=np.array([np.pi * 0.9, np.pi * 1.1]),
                    rotation=np.array([np.pi * (0.4), np.pi * (0.6)]),
                    # rotation=np.array([np.pi * (1.4), np.pi * (1.6)]),
                    reference=ReferenceConfig(self.table_origin),
                ),
            )
        )
        return [self.table_origin, self.table_target, self.box]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsInContact(self.box, self.table_target),
            IsUpright(self.box, threshold=0.95),
            IsStatic(self.box),
            IsGripperFar(self.box, threshold=0.1),
        )

    def _get_instruction(self) -> str:
        return "Pick up the box from one table and place it on the other."

    def get_object(self):
        return dict(
            box=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            table_target=dict(obj_name=self.table_target.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="box",
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
            object_ref="box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # each arm's subtask is to place the object on the target table
        task.task_spec_0.subtask_2 = dict(
            object_ref="table_target",
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
            object_ref="table_target",
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

        # Constraint for concurrent lift between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent place between both arms
        task_constraint.constraint_2 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
        )

        return task_constraint.to_dict()


class LMToteBoxPnP(LMBoxPnP):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(1, 1), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                # mjcf_path="objects/omniverse/locomanip/tote_f01/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01_handles/model.xml",
                mjcf_path="objects/omniverse/locomanip/tote_f01_handles_vertical/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles/model.xml",
                static=False,
                # scale=0.7,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.35, -0.3]),
                    y_range=np.array([-0.1, 0.1]),
                    # rotation=np.array([np.pi * 0.9, np.pi * 1.1]),
                    rotation=np.array([np.pi * (0.4), np.pi * (0.6)]),
                    # rotation=np.array([np.pi * (1.4), np.pi * (1.6)]),
                    reference=ReferenceConfig(self.table_origin),
                ),
            )
        )
        return [self.table_origin, self.table_target, self.box]


class LMBoxPalletsPnP(LMFactoryEnv):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 2.0), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.pallet_origin = SceneObject(
            ObjectConfig(
                name="pallet_left",
                mjcf_path="objects/omniverse/locomanip/pallet_b1/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1, 1, 0]),
                    rotation=np.array([-np.pi / 2 * 0.98, -np.pi / 2 * 1.02]),
                ),
            )
        )
        self.pallet_target = SceneObject(
            ObjectConfig(
                name="pallet_right",
                mjcf_path="objects/omniverse/locomanip/pallet_b1/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1, -1, 0]),
                    rotation=np.array([-np.pi / 2 * 0.98, -np.pi / 2 * 1.02]),
                ),
            )
        )
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles/model.xml",
                mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles_vertical/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                # scale=0.7,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.35, -0.3]),
                    # rotation=np.array([np.pi * 0.95, np.pi * 1.05]),
                    rotation=np.array([np.pi * (0.45), np.pi * (0.55)]),
                    # rotation=np.array([np.pi * (1.45), np.pi * (1.55)]),
                    reference=ReferenceConfig(self.pallet_origin),
                ),
            )
        )
        return [self.box, self.pallet_origin, self.pallet_target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsInContact(self.box, self.pallet_target),
            IsUpright(self.box, threshold=0.95),
            IsStatic(self.box),
            IsGripperFar(self.box, threshold=0.1),
        )

    def _get_instruction(self) -> str:
        return "Pick up the box from one pallet and place it on the other."


class LMBoxTableToShelfPnP(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 2), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1, -1, 0]),
                    rotation=np.array([-np.pi / 2 * 0.9, -np.pi / 2 * 1.1]),
                ),
            )
        )
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1, 1, 0]),
                    rotation=np.array([-np.pi / 2 * 0.98, -np.pi / 2 * 1.02]),
                ),
            )
        )
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles_vertical/model.xml",
                mjcf_path="objects/omniverse/locomanip/tote_f01_handles_vertical/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                # scale=0.7,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.35, -0.3]),
                    # rotation=np.array([np.pi * 0.9, np.pi * 1.1]),
                    rotation=np.array([np.pi * (0.4), np.pi * (0.6)]),
                    # rotation=np.array([np.pi * (1.4), np.pi * (1.6)]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        return [self.box, self.shelf, self.table]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsInContact(self.box, self.shelf),
            IsUpright(self.box, threshold=0.95),
            IsStatic(self.box),
            IsGripperFar(self.box, threshold=0.1),
        )

    def _get_instruction(self) -> str:
        return "Pick up the box from the table and place it on the shelf."

    def get_object(self):
        return dict(
            box=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            shelf=dict(obj_name=self.shelf.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="box",
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
            object_ref="box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # each arm's subtask is to place the object on the target table
        task.task_spec_0.subtask_2 = dict(
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
        task.task_spec_1.subtask_2 = dict(
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
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()

        # Constraint for concurrent lift between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent place between both arms
        task_constraint.constraint_2 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
        )

        return task_constraint.to_dict()


class LMBoxPalletToShelfPnP(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 2), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1, -1, 0]),
                    rotation=np.array([-np.pi / 2 * 0.9, -np.pi / 2 * 1.1]),
                ),
            )
        )
        self.pallet = SceneObject(
            ObjectConfig(
                name="pallet",
                mjcf_path="objects/omniverse/locomanip/pallet_b1/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1, 1, 0]),
                    rotation=np.array([-np.pi / 2 * 0.98, -np.pi / 2 * 1.02]),
                ),
            )
        )
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles_vertical/model.xml",
                mjcf_path="objects/omniverse/locomanip/tote_f01_handles_vertical/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                # scale=0.7,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.35, -0.3]),
                    # rotation=np.array([np.pi * 0.95, np.pi * 1.05]),
                    rotation=np.array([np.pi * (0.45), np.pi * (0.55)]),
                    # rotation=np.array([np.pi * (1.45), np.pi * (1.55)]),
                    reference=ReferenceConfig(self.pallet),
                ),
            )
        )
        return [self.box, self.shelf, self.pallet]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsInContact(self.box, self.shelf),
            IsUpright(self.box, threshold=0.95),
            IsStatic(self.box),
            IsGripperFar(self.box, threshold=0.1),
        )

    def _get_instruction(self) -> str:
        return "Pick up the box from the pallet and place it on the shelf."

    def get_object(self):
        return dict(
            box=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            pallet=dict(obj_name=self.pallet.mj_obj.root_body, obj_type="body"),
            shelf=dict(obj_name=self.shelf.mj_obj.root_body, obj_type="body")
        )

    def get_subtask_term_signals(self):
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="box",
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
            object_ref="box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # each arm's subtask is to place the object on the target table
        task.task_spec_0.subtask_2 = dict(
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
        task.task_spec_1.subtask_2 = dict(
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
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()

        # Constraint for concurrent lift between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent place between both arms
        task_constraint.constraint_2 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
        )

        return task_constraint.to_dict()


class LMBoxTableToShelfRampPnP(LMBoxTableToShelfPnP):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 3.6), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.ramp = SceneObject(
            ObjectConfig(
                name="ramp",
                mjcf_path="objects/omniverse/locomanip/dock_board_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.05, 0.05]),
                    y_range=np.array([1.1, 1.1]),
                    rotation=np.array([-np.pi / 2 * 0.9, -np.pi / 2 * 1.1]),
                    z_offset=0,
                    reference=ReferenceConfig(self.shelf, on_top=False),
                ),
            )
        )
        return [self.box, self.table, self.shelf, self.ramp]


class LMBoxesTableToShelfPnP(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 2.0), handedness=SceneHandedness.LEFT)

    def _get_objects(self) -> list[SceneObject]:
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([1, -1, 0]),
                    rotation=np.array([-np.pi / 2 * 0.9, -np.pi / 2 * 1.1]),
                ),
            )
        )
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1, 0.8, 0]),
                    rotation=np.array([-np.pi / 2 * 0.98, -np.pi / 2 * 1.02]),
                ),
            )
        )
        self.box_1 = SceneObject(
            ObjectConfig(
                name="obj_1",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles/model.xml",
                mjcf_path="objects/omniverse/locomanip/cardbox_a1_handles_vertical/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01_handles_vertical/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                # scale=0.7,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.3, -0.3]),
                    y_range=np.array([-0.35, -0.3]),
                    # rotation=np.array([np.pi * 0.9, np.pi * 1.1]),
                    rotation=np.array([np.pi * (0.4), np.pi * (0.6)]),
                    # rotation=np.array([np.pi * (1.4), np.pi * (1.6)]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        self.box_2 = SceneObject(
            ObjectConfig(
                name="obj_2",
                # mjcf_path="objects/omniverse/locomanip/tote_f01/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01_gripridge/model.xml",
                # mjcf_path="objects/omniverse/locomanip/tote_f01_handles/model.xml",
                mjcf_path="objects/omniverse/locomanip/tote_f01_handles_vertical/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles/model.xml",
                # mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                # scale=0.7,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([0.3, 0.3]),
                    y_range=np.array([-0.35, -0.3]),
                    # rotation=np.array([np.pi * 0.9, np.pi * 1.1]),
                    rotation=np.array([np.pi * (0.4), np.pi * (0.6)]),
                    # rotation=np.array([np.pi * (1.4), np.pi * (1.6)]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        return [self.box_1, self.box_2, self.shelf, self.table]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.box_1, threshold=0.95),
            IsStatic(self.box_1),
            IsGripperFar(self.box_1, threshold=0.1),
            IsUpright(self.box_2, threshold=0.95),
            IsStatic(self.box_2),
            IsGripperFar(self.box_2, threshold=0.1),
            IsInContact(self.box_1, self.shelf),
            IsInContact(self.box_2, self.shelf),
        )

    def _get_instruction(self) -> str:
        return "Pick up boxes from the table and place them on the shelf."

    def get_object(self):
        return dict(
            box_1=dict(obj_name=self.box_1.mj_obj.root_body, obj_type="body"),
            box_2=dict(obj_name=self.box_2.mj_obj.root_body, obj_type="body"),
            shelf=dict(obj_name=self.shelf.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="box_2",
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
            object_ref="box_2",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # each arm's subtask is to place the object on the target table
        task.task_spec_0.subtask_2 = dict(
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
        task.task_spec_1.subtask_2 = dict(
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
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_3 = dict(
            object_ref="box_1",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_3 = dict(
            object_ref="box_1",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # each arm's subtask is to place the object on the target table
        task.task_spec_0.subtask_4 = dict(
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
        task.task_spec_1.subtask_4 = dict(
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
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()

        # Constraint for concurrent lift between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent place between both arms
        task_constraint.constraint_2 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent lift between both arms
        task_constraint.constraint_3 = dict(
            subtasks=[("task_spec_0", "subtask_3"), ("task_spec_1", "subtask_3")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent place between both arms
        task_constraint.constraint_4 = dict(
            subtasks=[("task_spec_0", "subtask_4"), ("task_spec_1", "subtask_4")],
            constraint_type="temporal_concurrent",
        )

        return task_constraint.to_dict()


class LMDrillLift(LMBottleLift):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.bottle.update_cfg(
            scale=1.0,
            mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
            sampler_config=SamplingConfig(
                x_range=np.array([-0.28, -0.24]),
                y_range=np.array([-0.05, 0.05]),
                rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                reference=ReferenceConfig(obj=self.table),
            ),
        )
        return [self.table, self.bottle]

    def _get_instruction(self) -> str:
        return "Pick up the drill."


class LMDrillLiftBi(LMDrillLift):

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # right arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal="obj_lifted",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # mirror it for left arm
        task.task_spec_1.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal="obj_lifted",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()


class LMDrillLiftStatic(LMDrillLift):
    SCENE_SCALE = SceneScaleConfig(planar_scale=0.6)


class LMDrillLiftObstacle(LMDrillLift):
    SCENE_SCALE = SceneScaleConfig(planar_scale=2.0)

    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        self.bottle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.28, -0.24]),
                y_range=np.array([0.45, 0.50]),
                rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                reference=ReferenceConfig(obj=self.table),
            ),
        )

        self.obstacle = SceneObject(
            ObjectConfig(
                name="obstacle",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([0.35, 0, 0]),
                    rotation=np.array([np.pi * 0.98, np.pi * 1.02]),
                ),
                scale=0.7
            )
        )
        return [*objects, self.obstacle]


class LMNavDrillLiftObstacle(LMDrillLiftObstacle):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMDrillLiftObstacleBi(LMDrillLiftBi):
    SCENE_SCALE = SceneScaleConfig(planar_scale=2.0)

    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        self.bottle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.28, -0.24]),
                y_range=np.array([0.45, 0.50]),
                rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                reference=ReferenceConfig(obj=self.table),
            ),
        )

        self.obstacle = SceneObject(
            ObjectConfig(
                name="obstacle",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.1, 0.1]),
                    reference_pos=np.array([0.35, 0, 0]),
                    rotation=np.array([np.pi * 0.98, np.pi * 1.02]),
                ),
                scale=0.7
            )
        )
        return [*objects, self.obstacle]


class LMNavDrillLiftObstacleBi(LMDrillLiftObstacleBi):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMDrillPnP(LMBottlePnP):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.bottle.update_cfg(
            scale=1.0,
            mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
            sampler_config=SamplingConfig(
                x_range=np.array([-0.28, -0.24]),
                y_range=np.array([-0.05, 0.05]),
                rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                reference=ReferenceConfig(obj=self.table_origin),
            ),
        )
        return [self.table_origin, self.table_target, self.bottle]

    def _get_instruction(self) -> str:
        return "Pick up the drill from one table and place it on the other."

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsStatic(self.bottle),
            IsUpright(self.bottle, threshold=0.95),
            IsGripperFar(self.bottle, threshold=0.1),
            IsInContact(self.bottle, self.table_target),
        )


class LMDrillPnPCloser(LMDrillPnP):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.bottle.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.28, -0.24]),
                y_range=np.array([0.4, 0.45]),
                rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                reference=ReferenceConfig(obj=self.table_origin),
            ),
        )
        return [self.table_origin, self.table_target, self.bottle]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (0.3, 0.3), (-0.05, 0.05), (0.05, 0.05))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # right arm's subtasks is to pick up the object and then place it on the target table
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="table_target",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # # dummy subtask for left arm
        # task.task_spec_1.subtask_1 = dict(
        #     object_ref=None,
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )
        # left arm's subtasks are dummy subtasks relevant to the same objects
        task.task_spec_1.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_2 = dict(
            object_ref="table_target",
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


class LMNavDrillPnPCloser(LMDrillPnPCloser):

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (0.3 - 0.15, 0.3 + 0.15), (-0.05, 0.05), (0.05 - np.pi / 6, 0.05 + np.pi / 6))

class LMDrillPnP180(LMDrillPnP):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table_origin.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([1.2, 0, 0]),
                rotation=np.array([-np.pi, -np.pi]),
            ),
        )
        self.table_target.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([-1.2, 0, 0]),
                rotation=np.array([0, 0]),
            ),
        )
        return [self.table_origin, self.table_target, self.bottle]


class LMDrillPnP90(LMDrillPnP):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table_origin.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([1.2, 0, 0]),
                rotation=np.array([-np.pi, -np.pi]),
            ),
        )
        self.table_target.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([0, 1.2, 0]),
                rotation=np.array([-np.pi / 2, -np.pi / 2]),
            ),
        )
        return [self.table_origin, self.table_target, self.bottle]


class LMDrillPnP90Bi(LMDrillPnP90):

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # right arm's subtasks is to pick up the object and then place it on the target table
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="table_target",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # mirror for left arm
        task.task_spec_1.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_2 = dict(
            object_ref="table_target",
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


class LMDrillPnP90Low(LMDrillPnP90):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table_origin.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([1.2, 0, -0.05]),
                rotation=np.array([-np.pi, -np.pi]),
            ),
        )
        self.table_target.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([0, 1.2, -0.05]),
                rotation=np.array([-np.pi / 2, -np.pi / 2]),
            ),
        )
        return [self.table_origin, self.table_target, self.bottle]


class LMDrillPnP90LabTable(LMDrillPnP90):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table_origin.update_cfg(
            mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([1.2, 0, 0]),
                rotation=np.array([-np.pi, -np.pi]),
            ),
        )
        self.table_target.update_cfg(
            mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([0, 1.2, 0]),
                rotation=np.array([-np.pi / 2, -np.pi / 2]),
            ),
        )
        return [self.table_origin, self.table_target, self.bottle]


class LMDrillLiftLowShelf(LMBottleLiftLowShelf):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.bottle.update_cfg(
            scale=1.0,
            mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.24, -0.26]),
                rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                reference=ReferenceConfig(obj=self.shelf, spawn_id=0),
            ),
        )
        return [self.shelf, self.bottle]

    def _get_instruction(self) -> str:
        return "Pick up the drill."


class LMDrillLiftLowShelfStatic(LMDrillLiftLowShelf):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0.2, 0.6))


class LMDrillShelfLowToHighPnP(LMBottleShelfLowToHighPnP):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.bottle.update_cfg(
            scale=1.0,
            mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.24, -0.26]),
                rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                reference=ReferenceConfig(obj=self.shelf, spawn_id=0),
            ),
        )
        return [self.shelf, self.bottle]

    def _get_instruction(self) -> str:
        return "Pick up the drill and put it on a higher shelf."

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsStatic(self.bottle),
            IsUpright(self.bottle, threshold=0.95),
            IsGripperFar(self.bottle, threshold=0.1),
            IsPositionInRange(self.bottle, 2, 0.5, 1.0),
            IsInContact(self.bottle, self.shelf),
        )


class LMDrillShelfLowToHighPnPStatic(LMDrillShelfLowToHighPnP):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0.2, 0.6))


class LMPlaceDrillInHolder(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(1, 1), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.0, 0.8, 0]),
                    rotation=np.array([np.pi * 0.98, np.pi * 1.02]),
                ),
            )
        )
        self.drill = SceneObject(
            ObjectConfig(
                name="drill",
                mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
                static=False,
                friction=(1, 0.005, 0.0001),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.28, -0.24]),
                    y_range=np.array([-0.1, 0.1]),
                    rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                    reference=ReferenceConfig(obj=self.table),
                ),
            )
        )
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                friction=(0.0001, 0.005, 0.0001),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.2, -0.8, 0]),
                    rotation=np.array([np.pi * 0.98, np.pi * 1.02]),
                ),
            )
        )
        self.holder = SceneObject(
            ObjectConfig(
                name="holder",
                mjcf_path="objects/omniverse/locomanip/powerdrill_holder/model.xml",
                static=True,
                friction=(0, 0, 0),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.15, -0.15]),
                    y_range=np.array([-0.02, 0.02]),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                    z_offset=1.125,
                    reference=ReferenceConfig(self.shelf, on_top=False),
                ),
            )
        )
        return [self.table, self.drill, self.shelf, self.holder]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsInContact(self.drill, self.holder),
            IsStatic(self.drill),
            IsUpright(self.drill, threshold=0.8),
            IsGripperFar(self.drill, threshold=0.1),
        )

    def _get_instruction(self) -> str:
        return "Pick up the drill from the table and place it in the holder."

    def get_object(self):
        return dict(
            drill=dict(obj_name=self.drill.mj_obj.root_body, obj_type="body"),
            shelf=dict(obj_name=self.shelf.mj_obj.root_body, obj_type="body"),
            holder=dict(obj_name=self.holder.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.table.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # right arm's subtasks is to pick up the drill and then place drill in holder
        task.task_spec_0.subtask_1 = dict(
            object_ref="drill",
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="holder",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # # dummy subtask for left arm
        # task.task_spec_1.subtask_1 = dict(
        #     object_ref=None,
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )
        task.task_spec_1.subtask_1 = dict(
            object_ref="drill",
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_2 = dict(
            object_ref="holder",
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
        return task_constraint.to_dict()


class LMPlaceDrillInHolder90(LMPlaceDrillInHolder):
    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        self.shelf.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([0, -1.2, 0]),
                rotation=np.array([-np.pi / 2, -np.pi / 2]),
            ),
        )
        self.holder.update_cfg(
            sampler_config=SamplingConfig(
                y_range=np.array([0.15, 0.15]),
                x_range=np.array([-0.02, 0.02]),
                rotation=np.zeros(2),
                z_offset=1.125,
                reference=ReferenceConfig(self.shelf, on_top=False),
            ),
        )
        return objects


class LMPickDrillFromHolder(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(1, 1), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.35, 0, 0]),
                    rotation=np.array([np.pi / 2 * 0.98, np.pi / 2 * 1.02]),
                ),
            )
        )
        self.holder = SceneObject(
            ObjectConfig(
                name="holder",
                mjcf_path="objects/omniverse/locomanip/powerdrill_holder/model.xml",
                static=True,
                friction=(0.0001, 0.005, 0.0001),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.75, -0.75]),
                    y_range=np.array([-0.02, 0.02]),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                    z_offset=-0.1,
                    reference=ReferenceConfig(self.shelf, spawn_id=1),
                ),
            )
        )
        self.drill = SceneObject(
            ObjectConfig(
                name="drill",
                mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
                static=False,
                friction=(1, 0.005, 0.0001),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, -0.1]),
                    y_range=np.array([-0.055, -0.055]),
                    z_offset=-0.145,
                    reference=ReferenceConfig(obj=self.holder, on_top=False),
                ),
            )
        )
        return [self.drill, self.shelf, self.holder]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsInContact(self.drill, self.holder)),
            IsGrasped(self.drill),
        )

    def _get_instruction(self) -> str:
        return "Pick up the drill from the holder."

    def get_object(self):
        return dict(
            drill=dict(obj_name=self.drill.mj_obj.root_body, obj_type="body"),
            shelf=dict(obj_name=self.shelf.mj_obj.root_body, obj_type="body"),
            holder=dict(obj_name=self.holder.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # NOTE: instead of writing subtask term signals we will manually annotate skill segments
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # right arm's subtasks is to pick up the drill from the holder
        task.task_spec_0.subtask_1 = dict(
            object_ref="drill",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # # dummy subtask for left arm
        # task.task_spec_1.subtask_1 = dict(
        #     object_ref=None,
        #     subtask_term_signal=None,
        #     subtask_term_offset_range=None,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     action_noise=0.05,
        #     num_interpolation_steps=5,
        #     num_fixed_steps=0,
        #     apply_noise_during_interpolation=False,
        # )
        # left arm subtask (no-op, but we will copy from source demos)
        task.task_spec_1.subtask_1 = dict(
            object_ref="drill",
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
        return task_constraint.to_dict()


class LMBoxTableToShelfPnPAligned(LMBoxTableToShelfPnP):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, np.zeros(2), np.zeros(2), (np.pi / 2, np.pi / 2))


class LMBoxTableToShelfPnPAlignedSmall(LMBoxTableToShelfPnPAligned):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 1.0), handedness=SceneHandedness.RIGHT)


class LMBoxTableToShelfPnPAlignedNew(LMBoxTableToShelfPnPAligned):
    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.35, -0.3]),
                    # rotation=np.array([np.pi * 0.9, np.pi * 1.1]),
                    rotation=np.array([np.pi * (0.4), np.pi * (0.6)]),
                    # rotation=np.array([np.pi * (1.4), np.pi * (1.6)]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        objects[0] = self.box
        return objects

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # add special first subtask to prepare arms for approaching and lifting the box
        task.task_spec_0.subtask_1 = dict(
            object_ref="box",
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
            object_ref="box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_2 = dict(
            object_ref="box",
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
            object_ref="box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # each arm's subtask is to place the object on the target table
        task.task_spec_0.subtask_3 = dict(
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
        task.task_spec_1.subtask_3 = dict(
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
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()

        # Constraint for concurrent lift between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent place between both arms
        task_constraint.constraint_2 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
        )

        # Constraint for concurrent place between both arms
        task_constraint.constraint_3 = dict(
            subtasks=[("task_spec_0", "subtask_3"), ("task_spec_1", "subtask_3")],
            constraint_type="temporal_concurrent",
        )

        return task_constraint.to_dict()


class LMBoxTableToShelfPnPAlignedNewSmall(LMBoxTableToShelfPnPAlignedNew):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 1.4), handedness=SceneHandedness.RIGHT)


class LMBoxesTableToShelfPnPAligned(LMBoxesTableToShelfPnP):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, np.zeros(2), np.zeros(2), (np.pi / 2, np.pi / 2))


class LMBoxPalletToShelfPnPAligned(LMBoxPalletToShelfPnP):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, np.zeros(2), np.zeros(2), (np.pi / 2, np.pi / 2))


class LMBoxPalletToShelfPnPAlignedSmall(LMBoxPalletToShelfPnPAligned):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 1.3), handedness=SceneHandedness.RIGHT)


class LMBoxPalletToShelfPnPAlignedSimple(LMBoxPalletToShelfPnPAligned):
    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        self.box.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.1, 0.1]),
                y_range=np.array([-0.4, -0.4]),
                rotation=np.array([np.pi * 0.45, np.pi * 0.55]),
                reference=ReferenceConfig(self.pallet),
            ),
        )
        pallet_height = self.pallet.mj_obj.top_offset[2]
        self.pallet.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.1, 0.1]),
                y_range=np.array([-0.1, 0.1]),
                reference_pos=np.array([1, 1, pallet_height]),
                rotation=np.array([-np.pi / 2 * 0.98, -np.pi / 2 * 1.02]),
            ),
        )
        return objects


class LMBoxPalletToShelfPnPAlignedSimpleSmall(LMBoxPalletToShelfPnPAlignedSimple):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 1.3), handedness=SceneHandedness.RIGHT)


class LMBoxPalletToShelfPnPAlignedNew(LMBoxPalletToShelfPnPAligned):
    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.1, 0.1]),
                    y_range=np.array([-0.35, -0.3]),
                    # rotation=np.array([np.pi * 0.95, np.pi * 1.05]),
                    rotation=np.array([np.pi * (0.45), np.pi * (0.55)]),
                    # rotation=np.array([np.pi * (1.45), np.pi * (1.55)]),
                    reference=ReferenceConfig(self.pallet),
                ),
            )
        )
        objects[0] = self.box
        return objects


class LMBoxPalletToShelfPnPAlignedNewSmall(LMBoxPalletToShelfPnPAlignedNew):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 1.4), handedness=SceneHandedness.RIGHT)


class LMDrillLiftLowShelfAligned(LMDrillLiftLowShelf):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, np.zeros(2), np.zeros(2), (np.pi / 2, np.pi / 2))


class LMDrillLiftLowShelfAlignedStatic(LMDrillLiftLowShelfAligned):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0.2, 0.6))


class LMDrillShelfLowToHighPnPAligned(LMDrillShelfLowToHighPnP):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, np.zeros(2), np.zeros(2), (np.pi / 2, np.pi / 2))


class LMDrillShelfLowToHighPnPAlignedStatic(LMDrillShelfLowToHighPnPAligned):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0.2, 0.6))


# Nav variants with randomized robot base pose
# Parent classes have default pose (0, 0, 0), so we center randomization around that
class LMNavBoxLiftFloor(LMBoxLiftFloor):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavBoxLift(LMBoxLift):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavDrillLift(LMDrillLift):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavDrillLiftBi(LMDrillLiftBi):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavDrillPnP(LMDrillPnP):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavDrillShelfLowToHighPnPAligned(LMDrillShelfLowToHighPnPAligned):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's pose (0, 0, π/2)
            RobotPoseRandomizer.set_pose(
                self, (-0.15, 0.15), (-0.2, 0.2), (np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6)
            )


class LMMildNavPickDrillFromHolder(LMPickDrillFromHolder):
    """Mild robot pose randomization variant: small base pose variation without requiring navigation."""

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.05, 0.05), (-0.06, 0.06), (-np.pi / 18, np.pi / 18))


class LMNavPickDrillFromHolder(LMPickDrillFromHolder):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMWalkPickDrillFromHolder(LMPickDrillFromHolder):
    """Walk variant: robot starts further away from the drill holder, requiring walking to reach it.
    Default robot is at (0,0,0) and shelf at ~(1.35,0,0). This pushes the robot back 0.5-0.7m."""
    SCENE_SCALE = SceneScaleConfig(planar_scale=(2, 1), handedness=SceneHandedness.RIGHT)

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Push robot back from default (0,0,0) so it needs to walk to the shelf
            RobotPoseRandomizer.set_pose(
                self,
                (-0.7, -0.5),          # x: robot placed 0.5-0.7m further back
                (-0.1, 0.1),           # y: small lateral variation
                (-np.pi / 18, np.pi / 18)  # yaw: small rotation
            )


class LMNavWalkPickDrillFromHolder(LMWalkPickDrillFromHolder):
    """Nav variant of Walk: robot starts away with broader position/yaw randomization."""

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Broader randomization around the walk-back position
            RobotPoseRandomizer.set_pose(
                self,
                (-1.0, -0.3),          # x: wide range, always behind default
                (-0.4, 0.4),           # y: larger lateral variation
                (-np.pi / 6, np.pi / 6)  # yaw: larger rotation variation
            )


class LMNavDrillPnP90(LMDrillPnP90):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))

            # Same _reset_internal logic for LMBoxTableToShelfPnPAligned


class LMNavDrillPnP90Bi(LMDrillPnP90Bi):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavBoxTableToShelfPnPAligned(LMBoxTableToShelfPnPAligned):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavBoxTableToShelfPnPAlignedSmall(LMNavBoxTableToShelfPnPAligned):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 1.4), handedness=SceneHandedness.RIGHT)


class LMNavBoxPalletToShelfPnPAligned(LMBoxPalletToShelfPnPAligned):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavBoxPalletToShelfPnPAlignedSmall(LMNavBoxPalletToShelfPnPAligned):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 1.4), handedness=SceneHandedness.RIGHT)


class LMNavBoxPalletToShelfPnPAlignedSimpleSmall(LMBoxPalletToShelfPnPAlignedSimpleSmall):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))

class LMNavBoxTableToShelfPnPAlignedNew(LMBoxTableToShelfPnPAlignedNew):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavBoxTableToShelfPnPAlignedNewSmall(LMNavBoxTableToShelfPnPAlignedNew):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 1.4), handedness=SceneHandedness.RIGHT)

class LMNavBoxPalletToShelfPnPAlignedSimple(LMBoxPalletToShelfPnPAlignedSimple):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavBoxPalletToShelfPnPAlignedNew(LMBoxPalletToShelfPnPAlignedNew):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavBoxPalletToShelfPnPAlignedNewSmall(LMNavBoxPalletToShelfPnPAlignedNew):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(0, 1.4), handedness=SceneHandedness.RIGHT)


class LMPlaceDrillInHolder90Aligned(LMPlaceDrillInHolder90):
    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        # Centered placement of the table
        self.table.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.02, 0.02]),
                y_range=np.array([-0.02, 0.02]),
                reference_pos=np.array([1.2, 0, 0]),
                rotation=np.array([-np.pi, -np.pi]),
            ),
        )
        return objects


class LMPickDrillFromHolderHigh(LMPickDrillFromHolder):
    """LMPickDrillFromHolder with the drill/holder positioned 0.08m higher."""

    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        # Raise holder by 0.08m: original z_offset=-0.1, new z_offset=-0.02
        from dataclasses import replace as dc_replace
        new_sampler = dc_replace(self.holder.cfg.sampler_config, z_offset=-0.02)
        self.holder.update_cfg(sampler_config=new_sampler)
        return objects

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsInContact(self.drill, self.holder)),
            IsGrasped(self.drill),
        )

    def _get_instruction(self) -> str:
        return "Pick up the drill from the high holder."


class LMPickDrillFromHolderStanding(LMPickDrillFromHolder):
    """LMPickDrillFromHolder with drill at arm level (middle shelf) so robot doesn't bend."""

    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        from dataclasses import replace as dc_replace

        # Raise holder on lowest shelf (spawn_id=1): base z_offset=-0.1 -> +0.05 (~0.8m)
        new_holder_sampler = dc_replace(
            self.holder.cfg.sampler_config,
            z_offset=0.05,
        )
        self.holder.update_cfg(sampler_config=new_holder_sampler)

        # Shift drill lower in holder so head sits correctly (base z_offset=-0.145)
        new_drill_sampler = dc_replace(
            self.drill.cfg.sampler_config,
            z_offset=-0.24,
        )
        self.drill.update_cfg(sampler_config=new_drill_sampler)

        # Scale drill z-axis 1.5x taller for easier grasping at standing height
        self.drill.update_cfg(scale=(1.0, 1.0, 1.5))

        return objects

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsInContact(self.drill, self.holder)),
            IsGrasped(self.drill),
        )

    def _get_instruction(self) -> str:
        return "Pick up the drill from the holder at standing height."


class LMPickDrillFromHolderStandingEasy(LMPickDrillFromHolderStanding):
    """Standing variant with narrower drill (0.8x X) shifted closer to robot."""
    def _get_objects(self):
        objects = super()._get_objects()
        from dataclasses import replace as dc_replace
        # Narrow drill in X (0.8x), keep Y, keep Z=1.5
        self.drill.update_cfg(scale=(0.8, 1.0, 1.5))
        # Shift drill closer to robot (x_range: -0.1 -> -0.12)
        new_sampler = dc_replace(
            self.drill.cfg.sampler_config,
            x_range=np.array([-0.14, -0.14]),
        )
        self.drill.update_cfg(sampler_config=new_sampler)
        return objects


class LMNavPickDrillFromHolderStandingEasy(LMPickDrillFromHolderStandingEasy):
    """Nav variant of PickDrillFromHolderStandingEasy with robot pose randomization."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMPickDrillFromHolderStandingEasyFar(LMPickDrillFromHolderStandingEasy):
    """StandingEasy with robot starting ~0.5m further from drill/holder."""
    SCENE_SCALE = SceneScaleConfig(planar_scale=(1, 1))

    def _reset_internal(self):
        super()._reset_internal()
        RobotPoseRandomizer.set_pose(self, (-0.5, -0.5), (0, 0), (0, 0))


class LMNavPickDrillFromHolderStandingEasyFar(LMPickDrillFromHolderStandingEasyFar):
    """Nav variant of PickDrillFromHolderStandingEasyFar with robot pose randomization."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.65, -0.35), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMPickDrillFromHolderStandingEasyFarD1(LMPickDrillFromHolderStandingEasyFar):
    """D1 variant: shelf (holder-support) gets larger randomization.

    Base is ±2cm on each axis; D1 widens to ±15cm so the drill pose varies
    meaningfully while staying in ego-camera view from the robot's ~0.5m
    standoff.
    """
    def _get_objects(self):
        objects = super()._get_objects()
        from dataclasses import replace as dc_replace
        new_shelf_sampler = dc_replace(
            self.shelf.cfg.sampler_config,
            x_range=np.array([-0.15, 0.15]),
            y_range=np.array([-0.15, 0.15]),
        )
        self.shelf.update_cfg(sampler_config=new_shelf_sampler)
        return objects


class LMNavPickDrillFromHolderStandingEasyFarD1(LMPickDrillFromHolderStandingEasyFarD1):
    """Nav variant of D1 with robot pose randomization."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.65, -0.35), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMPickDrillFromHolderStandingEasyFarD2(LMPickDrillFromHolderStandingEasyFar):
    """D2 variant: shelf (holder-support) gets even larger randomization than D1.

    D1 widened to ±15cm; D2 goes to ±25cm so the drill pose varies more
    aggressively. At robot standoff ~0.5m this keeps the holder visible in
    ego view but pushes the task harder.
    """
    def _get_objects(self):
        objects = super()._get_objects()
        from dataclasses import replace as dc_replace
        new_shelf_sampler = dc_replace(
            self.shelf.cfg.sampler_config,
            x_range=np.array([-0.25, 0.25]),
            y_range=np.array([-0.25, 0.25]),
        )
        self.shelf.update_cfg(sampler_config=new_shelf_sampler)
        return objects


class LMNavPickDrillFromHolderStandingEasyFarD2(LMPickDrillFromHolderStandingEasyFarD2):
    """Nav variant of D2 with wider robot pose randomization than D1 nav."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.80, -0.20), (-0.30, 0.30), (-np.pi / 4, np.pi / 4))


class LMPickDrillFromHolderStandingEasyFarD3(LMPickDrillFromHolderStandingEasyFar):
    """D3 variant: shelf randomization wider than D2, especially laterally.

    D2 used ±25cm both axes. D3 widens depth to ±45cm and lateral (y) to
    ±90cm — shelf/drill can land well outside reach of a stationary robot,
    so the policy must learn to navigate to it.
    """
    def _get_objects(self):
        objects = super()._get_objects()
        from dataclasses import replace as dc_replace
        new_shelf_sampler = dc_replace(
            self.shelf.cfg.sampler_config,
            x_range=np.array([-0.45, 0.45]),
            y_range=np.array([-0.90, 0.90]),
        )
        self.shelf.update_cfg(sampler_config=new_shelf_sampler)
        return objects


class LMNavPickDrillFromHolderStandingEasyFarD3(LMPickDrillFromHolderStandingEasyFarD3):
    """Nav variant of D3 with widened robot pose randomization than D2 nav, esp. lateral."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.95, -0.05), (-0.90, 0.90), (-np.pi / 3, np.pi / 3))


class LMPickDrillFromHolderStandingEasyFarD4(LMPickDrillFromHolderStandingEasyFar):
    """D4 variant: shelf randomization even wider than D3.

    D3 had x ±45cm, y ±90cm. D4 bumps to x ±60cm, y ±130cm so the shelf can
    land fully outside the robot's standing-still reach in both axes.
    """
    def _get_objects(self):
        objects = super()._get_objects()
        from dataclasses import replace as dc_replace
        new_shelf_sampler = dc_replace(
            self.shelf.cfg.sampler_config,
            x_range=np.array([-0.60, 0.60]),
            y_range=np.array([-1.30, 1.30]),
        )
        self.shelf.update_cfg(sampler_config=new_shelf_sampler)
        return objects


class LMNavPickDrillFromHolderStandingEasyFarD4(LMPickDrillFromHolderStandingEasyFarD4):
    """Nav variant of D4 with widest robot pose randomization."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-1.10, 0.00), (-1.30, 1.30), (-np.pi / 2.5, np.pi / 2.5))


class LMNavPickDrillFromHolderStanding(LMPickDrillFromHolderStanding):
    """Nav variant of PickDrillFromHolderStanding with robot pose randomization."""
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMDrillLiftObstacleDT(LMDrillLift):
    """Drill lift with obstacle shelf layout matching real-world setup.

    Layout:
    - White lab_shelf (scaled to ~91cm tall) holds the drill
    - Brown/dark shelf_a12 (obstacle, ~170cm tall) stands next to/in front of white shelf
    - Dark grey wall behind both shelves
    - Robot is initialized behind the obstacle shelf
    """
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)

    def _get_objects(self) -> list[SceneObject]:
        # Obstacle shelf: custom digital twin (69.5cm wide, 38.5cm deep, 170.5cm tall)
        # Black metal poles + brown wooden shelves at 7cm, 45cm, 85cm
        # Already at real-world scale, no scaling needed
        self.obstacle = SceneObject(
            ObjectConfig(
                name="obstacle",
                mjcf_path="objects/omniverse/locomanip/obstacle_shelf/model.xml",
                static=True,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.55, 0.15, 0]),
                    rotation=np.array([np.pi / 2 * 0.98, np.pi / 2 * 1.02]),
                ),
            )
        )

        # White target shelf: 62cm long (x) x 24cm wide (y) x 91cm tall (z)
        # Native lab_shelf: 62cm x 29.2cm x 181cm
        # Scale: (1.0, 0.823, 0.503)
        # Positioned BEHIND obstacle shelf, open side faces robot
        self.white_shelf = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_shelf/model.xml",
                static=True,
                scale=(1.0, 0.823, 0.503),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.47, 0.15, 0]),
                    rotation=np.array([np.pi / 2 * 0.98, np.pi / 2 * 1.02]),
                ),
            )
        )
        self.table = self.white_shelf

        # Green cup on the white shelf top level
        # ±4.5cm randomization box for canister position
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/green_cup/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.045, 0.045]),
                    y_range=np.array([-0.225, -0.135]),
                    rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                    reference=ReferenceConfig(obj=self.white_shelf),
                ),
            )
        )

        return [self.white_shelf, self.bottle, self.obstacle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsInContact(self.bottle, self.white_shelf)),
            IsInContactWithRobot(self.bottle),
        )

    def _get_instruction(self) -> str:
        return "Pick up the drill from the white shelf, reaching over the obstacle shelf."

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # ±10cm position, ±15deg rotation
            RobotPoseRandomizer.set_pose(self, (-0.10, 0.10), (-0.10, 0.10), (-np.pi / 12, np.pi / 12))

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.white_shelf.mj_obj.root_body, obj_type="body"),
            obstacle=dict(obj_name=self.obstacle.mj_obj.root_body, obj_type="body"),
        )


class LMNavDrillLiftObstacleDT(LMDrillLiftObstacleDT):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMDrillLiftObstacleDTBi(LMDrillLiftObstacleDT, LMDrillLiftBi):
    pass


class LMDrillLiftDTBi(LMDrillLiftObstacleDTBi):
    """No-obstacle variant of LMDrillLiftObstacleDTBi. Inherits everything but drops the obstacle shelf from the scene."""

    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        return [obj for obj in objects if obj is not self.obstacle]

    def _get_instruction(self) -> str:
        return "Pick up the drill from the white shelf."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.white_shelf.mj_obj.root_body, obj_type="body"),
        )


class LMDrillLiftObstacleDTpt75(LMDrillLiftObstacleDT):
    SCENE_SCALE = SceneScaleConfig(planar_scale=0.75)


class LMNavDrillLiftObstacleDTpt75(LMDrillLiftObstacleDTpt75):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMCannisterLiftBiDT(LMDrillLiftBi):
    """Bimanual lift of the green canister off the white shelf — no obstacle."""
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)

    def _get_objects(self) -> list[SceneObject]:
        self.white_shelf = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_shelf/model.xml",
                static=True,
                scale=(1.0, 0.823, 0.503),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.47, 0.15, 0]),
                    rotation=np.array([np.pi / 2 * 0.98, np.pi / 2 * 1.02]),
                ),
            )
        )
        self.table = self.white_shelf

        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/green_cup/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.045, 0.045]),
                    y_range=np.array([-0.225, -0.135]),
                    rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                    reference=ReferenceConfig(obj=self.white_shelf),
                ),
            )
        )
        return [self.white_shelf, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsInContact(self.bottle, self.white_shelf)),
            IsPositionInRange(self.bottle, 2, self.white_shelf.mj_obj.top_offset[2] + 0.1),
            IsGrasped(self.bottle),
        )

    def _get_instruction(self) -> str:
        return "Pick up the canister from the white shelf with both hands."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.white_shelf.mj_obj.root_body, obj_type="body"),
        )


class LMNavDrillLiftObstacleDTBi(LMDrillLiftObstacleDTBi):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))
