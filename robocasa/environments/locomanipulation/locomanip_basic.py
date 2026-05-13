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
from robocasa.environments.locomanipulation.locomanip import LMSimpleEnv
from robocasa.models.scenes.factory_arena import FactoryArena
from robocasa.models.scenes.lab_arena import LabArena, LabArenaPlane
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
    SceneScaleConfig,
)
from robocasa.utils.scene.scene import SceneObject
from robocasa.utils.scene.success_criteria import (
    AllCriteria,
    AnyCriteria,
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
from robocasa.utils.visuals_utls import Gradient, randomize_materials_rgba


class LMPickBottle(LMSimpleEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)

    TABLE_GRADIENT: Gradient = Gradient(
        np.array([0.68, 0.34, 0.07, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    )
    LIFT_OFFSET = 0.2

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 0, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
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
                    x_range=np.array([-0.08, 0.04]),
                    y_range=np.array([-0.08, 0.08]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference_pos=np.array([0.4, 0, self.table.mj_obj.top_offset[2]]),
                ),
            )
        )
        return [self.table, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsGrasped(self.bottle, "right"),
            IsPositionInRange(
                self.bottle, 2, self.table.mj_obj.top_offset[2] + self.LIFT_OFFSET, 10
            ),
        )

    def _get_instruction(self) -> str:
        return "Pick up the bottle."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        signals["grasp_bottle"] = int(
            self._check_grasp(self.robots[0].gripper["right"], self.bottle.mj_obj.contact_geoms)
        )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
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

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            self._randomize_table_rgba()

    def _randomize_table_rgba(self):
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=self.table.mj_obj, gradient=self.TABLE_GRADIENT, linear=True
        )


class LMPickBottleHigh(LMPickBottle):
    TABLE_OFFSET = 0.1

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 0, self.TABLE_OFFSET]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
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
                    x_range=np.array([-0.08, 0.04]),
                    y_range=np.array([-0.08, 0.08]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference_pos=np.array(
                        [0.4, 0, self.TABLE_OFFSET + self.table.mj_obj.top_offset[2]]
                    ),
                    reference=ReferenceConfig(obj=self.table),
                ),
            )
        )
        return [self.table, self.bottle]


class LMNavPickBottle(LMPickBottle):
    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.3, -0.16), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMPickBottleGround(LMPickBottle):
    def _get_objects(self) -> list[SceneObject]:
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.08, 0.04]),
                    y_range=np.array([-0.08, 0.08]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference_pos=np.array(
                        [0.4, 0, 0.075]
                    ),  # Base position on ground (z=0.075 is bottle radius)
                ),
            )
        )
        return [self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsGrasped(self.bottle, "right"),
            IsPositionInRange(self.bottle, 2, self.LIFT_OFFSET, 10),
        )

    def _randomize_table_rgba(self):
        pass


class LMPnPBottle(LMPickBottle):
    LIFT_OFFSET = 0.15

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table_target = SceneObject(
            ObjectConfig(
                name="table_target",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 1.2, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        return [self.table, self.table_target, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.bottle, symmetric=True), IsInContact(self.bottle, self.table_target)
        )

    def _get_instruction(self) -> str:
        return "Pick up the bottle and place it on the other table."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            target_table=dict(obj_name=self.table_target.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        return {}

    def _randomize_table_rgba(self):
        for table in [self.table_target, self.table]:
            randomize_materials_rgba(
                rng=self.rng, mjcf_obj=table.mj_obj, gradient=self.TABLE_GRADIENT, linear=True
            )


class LMPickMultipleBottles(LMPickBottle):
    BOTTLE_COLOURS = [(0.3, 0.7, 0.8, 1.0), (0.8, 0.4, 0.3, 1.0)]
    BOTTLES_COUNT = 2
    Y_OFFSET_STEP = 0.1

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 0, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )

        self.bottles = []
        offsets = np.arange(self.BOTTLES_COUNT) - (self.BOTTLES_COUNT - 1) / 2.0
        for i in range(self.BOTTLES_COUNT):
            reference_pos = np.array([0.4, 0, self.table.mj_obj.top_offset[2]])
            reference_pos += np.array([0, self.Y_OFFSET_STEP * offsets[i], 0])
            bottle = SceneObject(
                ObjectConfig(
                    name=f"bottle_{i}",
                    mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                    static=False,
                    scale=0.6,
                    sampler_config=SamplingConfig(
                        x_range=np.array([-0.08, 0.04]),
                        y_range=np.array([-0.04, 0.04]),
                        # rotation=np.array([-np.pi, np.pi]),
                        rotation=np.array([0.0, 0.0]),
                        reference_pos=reference_pos,
                    ),
                    rgba=self.BOTTLE_COLOURS[i % len(self.BOTTLE_COLOURS)],
                )
            )
            self.bottles.append(bottle)
        return [self.table, *self.bottles]

    def _get_success_criteria(self) -> SuccessCriteria:
        criteria = []
        for bottle in self.bottles:
            criteria.append(AnyCriteria(IsGrasped(bottle, "right"), IsGrasped(bottle, "left")))
            criteria.append(
                IsPositionInRange(bottle, 2, self.table.mj_obj.top_offset[2] + self.LIFT_OFFSET, 10)
            )
        return AllCriteria(*criteria)

    def _get_instruction(self) -> str:
        return "Pick up bottles."

    def get_object(self):
        return {
            bottle.mj_obj.name: dict(obj_name=bottle.mj_obj.root_body, obj_type="body")
            for bottle in self.bottles
        }

    def get_subtask_term_signals(self):
        return {
            f"grasp_{bottle.mj_obj.name}": int(
                self._check_grasp(self.robots[0].gripper["right"], bottle.mj_obj)
                or self._check_grasp(self.robots[0].gripper["left"], bottle.mj_obj)
            )
            for bottle in self.bottles
        }

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        for i in range(LMPickMultipleBottles.BOTTLES_COUNT):
            subtask = dict(
                object_ref=f"bottle_{i}",
                subtask_term_signal=None,
                subtask_term_offset_range=None,
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
            setattr(task.task_spec_0, f"subtask_{i+1}", subtask)
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


class LMPnPMultipleBottles(LMPickMultipleBottles):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table_target = SceneObject(
            ObjectConfig(
                name="table_target",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 1.2, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        return [self.table, self.table_target, *self.bottles]

    def _get_success_criteria(self) -> SuccessCriteria:
        criteria = [
            AllCriteria(IsInContact(bottle, self.table_target), IsUpright(bottle, symmetric=True))
            for bottle in self.bottles
        ]
        return AllCriteria(*criteria)

    def _get_instruction(self) -> str:
        return "Pick up bottles from one table and place it on the other."

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        for i in range(LMPnPMultipleBottles.BOTTLES_COUNT):
            bottle_name = f"bottle_{i}"
            subtask = dict(
                object_ref=bottle_name,
                subtask_term_signal=f"{bottle_name}_off_table",
                subtask_term_offset_range=None,
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
            setattr(task.task_spec_0, f"subtask_{i+1}", subtask)
        # Next subtask for placing on target table
        task.task_spec_0.subtask_3 = dict(
            object_ref="target_table",
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

    def get_subtask_term_signals(self):
        signals = dict()
        for bottle in self.bottles:
            obj_z = self.sim.data.body_xpos[self.obj_body_id(self.bottle.mj_obj.name)][2]
            target_table_pos = self.sim.data.body_xpos[
                self.obj_body_id(self.table_target.mj_obj.name)
            ]
            target_table_z = target_table_pos[2] + self.table_target.mj_obj.top_offset[2]
            signals[f"{bottle.mj_obj.name}_off_table"] = int(
                obj_z - target_table_z > self.LIFT_OFFSET
            )
        return signals

    def _randomize_table_rgba(self):
        for table in [self.table_target, self.table]:
            randomize_materials_rgba(
                rng=self.rng, mjcf_obj=table.mj_obj, gradient=self.TABLE_GRADIENT, linear=True
            )


class LMPickBottleShelf(LMPickBottle):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/lab_shelf/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                    reference_pos=np.array([0.8 + 0.1, -0.4, 0]),  # to the right
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
                    x_range=np.array([-0.14, -0.06]),
                    y_range=np.array([-0.08, 0.08]),
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(self.shelf, spawn_id=2),
                ),
            )
        )
        return [self.shelf, self.bottle]


class LMNavPickBottleShelf(LMPickBottleShelf):
    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.1, -0.05), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsGrasped(self.bottle, "right"),
            NotCriteria(IsInContact(self.bottle, self.shelf)),
        )


class LMNavPickBottleShelfRight(LMNavPickBottleShelf):
    """Same as LMNavPickBottleShelf, but bottle is on the right side of the shelf"""

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.bottle.cfg.sampler_config.y_range = np.array([0.1, 0.2])


class LMBoxTableToShelf(LMSimpleEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)
    MUJOCO_ARENA_CLS = LabArenaPlane

    TABLE_CORNER = (-0.371296, 0.581634)
    BOX_OFFSET = (0.23, -0.56)
    BOX_CENTER = (TABLE_CORNER[0] + BOX_OFFSET[0], TABLE_CORNER[1] + BOX_OFFSET[1])

    def _get_objects(self) -> list[SceneObject]:
        self.shelf_cabinet = SceneObject(
            ObjectConfig(
                name="shelf_cabinet",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_cabinet/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.85 - 0.3, -0.58 - 0.15, 0]),
                ),
            )
        )

        self.shelf_board_0 = SceneObject(
            ObjectConfig(
                name="shelf_board_0",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=5),
                ),
            )
        )
        self.shelf_board_1 = SceneObject(
            ObjectConfig(
                name="shelf_board_1",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=14),
                ),
            )
        )
        self.shelf_board_2 = SceneObject(
            ObjectConfig(
                name="shelf_board_2",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=0),
                ),
            )
        )
        self.shelf_board_3 = SceneObject(
            ObjectConfig(
                name="shelf_board_3",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=24),
                ),
            )
        )
        self.shelf_boards = [
            self.shelf_board_0,
            self.shelf_board_1,
            self.shelf_board_2,
            self.shelf_board_3,
        ]

        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.85 + 0.37, 0, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )

        self.box = SceneObject(
            ObjectConfig(
                name="box",
                mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([self.BOX_CENTER[0] - 0.05, self.BOX_CENTER[0] + 0.05]),
                    y_range=np.array([self.BOX_CENTER[1] - 0.05, self.BOX_CENTER[1] + 0.05]),
                    rotation=np.array([np.pi - 0.174533, np.pi + 0.174533]),
                    reference=ReferenceConfig(self.table, on_top=True),
                ),
            )
        )
        return [*self.shelf_boards, self.shelf_cabinet, self.table, self.box]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsStatic(self.box),
            IsUpright(self.box),
            IsInContact(self.box, self.shelf_board_1),
        )

    def _get_instruction(self) -> str:
        return "Place box on shelf"

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.06, 0.06), (-0.06, 0.06), (-0.52, 0.52))

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
            object_ref="shelf_board_3",
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
            object_ref="shelf_board_3",
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

    def get_object(self):
        return dict(
            box=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            shelf_board_3=dict(obj_name=self.shelf_board_3.mj_obj.root_body, obj_type="body"),
            shelf_cabinet=dict(obj_name=self.shelf_cabinet.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.table.mj_obj.root_body, obj_type="body"),
        )


# variant where robot base starts at a close position
class LMBoxTableToShelfStaticDT(LMBoxTableToShelf, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)
    MUJOCO_ARENA_CLS = LabArenaPlane

    TABLE_CORNER = (-0.371296, 0.581634)
    BOX_OFFSET = (0.23, -0.56)
    BOX_CENTER = (TABLE_CORNER[0] + BOX_OFFSET[0], TABLE_CORNER[1] + BOX_OFFSET[1])

    def _get_objects(self) -> list[SceneObject]:
        self.shelf_cabinet = SceneObject(
            ObjectConfig(
                name="shelf_cabinet",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_cabinet/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.85 - 0.3, -0.58 - 0.15, 0]),
                ),
            )
        )

        self.shelf_board_0 = SceneObject(
            ObjectConfig(
                name="shelf_board_0",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=5),
                ),
            )
        )
        self.shelf_board_1 = SceneObject(
            ObjectConfig(
                name="shelf_board_1",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=14),
                ),
            )
        )
        self.shelf_board_2 = SceneObject(
            ObjectConfig(
                name="shelf_board_2",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=0),
                ),
            )
        )
        self.shelf_board_3 = SceneObject(
            ObjectConfig(
                name="shelf_board_3",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=24),
                ),
            )
        )
        self.shelf_boards = [
            self.shelf_board_0,
            self.shelf_board_1,
            self.shelf_board_2,
            self.shelf_board_3,
        ]

        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.85 + 0.37, 0, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )

        self.box = SceneObject(
            ObjectConfig(
                name="box",
                mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([self.BOX_CENTER[0], self.BOX_CENTER[0]]),
                    y_range=np.array([self.BOX_CENTER[1], self.BOX_CENTER[1]]),
                    rotation=np.array([np.pi, np.pi]),
                    reference=ReferenceConfig(self.table, on_top=True),
                ),
            )
        )
        return [*self.shelf_boards, self.shelf_cabinet, self.table, self.box]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsStatic(self.box),
            IsUpright(self.box, threshold=0.95),
            IsInContact(self.box, self.shelf_board_1),
        )

    def _get_instruction(self) -> str:
        return "Place box on shelf"

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
            object_ref="shelf_board_3",
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
            object_ref="shelf_board_3",
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

    def get_object(self):
        return dict(
            box=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            shelf_board_3=dict(obj_name=self.shelf_board_3.mj_obj.root_body, obj_type="body"),
            shelf_cabinet=dict(obj_name=self.shelf_cabinet.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.table.mj_obj.root_body, obj_type="body"),
        )

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (0.73, 0.73), (-0.06, 0.06), (-0.00, 0.00))

        #    Initialize arms to raised pose
        self._set_raised_arm_pose()

    def _set_raised_arm_pose(self):
        """Set robot arms to a raised pose after reset."""
        robot = self.robots[0]

        # Map joint name substring -> target value
        arm_joint_targets = {
            "left_shoulder_roll": 0.15,
            "left_elbow": -0.35,
            "right_shoulder_roll": -0.15,
            "right_elbow": -0.35,
        }

        for joint_name, qpos_idx in zip(robot.robot_joints, robot._ref_joint_pos_indexes):
            for target_name, target_value in arm_joint_targets.items():
                if target_name in joint_name:
                    self.sim.data.qpos[qpos_idx] = target_value
                    break

        self.sim.forward()

    def _set_default_arm_pose(self):
        """Set robot arms to a default pose after reset."""
        robot = self.robots[0]

        # Map joint name substring -> target value
        arm_joint_targets = {
            "left_shoulder_roll": 0.0,
            "left_elbow": 0.0,
            "right_shoulder_roll": 0.0,
            "right_elbow": 0.0,
        }

        for joint_name, qpos_idx in zip(robot.robot_joints, robot._ref_joint_pos_indexes):
            for target_name, target_value in arm_joint_targets.items():
                if target_name in joint_name:
                    self.sim.data.qpos[qpos_idx] = target_value
                    break

        self.sim.forward()


class LMBoxTableToCartStaticDT(LMBoxTableToShelfStaticDT):
    TABLE_CORNER = (-0.371296, 0.581634)
    BOX_OFFSET = (0.23, -0.56)
    BOX_CENTER = (TABLE_CORNER[0] + BOX_OFFSET[0], TABLE_CORNER[1] + BOX_OFFSET[1])

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()

        self.box = SceneObject(
            ObjectConfig(
                name="box",
                mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([self.BOX_CENTER[0] - 0.05, self.BOX_CENTER[0] + 0.05]),
                    y_range=np.array([self.BOX_CENTER[1] - 0.05, self.BOX_CENTER[1] + 0.05]),
                    rotation=np.array([np.pi - 0.174533, np.pi + 0.174533]),
                    reference=ReferenceConfig(self.table, on_top=True),
                ),
            )
        )
        self.shelf_cabinet.update_cfg(
            mjcf_path="objects/omniverse/locomanip/lab_cart/model.xml",
            sampler_config=SamplingConfig(
                reference_pos=np.array([0.85 - 0.3, -0.58 - 0.25, 0]),
                rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
            ),
            scale=1.0,
        )

        self.box_in_the_cart = SceneObject(
            ObjectConfig(
                name="box_in_the_cart",
                mjcf_path="objects/omniverse/locomanip/cardbox_a1_flat/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=2),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                ),
            )
        )

        self.shelf_boards = [self.shelf_board_0, self.shelf_board_1, self.shelf_board_2]

        spawn_count = len(self.shelf_cabinet.mj_obj.spawns)
        for i, board in enumerate(self.shelf_boards):
            spawn_id = max(0, min(i, spawn_count - 1))
            board.update_cfg(
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=spawn_id),
                    z_offset=0.1,
                ),
                rgba=(1, 0, 0, 0),
                scale=(0.5, 1.0, 1.0),
            )

        return [*self.shelf_boards, self.shelf_cabinet, self.table, self.box, self.box_in_the_cart]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsStatic(self.box),
            IsUpright(self.box, threshold=0.95),
            IsInContact(self.box, self.shelf_board_2),
        )

    def get_object(self):
        return dict(
            box=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            shelf_board_2=dict(obj_name=self.shelf_board_2.mj_obj.root_body, obj_type="body"),
            shelf_cabinet=dict(obj_name=self.shelf_cabinet.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.table.mj_obj.root_body, obj_type="body"),
        )

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
            object_ref="shelf_board_2",
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
            object_ref="shelf_board_2",
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

class LMNavBoxTableToCartStaticDT(LMBoxTableToCartStaticDT):
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.15, 0.15), (-np.pi / 6, np.pi / 6))

class LMBoxTableToCartStaticDTDC(LMBoxTableToCartStaticDT):
    # remove the randomization of the box position for DC

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()

        self.box = SceneObject(
            ObjectConfig(
                name="box",
                mjcf_path="objects/omniverse/locomanip/lab_box_handles_v2/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([self.BOX_CENTER[0] - 0, self.BOX_CENTER[0] + 0]),
                    y_range=np.array([self.BOX_CENTER[1] - 0, self.BOX_CENTER[1] + 0]),
                    rotation=np.array([np.pi - 0.0, np.pi + 0.0]),
                    reference=ReferenceConfig(self.table, on_top=True),
                ),
            )
        )
        self.shelf_cabinet.update_cfg(
            mjcf_path="objects/omniverse/locomanip/lab_cart/model.xml",
            sampler_config=SamplingConfig(
                reference_pos=np.array([0.85 - 0.3, -0.58 - 0.25, 0]),
                rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
            ),
            scale=1.0,
        )

        self.box_in_the_cart = SceneObject(
            ObjectConfig(
                name="box_in_the_cart",
                mjcf_path="objects/omniverse/locomanip/cardbox_a1_flat/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=2),
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                ),
            )
        )

        self.shelf_boards = [self.shelf_board_0, self.shelf_board_1, self.shelf_board_2]

        spawn_count = len(self.shelf_cabinet.mj_obj.spawns)
        for i, board in enumerate(self.shelf_boards):
            spawn_id = max(0, min(i, spawn_count - 1))
            board.update_cfg(
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=spawn_id),
                    z_offset=0.1,
                ),
                rgba=(1, 0, 0, 0),
                scale=(0.5, 1.0, 1.0),
            )

        return [*self.shelf_boards, self.shelf_cabinet, self.table, self.box, self.box_in_the_cart]

class LMBoxTableToShelfStaticIndustrial(LMBoxTableToShelfStaticDT):
    MUJOCO_ARENA_CLS = FactoryArena

    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        self.shelf_cabinet.update_cfg(
            mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
            sampler_config=SamplingConfig(
                reference_pos=np.array([0.85 - 0.3, -0.58 - 0.25, 0]),
                rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
            ),
            scale=0.8,
        )

        spawn_count = len(self.shelf_cabinet.mj_obj.spawns)
        for i, board in enumerate(self.shelf_boards):
            spawn_id = max(0, min(i, spawn_count - 1))
            board.update_cfg(
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=spawn_id),
                    z_offset=-0.015,
                ),
                rgba=(1, 0, 0, 0),
                scale=1.5,
            )

        return objects


class LMBoxTableToShelfStaticIndustrialStartBack(LMBoxTableToShelfStaticIndustrial):
    """Version where robot starts further back from the table for DC."""

    def _reset_internal(self):
        super()._reset_internal()

        self._set_default_arm_pose()
        if not self.deterministic_reset:
            # Move robot further back - increase x position from 0.7 to 1.0
            RobotPoseRandomizer.set_pose(self, (0.5, 0.5), (-0.06, 0.06), (-0.00, 0.00))


class LMNavBoxTableToShelfStaticIndustrial(LMBoxTableToShelfStaticIndustrial):
    """Navigation version of LMBoxTableToShelfStaticIndustrial with robot pose randomization."""

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Centered on parent's default (0, 0, 0)
            RobotPoseRandomizer.set_pose(self, (-0.15, 0.15), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMNavBoxTableToShelfStaticIndustrialStartBack(LMNavBoxTableToShelfStaticIndustrial):
    """Navigation version where robot starts further back from the table for DC."""

    def _reset_internal(self):
        # Call the parent's parent (LMBoxTableToShelfStaticIndustrial) reset
        super(LMNavBoxTableToShelfStaticIndustrial, self)._reset_internal()

        self._set_default_arm_pose()
        if not self.deterministic_reset:
            # Move robot further back and add navigation randomization
            RobotPoseRandomizer.set_pose(self, (0.4, 0.6), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class LMPnPBottleToBin(LMPnPBottle, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)
    MUJOCO_ARENA_CLS = LabArenaPlane

    TABLE_CORNER = (-0.371296, -0.581634)
    BOTTLE_OFFSET = (0.13, 0.27)
    BOTTLE_CENTER = (TABLE_CORNER[0] + BOTTLE_OFFSET[0], TABLE_CORNER[1] + BOTTLE_OFFSET[1])

    BOTTLE_GRADIENT: Gradient = Gradient(
        np.array([0.7, 0.7, 0.7, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    )

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.5 - self.TABLE_CORNER[0], 0.3, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        self.bin = SceneObject(
            ObjectConfig(
                name="bin",
                mjcf_path="objects/omniverse/locomanip/lab_bin/model.xml",
                static=True,
                scale=(0.815, 0.626, 0.775),
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.25, -0.45, 0]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/lab_bottle/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([self.BOTTLE_CENTER[0] - 0.05, self.BOTTLE_CENTER[0] + 0.05]),
                    y_range=np.array([self.BOTTLE_CENTER[1] - 0.05, self.BOTTLE_CENTER[1] + 0.05]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        return [self.table, self.bin, self.bottle]

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            subtask_term_preoffset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # each arm's subtask is to place the object on the target table
        task.task_spec_0.subtask_2 = dict(
            object_ref="bin",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            subtask_term_preoffset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            subtask_term_preoffset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_2 = dict(
            object_ref="bin",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            subtask_term_preoffset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsGripperFar(self.bottle),
            IsPositionInRange(self.bottle, 2, max_val=0.6),
            IsInContact(self.bottle, self.bin),
        )

    def _get_instruction(self) -> str:
        return "Pick up bottle and place in bin"

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.1, 0.1), (-0.1, 0.1), (-0.52, 0.52))

    # get object
    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            bin=dict(obj_name=self.bin.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.table.mj_obj.root_body, obj_type="body"),
        )

    def _randomize_table_rgba(self):
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=self.table.mj_obj, gradient=self.TABLE_GRADIENT, linear=True
        )
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=self.bottle.mj_obj, gradient=self.BOTTLE_GRADIENT, linear=True
        )


class LMPnPBottleToBinFar(LMPnPBottleToBin, DexMGConfigHelper):
    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.3, -0.3), (-0.1, 0.1), (-0.52, 0.52))

class LMNavPnPBottleToBinFar(LMPnPBottleToBinFar):

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.3 - 0.15, -0.3 + 0.15), (-0.1, 0.1), (-0.52, 0.52))


class LMNavPnPBottleToBinFarObjRandMatch(LMNavPnPBottleToBinFar):

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.3 - 0.15, -0.3 + 0.15), (-0.1, 0.1), (-0.52, 0.52))

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.5 - self.TABLE_CORNER[0], 0.3, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        self.bin = SceneObject(
            ObjectConfig(
                name="bin",
                mjcf_path="objects/omniverse/locomanip/lab_bin/model.xml",
                static=True,
                scale=(0.815, 0.626, 0.775),
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.25, -0.45, 0]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/lab_bottle/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([self.BOTTLE_CENTER[0] - 0.10, self.BOTTLE_CENTER[0] + 0.10]),
                    y_range=np.array([self.BOTTLE_CENTER[1] - 0.10, self.BOTTLE_CENTER[1] + 0.10]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        return [self.table, self.bin, self.bottle]

class LMNavPnPBottleToBinFarObjRandLarge(LMNavPnPBottleToBinFar):

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.3 - 0.15, -0.3 + 0.15), (-0.1, 0.1), (-0.52, 0.52))

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.5 - self.TABLE_CORNER[0], 0.3, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        self.bin = SceneObject(
            ObjectConfig(
                name="bin",
                mjcf_path="objects/omniverse/locomanip/lab_bin/model.xml",
                static=True,
                scale=(0.815, 0.626, 0.775),
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.25, -0.45, 0]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/lab_bottle/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([self.BOTTLE_CENTER[0] - 0.20, self.BOTTLE_CENTER[0] + 0.20]),
                    y_range=np.array([self.BOTTLE_CENTER[1] - 0.20, self.BOTTLE_CENTER[1] + 0.20]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        return [self.table, self.bin, self.bottle]

class LMPnPBottleToBinStatic(LMPnPBottleToBin, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)
    MUJOCO_ARENA_CLS = LabArenaPlane

    TABLE_CORNER = (-0.371296, -0.581634)
    BOTTLE_OFFSET = (0.13, 0.27)
    BOTTLE_CENTER = (TABLE_CORNER[0] + BOTTLE_OFFSET[0], TABLE_CORNER[1] + BOTTLE_OFFSET[1])

    BOTTLE_GRADIENT: Gradient = Gradient(
        np.array([0.7, 0.7, 0.7, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    )

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.5 - self.TABLE_CORNER[0], 0.3, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        self.bin = SceneObject(
            ObjectConfig(
                name="bin",
                mjcf_path="objects/omniverse/locomanip/lab_bin/model.xml",
                static=True,
                scale=(0.815, 0.626, 0.775),
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.25, -0.45, 0]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/lab_bottle/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    x_range=np.array([self.BOTTLE_CENTER[0] - 0.05, self.BOTTLE_CENTER[0] + 0.05]),
                    y_range=np.array([self.BOTTLE_CENTER[1] - 0.05, self.BOTTLE_CENTER[1] + 0.05]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        return [self.table, self.bin, self.bottle]

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # each arm's subtask is to pick up the object
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            subtask_term_preoffset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # each arm's subtask is to place the object on the target table
        task.task_spec_0.subtask_2 = dict(
            object_ref="bin",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            subtask_term_preoffset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            subtask_term_preoffset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_2 = dict(
            object_ref="bin",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            subtask_term_preoffset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsGripperFar(self.bottle),
            IsPositionInRange(self.bottle, 2, max_val=0.6),
            IsInContact(self.bottle, self.bin),
        )

    def _get_instruction(self) -> str:
        return "Pick up bottle and place in bin"

    # get object
    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            bin=dict(obj_name=self.bin.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.table.mj_obj.root_body, obj_type="body"),
        )

    def _randomize_table_rgba(self):
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=self.table.mj_obj, gradient=self.TABLE_GRADIENT, linear=True
        )
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=self.bottle.mj_obj, gradient=self.BOTTLE_GRADIENT, linear=True
        )

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (0.25, 0.30), (-0.1, 0.1), (-0.00, 0.00))


class LMStockingDT(LMBoxTableToShelfStaticDT):
    """Digital-twin stocking task.

    Scene:
      * white lab_shelf_cabinet with 4 shelf boards, back against the grey labarena wall
      * wheeled lab_cart ~1.5 m from the shelf
      * tp_link_box (13 x 9 x 9 cm, teal) initially on the cart's top tray
      * tape_roll (OD 11.5 cm, ID 7.5 cm, grey) also on the cart's top tray
      * G1 robot starts beside the cart

    Success: both box AND tape resting on any shelf board, both static, and no
    robot gripper still in contact with either object.
    """

    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)
    # Use the full LabArena (includes the grey wall at x=0.9212) so the shelf has
    # something to back up against, rather than the plane-only variant.
    MUJOCO_ARENA_CLS = LabArena

    # Wall at x ~= 0.92. Shelf depth half = 0.146; back panel stands ~10 cm off the wall
    # so there is a visible gap between the shelf back and the grey labarena wall.
    SHELF_POS = (0.92 - 0.146 - 0.10, -0.30, 0.0)
    SHELF_YAW = np.pi * 0.5                       # rotate so back (-y in local) faces +x wall
    # Robot is 182 cm from the shelf along the wall (+y direction); cart sits
    # between robot and shelf at 75 cm from the robot.
    # Orthogonal-to-wall distance: wall at x=0.9212, cart at ~0.16 -> ~76 cm from wall.
    CART_POS = (0.92 - 0.76, SHELF_POS[1] + 1.82 - 0.75, 0.0)
    CART_YAW = -np.pi * 0.5   # rotated 180 deg from the -y side arrangement

    def _get_objects(self) -> list[SceneObject]:
        self.shelf_cabinet = SceneObject(
            ObjectConfig(
                name="shelf_cabinet",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_cabinet/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array(self.SHELF_POS),
                    rotation=np.array([self.SHELF_YAW, self.SHELF_YAW]),
                ),
            )
        )

        # Board rotation matches the cabinet yaw so boards align with the interior.
        board_rot = np.array([self.SHELF_YAW, self.SHELF_YAW])
        self.shelf_board_0 = SceneObject(
            ObjectConfig(
                name="shelf_board_0",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=5),
                    rotation=board_rot,
                ),
            )
        )
        self.shelf_board_1 = SceneObject(
            ObjectConfig(
                name="shelf_board_1",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=14),
                    rotation=board_rot,
                ),
            )
        )
        self.shelf_board_2 = SceneObject(  # level 3 from bottom -- the target
            ObjectConfig(
                name="shelf_board_2",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=0),
                    rotation=board_rot,
                ),
            )
        )
        self.shelf_board_3 = SceneObject(
            ObjectConfig(
                name="shelf_board_3",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=24),
                    rotation=board_rot,
                ),
            )
        )
        self.shelf_boards = [
            self.shelf_board_0,
            self.shelf_board_1,
            self.shelf_board_2,
            self.shelf_board_3,
        ]

        self.cart = SceneObject(
            ObjectConfig(
                name="cart",
                mjcf_path="objects/omniverse/locomanip/lab_cart/model.xml",
                static=False,
                density=500,
                sampler_config=SamplingConfig(
                    reference_pos=np.array(self.CART_POS),
                    rotation=np.array([self.CART_YAW, self.CART_YAW]),
                ),
            )
        )

        # Both box and tape start on the cart's top tray (spawn_id=2), offset
        # symmetrically along world-x so they sit side-by-side with comfortable
        # clearance (separation ~15 cm > 12.25 cm minimum non-overlap) while
        # staying centered on the tray rather than near the rim.
        self.box = SceneObject(
            ObjectConfig(
                name="box",
                mjcf_path="objects/omniverse/locomanip/tp_link_box/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.cart, spawn_id=2),
                    x_range=np.array([-0.075, -0.075]),
                    y_range=np.array([0.0, 0.0]),
                    rotation=np.array([0.0, 0.0]),
                ),
            )
        )

        self.tape = SceneObject(
            ObjectConfig(
                name="tape",
                mjcf_path="objects/omniverse/locomanip/tape_roll/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.cart, spawn_id=2),
                    x_range=np.array([0.075, 0.075]),
                    y_range=np.array([0.0, 0.0]),
                    rotation=np.array([0.0, 0.0]),
                ),
            )
        )

        return [
            *self.shelf_boards,
            self.shelf_cabinet,
            self.cart,
            self.box,
            self.tape,
        ]

    def _get_success_criteria(self) -> SuccessCriteria:
        # Success = both objects resting (static) on any shelf board AND the
        # robot's grippers are no longer holding/touching either object.
        # We intentionally don't check robot contact with the cart / cabinet /
        # target board itself, since the robot commonly brushes those surfaces
        # while placing and that shouldn't fail the task.
        return AllCriteria(
            IsStatic(self.box),
            IsStatic(self.tape),
            AnyCriteria(*[IsInContact(self.box, b) for b in self.shelf_boards]),
            AnyCriteria(*[IsInContact(self.tape, b) for b in self.shelf_boards]),
            NotCriteria(IsInContactWithRobot(self.box)),
            NotCriteria(IsInContactWithRobot(self.tape)),
        )

    def _get_instruction(self) -> str:
        return "Place the box and the tape on the second shelf from the bottom"

    def _reset_internal(self):
        # Skip parent's RobotPoseRandomizer call (which places robot at x=0.73 near the
        # original shelf position). Call the grandparent reset directly.
        LMSimpleEnv._reset_internal(self)
        if not self.deterministic_reset:
            # Robot ~75 cm in +y of the cart (cart is between robot and shelf along
            # the wall); facing -y toward the cart.
            rx = self.CART_POS[0] + 0.05
            ry = self.CART_POS[1] + 0.75
            yaw = -np.pi / 2
            RobotPoseRandomizer.set_pose(
                self,
                (rx - 0.02, rx + 0.02),
                (ry - 0.03, ry + 0.03),
                (yaw - 0.05, yaw + 0.05),
            )
        self._set_raised_arm_pose()

    def get_object(self):
        return dict(
            box=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            tape=dict(obj_name=self.tape.mj_obj.root_body, obj_type="body"),
            shelf_board_1=dict(obj_name=self.shelf_board_1.mj_obj.root_body, obj_type="body"),
            shelf_cabinet=dict(obj_name=self.shelf_cabinet.mj_obj.root_body, obj_type="body"),
            cart=dict(obj_name=self.cart.mj_obj.root_body, obj_type="body"),
        )

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # Each arm first picks up one of the two objects, then places on shelf_board_1.
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
            object_ref="tape",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="shelf_board_1",
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
            object_ref="shelf_board_1",
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


class LMNavStockingDT(LMStockingDT):
    """Nav variant of LMStockingDT: wider randomization of the robot's starting pose
    and yaw, requiring the policy to locomote/re-orient before approaching the cart."""

    def _reset_internal(self):
        LMSimpleEnv._reset_internal(self)
        if not self.deterministic_reset:
            rx = self.CART_POS[0] + 0.05
            ry = self.CART_POS[1] + 0.75
            yaw = -np.pi / 2
            RobotPoseRandomizer.set_pose(
                self,
                (rx - 0.15, rx + 0.15),
                (ry - 0.15, ry + 0.15),
                (yaw - np.pi / 6, yaw + np.pi / 6),
            )
        self._set_raised_arm_pose()


class LMStockSimple(LMStockingDT):
    """Simplified stocking task with two items and a 2-of-2 shelf success condition.

    Scene:
      * shelf fixed against the wall, with cart centered directly in front of it
      * brown box and yellow tape on the cart top tray

    Success: both items are static, resting on any shelf board, and no longer
    touching the robot.
    """

    # Pull the cart off the side-wall lane used by LMStockingDT and center it on
    # the shelf opening so the approach is straight-on.
    CART_POS = (LMStockingDT.SHELF_POS[0] - 0.42, LMStockingDT.SHELF_POS[1] + 0.02, 0.0)
    CART_YAW = -np.pi * 0.5

    def _get_objects(self) -> list[SceneObject]:
        self.shelf_cabinet = SceneObject(
            ObjectConfig(
                name="shelf_cabinet",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_cabinet/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array(self.SHELF_POS),
                    rotation=np.array([self.SHELF_YAW, self.SHELF_YAW]),
                ),
            )
        )

        board_rot = np.array([self.SHELF_YAW, self.SHELF_YAW])
        self.shelf_board_0 = SceneObject(
            ObjectConfig(
                name="shelf_board_0",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=5),
                    rotation=board_rot,
                ),
            )
        )
        self.shelf_board_1 = SceneObject(
            ObjectConfig(
                name="shelf_board_1",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=14),
                    rotation=board_rot,
                ),
            )
        )
        self.shelf_board_2 = SceneObject(
            ObjectConfig(
                name="shelf_board_2",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=0),
                    rotation=board_rot,
                ),
            )
        )
        self.shelf_board_3 = SceneObject(
            ObjectConfig(
                name="shelf_board_3",
                mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.shelf_cabinet, spawn_id=24),
                    rotation=board_rot,
                ),
            )
        )
        self.shelf_boards = [
            self.shelf_board_0,
            self.shelf_board_1,
            self.shelf_board_2,
            self.shelf_board_3,
        ]

        self.cart = SceneObject(
            ObjectConfig(
                name="cart",
                mjcf_path="objects/omniverse/locomanip/lab_cart/model.xml",
                static=False,
                density=500,
                sampler_config=SamplingConfig(
                    reference_pos=np.array(self.CART_POS),
                    rotation=np.array([self.CART_YAW, self.CART_YAW]),
                ),
            )
        )

        self.brown_box = SceneObject(
            ObjectConfig(
                name="brown_box",
                mjcf_path="objects/omniverse/locomanip/brown_stock_box/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.cart, spawn_id=2),
                    x_range=np.array([-0.07, -0.07]),
                    y_range=np.array([0.07, 0.07]),
                    rotation=np.array([0.0, 0.0]),
                ),
            )
        )

        self.yellow_tape = SceneObject(
            ObjectConfig(
                name="yellow_tape",
                mjcf_path="objects/omniverse/locomanip/yellow_tape_roll/model.xml",
                static=False,
                sampler_config=SamplingConfig(
                    reference=ReferenceConfig(self.cart, spawn_id=2),
                    x_range=np.array([0.065, 0.065]),
                    y_range=np.array([0.0, 0.0]),
                    rotation=np.array([0.0, 0.0]),
                ),
            )
        )
        self.tape = self.yellow_tape
        self.box = self.brown_box
        self.stock_items = [self.brown_box, self.yellow_tape]

        return [
            *self.shelf_boards,
            self.shelf_cabinet,
            self.cart,
            *self.stock_items,
        ]

    def _item_on_any_shelf(self, item: SceneObject) -> SuccessCriteria:
        return AllCriteria(
            IsStatic(item),
            AnyCriteria(*[IsInContact(item, board) for board in self.shelf_boards]),
            NotCriteria(IsInContactWithRobot(item)),
        )

    def _get_success_criteria(self) -> SuccessCriteria:
        brown_box_done = self._item_on_any_shelf(self.brown_box)
        yellow_tape_done = self._item_on_any_shelf(self.yellow_tape)
        return AllCriteria(brown_box_done, yellow_tape_done)

    def _get_instruction(self) -> str:
        return "Place the brown box and yellow tape on the shelves."

    def _reset_internal(self):
        LMSimpleEnv._reset_internal(self)
        if not self.deterministic_reset:
            # Start the robot just behind the cart on the shelf approach axis so
            # both the cart and target shelf are directly ahead.
            rx = self.CART_POS[0] - 0.52
            ry = self.CART_POS[1]
            yaw = 0.0
            RobotPoseRandomizer.set_pose(
                self,
                (rx - 0.03, rx + 0.03),
                (ry - 0.04, ry + 0.04),
                (yaw - 0.06, yaw + 0.06),
            )
        self._set_raised_arm_pose()

    def get_object(self):
        return dict(
            brown_box=dict(obj_name=self.brown_box.mj_obj.root_body, obj_type="body"),
            yellow_tape=dict(obj_name=self.yellow_tape.mj_obj.root_body, obj_type="body"),
            shelf_board_1=dict(obj_name=self.shelf_board_1.mj_obj.root_body, obj_type="body"),
            shelf_cabinet=dict(obj_name=self.shelf_cabinet.mj_obj.root_body, obj_type="body"),
            cart=dict(obj_name=self.cart.mj_obj.root_body, obj_type="body"),
        )

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="yellow_tape",
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
            object_ref="brown_box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="shelf_board_1",
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
            object_ref="shelf_board_1",
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
