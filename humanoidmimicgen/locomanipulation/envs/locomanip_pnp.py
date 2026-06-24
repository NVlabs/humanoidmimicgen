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
    IsPositionInRange,
    IsStatic,
    IsUpright,
    NotCriteria,
    SuccessCriteria,
)


def _subtask(object_ref, subtask_term_signal=None, subtask_term_offset_range=None):
    return dict(
        object_ref=object_ref,
        subtask_term_signal=subtask_term_signal,
        subtask_term_offset_range=subtask_term_offset_range,
        selection_strategy="random",
        selection_strategy_kwargs=None,
        action_noise=0.05,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )


def _constraints(*pairs):
    task_constraint = DexMGConfigHelper.AttrDict()
    for index, pair in enumerate(pairs, start=1):
        task_constraint[f"constraint_{index}"] = dict(
            subtasks=pair,
            constraint_type="temporal_concurrent",
        )
    return task_constraint.to_dict()


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
        return dict(obj=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"))

    def get_subtask_term_signals(self):
        return dict(obj_lifted=int(self._check_success()))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = _subtask("obj", "obj_lifted")
        task.task_spec_1.subtask_1 = _subtask("obj", "obj_lifted")
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        return _constraints([("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")])


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
        )

    def _get_instruction(self) -> str:
        return "Lift up the box."

    def get_object(self):
        return dict(obj=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"))

    def get_subtask_term_signals(self):
        return dict(obj_lifted=int(self._check_success()))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = _subtask("obj", "obj_lifted")
        task.task_spec_1.subtask_1 = _subtask("obj", "obj_lifted")
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        return _constraints([("task_spec_0", "subtask_1"), ("task_spec_1", "subtask_1")])


class LMDrillLift(LMFactoryEnv, DexMGConfigHelper):
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
                mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.28, -0.24]),
                    y_range=np.array([-0.05, 0.05]),
                    rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
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
        return "Pick up the drill."

    def get_object(self):
        return dict(bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"))

    def get_subtask_term_signals(self):
        return dict(obj_lifted=int(self._check_success()))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = _subtask("bottle", "obj_lifted")
        task.task_spec_1.subtask_1 = _subtask(None)
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        return DexMGConfigHelper.AttrDict().to_dict()


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
                scale=0.7,
            )
        )
        return [*objects, self.obstacle]


class LMDrillPnP90(LMFactoryEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(1, 1), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.table_origin = SceneObject(
            ObjectConfig(
                name="table_origin",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.2, 0, 0]),
                    rotation=np.array([-np.pi, -np.pi]),
                ),
            )
        )
        self.table_target = SceneObject(
            ObjectConfig(
                name="table_target",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0, 1.2, 0]),
                    rotation=np.array([-np.pi / 2, -np.pi / 2]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.28, -0.24]),
                    y_range=np.array([-0.05, 0.05]),
                    rotation=np.array([-np.pi * 0.05, np.pi * 0.05]),
                    reference=ReferenceConfig(obj=self.table_origin),
                ),
            )
        )
        return [self.table_origin, self.table_target, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsStatic(self.bottle),
            IsUpright(self.bottle, threshold=0.95),
            IsGripperFar(self.bottle, threshold=0.1),
            IsInContact(self.bottle, self.table_target),
        )

    def _get_instruction(self) -> str:
        return "Pick up the drill from one table and place it on the other."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            table_target=dict(obj_name=self.table_target.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = _subtask("bottle", subtask_term_offset_range=(5, 10))
        task.task_spec_0.subtask_2 = _subtask("table_target")
        task.task_spec_1.subtask_1 = _subtask(None)
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        return DexMGConfigHelper.AttrDict().to_dict()


class LMPickDrillFromHolderStandingEasyFar(LMFactoryEnv, DexMGConfigHelper):
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
                    z_offset=0.05,
                    reference=ReferenceConfig(self.shelf, spawn_id=1),
                ),
            )
        )
        self.drill = SceneObject(
            ObjectConfig(
                name="drill",
                mjcf_path="objects/omniverse/locomanip/powerdrill_b01/model.xml",
                static=False,
                scale=(0.8, 1.0, 1.5),
                friction=(1, 0.005, 0.0001),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.14, -0.14]),
                    y_range=np.array([-0.055, -0.055]),
                    z_offset=-0.24,
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
        return "Pick up the drill from the holder at standing height."

    def get_object(self):
        return dict(
            drill=dict(obj_name=self.drill.mj_obj.root_body, obj_type="body"),
            shelf=dict(obj_name=self.shelf.mj_obj.root_body, obj_type="body"),
            holder=dict(obj_name=self.holder.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        return dict()

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = _subtask("drill")
        task.task_spec_1.subtask_1 = _subtask("drill")
        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        return DexMGConfigHelper.AttrDict().to_dict()

    def _reset_internal(self):
        super()._reset_internal()
        RobotPoseRandomizer.set_pose(self, (-0.5, -0.5), (0, 0), (0, 0))
