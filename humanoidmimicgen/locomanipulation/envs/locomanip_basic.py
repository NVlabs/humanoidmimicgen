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
from humanoidmimicgen.locomanipulation.models.scenes.factory_arena import FactoryArena
from humanoidmimicgen.locomanipulation.utils.hmg_config import HMGConfigHelper
from humanoidmimicgen.locomanipulation.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
    SceneScaleConfig,
)
from humanoidmimicgen.locomanipulation.utils.scene.scene import SceneObject
from humanoidmimicgen.locomanipulation.utils.scene.success_criteria import (
    AllCriteria,
    IsInContact,
    IsStatic,
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


class LMBoxTableToShelfStaticIndustrial(LMFactoryEnv, HMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)
    MUJOCO_ARENA_CLS = FactoryArena

    TABLE_CORNER = (-0.371296, 0.581634)
    BOX_OFFSET = (0.23, -0.56)
    BOX_CENTER = (TABLE_CORNER[0] + BOX_OFFSET[0], TABLE_CORNER[1] + BOX_OFFSET[1])

    def _get_objects(self) -> list[SceneObject]:
        self.shelf_cabinet = SceneObject(
            ObjectConfig(
                name="shelf_cabinet",
                mjcf_path="objects/omniverse/locomanip/shelf_a12/model.xml",
                static=True,
                scale=0.8,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.85 - 0.3, -0.58 - 0.25, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )

        spawn_count = len(self.shelf_cabinet.mj_obj.spawns)
        self.shelf_boards = []
        for i in range(4):
            spawn_id = max(0, min(i, spawn_count - 1))
            board = SceneObject(
                ObjectConfig(
                    name=f"shelf_board_{i}",
                    mjcf_path="objects/omniverse/locomanip/lab_shelf_board/model.xml",
                    static=True,
                    scale=1.5,
                    rgba=(1, 0, 0, 0),
                    sampler_config=SamplingConfig(
                        reference=ReferenceConfig(self.shelf_cabinet, spawn_id=spawn_id),
                        z_offset=-0.015,
                    ),
                )
            )
            self.shelf_boards.append(board)

        (
            self.shelf_board_0,
            self.shelf_board_1,
            self.shelf_board_2,
            self.shelf_board_3,
        ) = self.shelf_boards

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
        return dict()

    @staticmethod
    def task_config():
        task = HMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = _subtask("box")
        task.task_spec_1.subtask_1 = _subtask("box")
        task.task_spec_0.subtask_2 = _subtask("shelf_board_3")
        task.task_spec_1.subtask_2 = _subtask("shelf_board_3")
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
            RobotPoseRandomizer.set_pose(self, (0.73, 0.73), (-0.06, 0.06), (0.0, 0.0))

        self._set_raised_arm_pose()

    def _set_raised_arm_pose(self):
        robot = self.robots[0]
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
