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

from abc import abstractmethod
from typing import Optional

from humanoidmimicgen.locomanipulation.envs.base import LocoManipulationEnv
from humanoidmimicgen.locomanipulation.models.scenes import GroundArena
from humanoidmimicgen.locomanipulation.models.scenes.factory_arena import FactoryArena
from humanoidmimicgen.locomanipulation.utils.scene.configs import SceneConfig, SceneScaleConfig
from humanoidmimicgen.locomanipulation.utils.scene.scene import Scene, SceneObject
from humanoidmimicgen.locomanipulation.utils.scene.success_criteria import SuccessCriteria


class LMEnvBase(LocoManipulationEnv):
    SCENE_SCALE = SceneScaleConfig()

    def __init__(
        self,
        translucent_robot: bool = False,
        use_object_obs: bool = False,
        scene_scale: Optional[SceneScaleConfig] = None,
        *args,
        **kwargs,
    ):
        self.scene_scale = scene_scale or self.SCENE_SCALE
        super().__init__(translucent_robot, use_object_obs, *args, **kwargs)

    def _load_model(self):
        self.scene = Scene(self, self._get_env_config(), self.scene_scale)
        self.mujoco_objects = self.scene.mujoco_objects

        super()._load_model()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        if not self.deterministic_reset:
            self.scene.reset()

    def _setup_references(self):
        super()._setup_references()

        self.obj_body_id = {}
        for obj in self.mujoco_objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)

    def _get_env_config(self) -> SceneConfig:
        return SceneConfig(
            objects=self._get_objects(),
            success=self._get_success_criteria(),
            instruction=self._get_instruction(),
        )

    @abstractmethod
    def _get_objects(self) -> list[SceneObject]:
        raise NotImplementedError

    @abstractmethod
    def _get_success_criteria(self) -> SuccessCriteria:
        raise NotImplementedError

    @abstractmethod
    def _get_instruction(self) -> str:
        raise NotImplementedError

    def _check_success(self):
        return self.scene.success()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.scene.instruction
        return ep_meta


# noinspection PyAbstractClass
class LMSimpleEnv(LMEnvBase):
    MUJOCO_ARENA_CLS = GroundArena


# noinspection PyAbstractClass
class LMFactoryEnv(LMEnvBase):
    MUJOCO_ARENA_CLS = FactoryArena
