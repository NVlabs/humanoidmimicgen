from enum import Enum
from typing import Optional

import numpy as np

from humanoidmimicgen.locomanipulation.utils.visuals_utls import Gradient, randomize_materials_rgba
from humanoidmimicgen.locomanipulation.envs.base import RobotPoseRandomizer
from humanoidmimicgen.locomanipulation.envs.locomanip_basic import LMPnPBottle
from humanoidmimicgen.locomanipulation.envs.locomanip_pnp import (
    LMBottlePnP,
    LMBottleShelfLowToHighPnP,
    LMBoxPnP,
)
from humanoidmimicgen.locomanipulation.models.scenes.lab_arena import LabArena, LabArenaPlane
from humanoidmimicgen.locomanipulation.models.scenes.factory_arena import FactoryArena
from humanoidmimicgen.locomanipulation.utils.dexmg_utils import DexMGConfigHelper
from humanoidmimicgen.locomanipulation.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
)
from humanoidmimicgen.locomanipulation.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
    SceneScaleConfig,
)
from humanoidmimicgen.locomanipulation.envs.locomanip import LMSimpleEnv
from humanoidmimicgen.locomanipulation.utils.scene.scene import SceneObject
from humanoidmimicgen.locomanipulation.utils.dexmg_utils import DexMGConfigHelper
from humanoidmimicgen.locomanipulation.utils.scene.success_criteria import (
    AllCriteria,
    AnyCriteria,
    IsInContact,
    IsUpright,
    SuccessCriteria,
)


class TargetZoneType(str, Enum):
    SOURCE = "SOURCE"
    TARGET = "TARGET"


class TargetZoneFactory:
    _COLOURS_MAP = {
        TargetZoneType.SOURCE: (0.129, 0.588, 0.953, 1.0),
        TargetZoneType.TARGET: (1.000, 0.596, 0.000, 1.0),
    }

    @classmethod
    def build(
        cls,
        target_type: TargetZoneType,
        x_range: np.ndarray,
        y_range: np.ndarray,
        reference: ReferenceConfig,
        z_offset: float = 0.0,
        scale: float = 0.1,
        name: Optional[str] = None,
        rgba: Optional[tuple] = None,
    ) -> SceneObject:
        config = ObjectConfig(
            name=name or target_type.value.lower(),
            mjcf_path="objects/omniverse/locomanip/target_zone/model.xml",
            static=True,
            scale=scale,
            sampler_config=SamplingConfig(
                x_range=x_range,
                y_range=y_range,
                reference=reference,
                z_offset=z_offset,
            ),
            rgba=rgba or cls._COLOURS_MAP.get(target_type),
        )
        return SceneObject(config)


class LMBottleShelfLowToHighTargetPnP(LMBottleShelfLowToHighPnP):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.02, 0.02]),
            y_range=np.array([-0.22, -0.24]),
            reference=ReferenceConfig(obj=self.shelf, spawn_id=0),
        )
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.2, 0.2]),
            y_range=np.array([-0.22, -0.24]),
            reference=ReferenceConfig(self.shelf, spawn_id=1),
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(obj=self.source),
                ),
            )
        )
        return [self.shelf, self.bottle, self.source, self.target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return IsInContact(self.bottle, self.target)

    def _get_instruction(self) -> str:
        return super()._get_instruction() + " Place it in the target zone."


class LMBottleTargetPnP(LMBottlePnP):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.35, -0.3]),
            y_range=np.array([-0.4, 0.4]),
            reference=ReferenceConfig(self.table_origin),
        )
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.35, -0.3]),
            y_range=np.array([-0.4, 0.4]),
            reference=ReferenceConfig(self.table_target),
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(obj=self.source),
                ),
            )
        )

        return [self.table_origin, self.table_target, self.bottle, self.source, self.target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return IsInContact(self.bottle, self.target)

    def _get_instruction(self) -> str:
        return super()._get_instruction() + " Place it in the target zone."


class LMBoxTargetPnP(LMBoxPnP):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.source = TargetZoneFactory.build(
            scale=0.2,
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.4, -0.35]),
            y_range=np.array([-0.4, 0.4]),
            reference=ReferenceConfig(self.table_origin),
        )
        self.target = TargetZoneFactory.build(
            scale=0.2,
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.4, -0.35]),
            y_range=np.array([-0.4, 0.4]),
            reference=ReferenceConfig(self.table_target),
        )
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/cardbox_a1/model.xml",
                static=False,
                scale=0.7,
                sampler_config=SamplingConfig(
                    rotation=np.array([-np.pi, np.pi]),
                    reference=ReferenceConfig(self.source),
                ),
            )
        )
        return [self.table_origin, self.table_target, self.box, self.source, self.target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return IsInContact(self.box, self.target)

    def _get_instruction(self) -> str:
        return super()._get_instruction() + " Place it in the target zone."


class LMTargetPnPBottle(LMPnPBottle):
    SOURCE_OFFSET = 0.05

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.3, -0.28]),
            y_range=np.array([-0.08, 0.08]),
            reference=ReferenceConfig(self.table),
        )
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.3, -0.28]),
            y_range=np.array([-0.08, 0.08]),
            reference=ReferenceConfig(self.table_target),
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(self.source),
                ),
            )
        )
        return [self.table, self.table_target, self.bottle, self.source, self.target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.bottle, symmetric=True), IsInContact(self.bottle, self.target)
        )

    def _get_instruction(self) -> str:
        return super()._get_instruction() + " Place it in the target zone."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            target=dict(obj_name=self.target.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        bottle_xpos = self.sim.data.body_xpos[self.obj_body_id[self.bottle.mj_obj.name]]
        source_xpos = self.sim.data.body_xpos[self.obj_body_id[self.source.mj_obj.name]]
        distance = np.linalg.norm(bottle_xpos - source_xpos)
        is_close = distance <= self.SOURCE_OFFSET
        in_contact = self.check_contact(
            self.bottle.mj_obj.contact_geoms, self.source.mj_obj.contact_geoms
        )
        return dict(obj_off_source=int(not in_contact) and (not is_close))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal="obj_off_source",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
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


class LMTargetPnPBottleStatic(LMTargetPnPBottle):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.14, -0.13]),
            y_range=np.array([-0.1, -0.08]),
            reference=ReferenceConfig(self.table),
        )
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.3, -0.28]),
            y_range=np.array([0.08, 0.1]),
            reference=ReferenceConfig(self.table),
        )
        return [self.table, self.bottle, self.source, self.target]

    def _get_instruction(self) -> str:
        return "Pick up the bottle and place it in the target zone."


class LMTargetPnPBottleToPlate(LMTargetPnPBottle):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.1, -0.08]),
            y_range=np.array([0.03, 0.05]),
            reference=ReferenceConfig(self.table),
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/aigc/cylinder/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    rotation=np.array([np.pi, np.pi]),
                    reference=ReferenceConfig(self.source),
                ),
            )
        )
        self.plate = self.true_target = SceneObject(
            ObjectConfig(
                name="plate",
                mjcf_path="objects/aigc/lab_plate/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.2, -0.15]),
                    y_range=np.array([-0.2, -0.15]),
                    rotation=np.array([0, 0]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.0, -0.0]),
            y_range=np.array([0.0, 0.0]),
            reference=ReferenceConfig(self.plate),
        )
        return [self.table, self.bottle, self.plate, self.source, self.target]

    def _get_instruction(self) -> str:
        return "Pick up the bottle and place it in the target zone."

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            AnyCriteria(
                IsInContact(self.bottle, self.true_target), IsInContact(self.bottle, self.target)
            ),
            IsUpright(self.bottle, symmetric=True),
        )


class LMTargetNavPnPBottleToPlate(LMTargetPnPBottleToPlate):
    def _get_objects(self) -> list[SceneObject]:
        [self.table, self.bottle, self.plate, self.source, self.target] = super()._get_objects()
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.35, -0.25]),
            y_range=np.array([0.3, 0.4]),
            reference=ReferenceConfig(self.table),
            rgba=(0.3, 0.3, 0.3, 0.0),
        )
        self.plate = SceneObject(
            ObjectConfig(
                name="plate",
                mjcf_path="objects/aigc/lab_plate/model.xml",
                static=True,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.25, -0.15]),
                    y_range=np.array([-0.4, -0.3]),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.0, 0.0]),
            y_range=np.array([-0.0, 0.0]),
            reference=ReferenceConfig(self.plate),
            rgba=(0.3, 0.3, 0.3, 0.0),
        )  # invisible target
        return [self.table, self.bottle, self.plate, self.source, self.target]

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-1, -0.3), (-0.1, 0.1), (-np.pi / 12, np.pi / 6))


class LMTargetNavPnPBottleToPlateDigitalTwin(LMTargetNavPnPBottleToPlate):
    MUJOCO_ARENA_CLS = LabArena

    # REF_CORNER_X = -0.371296
    # REF_CORNER_Y = 0.581634
    # REF_CORNER_X = -0.368
    # REF_CORNER_X = -0.348
    # REF_CORNER_Y = 0.587
    # REF_CORNER_X = -0.348
    # REF_CORNER_Y = 0.587
    # REF_CORNER_X = -0.375
    # REF_CORNER_Y = 0.608
    REF_CORNER_X = -0.368
    REF_CORNER_Y = 0.587

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.5, 0, 0.014]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        # Reference position (from corner):
        # X: 0.07, Y: 0.07
        # Range:
        # X: 0.07, Y: 0.07
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([self.REF_CORNER_X + 0.07 - 0.07, self.REF_CORNER_X + 0.07 + 0.07]),
            y_range=np.array([self.REF_CORNER_Y - 0.07 - 0.07, self.REF_CORNER_Y - 0.07 + 0.07]),
            reference=ReferenceConfig(self.table),
            rgba=(0.3, 0.3, 0.3, 0.0),
        )
        # Reference position (from corner):
        # X: 0.14, Y: -0.69
        # Range:
        # X: 0.07, Y: 0.07
        self.plate = SceneObject(
            ObjectConfig(
                name="plate",
                mjcf_path="objects/aigc/lab_plate/model.xml",
                static=True,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array(
                        [self.REF_CORNER_X + 0.14 - 0.07, self.REF_CORNER_X + 0.14 + 0.07]
                    ),
                    y_range=np.array(
                        [self.REF_CORNER_Y - 0.69 - 0.07, self.REF_CORNER_Y - 0.69 + 0.07]
                    ),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        return [self.table, self.bottle, self.plate, self.source, self.target]

    def _reset_internal(self):
        super()._reset_internal()

        # if not self.deterministic_reset:
        #     # Reference position (from corner):
        #     # X: -21in, Y: 2in
        #     # Range:
        #     # X: 6in, Y: 6in
        #     RobotPoseRandomizer.set_pose(
        #         self,
        #         (
        #             0.5 + self.REF_CORNER_X - 0.5334 - 0.1524,
        #             0.5 + self.REF_CORNER_X - 0.5334 + 0.1524,
        #         ),
        #         (self.REF_CORNER_Y - 0.0508 - 0.1524, self.REF_CORNER_Y - 0.0508 + 0.1524),
        #         (-np.pi / 12, np.pi / 12),
        #     )


class LMTargetPnPBottleToPlateStaticDigitalTwin(LMTargetNavPnPBottleToPlateDigitalTwin):
    """
    Pick-and-Place Bottle environment with robot position at table corner at reset.
    Does not require walking to reach the bottle and place it on the plate.
    """

    def _get_objects(self) -> list[SceneObject]:
        [self.table, self.bottle, self.plate, self.source, self.target] = super()._get_objects()

        source_range_y = 0.04  # less left-right range for the bottle
        source_range_x = 0.06
        source_offset_x = 0.07
        source_offset_y = 0.07
        # Reference position (from corner):
        # X: 0.07, Y: source_range_y
        # Range:
        # X: 0.07, Y: 0.07
        # rety to see what new position is when all at 0s
        source_offset_x = 0.06
        source_range_x = 0.06
        source_offset_y = 0.04  # checking this is 0 0
        source_range_y = 0.04
        # set all to zeros for dc again
        # source_offset_x = 0.0
        # source_range_x = 0.0
        # source_offset_y = 0.0
        # source_range_y = 0.0
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array(
                [
                    self.REF_CORNER_X + source_offset_x - source_range_x,
                    self.REF_CORNER_X + source_offset_x + source_range_x,
                ]
            ),
            y_range=np.array(
                [
                    self.REF_CORNER_Y - source_offset_y - source_range_y,
                    self.REF_CORNER_Y - source_offset_y + source_range_y,
                ]
            ),
            reference=ReferenceConfig(self.table),
            rgba=(0.3, 0.3, 0.3, 0.0),
        )

        plate_offset_y = 0.26
        self.plate = SceneObject(
            ObjectConfig(
                name="plate",
                mjcf_path="objects/aigc/lab_plate/model.xml",
                static=True,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array(
                        [self.REF_CORNER_X + 0.15 - 0.07, self.REF_CORNER_X + 0.15 + 0.07]
                    ),
                    y_range=np.array(
                        [
                            self.REF_CORNER_Y - plate_offset_y - 0.02,
                            self.REF_CORNER_Y - plate_offset_y + 0.02,
                        ]
                    ),
                    reference=ReferenceConfig(self.table),
                ),
            )
        )
        return [self.table, self.bottle, self.plate, self.source, self.target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.bottle, symmetric=True, threshold=0.95),
            AnyCriteria(
                IsInContact(self.bottle, self.target), IsInContact(self.bottle, self.plate)
            ),
        )

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            dist_from_corner_x = 0.28
            dist_from_corner_y = -0.02
            x_val = 0.5 + self.REF_CORNER_X - dist_from_corner_x
            y_val = self.REF_CORNER_Y - dist_from_corner_y
            x_val = -0.15882
            y_val = 0.61577 - 0.05 + 0.01 + 0.01 + 0.02
            yaw_val = 0.005
            # navpnp values
            # x_val -= 0.5
            # y_val = 0.61577 - 0.05 + 0.01 + 0.01 + 0.02
            # yaw_val = -0.15
            RobotPoseRandomizer.set_pose(
                self,
                (x_val, x_val),
                (y_val, y_val),
                (yaw_val, yaw_val),
            )

    def _get_instruction(self) -> str:
        return "Pick up bottle and place it on the plate."


class LMTargetPnPBottleToPlateStaticDigitalTwinRandBasePose(
    LMTargetPnPBottleToPlateStaticDigitalTwin
):
    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            dist_from_corner_x = 0.28
            dist_from_corner_y = -0.02
            x_val = 0.5 + self.REF_CORNER_X - dist_from_corner_x
            y_val = self.REF_CORNER_Y - dist_from_corner_y
            x_val = -0.15882
            y_val = 0.61577 - 0.05 + 0.01 + 0.01 + 0.02
            yaw_val = 0.005
            yaw_range = (-np.pi / 12, np.pi / 12)
            x_range = (-0.1, 0.1)
            y_range = (-0.1, 0.1)
            RobotPoseRandomizer.set_pose(
                self,
                (x_val + x_range[0], x_val + x_range[1]),
                (y_val + y_range[0], y_val + y_range[1]),
                (yaw_val + yaw_range[0], yaw_val + yaw_range[1]),
            )


class LMTargetNavRandPnPBottleToPlateDigitalTwin(LMTargetPnPBottleToPlateStaticDigitalTwin):
    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            x_val = -0.15882 - 0.3
            y_val = 0.61577 - 0.05 + 0.01 + 0.01 + 0.02
            yaw_val = 0.005
            yaw_range = (-np.pi / 10, np.pi / 10)
            x_range = (-0.3, 0.3)
            y_range = (-0.45, 0.45)
            RobotPoseRandomizer.set_pose(
                self,
                (x_val + x_range[0], x_val + x_range[1]),
                (y_val + y_range[0], y_val + y_range[1]),
                (yaw_val + yaw_range[0], yaw_val + yaw_range[1]),
            )


class LMTargetPnPBottleShelfToTable(LMTargetPnPBottle):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 0.6, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/lab_shelf/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                    reference_pos=np.array([0.8, -0.4, 0]),
                ),
            )
        )
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.1, -0.06]),
            y_range=np.array([-0.08, 0.08]),
            reference=ReferenceConfig(self.shelf, spawn_id=2),
        )
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.3, -0.28]),
            y_range=np.array([-0.08, 0.08]),
            reference=ReferenceConfig(self.table),
        )
        return [self.shelf, self.table, self.bottle, self.source, self.target]


class LMTargetPnPTwoBottles(LMTargetPnPBottle):
    BOTTLE_COLOURS = [(0.3, 0.7, 0.8, 1.0), (0.8, 0.4, 0.3, 1.0)]

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.source_1 = TargetZoneFactory.build(
            name="source_1",
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.14, -0.13]),
            y_range=np.array([-0.1, -0.08]),
            reference=ReferenceConfig(self.table),
            rgba=self.BOTTLE_COLOURS[0],
        )
        self.source_2 = TargetZoneFactory.build(
            name="source_2",
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.14, -0.13]),
            y_range=np.array([0.1, 0.08]),
            reference=ReferenceConfig(self.table),
            rgba=self.BOTTLE_COLOURS[1],
        )
        self.target_1 = TargetZoneFactory.build(
            name="target_1",
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.22, -0.2]),
            y_range=np.array([-0.22, -0.18]),
            reference=ReferenceConfig(self.table_target),
            rgba=self.BOTTLE_COLOURS[0],
        )
        self.target_2 = TargetZoneFactory.build(
            name="target_2",
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.22, -0.2]),
            y_range=np.array([0.18, 0.22]),
            reference=ReferenceConfig(self.table_target),
            rgba=self.BOTTLE_COLOURS[1],
        )
        self.bottle_1 = SceneObject(
            ObjectConfig(
                name="bottle_1",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(self.source_1),
                ),
                rgba=self.BOTTLE_COLOURS[0],
            )
        )
        self.bottle_2 = SceneObject(
            ObjectConfig(
                name="bottle_2",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    # rotation=np.array([-np.pi, np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(self.source_2),
                ),
                rgba=self.BOTTLE_COLOURS[1],
            )
        )
        return [
            self.table,
            self.table_target,
            self.bottle_1,
            self.bottle_2,
            self.source_1,
            self.source_2,
            self.target_1,
            self.target_2,
        ]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.bottle_1, symmetric=True),
            IsInContact(self.bottle_1, self.target_1),
            IsUpright(self.bottle_2, symmetric=True),
            IsInContact(self.bottle_2, self.target_2),
        )

    def _get_instruction(self) -> str:
        return "Pick up bottles and place them in the target zones."

    def get_object(self):
        return dict(
            bottle_1=dict(obj_name=self.bottle_1.mj_obj.root_body, obj_type="body"),
            bottle_2=dict(obj_name=self.bottle_2.mj_obj.root_body, obj_type="body"),
            target_1=dict(obj_name=self.target_1.mj_obj.root_body, obj_type="body"),
            target_2=dict(obj_name=self.target_2.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = {}
        for i, (bottle, source) in enumerate(
            [(self.bottle_1, self.source_1), (self.bottle_2, self.source_2)]
        ):
            bottle_xpos = self.sim.data.body_xpos[self.obj_body_id(bottle.mj_obj.name)]
            source_xpos = self.sim.data.body_xpos[self.obj_body_id(source.mj_obj.name)]
            distance = np.linalg.norm(bottle_xpos - source_xpos)
            is_close = distance <= self.SOURCE_OFFSET
            in_contact = self.check_contact(bottle, source)
            signals[f"obj_off_source_{i}"] = int(not in_contact) and (not is_close)
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # Pick bottles
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle_1",
            subtask_term_signal="obj_off_source_1",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref="bottle_2",
            subtask_term_signal="obj_off_source_2",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Place bottles
        task.task_spec_0.subtask_2 = dict(
            object_ref="target_1",
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
            object_ref="target_2",
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


class LMTargetPnPBottleStaticSimpleBottle(LMTargetPnPBottleStatic):
    """Replaces the objaverse bottle with a simple cylinder bottle."""

    def _get_objects(self) -> list[SceneObject]:
        [self.table, self.bottle, self.source, self.target] = super()._get_objects()
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/aigc/cylinder/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    rotation=np.array([np.pi, np.pi]),
                    reference=ReferenceConfig(self.source),
                ),
            )
        )
        return [self.table, self.bottle, self.source, self.target]


class LMTargetPnPBottleStaticSimpleBottleDemo(LMTargetPnPBottleStaticSimpleBottle):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.16, -0.15]),
            y_range=np.array([-0.1, -0.08]),
            reference=ReferenceConfig(self.table),
        )
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.16, -0.15]),
            y_range=np.array([0.03, 0.05]),
            reference=ReferenceConfig(self.table),
        )
        return [self.table, self.bottle, self.source, self.target]

    def _get_instruction(self) -> str:
        return "Pick up the bottle and place it in the target zone."


class LMTargetPnPTwoBottlesStatic(LMTargetPnPTwoBottles):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.target_1 = TargetZoneFactory.build(
            name="target_1",
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.22, -0.2]),
            y_range=np.array([-0.22, -0.18]),
            reference=ReferenceConfig(self.table),
            rgba=self.BOTTLE_COLOURS[0],
        )
        self.target_2 = TargetZoneFactory.build(
            name="target_2",
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.22, -0.2]),
            y_range=np.array([0.18, 0.22]),
            reference=ReferenceConfig(self.table),
            rgba=self.BOTTLE_COLOURS[1],
        )
        return [
            self.table,
            self.bottle_1,
            self.bottle_2,
            self.source_1,
            self.source_2,
            self.target_1,
            self.target_2,
        ]


class LMTargetPnPBottleTableToTable(LMTargetPnPBottleShelfToTable):
    # arena type should be factory arena
    MUJOCO_ARENA_CLS = FactoryArena

    def _get_objects(self) -> list[SceneObject]:
        # Call the parent's parent to get the base bottle setup
        super(LMTargetPnPBottleShelfToTable, self)._get_objects()

        # Create table_origin using the same asset as LMBottlePnP
        self.table_origin = SceneObject(
            ObjectConfig(
                name="table_origin",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.2, -0.8, -0.08]),  # lower to avoid hand collision
                    rotation=np.array([np.pi, np.pi]),
                ),
            )
        )

        # Create table_target using the same asset as LMBottlePnP
        self.table_target = SceneObject(
            ObjectConfig(
                name="table_target",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([0.1 - 0.05, 0.1 + 0.05]),
                    y_range=np.array([-0.02 - 0.05, 0.02 + 0.05]),
                    # reference_pos=np.array([1.5, 2.0, -0.08]),
                    reference_pos=np.array([1.2, 1.0, -0.08]),
                    rotation=np.array([np.pi, np.pi]),
                ),
            )
        )

        # Create source zone on the origin table
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-0.4 - 0.05, -0.4 + 0.05]),
            # y_range=np.array([-0.4, 0.4]),
            # move to the left of the table
            y_range=np.array([0.5 - 0.05, 0.5 + 0.05]),
            reference=ReferenceConfig(self.table_origin),
            rgba=(1.0, 0.0, 0.0, 0.0),
        )

        # Create target zone on the target table
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            x_range=np.array([-0.35 - 0.05, -0.35 + 0.05]),
            # y_range=np.array([-0.4, 0.4]),
            # move to the right of the table
            y_range=np.array([-0.5 - 0.05, -0.5 + 0.05]),
            reference=ReferenceConfig(self.table_target),
            rgba=(1.0, 0.0, 0.0, 0.0),
        )

        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    # rotation=np.array([-np.pi, -np.pi]),
                    rotation=np.array([0.0, 0.0]),
                    reference=ReferenceConfig(self.source),
                ),
            ),
        )
        return [self.table_origin, self.table_target, self.bottle, self.source, self.target]

    def _get_instruction(self) -> str:
        return (
            "Pick up the bottle from one table and place it in the target zone on the other table."
        )

    # update success criteria to be in contact with the target zone
    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            AnyCriteria(
                IsInContact(self.bottle, self.table_target), IsInContact(self.bottle, self.target)
            ),
            IsUpright(self.bottle, symmetric=True),
        )


class LMTargetNavPnPPickBoxFloorTableDT(LMSimpleEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)
    MUJOCO_ARENA_CLS = LabArena
    REF_CORNER_X = -0.6  # centering point for the box's reset
    REF_CORNER_Y = 0.0
    TABLE_HEIGHT = 0.68

    TABLE_GRADIENT: Gradient = Gradient(
        np.array([0.68, 0.34, 0.07, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    )
    LIFT_OFFSET = 0.2

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.5, 0, 0.014]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([self.REF_CORNER_X - 0.25, self.REF_CORNER_X + 0.25]),
            y_range=np.array([self.REF_CORNER_Y - 0.25, self.REF_CORNER_Y + 0.25]),
            z_offset=-self.TABLE_HEIGHT,
            reference=ReferenceConfig(self.table),
            rgba=(0.3, 0.3, 0.3, 0.0),
        )

        # Create box on floor using the small_box model
        self.box = SceneObject(
            ObjectConfig(
                name="box",
                mjcf_path="objects/aigc/small_box/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    rotation=np.array([-np.pi / 4 + np.pi / 2, np.pi / 4 + np.pi / 2]),
                    reference=ReferenceConfig(obj=self.source),
                    z_offset=0.09,  # Half height of box to sit on floor
                ),
            )
        )

        # TODO? ideally make the target go in a straight line from the source ...
        # # Create target zone on table
        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            scale=0.2,
            x_range=np.array([-0.3, -0.25]),
            y_range=np.array([-0.1, 0.1]),
            reference=ReferenceConfig(obj=self.table),
            rgba=(0.3, 0.3, 0.3, 0.0),
            name="target_table",
        )
        return [self.table, self.box, self.source, self.target]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.box, symmetric=True),
            AnyCriteria(IsInContact(self.box, self.target), IsInContact(self.box, self.table)),
        )

    def _get_instruction(self) -> str:
        return "Pick up the box from the floor and place it in the target zone on the table."

    def get_object(self):
        return dict(
            box=dict(obj_name=self.box.mj_obj.root_body, obj_type="body"),
            target=dict(obj_name=self.target.mj_obj.root_body, obj_type="body"),
            table=dict(obj_name=self.table.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Returns signals for subtask termination.
        obj_off_floor: Box has been lifted off the floor
        """
        return {}

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            # Randomize robot starting position - far from the box
            x_range = (-1.0, -0.5)
            y_range = (-0.25, 0.25)
            yaw_range = (-np.pi / 6, np.pi / 6)
            RobotPoseRandomizer.set_pose(
                self,
                x_range=x_range,  # Random x position
                y_range=y_range,  # Behind the box
                yaw_range=yaw_range,  # Random orientation
            )

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        # Subtask 1: Pick up the box from the floor
        task.task_spec_0.subtask_1 = dict(
            object_ref="box",
            subtask_term_signal="obj_off_floor",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Subtask 2: Place the box on the target zone
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
        # For dual-arm robots (left arm can also do the task)
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()


class LMTargetNavPnPPickBoxFloorTableDTStatic(LMTargetNavPnPPickBoxFloorTableDT):
    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        self.source.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([self.REF_CORNER_X, self.REF_CORNER_X]),
                y_range=np.array([self.REF_CORNER_Y - 0.3, self.REF_CORNER_Y - 0.3]),
                z_offset=0.4,
                reference=ReferenceConfig(self.table, on_top=False),
            )
        )
        self.target.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.array([-0.3, -0.25]),
                y_range=np.array([-0.1, 0.1]),
                reference=ReferenceConfig(obj=self.table),
            )
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            # Randomize robot starting position - far from the box
            x_range = (-1.0, -0.5)
            y_range = (-0.25, 0.25)
            yaw_range = (-np.pi / 6 + np.pi / 18, np.pi / 6 + np.pi / 18)

            x_range = (x_range[0] + x_range[1]) / 2, (x_range[0] + x_range[1]) / 2
            y_range = (y_range[0] + y_range[1]) / 2, (y_range[0] + y_range[1]) / 2
            yaw_range = (yaw_range[0] + yaw_range[1]) / 2, (yaw_range[0] + yaw_range[1]) / 2
            RobotPoseRandomizer.set_pose(
                self,
                x_range=x_range,  # Random x position
                y_range=y_range,  # Behind the box
                yaw_range=yaw_range,  # Random orientation
            )

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.box, symmetric=True),
            IsInContact(self.box, self.target),
        )


class LMTargetNavBoxPlatformToTableDT(LMTargetNavPnPPickBoxFloorTableDT):
    MUJOCO_ARENA_CLS = LabArenaPlane
    PLATFORM_X = 0.158937
    PLATFORM_Y = 0.210934

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.37 + 0.371296, 0.0, 0.0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )

        self.platform = SceneObject(
            ObjectConfig(
                name="platform",
                mjcf_path="objects/omniverse/locomanip/tote_f01/model.xml",
                static=True,
                scale=(1.0, 1.0, 1.556),
                sampler_config=SamplingConfig(
                    reference_pos=np.array([0.0, -0.22 - self.PLATFORM_Y, 0.0]),
                ),
            )
        )

        self.source = TargetZoneFactory.build(
            target_type=TargetZoneType.SOURCE,
            x_range=np.array([-self.PLATFORM_X + 0.1, 0]),
            y_range=np.array([self.PLATFORM_Y - 0.1, self.PLATFORM_Y - 0.15]),
            reference=ReferenceConfig(self.platform, on_top=True),
            rgba=(0.3, 0.3, 0.3, 0.0),
        )

        self.box = SceneObject(
            ObjectConfig(
                name="box",
                mjcf_path="objects/aigc/small_box/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    rotation=np.array([-np.pi / 4 + np.pi / 2, np.pi / 4 + np.pi / 2]),
                    reference=ReferenceConfig(obj=self.source, on_top=False),
                    z_offset=0.09,
                ),
            )
        )

        self.target = TargetZoneFactory.build(
            target_type=TargetZoneType.TARGET,
            scale=0.15,
            x_range=np.array([-0.3, -0.25]),
            y_range=np.array([-0.1, 0.1]),
            reference=ReferenceConfig(obj=self.table),
            rgba=(0.3, 0.3, 0.3, 1.0),
            name="target_table",
        )

        return [self.table, self.platform, self.box, self.source, self.target]

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            x_range = (-0.52 - 0.05, -0.52 + 0.05)
            y_range = (-0.05, 0.05)
            yaw_range = (-np.pi / 6 + np.pi / 18, np.pi / 6 + np.pi / 18)
            RobotPoseRandomizer.set_pose(self, x_range, y_range, yaw_range)

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.box, symmetric=True),
            AnyCriteria(IsInContact(self.box, self.target), IsInContact(self.box, self.table)),
        )


class LMTargetNavBoxPlatformToTableStaticDT(LMTargetNavBoxPlatformToTableDT):
    def _get_objects(self) -> list[SceneObject]:
        objects = super()._get_objects()
        self.table.update_cfg(
            sampler_config=SamplingConfig(
                reference_pos=np.array([self.PLATFORM_X + 0.371296, 0.0, 0.0]),
                rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
            ),
        )
        self.source.update_cfg(
            sampler_config=SamplingConfig(
                x_range=np.zeros(2),
                y_range=np.array([self.PLATFORM_Y - 0.1, self.PLATFORM_Y - 0.15]),
                reference=ReferenceConfig(self.platform, on_top=True),
            ),
        )
        return objects

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            x_range = (-0.2, -0.2)
            y_range = (-0.01, 0.01)
            yaw_range = np.zeros(2)
            RobotPoseRandomizer.set_pose(self, x_range, y_range, yaw_range)
