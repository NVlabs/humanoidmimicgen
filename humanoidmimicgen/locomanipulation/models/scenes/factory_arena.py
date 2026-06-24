# SPDX-FileCopyrightText: Copyright (c) 2024 The RoboCasa Team
# SPDX-License-Identifier: MIT

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
import humanoidmimicgen.locomanipulation as locomanipulation


class FactoryArena(Arena):
    """Factory workspace."""

    def __init__(self):
        super().__init__(
            xml_path_completion(
                "arenas/gear_factory/gear_factory.xml", root=locomanipulation.models.assets_root
            )
        )
