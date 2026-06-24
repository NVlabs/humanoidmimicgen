# SPDX-FileCopyrightText: Copyright (c) 2024 The RoboCasa Team
# SPDX-License-Identifier: MIT

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion

import humanoidmimicgen.locomanipulation as locomanipulation


class GroundArena(Arena):
    """Empty workspace."""

    def __init__(self):
        super().__init__(
            xml_path_completion("arenas/ground_arena.xml", root=locomanipulation.models.assets_root)
        )
