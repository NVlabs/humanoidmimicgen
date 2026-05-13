# SPDX-FileCopyrightText: Copyright (c) 2024 The RoboCasa Team
# SPDX-License-Identifier: MIT

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
import robocasa


class LabArena(Arena):
    """Lab workspace."""

    def __init__(self):
        super().__init__(
            xml_path_completion("arenas/gear_lab/gear_lab.xml", root=robocasa.models.assets_root)
        )


class LabArenaPlane(Arena):
    """Lab workspace. Plane only."""

    def __init__(self):
        super().__init__(
            xml_path_completion(
                "arenas/gear_lab/gear_lab_plane.xml", root=robocasa.models.assets_root
            )
        )
