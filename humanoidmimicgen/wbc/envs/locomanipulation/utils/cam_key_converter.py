# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from humanoidmimicgen.wbc_constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH


@dataclass
class CameraConfig:
    width: int
    height: int
    mapped_key: str


class CameraKeyMapper:
    def __init__(self):
        # Default camera dimensions
        self.default_width = RS_VIEW_CAMERA_WIDTH
        self.default_height = RS_VIEW_CAMERA_HEIGHT

        # Camera key mapping with custom dimensions
        self.camera_configs: Dict[str, CameraConfig] = {
            "frontview": CameraConfig(self.default_width, self.default_height, "front_view"),
            "robot0_rs_egoview": CameraConfig(self.default_width, self.default_height, "ego_view"),
            "robot0_rs_tppview": CameraConfig(self.default_width, self.default_height, "tpp_view"),
            "robot0_oak_egoview": CameraConfig(self.default_width, self.default_height, "ego_view"),
            "robot0_oak_left_monoview": CameraConfig(
                self.default_width, self.default_height, "ego_view_left_mono"
            ),
            "robot0_oak_right_monoview": CameraConfig(
                self.default_width, self.default_height, "ego_view_right_mono"
            ),
            "robot0_left_eef_view": CameraConfig(
                self.default_width, self.default_height, "ego_view_left_eef"
            ),
            "robot0_right_eef_view": CameraConfig(
                self.default_width, self.default_height, "ego_view_right_eef"
            ),
        }

    def get_camera_config(self, key: str) -> Optional[Tuple[str, int, int]]:
        """
        Get the mapped camera key and dimensions for a given camera key.

        Args:
            key: The input camera key

        Returns:
            Tuple of (mapped_key, width, height) if key exists, None otherwise
        """
        config = self.camera_configs.get(key.lower())
        if config is None:
            return None
        return config.mapped_key, config.width, config.height
