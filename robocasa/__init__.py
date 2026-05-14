# SPDX-FileCopyrightText: Copyright (c) 2024 The RoboCasa Team
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""HumanoidMimicGen RoboCasa environment registration."""

import sys
import types

if "robosuite.macros_private" not in sys.modules:
    macros_private = types.ModuleType("robosuite.macros_private")
    macros_private.CACHE_NUMBA = False
    sys.modules["robosuite.macros_private"] = macros_private

from robosuite.environments.base import make

from . import models
from .environments.locomanipulation import *  # noqa: F401,F403
from .environments.locomanipulation import (
    ALL_LOCOMANIPULATION_ENVIRONMENTS,
    PAPER_LOCOMANIPULATION_ENVIRONMENTS,
    RETAINED_LOCOMANIPULATION_ENVIRONMENTS,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_LOCOMANIPULATION_ENVIRONMENTS",
    "PAPER_LOCOMANIPULATION_ENVIRONMENTS",
    "RETAINED_LOCOMANIPULATION_ENVIRONMENTS",
    "make",
    "models",
    *PAPER_LOCOMANIPULATION_ENVIRONMENTS,
]
