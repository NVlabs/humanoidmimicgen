# Copyright (c) 2024 the RoboCasa Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""HumanoidMimicGen loco-manipulation environment registration."""

import sys
import types

if "robosuite.macros_private" not in sys.modules:
    macros_private = types.ModuleType("robosuite.macros_private")
    macros_private.CACHE_NUMBA = False
    sys.modules["robosuite.macros_private"] = macros_private

from humanoidmimicgen.locomanipulation.utils.robosuite_compat import install_robosuite_compat

install_robosuite_compat()

from robosuite.environments.base import make

from . import models
from .envs import *  # noqa: F401,F403
from .envs import (
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
