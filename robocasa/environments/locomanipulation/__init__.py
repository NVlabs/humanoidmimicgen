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

from .base import REGISTERED_LOCOMANIPULATION_ENVS, RETAINED_LOCOMANIPULATION_ENV_NAMES
from .locomanip_basic import *  # noqa: F401,F403
from .locomanip_pnp import *  # noqa: F401,F403
from .locomanip_push import *  # noqa: F401,F403
from .locomanip_simple import *  # noqa: F401,F403
from .locomanip_targets import *  # noqa: F401,F403

PAPER_LOCOMANIPULATION_ENVIRONMENTS = RETAINED_LOCOMANIPULATION_ENV_NAMES
ALL_LOCOMANIPULATION_ENVIRONMENTS = PAPER_LOCOMANIPULATION_ENVIRONMENTS
RETAINED_LOCOMANIPULATION_ENVIRONMENTS = PAPER_LOCOMANIPULATION_ENVIRONMENTS

_missing_envs = set(PAPER_LOCOMANIPULATION_ENVIRONMENTS) - set(REGISTERED_LOCOMANIPULATION_ENVS)
if _missing_envs:
    raise RuntimeError(f"Retained loco-manipulation envs were not registered: {sorted(_missing_envs)}")

__all__ = [
    "ALL_LOCOMANIPULATION_ENVIRONMENTS",
    "PAPER_LOCOMANIPULATION_ENVIRONMENTS",
    "RETAINED_LOCOMANIPULATION_ENVIRONMENTS",
    *PAPER_LOCOMANIPULATION_ENVIRONMENTS,
]
