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
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod


class DexMGConfigHelper:
    """
    Helper class for multi-inheritance scenarios, specifically designed to support task configuration
    and environment interaction in robotic manipulation tasks.

    Example Usage:
        This class is intended to be used in multi-inheritance cases such as:
            class ExampleLocoManipulationEnv(LocoManipulationEnv, DexMGConfigHelper)
        which will automatically generate a new configuration class:
            class PnPCounterToSink_Config(MG_Config)

    Behavior:
    1. In the implementation of the MG_RoboSuiteHumanoidGeneric class:
        - `self.env` will be an instance of PnPCounterToSink, enabling access to:
            - `self.env.get_object()` to retrieve key objects.
            - `self.env.get_subtask_term_signals()` to obtain signal information.
    2. In the implementation of PnPCounterToSink_Config:
        - `self.task_config()` delegates to `PnPCounterToSink.task_config`, which defines
          subtask divisions based on objects and signals.

    Attributes:
        subclasses (list): A list storing tuples of subclass names and their respective classes.
    """

    class AttrDict(dict):
        def __getattr__(self, key):
            if key not in self:
                self[key] = (
                    DexMGConfigHelper.AttrDict()
                )  # Create a new AttrDict if key doesn't exist
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

        def to_dict(self):
            """Recursively converts AttrDict to a normal dictionary"""
            return {
                key: (value.to_dict() if isinstance(value, DexMGConfigHelper.AttrDict) else value)
                for key, value in self.items()
            }

    subclasses = []

    def __init_subclass__(cls, **kwargs):
        """
        Automatically registers each subclass of DexMGConfigHelper.

        Args:
            cls: The subclass being registered.
            kwargs: Additional keyword arguments.
        """
        super().__init_subclass__(**kwargs)
        DexMGConfigHelper.subclasses.append((cls.__name__, cls))

    def __init__(self):
        """
        Initialize a DexMGConfigHelper instance.
        Note: This is an abstract class and should not be instantiated directly.
        """
        pass

    # Helper methods used by mimicgen-compatible robosuite humanoid interfaces.
    def get_grippers(self):
        """
        Get grippers for the environment.
        """
        return self.robots[0].gripper["right"], self.robots[0].gripper["left"]

    @abstractmethod
    def get_object(self):
        """
        Retrieve a key object required for the task.
        This method must be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_subtask_term_signals(self):
        """
        Retrieve signals used to define subtask termination conditions.
        This method must be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def task_config():
        """
        Define the configuration for dividing a task into subtasks.
        This method must be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError
