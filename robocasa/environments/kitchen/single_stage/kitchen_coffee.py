import numpy as np

from robocasa.environments.kitchen.kitchen import *
from robocasa.utils.dexmg_utils import DexMGConfigHelper


class PnPCoffee(Kitchen):
    """
    Class encapsulating the atomic pick and place coffee tasks.

    Args:
        behavior (str): "counter_to_machine" or "machine_to_counter". Used to define the desired
            pick and place behavior for the task.
    """

    def __init__(self, behavior="machine_to_counter", *args, **kwargs):
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the coffee tasks. (Coffee machine and counter)
        """
        super()._setup_kitchen_references()
        self.coffee_machine = self.get_fixture("coffee_machine")
        self.counter = self.get_fixture(FixtureType.COUNTER, ref=self.coffee_machine)
        self.init_robot_base_pos = self.coffee_machine

    def get_ep_meta(self):
        """
        Get the episode metadata for the coffee tasks.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        if self.behavior == "counter_to_machine":
            ep_meta["lang"] = (
                f"pick the {obj_lang} from the counter and place it under the coffee machine dispenser"
            )
        elif self.behavior == "machine_to_counter":
            ep_meta["lang"] = (
                f"pick the {obj_lang} from under the coffee machine dispenser and place it on the counter"
            )
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the coffee tasks. This includes the object placement configurations.
        Place the mug on the counter or under the coffee machine dispenser based on the behavior.

        Returns:
            list: List of object configurations.
        """
        cfgs = []
        if self.behavior == "counter_to_machine":
            cfgs.append(
                dict(
                    name="obj",
                    obj_groups="mug",
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.coffee_machine,
                        ),
                        size=(0.30, 0.40),
                        pos=("ref", -1.0),
                        rotation=[np.pi / 4, np.pi / 2],
                    ),
                )
            )
        elif self.behavior == "machine_to_counter":
            cfgs.append(
                dict(
                    name="obj",
                    obj_groups="mug",
                    placement=dict(
                        fixture=self.coffee_machine,
                        ensure_object_boundary_in_range=False,
                        margin=0.0,
                        ensure_valid_placement=False,
                        rotation=(np.pi / 8, np.pi / 4),
                    ),
                )
            )
        else:
            raise NotImplementedError

        return cfgs

    def _check_success(self):
        """
        Check if the coffee task is successful.
        This includes checking if the gripper is far from the object and the object is in corretly placed
        on the desired fixture (counter or coffee machine).
        """
        gripper_obj_far = OU.gripper_obj_far(self)

        if self.behavior == "counter_to_machine":
            contact_check = self.coffee_machine.check_receptacle_placement_for_pouring(self, "obj")
        elif self.behavior == "machine_to_counter":
            contact_check = OU.check_obj_fixture_contact(self, "obj", self.counter)
        return contact_check and gripper_obj_far


class CoffeeSetupMug(PnPCoffee, DexMGConfigHelper):
    """
    Class encapsulating the coffee setup task. Pick the mug from the counter and place it under the coffee machine dispenser.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="counter_to_machine", *args, **kwargs)

    def get_object(self):
        return dict(
            obj=dict(obj_name=self.objects["obj"].root_body, obj_type="body"),
            coffee_machine=dict(
                obj_name="{}_{}".format(self.coffee_machine.name, "receptacle_place_site"),
                obj_type="site",
            ),
        )

    def get_subtask_term_signals(self):
        signals = dict()

        contact_obj = self.check_contact(
            self.robots[0].gripper["right"],
            self.objects["obj"],
        )
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self._check_success())

        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="stage_contact_obj",
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        task.task_spec.subtask_2 = dict(
            object_ref="coffee_machine",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        return task.to_dict()


class CoffeeServeMug(PnPCoffee, DexMGConfigHelper):
    """
    Class encapsulating the coffee serve task. Pick the mug from under the coffee machine dispenser and place it on the counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="machine_to_counter", *args, **kwargs)

    def get_object(self):
        return dict(
            obj=dict(obj_name=self.objects["obj"].root_body, obj_type="body"),
            coffee_machine=dict(
                obj_name="{}_{}".format(self.coffee_machine.name, "receptacle_place_site"),
                obj_type="site",
            ),
        )

    def get_subtask_term_signals(self):
        signals = dict()

        contact_obj = self.check_contact(
            self.robots[0].gripper["right"],
            self.objects["obj"],
        )
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self._check_success())

        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="stage_contact_obj",
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        task.task_spec.subtask_2 = dict(
            object_ref="coffee_machine",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        return task.to_dict()


class CoffeePressButton(Kitchen, DexMGConfigHelper):
    """
    Class encapsulating the coffee press button task. Press the button on the coffee machine to serve coffee.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the coffee press button task. (Coffee machine and counter the coffee machine is on)
        """
        super()._setup_kitchen_references()
        self.coffee_machine = self.get_fixture("coffee_machine")
        self.counter = self.get_fixture(FixtureType.COUNTER, ref=self.coffee_machine)
        self.init_robot_base_pos = self.coffee_machine

    def get_ep_meta(self):
        """
        Get the episode metadata for the coffee press button task.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "press the button on the coffee machine to serve coffee"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the coffee press button task. This includes the object placement configurations.
        Places the mug under the coffee machine dispenser.

        Returns:
            list: List of object configurations.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="mug",
                placement=dict(
                    fixture=self.coffee_machine,
                    ensure_object_boundary_in_range=False,
                    margin=0.0,
                    ensure_valid_placement=False,
                    rotation=(np.pi / 8, np.pi / 4),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the coffee press button task is successful.
        This includes checking if the gripper is far from the object and the coffee machine is turned on/button has been pressed.
        """
        gripper_button_far = self.coffee_machine.gripper_button_far(self)

        turned_on = self.coffee_machine.get_state()["turned_on"]
        return turned_on and gripper_button_far

    def get_object(self):
        return dict(
            button=dict(
                obj_name="{}_{}".format(self.coffee_machine.name, "start_button"),
                obj_type="geom",
            ),
        )

    def get_subtask_term_signals(self):
        signals = dict()

        signals["success"] = int(self._check_success())

        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec.subtask_1 = dict(
            object_ref="button",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        return task.to_dict()
