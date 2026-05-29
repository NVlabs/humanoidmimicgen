from robocasa.environments.kitchen.kitchen import *
from robocasa.utils.dexmg_utils import DexMGConfigHelper


class ManipulateDrawer(Kitchen):
    """
    Class encapsulating the atomic manipulate drawer tasks.

    Args:
        behavior (str): "open" or "close". Used to define the desired
            drawer manipulation behavior for the task.

        drawer_id (str): The drawer fixture id to manipulate
    """

    def __init__(self, behavior="open", drawer_id=FixtureType.TOP_DRAWER, *args, **kwargs):
        # for open and close tasks the robot will next to the drawer.
        # since for open tasks all drawers will be close it is ambigious which drawer to open,
        # so we specify a which side the drawer is on to open. The side is determined in _load_model
        self.robot_side = ""
        self.drawer_id = drawer_id
        assert behavior in ["open", "close"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        robot_model = self.robots[0].robot_model
        x_ofs = (self.drawer.width / 2) + 0.3
        inits = []

        # compute where the robot placement if it is to the left of the drawer
        (
            robot_base_pos_left,
            robot_base_ori_left,
        ) = self.compute_robot_base_placement_pose(ref_fixture=self.drawer, offset=(-x_ofs, -0.23))
        # get a test point to check if the robot is in contact with any fixture.
        test_pos_left, _ = self.compute_robot_base_placement_pose(
            ref_fixture=self.drawer, offset=(-x_ofs - 0.3, -0.23)
        )

        # check if the robot will be in contact with any fixture or wall during initialization
        if not self.check_fxtr_contact(test_pos_left) and not self.check_sidewall_contact(
            test_pos_left
        ):
            # drawer is to the right of the robot
            inits.append((robot_base_pos_left, robot_base_ori_left, "right"))

        # compute where the robot placement if it is to the right of the drawer
        (
            robot_base_pos_right,
            robot_base_ori_right,
        ) = self.compute_robot_base_placement_pose(ref_fixture=self.drawer, offset=(x_ofs, -0.23))
        # get a test point to check if the robot is in contact with any fixture if initialized to the right of the drawer
        test_pos_right, _ = self.compute_robot_base_placement_pose(
            ref_fixture=self.drawer, offset=(x_ofs + 0.3, -0.23)
        )

        if not self.check_fxtr_contact(test_pos_right) and not self.check_sidewall_contact(
            test_pos_right
        ):
            inits.append((robot_base_pos_right, robot_base_ori_right, "left"))

        assert len(inits) > 0
        random_index = self.rng.integers(len(inits))
        robot_base_pos, robot_base_ori, side = inits[random_index]
        self.drawer_side = side
        robot_model.set_base_xpos(robot_base_pos)
        robot_model.set_base_ori(robot_base_ori)

    def _reset_internal(self):
        """
        Reset the environment internal state for the drawer tasks.
        This includes setting the drawer state based on the behavior
        """
        if self.behavior == "open":
            self.drawer.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        elif self.behavior == "close":
            self.drawer.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        # set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened
        super()._reset_internal()

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the drawer tasks
        """
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref("drawer", dict(id=self.drawer_id))
        self.init_robot_base_pos = self.drawer

    def get_ep_meta(self):
        """
        Get the episode metadata for the drawer tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"{self.behavior} the {self.drawer_side} drawer"
        return ep_meta

    def check_fxtr_contact(self, pos):
        """
        Check if the point is in contact with any fixture

        Args:
            pos (tuple): The position of the point to check

        Returns:
            bool: True if the point is in contact with any fixture, False otherwise
        """
        fxtrs = [
            fxtr
            for fxtr in self.fixtures.values()
            if isinstance(fxtr, Counter)
            or isinstance(fxtr, Stove)
            or isinstance(fxtr, Stovetop)
            or isinstance(fxtr, HousingCabinet)
            or isinstance(fxtr, SingleCabinet)
            or isinstance(fxtr, HingeCabinet)
        ]

        for fxtr in fxtrs:
            # get bounds of fixture
            if OU.point_in_fixture(point=pos, fixture=fxtr, only_2d=True):
                return True
        return False

    def check_sidewall_contact(self, pos):
        """
        Check if the point is in contact with any wall.
        This is separate from check_fxtr_contact because the point could be outside of the wall's
        bounds but still in contact with the wall since the wall is thin.

        Args:
            pos (tuple): The position of the point to check

        Returns:
            bool: True if the point is in contact with any wall, False otherwise
        """

        walls = [fxtr for (name, fxtr) in self.fixtures.items() if isinstance(fxtr, Wall)]
        for wall in walls:
            if wall.wall_side == "right" and pos[0] > wall.pos[0]:
                return True
            # terrible hack change later we only want to check if its outside the main left wall not a secondary left wall
            if wall.wall_side == "left" and "2" not in wall.name and pos[0] < wall.pos[0]:
                return True
            if wall.wall_side == "back" and pos[1] > wall.pos[1]:
                return True
        return False

    def _check_success(self):
        """
        Check if the drawer manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        door_state = self.drawer.get_door_state(env=self)

        success = True
        for joint_p in door_state.values():
            if self.behavior == "open":
                if joint_p < 0.95:
                    success = False
                    break
            elif self.behavior == "close":
                if joint_p > 0.05:
                    success = False
                    break

        return success


class OpenDrawer(ManipulateDrawer, DexMGConfigHelper):
    """
    Class encapsulating the atomic open drawer task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="open", *args, **kwargs)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer tasks. This includes the object placement configurations.
        Place the object inside the drawer and 1-4 distractors on the counter.

        Returns:
            list: List of object configurations.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="drawer_obj",
                obj_groups="all",
                graspable=True,
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, -0.75),
                ),
            )
        )

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i+1}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.drawer),
                        sample_region_kwargs=dict(
                            ref=self.drawer,
                        ),
                        size=(1.0, 0.50),
                        pos=(None, -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )

        return cfgs

    def get_object(self):
        return dict(
            handle=dict(obj_name=self.drawer.handle_name, obj_type="geom"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        contact_handle = self.check_contact(
            self.robots[0].gripper["right"],
            self.drawer.handle_name,
        )
        signals["stage_contact_handle"] = int(contact_handle)
        signals["success"] = int(self._check_success())
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec.subtask_1 = dict(
            object_ref="handle",
            subtask_term_signal="stage_contact_handle",
            subtask_term_offset_range=(5, 10),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        task.task_spec.subtask_2 = dict(
            object_ref="handle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,  # (10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        return task.to_dict()


class CloseDrawer(ManipulateDrawer, DexMGConfigHelper):
    """
    Class encapsulating the atomic close drawer task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer tasks. This includes the object placement configurations.
        Place the object inside the drawer and 1-4 distractors on the counter.

        Returns:
            list: List of object configurations.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="drawer_obj",
                obj_groups="all",
                graspable=True,
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, -0.75),
                    offset=(0, -self.drawer.size[1] * 0.55),
                ),
            )
        )

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i+1}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.drawer),
                        sample_region_kwargs=dict(
                            ref=self.drawer,
                        ),
                        size=(1.0, 0.50),
                        pos=(None, -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )

        return cfgs

    def get_object(self):
        return dict(
            handle=dict(obj_name=self.drawer.handle_name, obj_type="geom"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        drawer_geoms = find_elements(self.drawer.worldbody, tags="geom", return_first=False)
        drawer_geom_names = [e.get("name") for e in drawer_geoms]
        contact_drawer = self.check_contact(
            self.robots[0].gripper["right"],
            drawer_geom_names,
        )
        signals["stage_contact_drawer"] = int(contact_drawer)
        signals["success"] = int(self._check_success())
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec.subtask_1 = dict(
            object_ref="handle",
            subtask_term_signal="stage_contact_drawer",
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        task.task_spec.subtask_2 = dict(
            object_ref="handle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,  # (10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        return task.to_dict()
