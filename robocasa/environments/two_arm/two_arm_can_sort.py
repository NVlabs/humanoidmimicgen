import os

import numpy as np
import robosuite
from robosuite.models.objects import CylinderObject
from robosuite.models.objects.composite.bin import Bin
from robosuite.models.tasks import ManipulationTask

from robocasa.environments.two_arm.two_arm_dexmg_env import (
    TwoArmDexMGEnv,
)
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)


class TwoArmCanSortRandom(TwoArmDexMGEnv, DexMGConfigHelper):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 1.2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        red_prob=0.5,
        use_cylinder=True,
        *args,
        **kwargs,
    ):

        self.red_prob = red_prob
        self.is_red = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.9))
        self.use_cylinder = use_cylinder
        self.can_height = 0.18

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self._initialize_object_states()

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            *args,
            **kwargs,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # initialize objects of interest
        self.red_box = Bin(
            name="red_box",
            bin_size=(0.25, 0.2, 0.05),
            wall_thickness=0.01,
            transparent_walls=False,
            friction=None,
            density=100000.0,
            # use_texture=True,
            use_texture=False,
            rgba=(0.58, 0.15, 0.10, 1.0),
            # material=redwood,
        )

        self.blue_box = Bin(
            name="blue_box",
            bin_size=(0.25, 0.2, 0.05),
            wall_thickness=0.01,
            transparent_walls=False,
            friction=None,
            density=100000.0,
            # use_texture=True,
            use_texture=False,
            rgba=(0.53, 0.77, 0.95, 1.0),
            # material=bluewood,
        )

        if self.use_cylinder:
            if self.rng.random() < self.red_prob:
                self.is_red = True
                can_rgba = (0.58, 0.15, 0.10, 1.0)
            else:
                self.is_red = False
                can_rgba = (0.53, 0.77, 0.95, 1.0)
            self.can = CylinderObject(
                name="cube",
                # outer_radius=0.03,
                # inner_radius=0.025,
                # height=self.can_height/2,
                size=[0.03, self.can_height / 2],
                # ngeoms=64,
                rgba=can_rgba,
            )
        else:
            base_mjcf_path = os.path.join(robosuite.__path__[0], "models/assets/objects/objaverse/")
            from mimicgen.models.robosuite.objects import BlenderObject

            def _create_obj(cfg):
                print(f"Creating object {cfg['name']} with mjcf path {cfg['mjcf_path']}")
                object = BlenderObject(
                    name=cfg["name"],
                    mjcf_path=cfg["mjcf_path"],
                    scale=cfg["scale"],
                    solimp=(0.998, 0.998, 0.001),
                    solref=(0.001, 1),
                    density=100,
                    # friction=(0.95, 0.3, 0.1),
                    friction=(1, 1, 1),
                    margin=0.001,
                )
                return object

            # red
            cfg0 = {
                "name": "can",
                "mjcf_path": os.path.join(base_mjcf_path, "can_15/model.xml"),
                "scale": 1.5,
            }

            # blue
            cfg1 = {
                "name": "can",
                "mjcf_path": os.path.join(base_mjcf_path, "can_3/model.xml"),
                "scale": 1.5,
            }

            if self.rng.random() < self.red_prob:
                self.is_red = True
                self.can = _create_obj(cfg0)
            else:
                self.is_red = False
                self.can = _create_obj(cfg1)

        objects = [self.red_box, self.blue_box, self.can]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )
        self._modify_camera_view()

    def _modify_camera_view(self):
        # Modify the agentview camera to have a higher z-axis position
        self.model.mujoco_arena.set_camera(
            camera_name="agentview",
            pos=(-0.4, 0, 1.6),  # Increased z-axis from 1.35 to 1.8
            quat=(0.6818, 0.189, -0.189, -0.6818),
            camera_attribs={"fovy": "77"},
        )
        self.model.mujoco_arena.set_camera(
            camera_name="egoview",
            pos=(-0.4, 0, 1.6),  # Increased z-axis from 1.35 to 1.8
            quat=(0.6818, 0.189, -0.189, -0.6818),
            camera_attribs={"fovy": "77"},
        )

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler", rng=self.rng)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="RedBoxSampler",
                mujoco_objects=self.red_box,
                x_range=(-0.07, -0.07),
                y_range=(0.4, 0.4),
                rotation=(0.0, 0.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.0,
                rng=self.rng,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="BlueBoxSampler",
                mujoco_objects=self.blue_box,
                x_range=(-0.07, -0.07),
                y_range=(0.2, 0.2),
                rotation=(0.0, 0.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.001,
                rng=self.rng,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CanSampler",
                mujoco_objects=self.can,
                x_range=(-0.2, -0.1),
                # y_range=(-0.2, 0.0),
                y_range=(-0.1, 0.1),
                rotation=(0.0, 0.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
                rng=self.rng,
            )
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # TODO: replace

        # Additional object references from this env
        self.obj_body_id = dict(
            red_box=self.sim.model.body_name2id(self.red_box.root_body),
            blue_box=self.sim.model.body_name2id(self.blue_box.root_body),
            can=self.sim.model.body_name2id(self.can.root_body),
        )

    def _initialize_object_states(self):
        """
        Initialize object states (pose estimates that are observed once at start of episode).
        """
        self.initial_object_states = dict()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        return observables

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the object.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # TODO: replace object ref

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper["right"], target=self.can
            )

    def reward(self, action=None):
        """
        Reward function for the task.

        The sparse reward only consists of the threading component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            pass

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def get_ep_meta(self):
        """
        Get episode metadata.

        Returns:
            dict: dictionary of episode metadata
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "pick the red can and put it in the red box, pick the blue can and put it in the blue box"
        )
        return ep_meta

    def get_object(self):
        """
        Retrieve key objects required for the task.

        Returns:
            dict: Dictionary mapping object names to object information
        """
        objects = dict()
        objects["obj"] = dict(obj_name=self.can.root_body, obj_type="body", obj_joint=None)
        objects["red_box"] = dict(
            obj_name=self.red_box.root_body,
            obj_type="body",
            obj_joint=None,
        )
        objects["blue_box"] = dict(
            obj_name=self.blue_box.root_body,
            obj_type="body",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        """
        Retrieve signals used to define subtask termination conditions.

        Returns:
            dict: Dictionary mapping signal names to their current values
        """
        signals = dict()

        # Check if can is lifted off the ground
        can_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["can"]])
        ground_threshold = self.table_offset[2] + 0.05  # 5cm above table
        signals["obj_off_ground"] = int(can_pos[2] > ground_threshold)

        # Check if handover is happening (both grippers are near the can)
        right_gripper = self.robots[0].gripper["right"]
        left_gripper = self.robots[0].gripper["left"]

        # Get gripper positions
        right_gripper_pos = np.array(
            self.sim.data.site_xpos[
                self.sim.model.site_name2id(right_gripper.important_sites["grip_site"])
            ]
        )
        left_gripper_pos = np.array(
            self.sim.data.site_xpos[
                self.sim.model.site_name2id(left_gripper.important_sites["grip_site"])
            ]
        )

        # Distance between grippers and can
        right_to_can_dist = np.linalg.norm(right_gripper_pos - can_pos)
        left_to_can_dist = np.linalg.norm(left_gripper_pos - can_pos)

        # Handover is considered done when both grippers are close to the can
        handover_threshold = 0.1  # 10cm
        signals["handover_done"] = int(
            right_to_can_dist < handover_threshold and left_to_can_dist < handover_threshold
        )

        # Check if can is in the correct box (same as success check)
        signals["task_complete"] = int(self._check_success())

        return signals

    @staticmethod
    def task_config():
        """
        Define the configuration for dividing the task into subtasks.

        Returns:
            dict: Task configuration dictionary
        """
        task = DexMGConfigHelper.AttrDict()

        # Task spec for first arm (right arm - pick up)
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="obj_off_ground",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Handover subtask
        task.task_spec_0.subtask_2 = dict(
            object_ref="obj",
            subtask_term_signal="handover_done",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Place in correct box
        task.task_spec_0.subtask_3 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Task spec for second arm (left arm - assist and place)
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal="obj_off_ground",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Handover from right arm
        task.task_spec_1.subtask_2 = dict(
            object_ref="obj",
            subtask_term_signal="handover_done",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Place in correct box
        task.task_spec_1.subtask_3 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for coordinating between arms.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()

        # Constraint for concurrent handover between both arms
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
            coordination_scheme={
                "scheme": "replay",
                "pos_noise_scale": 0.0,
                "rot_noise_scale": 0.0,
            },
        )

        return task_constraint.to_dict()

    def _check_success(self):
        """
        Check success.
        """
        if self.is_red is None:
            return False
        # TODO: implement success check

        red_box_base_pos = np.array(
            self.sim.data.geom_xpos[self.sim.model.geom_name2id("red_box_base")]
        )
        blue_box_base_pos = np.array(
            self.sim.data.geom_xpos[self.sim.model.geom_name2id("blue_box_base")]
        )
        can_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["can"]])

        x_threshold = self.blue_box.bin_size[0] / 2
        y_threshold = self.blue_box.bin_size[1] / 2
        z_threshold = self.can_height / 2
        x_check_red = np.abs(red_box_base_pos[0] - can_pos[0]) < x_threshold
        x_check_blue = np.abs(blue_box_base_pos[0] - can_pos[0]) < x_threshold
        y_check_red = np.abs(red_box_base_pos[1] - can_pos[1]) < y_threshold
        y_check_blue = np.abs(blue_box_base_pos[1] - can_pos[1]) < y_threshold
        z_check_red = can_pos[2] - red_box_base_pos[2] < z_threshold
        z_check_blue = can_pos[2] - blue_box_base_pos[2] < z_threshold

        if self.is_red:
            return x_check_red and y_check_red and z_check_red
        else:
            return x_check_blue and y_check_blue and z_check_blue


class TwoArmCanSortRed(TwoArmCanSortRandom, DexMGConfigHelper):
    def __init__(self, **kwargs):
        super().__init__(red_prob=1.1, **kwargs)

    def get_ep_meta(self):
        """
        Get episode metadata.

        Returns:
            dict: dictionary of episode metadata
        """
        return {
            "lang": "pick the red can and put it in the red box",
        }

    @staticmethod
    def task_config():
        """
        Define the configuration for red can sorting task.

        Returns:
            dict: Task configuration dictionary
        """
        task = DexMGConfigHelper.AttrDict()

        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="obj_off_ground",
            subtask_term_offset_range=(5, 10),
            # avoiding the other subtasks increases success rate ... lol?
            # subtask_term_signal=None,
            # subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=20,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # hand over
        task.task_spec_0.subtask_2 = dict(
            object_ref="obj",
            subtask_term_signal="handover_done",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=20,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Move towards red box
        task.task_spec_0.subtask_3 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=20,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal="obj_off_ground",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=20,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # hand over
        task.task_spec_1.subtask_2 = dict(
            object_ref="obj",
            subtask_term_signal="handover_done",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=20,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # put lid on box
        task.task_spec_1.subtask_3 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=20,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Note: task_constraint is not supported by the mimicgen framework
        # The framework only supports task_spec, task_spec_0, and task_spec_1 keys

        # task.task_spec_0.do_not_lock_keys()
        # task.task_spec_1.do_not_lock_keys()
        print(f"task: {task}")
        return task.to_dict()

    def get_object(self):
        """
        Retrieve key objects required for the task.

        Returns:
            dict: Dictionary mapping object names to RobosuiteObject instances
        """
        objects = dict()
        objects["obj"] = dict(obj_name=self.can.root_body, obj_type="body", obj_joint=None)
        objects["red_box"] = dict(
            obj_name=self.red_box.root_body,
            obj_type="body",
            obj_joint=None,
        )
        objects["blue_box"] = dict(
            obj_name=self.blue_box.root_body,
            obj_type="body",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        obj_height = self.can.size[1]
        obj_z = self.get_object_pose(self.can.root_body, obj_type="body")[2, 3] - obj_height
        table_z = self.table_offset[2]
        th = 0.05

        signals["obj_off_ground"] = int(obj_z - table_z > th)

        grasping_g0 = self._check_grasp(gripper=self.get_grippers()[0], object_geoms=self.can)
        grasping_g1 = self._check_grasp(gripper=self.get_grippers()[1], object_geoms=self.can)

        signals["handover_done"] = int(not grasping_g0 and grasping_g1)

        return signals

    # todo: investigate how to correctly add task_constraint_config
    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for red can sorting coordination.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()

        # Constraint for concurrent handover between both arms during red can task
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
            coordination_scheme={
                "scheme": "replay",
                "pos_noise_scale": 0.0,
                "rot_noise_scale": 0.0,
            },
        )

        return task_constraint.to_dict()


class TwoArmCanSortBlue(TwoArmCanSortRandom, DexMGConfigHelper):
    def __init__(self, **kwargs):
        super().__init__(red_prob=-0.1, **kwargs)

    def get_ep_meta(self):
        """
        Get episode metadata.

        Returns:
            dict: dictionary of episode metadata
        """
        return {
            "lang": "pick the blue can and put it in the blue box",
        }

    @staticmethod
    def task_config():
        """
        Define the configuration for blue can sorting task.

        Returns:
            dict: Task configuration dictionary
        """
        task = DexMGConfigHelper.AttrDict()

        # Task spec for first arm (right arm - pick up blue can)
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="obj_off_ground",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Handover to left arm
        task.task_spec_0.subtask_2 = dict(
            object_ref="obj",
            subtask_term_signal="handover_done",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Move towards blue box
        task.task_spec_0.subtask_3 = dict(
            object_ref="blue_box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Task spec for second arm (left arm - assist and place in blue box)
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal="obj_off_ground",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Receive handover from right arm
        task.task_spec_1.subtask_2 = dict(
            object_ref="obj",
            subtask_term_signal="handover_done",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Place blue can in blue box
        task.task_spec_1.subtask_3 = dict(
            object_ref="blue_box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        return task.to_dict()

    @staticmethod
    def task_constraint_config():
        """
        Define task constraints for blue can sorting coordination.

        Returns:
            dict: Task constraint configuration dictionary
        """
        task_constraint = DexMGConfigHelper.AttrDict()

        # Constraint for concurrent handover between both arms during blue can task
        task_constraint.constraint_1 = dict(
            subtasks=[("task_spec_0", "subtask_2"), ("task_spec_1", "subtask_2")],
            constraint_type="temporal_concurrent",
            coordination_scheme={
                "scheme": "replay",
                "pos_noise_scale": 0.0,
                "rot_noise_scale": 0.0,
            },
        )

        return task_constraint.to_dict()
