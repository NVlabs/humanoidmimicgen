import datetime
import os
import time
from collections import OrderedDict

import h5py
import numpy as np
import robosuite
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject, HollowCylinderObject
from robosuite.models.objects.composite.bin import Bin
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import (
    CustomMaterial,
    add_material,
    find_elements,
    string_to_array,
)
from robosuite.utils.observables import Observable, sensor

import robosuite
import robosuite.utils.transform_utils as T
from robocasa.environments.two_arm.two_arm_dexmg_env import (
    TwoArmDexMGEnv,
)
from robocasa.models.objects.xml_objects import BlenderObject
from robocasa.models.scenes.two_arm_arena import GearKitchenTableArena
from robocasa.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)


class TwoArmPickPlace(TwoArmDexMGEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.75, 1.2, 0.05),
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
        use_cylinder=True,
        *args,
        **kwargs,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # add robot base randomization
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
        # load model for table top workspace
        self.table_offset = np.array((-0.02, -0.025, 1.0)) + self.rng.uniform(0, 0.02, 3)

        super()._load_model()

        self.randomize_camera(arena=self.mujoco_arena)
        self.randomize_table_texture(arena=self.mujoco_arena)

        import robocasa

        base_mjcf_path = os.path.join(robocasa.__path__[0], "models/assets/objects/objaverse/")

        # self.can = CylinderObject(
        #     name="can",
        #     size=[0.03, self.can_height/2],
        #     rgba=(0.58, 0.15, 0.10, 1.0),
        # )
        self.can = HollowCylinderObject(
            name="can",
            inner_radius=0.02,
            outer_radius=0.03,
            height=self.can_height / 2,
            ngeoms=16,
            rgba=(0.58, 0.15, 0.10, 1.0),
        )

        def _create_obj(cfg):
            object = BlenderObject(
                name=cfg["name"],
                mjcf_path=cfg["mjcf_path"],
                scale=cfg["scale"],
                solimp=(0.998, 0.998, 0.001),
                solref=(0.001, 1),
                density=cfg.get("density", 100),
                # friction=(0.95, 0.3, 0.1),
                friction=(1, 1, 1),
                margin=0.001,
            )
            return object

        # initialize objects of interest
        self.plate = _create_obj(
            {
                "name": "plate",
                "mjcf_path": os.path.join(base_mjcf_path, "plate/plate_12/model.xml"),
                "scale": 1.0,
                "density": 5000,
            }
        )

        objects = [self.plate, self.can]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler", rng=self.rng)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="PlateSampler",
                mujoco_objects=self.plate,
                x_range=(-0.25, -0.10),
                y_range=(-0.4, -0.05),
                rotation=(0.0, 0.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
                rng=self.rng,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CanSampler",
                mujoco_objects=self.can,
                x_range=(-0.25, -0.10),
                y_range=(-0.4, -0.10),
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
            plate=self.sim.model.body_name2id(self.plate.root_body),
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

    def _check_success(self):
        """
        Check success.
        """
        plate_base_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["plate"]])
        can_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["can"]])

        radius = self.can_height / 2
        # z_threshold = self.can_height / 2
        x_check = np.abs(plate_base_pos[0] - can_pos[0]) < radius
        y_check = np.abs(plate_base_pos[1] - can_pos[1]) < radius
        # z_check = 0 < can_pos[2] - plate_base_pos[2] < z_threshold

        can_touch = self.check_contact(self.plate, self.can)
        hand_touch = self.check_contact(self.robots[0].gripper["right"], self.can)

        return (x_check and y_check) and can_touch and not hand_touch

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
        ep_meta["lang"] = "pick the object and place it on the plate"
        return ep_meta
