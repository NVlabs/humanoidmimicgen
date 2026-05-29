import datetime
import os
import glob
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import robocasa
import robosuite
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robosuite.examples.third_party_controller.mink_controller import IKSolverMink
from robosuite.models.arenas import TableArena
from robosuite.models.objects.composite.bin import Bin
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import find_elements
from robosuite.utils.observables import Observable, sensor

import robosuite
import robosuite.utils.transform_utils as T
from robocasa.environments.two_arm.two_arm_dexmg_env import (
    TwoArmDexMGEnv,
)
from robocasa.models.objects.xml_objects import BlenderObject
from robocasa.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robocasa.models.scenes.two_arm_arena import GearKitchenTableArena


class TwoArmManipulateCarPart(TwoArmDexMGEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.75, 1.8, 0.05),
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
        car_part_name="",
        placement_initializer_parameters=None,
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

        self.car_part_name = car_part_name
        self.placement_initializer_parameters = placement_initializer_parameters
        self.obj_names = []
        self.box_names = []

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

        self.mujoco_arena.set_camera(
            camera_name="egoview",
            pos=(-0.4, 0, 1.6),
            quat=(0.6818, 0.189, -0.189, -0.6818),
            camera_attribs={"fovy": "77"},
        )

        self.box = Bin(
            name="box",
            bin_size=(0.39, 0.39, 0.1),
            wall_thickness=0.01,
            transparent_walls=False,
            friction=None,
            density=100000.0,
            use_texture=False,
            rgba=(0.3, 0.7, 0.3, 1.0),
        )

        if self.placement_initializer_parameters["include_trash_bin"]:
            self.trash_bin = Bin(
                name="trash_bin",
                bin_size=(0.39, 0.39, 0.1),
                wall_thickness=0.01,
                transparent_walls=False,
                friction=None,
                density=100000.0,
                use_texture=False,
                rgba=(0.7, 0.3, 0.3, 1.0),
            )

        mjcf_path = glob.glob(
            f"{robocasa.__path__[0]}/models/assets/gear_kitchen/objects/car_parts/{self.car_part_name}/{self.car_part_name}.xml"
        )[0]

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
                rgba=(0.1, 0.1, 0.1, 1.0),
            )
            return object

        # initialize objects of interest
        self.obj = _create_obj(
            {
                "name": "obj",
                "mjcf_path": mjcf_path,
                "scale": 0.7,
            }
        )

        objects = [self.box, self.obj]
        self.objects = {
            "box": self.box,
            "obj": self.obj,
        }
        if self.placement_initializer_parameters["include_trash_bin"]:
            objects.append(self.trash_bin)
            self.objects["trash_bin"] = self.trash_bin

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

        if "BDM1" in self.robots[0].name:
            robot_model = self.robots[0].robot_model
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            xpos = (xpos[0], xpos[1], 1.00)
            robot_model.set_base_xpos(xpos)

    def _reset_internal(self):
        super()._reset_internal()

        # Only needed for GR1
        if "GR1" in self.robots[0].name:
            for joint_name in ["robot0_l_elbow_pitch", "robot0_r_elbow_pitch"]:
                l_elbow_pitch_id = self.sim.model.joint_name2id(joint_name)
                l_elbow_pitch_range = self.sim.model.jnt_range[l_elbow_pitch_id]
                self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(joint_name)] = (
                    l_elbow_pitch_range[0] * 0.7 + l_elbow_pitch_range[1] * 0.3
                )
        elif "BDM1" in self.robots[0].name:
            for joint_name in ["robot0_larm.el", "robot0_rarm.el"]:
                l_elbow_pitch_id = self.sim.model.joint_name2id(joint_name)
                l_elbow_pitch_range = self.sim.model.jnt_range[l_elbow_pitch_id]
                self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(joint_name)] = (
                    l_elbow_pitch_range[0] * 0.1 + l_elbow_pitch_range[1] * 0.9
                )

        geoms = find_elements(self.model.root, tags="geom", return_first=False)
        self.obj_names = []
        self.box_names = []
        for g in geoms:
            name = g.get("name", None)
            if name is not None and name.startswith("box_base"):
                self.box_names.append(name)
            if name is not None and name.startswith("obj_"):
                self.obj_names.append(name)

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler", rng=self.rng)
        box_y_offset = 0.35
        if self.placement_initializer_parameters["include_trash_bin"]:
            box_y_offset = self.rng.choice([0.15, 0.55])
            trash_bin_y_offset = 0.7 - box_y_offset
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="TrashBinSampler",
                    mujoco_objects=self.trash_bin,
                    x_range=(-0.1, -0.1),
                    y_range=(trash_bin_y_offset, trash_bin_y_offset),
                    rotation=(0.0, 0.0),
                    rng=self.rng,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_offset,
                    z_offset=0.0,
                )
            )

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="BoxSampler",
                mujoco_objects=self.box,
                x_range=(-0.1, -0.1),
                y_range=(box_y_offset, box_y_offset),
                rotation=(0.0, 0.0),
                rng=self.rng,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjSampler",
                mujoco_objects=self.obj,
                x_range=self.placement_initializer_parameters["x_range"],
                y_range=self.placement_initializer_parameters["y_range"],
                rotation=self.placement_initializer_parameters["rotation"],
                rng=self.rng,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=self.placement_initializer_parameters["z_offset"],
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
            box=self.sim.model.body_name2id(self.box.root_body),
            obj=self.sim.model.body_name2id(self.obj.root_body),
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

        return self.check_contact(self.box_names, self.obj_names)
        # TODO: implement success check

        box_base_pos = np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id("box_base")])
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["obj"]])

        x_threshold = self.box.bin_size[0] / 2
        y_threshold = self.box.bin_size[1] / 2
        z_threshold = self.box.bin_size[2]
        x_check = np.abs(box_base_pos[0] - obj_pos[0]) < x_threshold
        y_check = np.abs(box_base_pos[1] - obj_pos[1]) < y_threshold
        z_check = np.abs(box_base_pos[2] - obj_pos[2]) < z_threshold

        # print(f"x_check: {x_check}, y_check: {y_check}, z_check: {z_check}")
        return x_check and y_check and z_check

    def _check_obj_lifted(self):
        # TODO: use table top
        box_base_pos = np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id("box_base")])
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["obj"]])

        return obj_pos[2] - box_base_pos[2] > 0.1

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
                gripper=self.robots[0].gripper["right"], target=self.obj
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


class TwoArmPnPCarPart(TwoArmManipulateCarPart, DexMGConfigHelper):
    def get_object(self):
        objects = dict()
        objects["obj"] = dict(obj_name=self.obj.root_body, obj_type="body", obj_joint=None)

        objects["box"] = dict(
            obj_name=self.box.root_body,
            obj_type="body",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        signals = dict()
        signals["grasp_object"] = int(
            self._check_grasp(gripper=self.robots[0].gripper["right"], object_geoms=self.obj)
        )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="grasp_object",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        return task.to_dict()


class TwoArmTransportCarPart(TwoArmManipulateCarPart, DexMGConfigHelper):
    def get_object(self):
        objects = dict()
        objects["obj"] = dict(obj_name=self.obj.root_body, obj_type="body", obj_joint=None)
        objects["trash_bin"] = dict(
            obj_name=self.trash_bin.root_body, obj_type="body", obj_joint=None
        )
        objects["box"] = dict(
            obj_name=self.box.root_body,
            obj_type="body",
            obj_joint=None,
        )
        return objects

    def get_subtask_term_signals(self):
        signals = dict()
        signals["lift_obj_right"] = int(self._check_obj_lifted())
        g0, g1 = self.get_grippers()
        grasping_payload_g0 = self._check_grasp(gripper=g0, object_geoms=self.obj_names)
        grasping_payload_g1 = self._check_grasp(gripper=g1, object_geoms=self.obj_names)
        signals["transport_obj"] = int(not grasping_payload_g0 and grasping_payload_g1)
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="lift_obj_right",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="obj",
            subtask_term_signal="transport_obj",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_0.subtask_3 = dict(
            object_ref="obj",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref="obj",
            subtask_term_signal="transport_obj",
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        task.task_spec_1.subtask_2 = dict(
            object_ref="box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=True,
        )
        return task.to_dict()


class TwoArmPnPCarPartBrakepedal(TwoArmPnPCarPart):
    def __init__(self, *args, **kwargs):
        placement_initializer_parameters = dict(
            x_range=(-0.15, -0.05),
            y_range=(-0.45, -0.25),
            rotation=(-np.pi, np.pi),
            z_offset=0.06,
            include_trash_bin=False,
        )
        super().__init__(
            car_part_name="brake_pedal",
            placement_initializer_parameters=placement_initializer_parameters,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the brake pedal into the box."
        return ep_meta


class TwoArmTransportCarPartBrakepedal(TwoArmTransportCarPart):
    def __init__(self, *args, **kwargs):
        placement_initializer_parameters = dict(
            x_range=(-0.1, -0.0),
            y_range=(-0.4, -0.3),
            rotation=(0.25 * np.pi, 0.75 * np.pi),
            z_offset=0.06,
            include_trash_bin=True,
        )
        super().__init__(
            car_part_name="brake_pedal",
            placement_initializer_parameters=placement_initializer_parameters,
            *args,
            **kwargs,
        )


class TwoArmPnPCarPartFrontmuffler(TwoArmPnPCarPart):
    def __init__(self, *args, **kwargs):
        placement_initializer_parameters = dict(
            x_range=(-0.2, 0.0),
            y_range=(-0.3, 0.0),
            rotation=(-np.pi, np.pi),
            z_offset=0.05,
            include_trash_bin=False,
        )
        super().__init__(
            car_part_name="front_muffler_variant_1",
            placement_initializer_parameters=placement_initializer_parameters,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the front muffler into the box."
        return ep_meta


class TwoArmTransportCarPartFrontmuffler(TwoArmTransportCarPart):
    def __init__(self, *args, **kwargs):
        placement_initializer_parameters = dict(
            x_range=(-0.1, -0.0),
            y_range=(-0.4, -0.3),
            rotation=(0.25 * np.pi, 0.75 * np.pi),
            z_offset=0.05,
            include_trash_bin=True,
        )
        super().__init__(
            car_part_name="front_muffler_variant_1",
            placement_initializer_parameters=placement_initializer_parameters,
            *args,
            **kwargs,
        )


class TwoArmPnPCarPartSteeringwheel(TwoArmPnPCarPart):
    def __init__(self, *args, **kwargs):
        placement_initializer_parameters = dict(
            x_range=(-0.2, 0.0),
            y_range=(-0.3, 0.0),
            rotation=(-np.pi, np.pi),
            z_offset=0.06,
            include_trash_bin=False,
        )
        super().__init__(
            car_part_name="steering_wheel",
            placement_initializer_parameters=placement_initializer_parameters,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the steering wheel into the box."
        return ep_meta


class TwoArmTransportCarPartSteeringwheel(TwoArmTransportCarPart):
    def __init__(self, *args, **kwargs):
        placement_initializer_parameters = dict(
            x_range=(-0.1, -0.0),
            y_range=(-0.4, -0.3),
            rotation=(0.25 * np.pi, 0.75 * np.pi),
            z_offset=0.06,
            include_trash_bin=True,
        )
        super().__init__(
            car_part_name="steering_wheel",
            placement_initializer_parameters=placement_initializer_parameters,
            *args,
            **kwargs,
        )
