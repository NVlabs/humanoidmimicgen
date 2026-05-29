import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import (
    CustomMaterial,
    add_material,
    find_elements,
    string_to_array,
)
from robosuite.utils.observables import Observable, sensor

import robosuite.utils.transform_utils as T
from robocasa.environments.two_arm.two_arm_dexmg_env import (
    TwoArmDexMGEnv,
)
from robocasa.models.scenes.two_arm_arena import GearKitchenTableArena
from robocasa.models.objects import (
    CoffeeMachineObject,
    CoffeeMachinePodObject,
    LongDrawerObject,
)
from robocasa.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)


class TwoArmCoffee(TwoArmDexMGEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
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
        lid_start_value=None,
        make_coffee_machine_fixture=False,
        *args,
        **kwargs,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.9))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # starting value of coffee machine lid joint
        if lid_start_value is None:
            lid_start_value = 2.0 * np.pi / 3.0
        self.lid_start_value = lid_start_value

        # whether to make coffee machine a fixture
        self.make_coffee_machine_fixture = make_coffee_machine_fixture

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

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

    def get_coffee_machine_pod_margin(self):
        """
        Subclasses can override this to ensure more of a margin for object placements near the coffee pod.
        """
        return 1.0

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

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # initialize objects of interest
        self.coffee_pod = CoffeeMachinePodObject(name="coffee_pod")
        if self.make_coffee_machine_fixture:
            # coffee machine cannot move and is a fixture
            self.coffee_machine = CoffeeMachineObject(
                name="coffee_machine", joints=None, pod_holder_friction=(0.1, 0.1, 0.1)
            )
        else:
            # coffee machine can move
            self.coffee_machine = CoffeeMachineObject(name="coffee_machine", density=100000000.0)
        objects = [self.coffee_pod, self.coffee_machine]

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
            camera_name="egoview",
            pos=(-0.4, 0, 1.6),  # Increased z-axis from 1.35 to 1.8
            quat=(0.6818, 0.189, -0.189, -0.6818),
            camera_attribs={"fovy": "77"},
        )
        self.model.mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array(
                "0.753078462147161 2.062036796036723e-08 1.5194726087166726"
            ),  # Increased z-axis from 1.35 to 1.8
            quat=string_to_array(
                "0.6432409286499023 0.293668270111084 0.2936684489250183 0.6432408690452576"
            ),
            # camera_attribs={"fovy": "60"},
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """

        return dict(
            coffee_machine=dict(
                x=(0.0, 0.0),
                y=(-0.1, -0.1),
                z_rot=(-np.pi / 6.0, -np.pi / 6.0),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.13, -0.07),
                y=(0.17, 0.23),
                z_rot=(0.0, 0.0),
                reference=self.table_offset,
            ),
        )

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler", rng=self.rng)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeeMachineSampler",
                mujoco_objects=self.coffee_machine,
                x_range=bounds["coffee_machine"]["x"],
                y_range=bounds["coffee_machine"]["y"],
                rotation=bounds["coffee_machine"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["coffee_machine"]["reference"],
                z_offset=0.0,
                rng=self.rng,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeePodSampler",
                mujoco_objects=self.coffee_pod,
                x_range=bounds["coffee_pod"]["x"],
                y_range=bounds["coffee_pod"]["y"],
                rotation=bounds["coffee_pod"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["coffee_pod"]["reference"],
                z_offset=0.0,
                rng=self.rng,
            )
        )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super(TwoArmDexMGEnv, self)._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if (obj is self.coffee_machine) and self.make_coffee_machine_fixture:
                    # fixtures - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )

        # Always reset the hinge joint position
        self.sim.data.qpos[self.hinge_qpos_addr] = self.lid_start_value
        self.sim.forward()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        return observables

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references for this env
        self.obj_body_id = dict(
            coffee_pod=self.sim.model.body_name2id(self.coffee_pod.root_body),
            coffee_machine=self.sim.model.body_name2id(self.coffee_machine.root_body),
            coffee_pod_holder=self.sim.model.body_name2id("coffee_machine_pod_holder_root"),
            coffee_machine_lid=self.sim.model.body_name2id("coffee_machine_lid_main"),
        )
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr("coffee_machine_lid_main_joint0")

        # for checking contact (used in reward function, and potentially observation space)
        self.pod_geom_id = self.sim.model.geom_name2id("coffee_pod_g0")
        self.lid_geom_id = self.sim.model.geom_name2id("coffee_machine_lid_g0")
        pod_holder_geom_names = [
            "coffee_machine_pod_holder_cup_body_hc_{}".format(i) for i in range(64)
        ]
        self.pod_holder_geom_ids = [self.sim.model.geom_name2id(x) for x in pod_holder_geom_names]

        # size of bounding box for pod holder
        self.pod_holder_size = self.coffee_machine.pod_holder_size

        # size of bounding box for pod
        # self.pod_size = self.coffee_pod.horizontal_radius
        self.pod_size = self.coffee_pod.get_bounding_box_half_size()

    def _check_success(self):
        """
        Check if task is complete.
        """
        metrics = self._get_partial_task_metrics()
        return metrics["task"]

    def _check_lid(self):
        # lid should be closed (angle should be less than 5 degrees)
        hinge_tolerance = 15.0 * np.pi / 180.0
        hinge_angle = self.sim.data.qpos[self.hinge_qpos_addr]
        lid_check = hinge_angle < hinge_tolerance
        return lid_check

    def _check_pod(self):
        # pod should be in pod holder
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])
        pod_check = True

        # center of pod cannot be more than the difference of radii away from the center of pod holder
        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        if np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) > r_diff:
            pod_check = False

        # make sure vertical pod dimension is above pod holder lower bound and below the lid lower bound
        lid_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_machine_lid"]])
        z_lim_low = pod_holder_pos[2] - self.pod_holder_size[2]
        z_lim_high = lid_pos[2] - self.coffee_machine.lid_size[2]
        if (pod_pos[2] - self.pod_size[2] < z_lim_low) or (
            pod_pos[2] + self.pod_size[2] > z_lim_high
        ):
            pod_check = False
        return pod_check

    def _get_partial_task_metrics(self):
        metrics = dict()

        lid_check = self._check_lid()
        pod_check = self._check_pod()

        # pod should be in pod holder
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])
        pod_horz_check = True

        # center of pod cannot be more than the difference of radii away from the center of pod holder
        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        if np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) > r_diff:
            pod_horz_check = False

        # make sure vertical pod dimension is above pod holder lower bound and below the lid lower bound
        # lid_pos = np.array(
        #     self.sim.data.body_xpos[self.obj_body_id["coffee_machine_lid"]]
        # )
        z_lim_low = pod_holder_pos[2] - self.pod_holder_size[2]

        metrics["task"] = lid_check and pod_check

        # for pod insertion check, just check that bottom of pod is within some tolerance of bottom of container
        pod_insertion_z_tolerance = 0.02
        pod_z_check = (pod_pos[2] - self.pod_size[2] > z_lim_low) and (
            pod_pos[2] - self.pod_size[2] < z_lim_low + pod_insertion_z_tolerance
        )
        metrics["insertion"] = pod_horz_check and pod_z_check

        # pod grasp check
        metrics["grasp"] = self._check_pod_is_grasped()

        # check is True if the pod is on / near the rim of the pod holder
        rim_horz_tolerance = 0.03
        rim_horz_check = np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) < rim_horz_tolerance

        rim_vert_tolerance = 0.026
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length < rim_vert_tolerance) and (rim_vert_length > 0.0)
        metrics["rim"] = rim_horz_check and rim_vert_check

        return metrics

    def _check_pod_is_grasped(self):
        """
        check if pod is grasped by robot
        """
        return self._check_grasp(
            gripper=self.robots[0].gripper["right"],
            object_geoms=[g for g in self.coffee_pod.contact_geoms],
        )

    def _check_pod_and_pod_holder_contact(self):
        """
        check if pod is in contact with the container
        """
        pod_and_pod_holder_contact = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                (contact.geom1 == self.pod_geom_id) and (contact.geom2 in self.pod_holder_geom_ids)
            ) or (
                (contact.geom2 == self.pod_geom_id) and (contact.geom1 in self.pod_holder_geom_ids)
            ):
                pod_and_pod_holder_contact = True
                break
        return pod_and_pod_holder_contact

    def _check_pod_on_rim(self):
        """
        check if pod is on pod container rim and not being inserted properly (for reward check)
        """
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])

        # check if pod is in contact with the container
        pod_and_pod_holder_contact = self._check_pod_and_pod_holder_contact()

        # check that pod vertical position is not too low or too high
        rim_vert_tolerance_1 = 0.022
        rim_vert_tolerance_2 = 0.026
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length > rim_vert_tolerance_1) and (
            rim_vert_length < rim_vert_tolerance_2
        )

        return pod_and_pod_holder_contact and rim_vert_check

    def _check_pod_being_inserted(self):
        """
        check if robot is in the process of inserting the pod into the container
        """
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])

        rim_horz_tolerance = 0.005
        rim_horz_check = np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) < rim_horz_tolerance

        rim_vert_tolerance_1 = -0.01
        rim_vert_tolerance_2 = 0.023
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length < rim_vert_tolerance_2) and (
            rim_vert_length > rim_vert_tolerance_1
        )

        return rim_horz_check and rim_vert_check

    def _check_pod_inserted(self):
        """
        check if pod has been inserted successfully
        """
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])

        # center of pod cannot be more than the difference of radii away from the center of pod holder
        pod_horz_check = True
        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        pod_horz_check = np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) <= r_diff

        # check that bottom of pod is within some tolerance of bottom of container
        pod_insertion_z_tolerance = 0.02
        z_lim_low = pod_holder_pos[2] - self.pod_holder_size[2]
        pod_z_check = (pod_pos[2] - self.pod_size[2] > z_lim_low) and (
            pod_pos[2] - self.pod_size[2] < z_lim_low + pod_insertion_z_tolerance
        )
        return pod_horz_check and pod_z_check

    def _check_lid_being_closed(self):
        """
        check if lid is being closed
        """

        # (check for hinge angle being less than default angle value, 120 degrees)
        hinge_angle = self.sim.data.qpos[self.hinge_qpos_addr]
        return hinge_angle < 2.09

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
                gripper=self.robots[0].gripper["right"], target=self.coffee_machine
            )

    def get_ep_meta(self):
        """
        Get episode metadata.

        Returns:
            dict: dictionary of episode metadata
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "insert the coffee pod into the coffee machine and close the lid"
        return ep_meta


class TwoArmCoffeeLidClosed(TwoArmCoffee):
    """
    Harder version of coffee task where lid starts closed.
    """

    def __init__(self, **kwargs):
        assert "lid_start_value" not in kwargs, "invalid set of arguments"
        super().__init__(lid_start_value=0.0, **kwargs)


class TwoArmCoffeeFullPreparation(TwoArmCoffee):
    """
    Harder coffee task where the task starts with materials in drawer and coffee machine closed. The robot
    needs to retrieve the coffee pod and mug from the drawer, open the coffee machine, place the pod and mug
    in the machine, and then close the lid.
    """

    def _get_mug_model(self):
        """
        Allow subclasses to override which mug to use.
        """
        shapenet_id = (
            "3143a4ac"  # beige round mug, works well and matches color scheme of other assets
        )
        shapenet_scale = 1.0
        import robosuite_model_zoo
        from robosuite_model_zoo.utils.mjcf_obj import MJCFObject

        base_mjcf_path = os.path.join(robosuite_model_zoo.__path__[0], "assets/shapenet_core/mugs")
        mjcf_path = os.path.join(base_mjcf_path, "{}/model.xml".format(shapenet_id))

        self.mug = MJCFObject(
            name="mug",
            mjcf_path=mjcf_path,
            scale=shapenet_scale,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            # friction=(0.95, 0.3, 0.1),
            friction=(1, 1, 1),
            margin=0.001,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        TwoArmDexMGEnv._load_model(self)

        # Create drawer object
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )
        self.cabinet_object = LongDrawerObject(name="CabinetObject")

        obj_body = self.cabinet_object
        for material in [redwood, ceramic, lightwood]:
            tex_element, mat_element, _, used = add_material(
                root=obj_body.worldbody,
                naming_prefix=obj_body.naming_prefix,
                custom_material=deepcopy(material),
            )
            obj_body.asset.append(tex_element)
            obj_body.asset.append(mat_element)

        # Create mug
        self._get_mug_model()

        # Create coffee pod and machine (note that machine no longer has a cup!)
        # self.coffee_pod = CoffeeMachinePodObject(name="coffee_pod")
        self.coffee_pod = CoffeeMachinePodObject(
            name="coffee_pod", margin=self.get_coffee_machine_pod_margin()
        )
        if self.make_coffee_machine_fixture:
            # coffee machine cannot move and is a fixture
            self.coffee_machine = CoffeeMachineObject(
                name="coffee_machine",
                add_cup=False,
                joints=None,
            )
        else:
            # coffee machine can move
            self.coffee_machine = CoffeeMachineObject(
                name="coffee_machine",
                add_cup=False,
            )
        # self.coffee_machine = CoffeeMachineObject(name="coffee_machine", add_cup=False)
        # objects = [self.coffee_pod, self.coffee_machine, self.cabinet_object, self.mug]
        objects = [self.coffee_pod, self.coffee_machine, self.cabinet_object]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )
        # HACK: merge in mug afterwards because its number of geoms may change
        #       and this may break the generate_id_mappings function in task.py
        self.model.merge_objects([self.mug])  # add cleanup object to model

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            drawer=dict(
                # x=(0.1, 0.1),
                # y=(0.3, 0.3),
                # z_rot=(0.0, 0.0),
                x=(0.15, 0.15),
                y=(-0.35, -0.35),
                z_rot=(np.pi, np.pi),
                reference=self.table_offset,
            ),
            coffee_machine=dict(
                x=(-0.15, -0.15),
                y=(-0.25, -0.25),
                z_rot=(-np.pi / 6.0, -np.pi / 6.0),
                # put vertical
                # z_rot=(-np.pi / 2., -np.pi / 2.),
                reference=self.table_offset,
            ),
            mug=dict(
                # upper right
                # x=(-0.2, -0.2),
                # y=(0.17, 0.23),
                # z_rot=(0.0, 0.0),
                # lower right
                # x=(0.05, 0.20),
                # y=(-0.25, -0.05),
                # z_rot=(0.0, 0.0),
                x=(0.05, 0.20),
                y=(0.05, 0.25),
                z_rot=(0.0, 0.0),
                reference=self.table_offset,
            ),
            coffee_pod=dict(
                x=(-0.03, 0.03),
                y=(-0.05, 0.03),
                z_rot=(0.0, 0.0),
                reference=np.array((0.0, 0.0, 0.0)),
            ),
        )

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler", rng=self.rng)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="DrawerSampler",
                mujoco_objects=self.cabinet_object,
                x_range=bounds["drawer"]["x"],
                y_range=bounds["drawer"]["y"],
                rotation=bounds["drawer"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["drawer"]["reference"],
                z_offset=0.03,
                rng=self.rng,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeeMachineSampler",
                mujoco_objects=self.coffee_machine,
                x_range=bounds["coffee_machine"]["x"],
                y_range=bounds["coffee_machine"]["y"],
                rotation=bounds["coffee_machine"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["coffee_machine"]["reference"],
                z_offset=0.0,
                rng=self.rng,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="MugSampler",
                mujoco_objects=self.mug,
                x_range=bounds["mug"]["x"],
                y_range=bounds["mug"]["y"],
                rotation=bounds["mug"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["mug"]["reference"],
                z_offset=0.02,
                rng=self.rng,
            )
        )

        # Note: coffee pod gets its own placement sampler to sample within a box within the drawer.

        # First, got drawer geom size with self.sim.model.geom_size[self.drawer_bottom_geom_id]
        # Value is array([0.08 , 0.09 , 0.008])
        # Then, used this to set reasonable box for pod init within drawer.
        self.pod_placement_initializer = UniformRandomSampler(
            name="CoffeePodInDrawerSampler",
            mujoco_objects=self.coffee_pod,
            x_range=bounds["coffee_pod"]["x"],
            y_range=bounds["coffee_pod"]["y"],
            rotation=bounds["coffee_pod"]["z_rot"],
            rotation_axis="z",
            # ensure_object_boundary_in_range=True, # make sure pod fits within the box
            ensure_object_boundary_in_range=False,  # make sure pod fits within the box
            ensure_valid_placement=True,
            reference_pos=bounds["coffee_pod"]["reference"],
            z_offset=0.0,
            rng=self.rng,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.cabinet_qpos_addr = self.sim.model.get_joint_qpos_addr(self.cabinet_object.joints[0])
        self.obj_body_id["drawer"] = self.sim.model.body_name2id(self.cabinet_object.root_body)
        self.obj_body_id["mug"] = self.sim.model.body_name2id(self.mug.root_body)
        self.drawer_bottom_geom_id = self.sim.model.geom_name2id("CabinetObject_drawer_bottom")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super(TwoArmDexMGEnv, self)._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                obj_pos_to_set = np.array(obj_pos)
                have_fixture = (obj is self.cabinet_object) or (
                    (obj is self.coffee_machine) and self.make_coffee_machine_fixture
                )
                if obj is self.cabinet_object:
                    # obj_pos_to_set[2] = 0.905 # hardcode z-value to correspond to parent class
                    # obj_pos_to_set[2] = 0.805 # hardcode z-value to make sure it lies on table surface
                    obj_pos_to_set[2] = (
                        self.table_offset[2] + 0.005
                    )  # hardcode z-value to make sure it lies on table surface
                if have_fixture:
                    # fixtures - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )

        # Always reset the hinge joint position, and the cabinet slide joint position

        # Hinge for coffee machine starts closed
        self.sim.data.qpos[self.hinge_qpos_addr] = 0.0
        # self.sim.data.qpos[self.hinge_qpos_addr] = 2. * np.pi / 3.

        # Cabinet should start closed (0.) but can set to open (-0.135) for debugging.
        self.sim.data.qpos[self.cabinet_qpos_addr] = 0.0
        # self.sim.data.qpos[self.cabinet_qpos_addr] = -0.135
        self.sim.forward()

        if not self.deterministic_reset:

            # sample pod location relative to center of drawer bottom geom surface
            coffee_pod_placement = self.pod_placement_initializer.sample(on_top=False)
            assert len(coffee_pod_placement) == 1
            rel_pod_pos, rel_pod_quat, pod_obj = list(coffee_pod_placement.values())[0]
            rel_pod_pos, rel_pod_quat = np.array(rel_pod_pos), np.array(rel_pod_quat)
            assert pod_obj is self.coffee_pod

            # center of drawer bottom
            drawer_bottom_geom_pos = np.array(self.sim.data.geom_xpos[self.drawer_bottom_geom_id])

            # our x-y relative position is sampled with respect to drawer geom frame. Here, we use the drawer's rotation
            # matrix to convert this relative position to a world relative position, so we can add it to the drawer world position
            drawer_rot_mat = T.quat2mat(
                T.convert_quat(
                    self.sim.model.body_quat[
                        self.sim.model.body_name2id(self.cabinet_object.root_body)
                    ],
                    to="xyzw",
                )
            )
            rel_pod_pos[:2] = drawer_rot_mat[:2, :2].dot(rel_pod_pos[:2])

            # also convert the sampled pod rotation to world frame
            rel_pod_mat = T.quat2mat(T.convert_quat(rel_pod_quat, to="xyzw"))
            pod_mat = drawer_rot_mat.dot(rel_pod_mat)
            pod_quat = T.convert_quat(T.mat2quat(pod_mat), to="wxyz")

            # get half-sizes of drawer geom and coffee pod to place coffee pod at correct z-location (on top of drawer bottom geom)
            drawer_bottom_geom_z_offset = self.sim.model.geom_size[self.drawer_bottom_geom_id][
                -1
            ]  # half-size of geom in z-direction
            coffee_pod_bottom_offset = np.abs(self.coffee_pod.bottom_offset[-1])
            coffee_pod_z = (
                drawer_bottom_geom_pos[2]
                + drawer_bottom_geom_z_offset
                + coffee_pod_bottom_offset
                + 0.001
            )

            # set coffee pod in center of drawer
            pod_pos = np.array(drawer_bottom_geom_pos) + rel_pod_pos
            pod_pos[-1] = coffee_pod_z

            self.sim.data.set_joint_qpos(
                pod_obj.joints[0],
                np.concatenate([np.array(pod_pos), np.array(pod_quat)]),
            )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """

        # super class will populate observables for "mug" and "drawer" poses in addition to coffee machine and coffee pod
        observables = super()._setup_observables()

        return observables

    def _check_mug_placement(self):
        """
        Returns true if mug has been placed successfully on the coffee machine.
        """

        # check z-axis alignment by checking z unit-vector of obj pose and dot with (0, 0, 1)
        # then take cosine dist (1 - dot-prod)
        obj_rot = self.sim.data.body_xmat[self.obj_body_id["mug"]].reshape(3, 3)
        z_axis = obj_rot[:3, 2]
        dist_to_z_axis = 1.0 - z_axis[2]
        mug_upright = dist_to_z_axis < 1e-3

        # to check if mug is placed on the machine successfully, we check that the mug is upright, and that it is
        # making contact with the coffee machine base plate
        coffee_base_plate_geom = "coffee_machine_base_g0"
        mug_on_machine = self.check_contact(coffee_base_plate_geom, self.mug)

        return mug_upright and mug_on_machine

    def _get_partial_task_metrics(self):
        """
        Returns a dictionary of partial task metrics which correspond to different parts of the task being done.
        """

        # populate with superclass metrics that concern the coffee pod and coffee machine
        metrics = super()._get_partial_task_metrics()

        # whether mug is grasped (NOTE: the use of tolerant grasp function for mug, due to problems with contact)
        metrics["mug_grasp"] = self._check_grasp_tolerant(
            gripper=self.robots[0].gripper["right"],
            object_geoms=[g for g in self.mug.contact_geoms],
        )

        # whether mug has been placed on coffee machine
        metrics["mug_place"] = self._check_mug_placement()

        # new task success includes mug placement
        metrics["task"] = metrics["task"] and metrics["mug_place"]

        # can have a check on drawer being closed here, to make the task even harder
        # print(self.sim.data.qpos[self.cabinet_qpos_addr])

        return metrics
