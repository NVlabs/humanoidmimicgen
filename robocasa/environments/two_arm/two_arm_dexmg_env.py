import os
from copy import deepcopy
from typing import List
import xml.etree.ElementTree as ET

import numpy as np
import robosuite
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import (
    array_to_string,
    find_elements,
    new_element,
    string_to_array,
    xml_path_completion,
)
import robosuite.utils.transform_utils as T

import robocasa
import robocasa.utils.camera_utils as CamUtils
from robocasa.models.scenes.two_arm_arena import GearKitchenTableArena


class TwoArmDexMGEnv(TwoArmEnv):
    def __init__(self, translucent_robot=False, use_gear_kitchen=False, *args, **kwargs):

        self.translucent_robot = translucent_robot
        self.use_gear_kitchen = use_gear_kitchen
        self.mujoco_arena = None
        if kwargs.get("groot_commit_id", None) is not None:
            del kwargs["groot_commit_id"]  # causes error when present in kwargs
        super().__init__(*args, **kwargs)

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

        for robot in self.robots:
            robot_model = robot.robot_model
            visual_geom_names = robot_model.visual_geoms

            for name in visual_geom_names:
                rgba = self.sim.model.geom_rgba[self.sim.model.geom_name2id(name)]
                if self.translucent_robot:
                    rgba[-1] = 0.05
                else:
                    rgba[-1] = 1.0

    def set_cameras(self):
        """
        Adds new two-arm-relevant cameras to the environment.
        """

        self._cam_configs = deepcopy(CamUtils.CAM_CONFIGS)

        for robot in self.robots:
            if hasattr(robot.robot_model, "get_camera_configs"):
                self._cam_configs.update(robot.robot_model.get_camera_configs())

        for cam_name, cam_cfg in self._cam_configs.items():
            if cam_cfg.get("parent_body", None) is not None:
                continue

            self.mujoco_arena.set_camera(
                camera_name=cam_name,
                pos=cam_cfg["pos"],
                quat=cam_cfg["quat"],
                camera_attribs=cam_cfg.get("camera_attribs", None),
            )

    def _load_model(self):
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            for robot, offset in zip(self.robots, (-0.25, 0.25)):
                xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                xpos = np.array(xpos) + np.array((0, offset, 0))
                robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        if self.use_gear_kitchen:
            self.mujoco_arena = GearKitchenTableArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
            )
        else:
            self.mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
                xml=xml_path_completion(
                    "arenas/table_arena.xml",
                ),
            )

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])
        self.set_cameras()

        egoview_camera = new_element(
            name="egoview",
            tag="camera",
            pos="-0.3 0 1.5",
            quat="0.67397475 0.21391128 -0.21391128 -0.6739747",
        )
        self.mujoco_arena.worldbody.append(egoview_camera)

        if self.translucent_robot:
            material_names = ["LightGreyTrans", "BlackTrans"]
            for robot in self.robots:
                for material_name in material_names:
                    material = find_elements(
                        root=robot.robot_model.asset,
                        tags="material",
                        attribs={"name": robot.robot_model.naming_prefix + material_name},
                        return_first=True,
                    )
                    if material is None:
                        continue
                    rgba = string_to_array(material.get("rgba"))
                    rgba[-1] = 0.000001
                    material.set("rgba", array_to_string(rgba))

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """

        # NOTE: fixed with mujoco >= 2.2
        # # NOTE: we changed this to the robosuite implementation (instead of mujoco binding) because of
        # #       an issue with the DM mujoco binding where .obj meshes dump vertices and increase the
        # #       size of the xml substantially, and also cause some simulation issues when reloading
        # #       from xml (such as slippage during grasping the obj meshes)
        # xml = self.env.model.get_xml()
        xml = self.sim.model.get_xml()  # model xml file
        state = np.array(self.sim.get_state().flatten())  # simulator state
        return dict(model=xml, states=state)

    def edit_model_xml(self, xml_str):
        """
        This function postprocesses the model.xml collected from a MuJoCo demonstration
        for retrospective model changes.

        Args:
            xml_str (str): Mujoco sim demonstration XML file as string

        Returns:
            str: Post-processed xml file as string
        """
        xml_str = super().edit_model_xml(xml_str)

        tree = ET.fromstring(xml_str)
        root = tree
        worldbody = root.find("worldbody")
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        robosuite_path_split = os.path.split(robosuite.__file__)[0].split("/")
        robocasa_path_split = os.path.split(robocasa.__file__)[0].split("/")

        # replace robocasa-specific asset paths
        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # maybe replace all paths to robosuite assets
            if "models/assets" in old_path:
                if "/robosuite/" in old_path:
                    check_lst = [
                        loc for loc, val in enumerate(old_path_split) if val == "robosuite"
                    ]
                    ind = max(check_lst)  # last occurrence index
                    new_path_split = robosuite_path_split + old_path_split[ind + 1 :]
                elif "/robocasa/" in old_path:
                    check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robocasa"]
                    ind = max(check_lst)  # last occurrence index
                    new_path_split = robocasa_path_split + old_path_split[ind + 1 :]
                elif "/dexmimicgen_environments/" in old_path:
                    check_lst = [
                        loc
                        for loc, val in enumerate(old_path_split)
                        if val == "dexmimicgen_environments"
                    ]
                    ind = max(check_lst)
                else:
                    raise ValueError

                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        # set cameras
        for cam_name, cam_config in self._cam_configs.items():
            parent_body = cam_config.get("parent_body", None)

            cam_root = worldbody
            if parent_body is not None:
                cam_root = find_elements(root=worldbody, tags="body", attribs={"name": parent_body})
                if cam_root is None:
                    # camera config refers to body that doesnt exist on the robot
                    continue

            cam = find_elements(root=cam_root, tags="camera", attribs={"name": cam_name})

            if cam is None:
                old_cam = find_elements(root=worldbody, tags="camera", attribs={"name": cam_name})
                if old_cam is not None:
                    # old camera associated with different body
                    continue

                cam = ET.Element("camera")
                cam.set("mode", "fixed")
                cam.set("name", cam_name)
                cam_root.append(cam)

            cam.set("pos", array_to_string(cam_config["pos"]))
            cam.set("quat", array_to_string(cam_config["quat"]))
            for k, v in cam_config.get("camera_attribs", {}).items():
                cam.set(k, v)

        result = ET.tostring(root).decode("utf8")
        return result

    @staticmethod
    def quat_to_xyaxes(quat: List[float]) -> List[float]:
        """
        Convert quaternion to xyaxes.
        """
        w, x, y, z = quat
        mat = T.quat2mat([x, y, z, w])
        x_axis = mat[:, 0]
        y_axis = mat[:, 1]
        return x_axis.tolist() + y_axis.tolist()

    def randomize_camera(self, arena):
        """
        Randomize camera view.
        """
        # Fixed camera name
        # camera_name = "teleopview"
        # angle_type = "xyaxes"
        camera = find_elements(
            root=arena.worldbody,
            tags="camera",
            attribs={"name": self.render_camera},
            return_first=True,
        )
        assert camera is not None
        camera_pos = string_to_array(camera.get("pos"))
        camera_pos += self.rng.uniform(-0.025, 0.025, 3)
        camera.set("pos", array_to_string(camera_pos))

        if "xyaxes" in camera.attrib:
            camera_orientation = string_to_array(camera.get("xyaxes"))
        elif "quat" in camera.attrib:
            quat = camera.get("quat").strip().split(" ")
            quat = [float(q) for q in quat]
            camera_orientation = self.quat_to_xyaxes(quat)
            camera.attrib.pop("quat")  # multiple orientation options are not allowed
        else:
            raise ValueError("No camera orientation specified.")
        camera_orientation += self.rng.uniform(-0.025, 0.025, 6)
        camera.set("xyaxes", array_to_string(camera_orientation))
        camera_fov = string_to_array(camera.get("fovy", "70"))
        camera_fov += self.rng.uniform(-5, 5, 1)
        camera.set("fovy", array_to_string(camera_fov))

    def randomize_table_texture(self, arena):
        material_name = "table_ceramic"
        material = find_elements(
            root=arena.asset,
            tags="material",
            attribs={"name": material_name},
            return_first=True,
        )
        rgb_base = np.array([0.7176, 0.6509, 0.5960, 1])
        # do not randomize alpha
        new_rgb = rgb_base + self.rng.uniform(-0.05, 0.05, 4)
        new_rgb[-1] = 1

        new_material = {
            "name": material_name,
            "specular": "0.1",
            "shininess": "0.1",
            "reflectance": "0.1",
            "rgba": array_to_string(new_rgb),
        }
        old_keys = list(material.attrib.keys())
        # pop all old keys
        for key in old_keys:
            material.attrib.pop(key)
        for key, value in new_material.items():
            material.set(key, value)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        # TODO: move this to a more appropriate place
        ep_meta["seed"] = self.seed
        return ep_meta
