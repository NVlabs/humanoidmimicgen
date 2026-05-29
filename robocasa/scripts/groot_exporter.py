from collections import defaultdict
import datetime
import imageio
import json
import numpy as np
import os
import shutil
import subprocess
import time

from robocasa.models.robots import (
    GROOT_ROBOCASA_ENVS_ROBOTS,
    reconstruct_latest_actions,
    gather_robot_observations,
    make_key_converter,
)
import robocasa
import robocasa.utils.transform_utils as T
import robosuite
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER


def get_git_commit_id(folder: str):
    """Gets the latest Git commit ID from the given folder if it is part of a Git repository."""
    if not os.path.isdir(folder):
        return ""  # Invalid directory
    try:
        # Change to the target directory
        original_dir = os.getcwd()
        os.chdir(folder)
        # Check if it's inside a Git repository
        is_git_repo = (
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if not is_git_repo:
            return ""  # Not in a Git repository
        # Get the commit ID
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .strip()
            .decode("utf-8")
        )
        return commit_id
    except subprocess.CalledProcessError:
        return ""  # Error executing Git commands
    finally:
        # Restore the original working directory
        os.chdir(original_dir)


class GrootExporter:
    def __init__(
        self,
        groot_dir,
        env,
        groot_env_meta,
    ):
        if os.path.exists(groot_dir):
            shutil.rmtree(groot_dir)
        os.makedirs(groot_dir, exist_ok=True)
        self.groot_dir = groot_dir
        self.groot_env_meta = groot_env_meta

        # TODO: Fix so many if conditions
        robots_name = "_".join(robot.name for robot in env.robots)
        embodiment_tag = f"robocasa_{GROOT_ROBOCASA_ENVS_ROBOTS[robots_name]}"

        # TODO: Fix so many if conditions
        import groot.vla.data.schema.raw

        if robots_name.startswith("GR1"):
            from groot.vla.data.schema.raw.gr1 import (
                RawTrajectory_GR1_V1_0,
            )

            schema_class = RawTrajectory_GR1_V1_0
        elif robots_name == "PandaDexRH_PandaDexLH":
            from groot.vla.data.schema.raw.panda import (
                RawTrajectory_BimanualPandaDex_V1_0,
            )

            schema_class = RawTrajectory_BimanualPandaDex_V1_0
        elif robots_name == "Panda_Panda":
            from groot.vla.data.schema.raw.panda import (
                RawTrajectory_BimanualPanda_V1_0,
            )

            schema_class = RawTrajectory_BimanualPanda_V1_0
        elif robots_name == "PandaOmron":
            from groot.vla.data.schema.raw.panda import (
                RawTrajectory_SingleArmPanda_V1_0,
            )

            schema_class = RawTrajectory_SingleArmPanda_V1_0
        else:
            raise ValueError(f"Unknown robot name: {robots_name}")

        # TODO: update
        try:
            operator = os.getenv("USER")
            if not isinstance(operator, str) or len(operator) == 0:
                operator = "unknown"
        except:
            operator = "unknown"

        self.groot_metadata = {
            "modalities": {
                "video": dict(),
                "state": {
                    "body": dict(),
                    "hand": dict(),
                },
                "action": {
                    "body": dict(),
                    "hand": dict(),
                },
            },
            "embodiment": {
                "robot_name": "_".join(robot.name for robot in env.robots),
                "robot_type": "_".join(robot.name for robot in env.robots),
                "record_frequency": 20.0,
                "body_controller_frequency": 20.0,
                "hand_controller_frequency": 20.0,
                "embodiment_tag": embodiment_tag,
            },
            "session": {
                "operator": operator,
                "description": groot_env_meta["env_name"],  # TODO: update
                "remarks": env.get_ep_meta().get("lang", ""),
                "session_start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trajectory_start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": "America/Los_Angeles",
            },
            "version": {
                "path_robosuite": os.path.dirname(robosuite.__file__),
                "path_robocasa": os.path.dirname(robocasa.__file__),
                "commit_hash_robosuite": get_git_commit_id(os.path.dirname(robosuite.__file__)),
                "commit_hash_robocasa": get_git_commit_id(os.path.dirname(robocasa.__file__)),
                "commit_hash_groot": get_git_commit_id(os.path.dirname(groot.__file__)),
                "state_action_schema_class": f"{schema_class.__module__}.{schema_class.__name__}",
            },
            "config": {
                "env_meta": groot_env_meta,
                "ep_meta": env.get_ep_meta(),
                "model_file": env.sim.model.get_xml(),
                "initial_state": env.sim.get_state().flatten().tolist(),
            },
        }

        self.groot_current_time = time.time()
        self.groot_episode_data = schema_class()
        self.groot_episode_data.set_start_time(self.groot_current_time)
        self.groot_hdf5_file = os.path.join(groot_dir, "state_action.hdf5")

        self.groot_video_writers = {}
        for cam_name, camera_width, camera_height in zip(
            env.camera_names, env.camera_widths, env.camera_heights
        ):
            groot_video_file = os.path.join(groot_dir, f"{cam_name}.mp4")
            self.groot_video_writers[cam_name] = imageio.get_writer(
                groot_video_file, fps=20, codec="libx264"
            )
            self.groot_metadata["modalities"]["video"][cam_name] = {
                "resolution": [camera_width, camera_height],
                "channels": 3,
                "fps": 20.0,
            }
        self.groot_info = []

    def add_record_before_step(self, env, actions, verbose=False):
        default_value = {
            "absolute": True,
            "rotation_type": None,
        }
        robots_name = "_".join(robot.name for robot in env.robots)
        key_converter = make_key_converter(robots_name)

        nested_dict = lambda: defaultdict(nested_dict)
        self.groot_record = nested_dict()

        ROBOSUITE_DEFAULT_LOGGER.warning(
            "Calling env._get_observations(force_update=True) will pollute the observables."
        )
        obs_dict = (
            env.viewer._get_observations(force_update=True)
            if env.viewer_get_obs
            else env._get_observations(force_update=True)
        )
        for cam_name in env.camera_names:
            im = obs_dict[f"{cam_name}_image"][::-1]
            self.groot_video_writers[cam_name].append_data(im)

        obs_dict.update(gather_robot_observations(env))
        obs_dict = key_converter.map_obs(obs_dict)
        obs_dict.update(key_converter.get_missing_keys_in_dumping_dataset())
        obs_dict = key_converter.convert_to_float64(obs_dict)
        for k, v in obs_dict.items():
            group, real_name = k.split(".")
            self.groot_record["state"][group][real_name] = np.array(v).reshape((1, -1))
            self.groot_record["state"][group]["time"] = np.array([self.groot_current_time])
            self.groot_metadata["modalities"]["state"][group][real_name] = (
                key_converter.get_metadata(real_name)
            )
            self.groot_metadata["modalities"]["state"][group]["time"] = default_value
        obj_info = {}
        for obj_name in env.obj_body_id.keys():
            obj_info[f"{obj_name}_pos"] = list(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
            obj_info[f"{obj_name}_quat_xyzw"] = list(
                T.convert_quat(
                    np.array(env.sim.data.body_xquat[env.obj_body_id[obj_name]]),
                    to="xyzw",
                )
            )
        self.groot_info.append(obj_info)

    def add_record_after_step(self, env, actions, verbose=False):
        default_value = {
            "absolute": True,
            "rotation_type": None,
        }
        robots_name = "_".join(robot.name for robot in env.robots)
        key_converter = make_key_converter(robots_name)

        action_dict = reconstruct_latest_actions(env, actions)
        action_dict = key_converter.map_action(action_dict)
        action_dict.update(key_converter.get_missing_keys_in_dumping_dataset())
        action_dict = key_converter.convert_to_float64(action_dict)
        for (
            k,
            v,
        ) in action_dict.items():
            group, real_name = k.split(".")
            self.groot_record["action"][group][real_name] = np.array(v).reshape((1, -1))
            self.groot_record["action"][group]["time"] = np.array([self.groot_current_time])
            self.groot_metadata["modalities"]["action"][group][real_name] = (
                key_converter.get_metadata(real_name)
            )
            self.groot_metadata["modalities"]["action"][group]["time"] = default_value

        self.groot_episode_data.append_and_save(self.groot_record, self.groot_hdf5_file)

        # update current time
        self.groot_current_time += env.control_timestep

    def finish(self):
        from groot.vla.data.schema.raw.metadata import (
            RawMetadata_V1_2,
        )

        metadata_checked = RawMetadata_V1_2.model_validate(self.groot_metadata)
        with open(os.path.join(self.groot_dir, "metadata.json"), "w") as f:
            f.write(metadata_checked.model_dump_json(indent=4))

        with open(os.path.join(self.groot_dir, "info.json"), "w") as f:
            json.dump(self.groot_info, f, indent=4)

        for _, writer in self.groot_video_writers.items():
            writer.close()
