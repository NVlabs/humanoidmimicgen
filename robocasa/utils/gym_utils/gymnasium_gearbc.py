import sys
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from robocasa.models.robots import GROOT_ROBOCASA_ENVS_ROBOTS
from .gymnasium_basic import (
    REGISTERED_ENVS,
    RoboCasaEnv,
)

ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)
FINAL_IMAGE_RESOLUTION = (224, 224)
RESIZED_IMAGE_RESOLUTION = (720, 480)


class GearBCRoboCasaEnv(RoboCasaEnv):
    def __init__(
        self,
        image_crop: bool | None = None,
        image_crop_size: Tuple[int, int, int, int] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        temp_observation_space = self.key_converter.deduce_observation_space(self.env)
        self.observation_space = spaces.Dict()
        for k, v in temp_observation_space.items():
            assert k.startswith("state.")
            self.observation_space[k[6:] + "_state"] = v
        mapped_names, camera_names, _, _ = self.key_converter.get_camera_config()
        for camera_name in camera_names:
            self.observation_space[camera_name + "_image"] = spaces.Box(
                low=0, high=1, shape=(3, *FINAL_IMAGE_RESOLUTION), dtype=np.float32
            )
        self.observation_space["caption"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )

        temp_action_space = self.key_converter.deduce_action_space(self.env)
        self.action_space = spaces.Dict()
        for k, v in temp_action_space.items():
            assert k.startswith("action.")
            if isinstance(v, spaces.Box):
                self.action_space[k[7:] + "_action"] = v
            else:
                self.action_space[k[7:] + "_action"] = spaces.Box(
                    low=-1, high=1, shape=(1,), dtype=np.float32
                )

        self.verbose = False
        for k, v in self.observation_space.items():
            self.verbose and print("{OBS}", k, v)
        for k, v in self.action_space.items():
            self.verbose and print("{ACTION}", k, v)

        self.image_crop = image_crop
        self.image_crop_size = image_crop_size

        if self.image_crop:
            assert self.image_crop_size is not None

        self.process_img = self.process_img_w_crop if self.image_crop else self.process_img_no_crop

    def process_img_no_crop(self, img):
        h, w, _ = img.shape
        if h != w:
            dim = max(h, w)
            y_offset = (dim - h) // 2
            x_offset = (dim - w) // 2
            img = np.pad(
                img,
                ((y_offset, y_offset), (x_offset, x_offset), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            h, w = dim, dim
        if (h, w) != FINAL_IMAGE_RESOLUTION:
            img = cv2.resize(img, FINAL_IMAGE_RESOLUTION, cv2.INTER_AREA)
        # Convert from (H, W, C) to (C, H, W)
        img = np.copy((np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0).clip(0.0, 1.0))
        return np.copy(img)

    def process_img_w_crop(self, img):
        h, w, _ = img.shape
        if (h, w) == FINAL_IMAGE_RESOLUTION:
            return img

        crop_size = self.image_crop_size
        # print(f"Cropping image to {crop_size}")
        img = img[crop_size[0] : crop_size[1], crop_size[2] : crop_size[3]]
        img_resized = cv2.resize(img, RESIZED_IMAGE_RESOLUTION, cv2.INTER_AREA)

        h, w = img_resized.shape[:2]
        if h != w:
            dim = max(h, w)
            y_offset = (dim - h) // 2
            x_offset = (dim - w) // 2
            img_resized = np.pad(
                img_resized,
                ((y_offset, y_offset), (x_offset, x_offset), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            h, w = dim, dim
        if (h, w) != FINAL_IMAGE_RESOLUTION:
            img_resized = cv2.resize(img_resized, FINAL_IMAGE_RESOLUTION, cv2.INTER_AREA)

        # Convert from (H, W, C) to (C, H, W)
        img_resized = np.copy(
            (np.transpose(img_resized, (2, 0, 1)).astype(np.float32) / 255.0).clip(0.0, 1.0)
        )
        return np.copy(img_resized)

    def get_gearbc_observation(self, raw_obs, reward=-1):
        obs = {}
        temp_obs = self.key_converter.map_obs(raw_obs)
        for k, v in temp_obs.items():
            if k.startswith("hand.") or k.startswith("body."):
                obs[k[5:] + "_state"] = v
            else:
                raise ValueError(f"Unknown key: {k}")
        mapped_names, camera_names, _, _ = self.key_converter.get_camera_config()
        for mapped_name, camera_name in zip(mapped_names, camera_names):
            obs[camera_name + "_image"] = self.process_img(raw_obs[camera_name + "_image"])
        obs["caption"] = raw_obs["language"]

        self.render_cache = np.copy(
            (np.transpose(obs[self.render_obs_key], (1, 2, 0)) * 255.0).astype(np.uint8)
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(FINAL_IMAGE_RESOLUTION) / 1000
        color = (255, 255, 0)  # Yellow text
        thickness = 1

        # Split text into lines of max 50 characters
        text = self.env.get_ep_meta().get("lang", "")
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= 50:  # +1 for space
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        if current_line:
            lines.append(" ".join(current_line))

        # Draw each line
        y_position = int(FINAL_IMAGE_RESOLUTION[0] * 0.9)
        x_position = int(FINAL_IMAGE_RESOLUTION[1] * 0.1)
        line_spacing = int(30 * font_scale)  # Adjust spacing between lines

        for line in lines:
            cv2.putText(
                self.render_cache,
                line,
                (x_position, y_position),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y_position += line_spacing  # Move down for next line

        # write reward
        cv2.putText(
            self.render_cache,
            f"Reward: {reward}",
            (
                int(0.1 * FINAL_IMAGE_RESOLUTION[1]),
                int(0.1 * FINAL_IMAGE_RESOLUTION[0]),
            ),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        # obs["annotation.human.action.task_description"] = raw_obs["language"]
        return obs

    def reset(self, seed=None, options=None):
        raw_obs, info = super().reset(seed=seed, options=options)
        obs = self.get_gearbc_observation(raw_obs)
        return obs, info

    def step(self, action):
        temp_action = action.copy()
        action = {}
        for k, v in temp_action.items():
            assert k.endswith("_action")
            action["action." + k[:-7]] = v
        for k, v in action.items():
            self.verbose and print("<ACTION>", k, v)

        action = self.key_converter.unmap_action(action)
        raw_obs, reward, terminated, truncated, info = super().step(action)
        obs = self.get_gearbc_observation(raw_obs, reward)

        for k, v in obs.items():
            self.verbose and print("<OBS>", k, v.shape if k.endswith("image") else v)
        self.verbose = False

        # if `reward > 0:
        #     import os
        #     import random
        #     import string
        #     # save the render cache
        #     random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        #     os.makedirs(f"/mnt/amlfs-01/home/runyud/workspace/outputs/random", exist_ok=True)
        #     cv2.imwrite(f"`/mnt/amlfs-01/home/runyud/workspace/outputs/random/render_{random_str}.png", self.render_cache[..., ::-1])

        return obs, reward, terminated, truncated, info


def create_gearbcrobocasa_env_class(env, robot, robot_alias):
    class_name = f"{env}_{robot}_Env"
    id_name = f"gearbc/{class_name}"

    env_class_type = type(
        class_name,
        (GearBCRoboCasaEnv,),
        {
            "__init__": lambda self, **kwargs: super(self.__class__, self).__init__(
                env_name=env,
                robots_name=robot,
                **kwargs,
            )
        },
    )

    current_module = sys.modules["robocasa.utils.gym_utils.gymnasium_gearbc"]
    setattr(current_module, class_name, env_class_type)
    register(
        id=id_name,  # Unique ID for the environment
        entry_point=f"robocasa.utils.gym_utils.gymnasium_gearbc:{class_name}",  # Path to your environment class
    )


for ENV in REGISTERED_ENVS:
    for ROBOT, ROBOT_ALIAS in GROOT_ROBOCASA_ENVS_ROBOTS.items():
        create_gearbcrobocasa_env_class(ENV, ROBOT, ROBOT_ALIAS)
