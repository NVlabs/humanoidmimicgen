# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import yaml


def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set the path to the LEGGED_GYM_ROOT_DIR using relative path
    current_file_dir = os.path.dirname(os.path.abspath(config_path))
    LEGGED_GYM_ROOT_DIR = os.path.join(current_file_dir, "..", "HomieRL", "legged_gym")
    LEGGED_GYM_ROOT_DIR = os.path.abspath(LEGGED_GYM_ROOT_DIR)

    # Process paths with LEGGED_GYM_ROOT_DIR
    for path_key in ["policy_path", "xml_path", "onnx_policy_path"]:
        if path_key in config:
            config[path_key] = config[path_key].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

    # Convert lists to numpy arrays where needed
    array_keys = ["kps", "kds", "default_angles", "cmd_scale", "cmd_init"]
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)

    return config, LEGGED_GYM_ROOT_DIR


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    q_conj = np.array([w, -x, -y, -z])

    return np.array(
        [
            v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
            + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
            v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
            + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
            v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
            + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
            + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
        ]
    )


def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)
