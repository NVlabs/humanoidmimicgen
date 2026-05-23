IMAGE_TOPIC_NAME = "realsense/color/image_raw"
STATE_TOPIC_NAME = "G1Env/env_state_act"
CONTROL_GOAL_TOPIC = "ControlPolicy/upper_body_pose"
NAV_CMD_TOPIC = "NavigationPolicy/navigate_cmd"
ROBOT_CONFIG_TOPIC = "WBCPolicy/robot_config"
KEYBOARD_INPUT_TOPIC = "/keyboard_input"

# Camera topic definitions.
# Dictionary mapping camera configurations to their ROS topics.
# Structure:
# - Each key represents a camera configuration (e.g., "realsense", "zed")
# - Each value is a nested dictionary where:
#   - Multiple entries in a list represent different devices of the same type
#   - Different keys in the nested dict represent different data types from the same device
CAMERA_TOPICS = {
    "realsense": {
        "color_image": ["realsense/color/image_raw"],
        "depth_image": ["realsense/depth/image_raw"],
    },
    "realsense_dual": {
        "color_image": ["/realsense1/color/image_raw", "/realsense2/color/image_raw"],
        "depth_image": ["/realsense1/depth/image_raw", "/realsense2/depth/image_raw"],
    },
    "zed": {
        "left_image": ["zed/left/image_raw"],
        "right_image": ["zed/right/image_raw"],
    },
    "oak": {
        "color_image": ["oak/color/image_raw"],
    },
    "oak_with_mono": {
        "color_image": ["oak/color/image_raw"],
        "mono_left_image": ["oak/mono_left/image_raw"],
        "mono_right_image": ["oak/mono_right/image_raw"],
    },
    "dummy": {
        "color_image": ["dummy/color/image_raw"],
        "depth_image": ["dummy/depth/image_raw"],
    },
    "replay_dummy": {
        "color_image": ["realsense/color/image_raw"],
        "depth_image": ["realsense/depth/image_raw"],
    },
}

DEFAULT_NAV_CMD = [0.0, 0.0, 0.0]
DEFAULT_BASE_HEIGHT = 0.74  # TODO: Homie's default base height is 0.74
DEFAULT_WRIST_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2  # x, y, z + w, x, y, z
