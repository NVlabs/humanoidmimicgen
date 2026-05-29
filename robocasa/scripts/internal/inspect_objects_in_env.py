import argparse
from copy import deepcopy
import os
import time
import json
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict

import mujoco
import mujoco.viewer
import numpy as np
from termcolor import colored
from pynput.keyboard import Listener
import robosuite
import robocasa
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.mjcf_utils import find_elements
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper


class ObjectValidator:
    def __init__(self, asset_folder=None):
        self.asset_folder = asset_folder
        self.results = {}
        self.stats = defaultdict(lambda: {"good": 0, "bad": 0})
        self._reset_state = 0
        self._valid_state = 0
        self._invalid_state = 0
        self._quit_state = 0

        self.listener = Listener(on_press=self.on_press)
        self.listener.start()
        controller_config = load_composite_controller_config(robot="GR1FixedLowerBody")
        self.env_config = {
            "env_name": "TabletopObjectShowcase",
            "robots": "GR1FixedLowerBody",
            "controller_configs": controller_config,
            "has_renderer": True,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "camera_names": "robot0_agentview_center",
            "control_freq": 20,
            "ignore_done": True,
            "translucent_robot": True,
            "renderer": "mjviewer",
            "obj_registries": [
                "objaverse",
                "aigen",
                "infinigen",
                "sketchfab",
                "lightwheel",
            ],
        }

    def on_press(self, key):
        try:
            if key.char == "y":
                self._valid_state = 1
            elif key.char == "n":
                self._invalid_state = 1
            elif key.char == "q":
                self._quit_state = 1
        except AttributeError:
            pass

    def get_all_objects(self):
        """Get all object XML paths recursively"""
        object_paths = []
        for root, dirs, files in os.walk(self.asset_folder):
            for file in files:
                if file == "model.xml":
                    full_path = os.path.join(root, file)
                    category = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
                    object_paths.append((full_path, category))
        return object_paths

    def read_model(self, filepath):
        """Load the object in a robosuite tabletop environment"""
        try:
            filepath = os.path.abspath(filepath)
            self.env_config["obj_groups"] = filepath
            env = robosuite.make(**self.env_config)
            env = VisualizationWrapper(env)
            zero_action = np.zeros(env.action_dim)
            env.step(zero_action)
            return env

        except Exception as e:
            traceback.print_exc()
            print(f"Error loading model: {e}")
            return None

    def validate_objects(self, input_device: str, mjcf=None):
        if mjcf:
            objects = [(mjcf, os.path.basename(os.path.dirname(os.path.dirname(mjcf))))]
        else:
            objects = self.get_all_objects()
            print(colored(f"\nFound {len(objects)} objects to validate", "cyan"))

        print(colored("Press Y for valid object, N for broken object, Q to quit\n", "yellow"))

        for xml_path, category in objects:
            obj_name = os.path.basename(os.path.dirname(xml_path))
            print(colored(f"\nTesting: {category}/{obj_name}", "cyan"))

            env = self.read_model(xml_path)
            if env is None:
                self.results[f"{category}/{obj_name}"] = False
                self.stats[category]["bad"] += 1
                continue

            self._valid_state = 0
            self._invalid_state = 0

            if input_device == "keyboard":
                from robosuite.devices import Keyboard

                device = Keyboard(
                    env=env,
                )
            elif input_device == "spacemouse":
                from robosuite.devices import SpaceMouse

                device = SpaceMouse(
                    env=env,
                )
            else:
                raise ValueError

            # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
            all_prev_gripper_actions = [
                {
                    f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                    for robot_arm in robot.arms
                    if robot.gripper[robot_arm].dof > 0
                }
                for robot in env.robots
            ]

            device.start_control()
            # Keep rendering until user input
            while True:
                start = time.time()
                active_robot = env.robots[device.active_robot]
                active_arm = device.active_arm
                input_ac_dict = device.input2action(mirror_actions=True)
                action_dict = deepcopy(input_ac_dict)
                for arm in active_robot.arms:
                    controller_input_type = active_robot.part_controllers[arm].input_type
                    if controller_input_type == "delta":
                        action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                    elif controller_input_type == "absolute":
                        action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                    else:
                        raise ValueError
                env_action = [
                    robot.create_action_vector(all_prev_gripper_actions[i])
                    for i, robot in enumerate(env.robots)
                ]
                env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)
                obs, _, _, _ = env.step(env_action)
                env.render()

                if self._valid_state:
                    self.results[f"{category}/{obj_name}"] = True
                    self.stats[category]["good"] += 1
                    break
                elif self._invalid_state:
                    self.results[f"{category}/{obj_name}"] = False
                    self.stats[category]["bad"] += 1
                    break
                elif self._quit_state:
                    env.close()
                    return

                elapsed = time.time() - start
                diff = 1 / 20 - elapsed
                if diff > 0:
                    time.sleep(diff)

            env.close()

    def print_summary(self):
        print("\n" + "=" * 50)
        print(colored("VALIDATION RESULTS:", "green"))
        print("=" * 50)

        print("\nDetailed Results:")
        for obj, is_valid in self.results.items():
            status = colored("✓ GOOD", "green") if is_valid else colored("✗ BAD", "red")
            print(f"{obj}: {status}")

        print("\nSummary by Category:")
        total_good = 0
        total_bad = 0
        for category, counts in self.stats.items():
            good = counts["good"]
            bad = counts["bad"]
            total = good + bad
            if total > 0:
                print(f"\n{category}:")
                print(f"  Total: {total}")
                print(f"  Good:  {colored(good, 'green')} ({good/total*100:.1f}%)")
                print(f"  Bad:   {colored(bad, 'red')} ({bad/total*100:.1f}%)")
                total_good += good
                total_bad += bad

        print("\nOverall Summary:")
        total = total_good + total_bad
        if total > 0:
            print(f"Total objects tested: {total}")
            print(f"Good objects: {colored(total_good, 'green')} ({total_good/total*100:.1f}%)")
            print(f"Bad objects:  {colored(total_bad, 'red')} ({total_bad/total*100:.1f}%)")

        # Save results to JSON
        json_output = {
            "detailed_results": self.results,
            "category_stats": dict(self.stats),
            "overall_stats": {
                "total_objects": total,
                "total_good": total_good,
                "total_bad": total_bad,
                "good_percentage": (round(total_good / total * 100, 1) if total > 0 else 0),
                "bad_percentage": round(total_bad / total * 100, 1) if total > 0 else 0,
            },
        }

        output_file = "object_validation_results.json"
        with open(output_file, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tool for manual inspection of objects in a robocasa environment. Loads objects sequentially from an object registry folder or XML file, allows users to validate their appearance and behavior, and exports the validation results as JSON."
    )
    parser.add_argument(
        "--asset_dir",
        type=str,
        required=False,
        help="Path to the folder containing object assets",
    )
    parser.add_argument(
        "--mjcf",
        type=str,
        required=False,
        help="Path to a specific XML file to validate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        help="Device to use for input, either keyboard or spacemouse",
    )
    args = parser.parse_args()

    if not args.asset_dir and not args.mjcf:
        parser.error("Either --asset_dir or --mjcf must be provided")

    validator = ObjectValidator(args.asset_dir)
    validator.validate_objects(args.device, mjcf=args.mjcf)
    validator.print_summary()


if __name__ == "__main__":
    main()
