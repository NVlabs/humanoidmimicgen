import argparse
import os
import time
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

import mujoco
import mujoco.viewer
import numpy as np
from termcolor import colored
from pynput.keyboard import Listener
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.mjcf_utils import find_elements


def edit_model_xml(xml_str):
    """
    Edit the model xml with custom changes, including resolving relative paths.
    """
    tree = ET.fromstring(xml_str)
    root = tree
    return ET.tostring(root, encoding="utf8").decode("utf8")


class ObjectValidator:
    def __init__(self, asset_folder):
        self.asset_folder = asset_folder
        self.results = {}
        self.stats = defaultdict(lambda: {"good": 0, "bad": 0})
        self._reset_state = 0
        self._valid_state = 0
        self._invalid_state = 0
        self._quit_state = 0

        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

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
        """Load the MJCF model"""
        try:
            # Store original working directory
            original_dir = os.getcwd()

            # Convert to absolute path
            filepath = os.path.abspath(filepath)

            with open(filepath, "r") as file:
                xml = file.read()

            xml = edit_model_xml(xml)
            root = ET.fromstring(xml)

            # add white background
            asset = find_elements(root, tags="asset")
            skybox = ET.fromstring(
                """<texture builtin="flat" height="256" rgb1="1 1 1" rgb2="1 1 1" type="skybox" width="256"/>"""
            )
            asset.append(skybox)

            # add lighting
            worldbody = find_elements(root, tags="worldbody")
            light = ET.fromstring(
                """<light pos="2.0 -2.0 2.0" dir="0.01 0.01 -1" specular="0.3 0.3 0.3" ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" directional="true" castshadow="false"/>"""
            )
            worldbody.append(light)

            # Change to model directory for loading
            os.chdir(os.path.dirname(filepath))

            xml = ET.tostring(root, encoding="unicode")
            model = mujoco.MjModel.from_xml_string(xml)
            sim = MjSim(model)

            # Restore original working directory
            os.chdir(original_dir)

            return sim

        except Exception as e:
            print(f"Error loading model: {e}")
            # Restore original working directory in case of error
            os.chdir(original_dir)
            return None

    def render_model(self, sim):
        """Render the model"""
        viewer = mujoco.viewer.launch_passive(
            sim.model._model,
            sim.data._data,
            show_right_ui=False,
        )

        viewer.cam.distance = 0.3
        viewer.cam.elevation = -30

        return viewer

    def validate_objects(self):
        objects = self.get_all_objects()
        print(colored(f"\nFound {len(objects)} objects to validate", "cyan"))
        print(colored("Press Y for valid object, N for broken object, Q to quit\n", "yellow"))

        for xml_path, category in objects:
            obj_name = os.path.basename(os.path.dirname(xml_path))
            print(colored(f"\nTesting: {category}/{obj_name}", "cyan"))

            sim = self.read_model(xml_path)
            if sim is None:
                self.results[f"{category}/{obj_name}"] = False
                self.stats[category]["bad"] += 1
                continue

            self._valid_state = 0
            self._invalid_state = 0
            viewer = self.render_model(sim)

            while viewer.is_running():
                if self._valid_state:
                    self.results[f"{category}/{obj_name}"] = True
                    self.stats[category]["good"] += 1
                    break
                elif self._invalid_state:
                    self.results[f"{category}/{obj_name}"] = False
                    self.stats[category]["bad"] += 1
                    break
                elif self._quit_state:
                    viewer.close()
                    return

                time.sleep(0.1)

            viewer.close()

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
                "good_percentage": round(total_good / total * 100, 1) if total > 0 else 0,
                "bad_percentage": round(total_bad / total * 100, 1) if total > 0 else 0,
            },
        }

        output_file = "object_validation_results.json"
        with open(output_file, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tool for manual inspection of objects in mujoco. Loads objects sequentially from an object registry folder or XML file, allows users to validate their appearance, and exports the validation results as JSON."
    )
    parser.add_argument(
        "--asset_dir",
        type=str,
        required=True,
        help="Path to the folder containing object assets",
    )
    args = parser.parse_args()

    validator = ObjectValidator(args.asset_dir)
    validator.validate_objects()
    validator.print_summary()


if __name__ == "__main__":
    main()
