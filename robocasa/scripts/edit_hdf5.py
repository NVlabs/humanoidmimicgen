import argparse
import h5py


def print_hdf5_info(file_path):
    """Print a brief introduction to an HDF5 file, including groups, datasets, and attributes."""
    with h5py.File(file_path, "r") as hdf_file:
        print(f"HDF5 File: {file_path}")

        def explore_group(group, indent=0):
            """Recursively explore groups and datasets in the HDF5 file."""
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Group):
                    print(" " * indent + f"📁 Group: {key}")
                    explore_group(item, indent + 4)
                elif isinstance(item, h5py.Dataset):
                    print(
                        " " * indent
                        + f"📄 Dataset: {key}, Shape: {item.shape}, Dtype: {item.dtype}"
                    )

        explore_group(hdf_file)

        print("\nFile Attributes:")
        for attr, value in hdf_file.attrs.items():
            print(f"  {attr}: {value}")
        print("File Attributes End\n")


def add_missing_info(file_path):
    """Add environment information to an HDF5 file."""
    with h5py.File(file_path, "a") as f:
        if "env_info" not in f["data"].attrs:
            env_name = f["data"].attrs["env"]
            f["data"].attrs["env_info"] = (
                '{{"env_name": "{}", "robots": "PandaMobile", "controller_configs": {{'
                '"type": "OSC_POSE", "input_max": 1, "input_min": -1, '
                '"output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], '
                '"output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], '
                '"kp": 150, "damping_ratio": 1, "impedance_mode": "fixed", '
                '"kp_limits": [0, 300], "damping_ratio_limits": [0, 10], '
                '"position_limits": null, "orientation_limits": null, '
                '"uncouple_pos_ori": true, "control_delta": true, '
                '"interpolation": null, "ramp_ratio": 0.2'
                '}}, "layout_ids": -1, "style_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 11], '
                '"translucent_robot": true, "obj_instance_split": "A"}}'
            ).format(env_name)
        if "repository_version" not in f["data"].attrs:
            f["data"].attrs["repository_version"] = "1.4.1"


def remove_hdf5_groups(file_path, group_paths, is_one_based):
    """Remove multiple specified groups or datasets from an HDF5 file."""
    with h5py.File(file_path, "a") as hdf_file:  # Open in append mode to modify
        for group_path in group_paths:
            if group_path in hdf_file:
                del hdf_file[group_path]
                print(f"Removed: {group_path}")
            else:
                print(f"Not found: {group_path}")
        old_keys = list(hdf_file["data"].keys())
        old_ids = [int(key.split("_")[-1]) for key in old_keys]
        sorted_old_keys = [key for _, key in sorted(zip(old_ids, old_keys))]
        print("Sorted old keys:", sorted_old_keys, "\n")
        for id, old_key in enumerate(sorted_old_keys):
            new_key = "demo_" + str(id + int(is_one_based))
            hdf_file["data"].move(old_key, new_key)
            print(f"Renamed: {old_key} -> {new_key}")


def merge_hdf5_groups(file_paths):
    """Merge multiple HDF5 files into one, keeping only the 'data' group."""
    with h5py.File(file_paths[0], "a") as merged_file:
        max_id = 0
        for key in merged_file["data"].keys():
            max_id = max(max_id, int(key.split("_")[-1]))
        for file_path in file_paths[1:]:
            with h5py.File(file_path, "r") as hdf_file:
                assert "data" in hdf_file, f"{file_path} does not contain 'data' group."
                for key in hdf_file["data"]:
                    max_id += 1
                    new_key = "demo_" + str(max_id)
                    merged_file["data"].copy(hdf_file["data"][key], new_key)
    print(file_paths[0], max_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read and display the structure of an HDF5 file. For example: python robocasa/scripts/edit_hdf5.py --filepath ../collected_demo/PandaOmron_TurnOnStove.hdf5 --add-missing-info --keys data/demo_12. Note that the demo can start with index 0 or 1."
    )
    parser.add_argument("--filepath", type=str, help="Path to the HDF5 file.")
    parser.add_argument(
        "--is-one-based",
        action="store_true",
        help="Whether the demo index starts from 1.",
    )
    parser.add_argument(
        "--add-missing-info",
        action="store_true",
        help="Add missing information to the HDF5 file.",
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="*",
        default=[],
        help="One or more group/dataset paths to delete.",
    )

    args = parser.parse_args()
    print_hdf5_info(args.filepath)
    if args.add_missing_info:
        add_missing_info(args.filepath)
    if len(args.keys) > 0:
        remove_hdf5_groups(args.filepath, args.keys, args.is_one_based)
    from robocasa.utils.robomimic.robomimic_dataset_utils import (
        convert_to_robomimic_format,
    )

    convert_to_robomimic_format(args.filepath)
    # print_hdf5_info(args.filepath)
