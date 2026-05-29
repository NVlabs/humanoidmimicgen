# RoboCasa Scripts

A collection of scripts for downloading assets, collecting demos, and managing objects in RoboCasa.

## Common Scripts

- **`download_kitchen_assets.py`**
  > **⚠️ Legacy Script:** Please switch to `assetkit pull` if possible.

  Downloads original RoboCasa kitchen assets.  
  *Example Usage:*

  ```bash
  python robocasa/scripts/download_kitchen_assets.py -y
  ```

- **`download_groot_assets.py`**  
  > **⚠️ Legacy Script:** Please switch to `assetkit pull` if possible.

  Downloads groot assets, including Sketchfab, Infinigen, and Omniverse.  
  *Example Usage:*

  ```bash
  python robocasa/scripts/download_groot_assets.py -y
  ```

- **`download_datasets.py`**
  Downloads datasets for the original RoboCasa tasks.  
  See the [Dataset Docs](https://robocasa.ai/docs/use_cases/downloading_datasets.html) for more details.  
  *Example Usage:*

  ```bash
  python robocasa/scripts/download_datasets.py --ds_types human_im
  ```

- **`collect_demos.py`**  
  Collects demonstration trajectories for any task and environment.  
  *Example Usage:*

  ```bash
  python robocasa/scripts/collect_demos.py \
    --robot GR1ArmsOnly \
    --controller ../robosuite/robosuite/controllers/config/robots/default_gr1_fixed_lower_body.json \
    --device spacemouse \
    --pos-sensitivity 1 \
    --rot-sensitivity 1 \
    --obj_registries sketchfab objaverse \
    --ik_indicator \
    --camera "egoview" \
    --environment PnPOnionToBowl
  ```

- **`playback_dataset.py`**  
  Playbacks recorded trajectories from an hdf5 file, either rendered on screen or saved to a video.
  *Example Usage:*

  ```bash
  python robocasa/scripts/playback_dataset.py --dataset ./robocasa/models/assets/demonstrations_private/2025-01-16-17-59-06/demo.hdf5
  ```

- **`get_dataset_info.py`**  
  Reports information about a dataset, including trajectory length statistics, maximum and minimum action element, filter keys present, environment metadata, and the structure of the first demonstration.
  *Example Usage:*

  ```bash
  python robocasa/scripts/get_dataset_info.py --dataset ./robocasa/models/assets/demonstrations_private/2025-01-16-17-59-06/demo.hdf5
  ```

- **`setup_macros.py`**  
  Sets up a private macros file, allowing user-specific settings that are not tracked by git.
  *Example Usage:*

  ```bash
  python robocasa/scripts/setup_macros.py
  ```

---

## Internal Scripts

> **Note:** These scripts are primarily for internal development use.

- **`internal/fix_converted_objects.py`**  
  Fixes common issues in objects imported from Objaverse using older pipelines.  
  *Example Usage:*

  ```bash
  python robocasa/scripts/internal/fix_converted_objects.py --obj-registries sketchfab
  ```

- **`internal/inspect_objects_in_env.py`**  
  Iterates through all objects in a registry, loading each onto a tabletop scene for manual robot interaction and labeling (stable/unstable).  
  *Example Usage:*

  ```bash
  python robocasa/scripts/internal/inspect_objects_in_env.py \
    --asset_dir robocasa/models/assets/objects/objaverse
  ```

- **`internal/test_objects_manual.py`**  
  Loads objects into the MuJoCo viewer for manual inspection and labeling (good/bad).  
  *Example Usage:*

  ```bash
  python robocasa/scripts/internal/test_objects_manual.py \
    --asset_dir robocasa/models/assets/objects/objaverse
  ```
