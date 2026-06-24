# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the nine paper loco-manipulation environments from dataset states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PAPER_TASKS = [
    (
        "01_box_lift_floor",
        "Box Lift Floor",
        "benchmark_dec_7_mid_conservative/G1_LMBoxLiftFloor",
    ),
    ("02_push_button", "Push Button", "G1_LMPushButton_20260129_234556"),
    ("03_box_lift", "Box Lift", "G1_LMBoxLift_20260129_230924"),
    (
        "04_push_shelf_forward",
        "Push Shelf Forward",
        "benchmark_dec_7_mid_conservative/G1_LMPushShelfForward",
    ),
    ("05_drill_lift", "Drill Lift", "G1_LMDrillLift_20260129_231155"),
    ("06_drill_pnp", "Drill Pick and Place", "G1_LMDrillPnP90_20260129_231446"),
    (
        "07_box_table_to_shelf",
        "Box Table to Shelf",
        "new_src_demos_jan_21/G1_LMBoxTableToShelfStaticIndustrial_Again",
    ),
    (
        "08_pick_drill_from_holder",
        "Pick Drill from Holder",
        "G1_LMPickDrillFromHolderStandingEasyFar_20260416_090757",
    ),
    (
        "09_obstacle_aware_pick_drill",
        "Obstacle-Aware Drill Lift",
        "new_src_demos_jan_21/G1_LMDrillLiftObstacle",
    ),
]

PLAYBACK_CONFIG_KEYS = [
    "wbc_version",
    "wbc_model_path",
    "wbc_policy_class",
    "control_frequency",
    "enable_waist",
    "robot",
    "task_name",
    "data_collection_frequency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one representative frame for each paper environment. DATA_ROOT "
            "should point at the local retained LeRobot dataset root used by "
            "scripts/run_wbc_goal_benchmark.sh."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else None,
        help="Root containing the retained LeRobot datasets. Defaults to DATA_ROOT.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/hmg_paper_9env_renderings"),
        help="Directory for individual PNGs and manifest.json.",
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        default=Path("/tmp/hmg_paper_9env_render_grid.png"),
        help="Output path for the 3x3 PNG grid.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("/tmp/hmg_paper_9env_renderings.zip"),
        help="Output path for the individual PNG bundle.",
    )
    parser.add_argument("--sim-frequency", type=int, default=200)
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def resolve_dataset(data_root: Path, dataset_rel: str) -> Path:
    dataset = (data_root / dataset_rel).expanduser().resolve()
    if not (dataset / "meta" / "episodes.jsonl").exists():
        raise FileNotFoundError(f"Missing LeRobot dataset metadata: {dataset}")
    return dataset


def load_first_state(dataset_path: Path):
    from humanoidmimicgen.lerobot_dataset import TypedLeRobotDataset

    with (dataset_path / "meta" / "episodes.jsonl").open() as file:
        first_episode = json.loads(next(file))

    dataset = TypedLeRobotDataset(repo_id="tmp/render", root=dataset_path, load_video=False)
    frame = dataset[first_episode.get("start_index", 0)]
    state_len = int(np.array(frame["observation.sim.mujoco_state_len"]).item())
    state = np.array(frame["observation.sim.mujoco_state"][:state_len])
    seed = np.array(frame["observation.sim.seed"]).item()
    return seed, state, dataset.meta.info["script_config"]


def make_config(dataset_path: Path, script_config: dict, sim_frequency: int):
    from humanoidmimicgen.wbc_goal_playback import SyncSimPlaybackConfig

    config = SyncSimPlaybackConfig()
    config.dataset = str(dataset_path)
    config.enable_offscreen = True
    config.enable_onscreen = False
    config.save_video = False
    config.sim_frequency = sim_frequency
    config.update(script_config, allowed_keys=PLAYBACK_CONFIG_KEYS)
    return config


def render_state(config, seed, state, width: int, height: int) -> np.ndarray:
    from humanoidmimicgen.wbc_runtime import get_env

    env = get_env(config, onscreen=False, offscreen=True)
    try:
        env.reset(seed=seed)
        env.reset_to({"states": state})
        base_env = env.base_env if hasattr(env, "base_env") else env
        render_camera = getattr(base_env, "render_camera", ["frontview"])
        camera_name = render_camera[0] if isinstance(render_camera, list) else render_camera
        return np.flipud(
            base_env.sim.render(width=width, height=height, camera_name=camera_name)
        )
    finally:
        env.close()


def label_image(image: Image.Image, title: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    margin = 10
    bbox = draw.textbbox((0, 0), title)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    rect = [margin - 5, margin - 5, margin + text_width + 8, margin + text_height + 8]
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((margin, margin), title, fill=(255, 255, 255))
    return image


def write_grid(images: list[Image.Image], grid_path: Path) -> None:
    if len(images) != 9:
        raise ValueError(f"Expected 9 images, got {len(images)}")
    width, height = images[0].size
    grid = Image.new("RGB", (width * 3, height * 3))
    for index, image in enumerate(images):
        grid.paste(image, ((index % 3) * width, (index // 3) * height))
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(grid_path)


def write_zip(out_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out_dir.glob("*.png")) + [out_dir / "manifest.json"]:
            bundle.write(path, arcname=path.name)


def main() -> int:
    args = parse_args()
    if args.data_root is None:
        raise SystemExit("Set DATA_ROOT or pass --data-root.")

    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    os.environ["HMG_SIMULATION_TIMESTEP"] = str(1.0 / float(args.sim_frequency))
    data_root = args.data_root.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    images = []
    for label, title, dataset_rel in PAPER_TASKS:
        dataset_path = resolve_dataset(data_root, dataset_rel)
        seed, state, script_config = load_first_state(dataset_path)
        config = make_config(dataset_path, script_config, args.sim_frequency)
        rgb = render_state(config, seed, state, args.width, args.height)
        image = label_image(Image.fromarray(rgb), title)
        png_path = args.out_dir / f"{label}.png"
        image.save(png_path)
        images.append(image)
        manifest.append(
            {
                "label": label,
                "title": title,
                "env_name": script_config["task_name"],
                "dataset": str(dataset_path),
                "seed": int(seed),
                "png": str(png_path),
            }
        )
        print(f"rendered {label}: {png_path}")

    with (args.out_dir / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    write_grid(images, args.grid_path)
    write_zip(args.out_dir, args.zip_path)
    print(f"grid: {args.grid_path}")
    print(f"zip: {args.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
