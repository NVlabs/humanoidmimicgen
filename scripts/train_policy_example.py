#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download, prepare, and train an HMG 64/50 policy using upstream LeRobot.

Install the exact tested runtime first::

    pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@8fff0fde7c79f23a93d845d1a50e985de01f8b8a"

Select one of the nine released tasks. This script downloads all eight shards
from the pinned public HMG dataset, converts them to LeRobot v3, projects them
to ``observation.state`` (43), ``observation.images.ego_view``, and ``action``
(35), and then starts training.
"""

from __future__ import annotations

import argparse
import copy
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


LEROBOT_REPOSITORY = "https://github.com/huggingface/lerobot.git"
LEROBOT_VERSION = "0.4.4"
LEROBOT_COMMIT = "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
DATASET_REPOSITORY = "linkenv/humanoidmimicgen-g1-benchmark"
DATASET_REVISION = "acf6b53853dd0199bf852286ca68067cc37472ff"
TASKS = (
    "01_box_lift_floor",
    "02_push_button",
    "03_box_lift",
    "04_push_shelf_forward",
    "05_drill_lift",
    "06_drill_pnp",
    "07_box_table_to_shelf",
    "08_pick_drill_from_holder",
    "09_obstacle_aware_pick_drill",
)
STATE_KEY = "observation.state"
IMAGE_KEY = "observation.images.ego_view"
ACTION_KEY = "action"
ACTION_SOURCE_KEY = "observation.sim.target_upper_body_pose"
NAVIGATION_SOURCE_KEY = "teleop.navigate_command"
BASE_HEIGHT_SOURCE_KEY = "teleop.base_height_command"
SYSTEM_FEATURE_KEYS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
UPPER_BODY_DIM = 31
NAVIGATION_DIM = 3
BASE_HEIGHT_DIM = 1
ACTION_DIM = UPPER_BODY_DIM + NAVIGATION_DIM + BASE_HEIGHT_DIM
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}
_MULTI_REPO_IDS: list[str] = []


def require_lerobot_runtime() -> dict[str, str]:
    """Fail unless LeRobot came from the exact tested source commit."""
    try:
        distribution = metadata.distribution("lerobot")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Install the pinned LeRobot runtime shown in --help") from exc
    direct_url = distribution.read_text("direct_url.json")
    if distribution.version != LEROBOT_VERSION or not direct_url:
        raise RuntimeError(
            f"LeRobot must be {LEROBOT_VERSION} installed from {LEROBOT_COMMIT}"
        )
    payload = json.loads(direct_url)
    commit = (payload.get("vcs_info") or {}).get("commit_id")
    if payload.get("url") != LEROBOT_REPOSITORY or commit != LEROBOT_COMMIT:
        raise RuntimeError(
            f"LeRobot must come from {LEROBOT_REPOSITORY}@{LEROBOT_COMMIT}"
        )
    return {
        "repository": LEROBOT_REPOSITORY,
        "version": LEROBOT_VERSION,
        "commit": LEROBOT_COMMIT,
    }


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _release_features(features: dict) -> dict:
    required = {
        STATE_KEY: 43,
        ACTION_SOURCE_KEY: UPPER_BODY_DIM,
        NAVIGATION_SOURCE_KEY: NAVIGATION_DIM,
        BASE_HEIGHT_SOURCE_KEY: BASE_HEIGHT_DIM,
    }
    for key, dimension in required.items():
        if features.get(key, {}).get("shape") != [dimension]:
            raise ValueError(f"Source feature {key} must have shape [{dimension}]")
    if features.get(IMAGE_KEY, {}).get("dtype") not in {"image", "video"}:
        raise ValueError(f"Source dataset is missing {IMAGE_KEY}")
    action_feature = copy.deepcopy(features[ACTION_SOURCE_KEY])
    action_feature.update(
        dtype="float32",
        shape=[ACTION_DIM],
        names=(
            [f"target_upper_body_pose_{index}" for index in range(UPPER_BODY_DIM)]
            + ["navigate_x", "navigate_y", "navigate_yaw", "base_height"]
        ),
    )
    projected = {
        STATE_KEY: copy.deepcopy(features[STATE_KEY]),
        IMAGE_KEY: copy.deepcopy(features[IMAGE_KEY]),
        ACTION_KEY: action_feature,
    }
    projected.update(
        {key: copy.deepcopy(features[key]) for key in SYSTEM_FEATURE_KEYS if key in features}
    )
    return projected


def _release_stats(stats: dict) -> dict:
    action_keys = (ACTION_SOURCE_KEY, NAVIGATION_SOURCE_KEY, BASE_HEIGHT_SOURCE_KEY)
    for key in (STATE_KEY, *action_keys):
        if key not in stats:
            raise ValueError(f"Source statistics are missing {key}")
    stat_names = set.intersection(*(set(stats[key]) for key in action_keys))
    action_stats = {}
    for stat_name in sorted(stat_names):
        values = [stats[key][stat_name] for key in action_keys]
        if all(isinstance(value, list) for value in values):
            action_stats[stat_name] = [item for value in values for item in value]
    if not action_stats:
        raise ValueError("WBC goal fields have no compatible vector statistics")
    return {
        STATE_KEY: copy.deepcopy(stats[STATE_KEY]),
        IMAGE_KEY: copy.deepcopy(stats.get(IMAGE_KEY, IMAGENET_STATS)),
        ACTION_KEY: action_stats,
    }


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def project_dataset(source: Path, output: Path) -> None:
    """Materialize the model-facing 43D/image/35D dataset atomically."""
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_info = _load_json(source / "meta/info.json")
    source_stats = _load_json(source / "meta/stats.json")
    features = _release_features(source_info.get("features", {}))
    stats = _release_stats(source_stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        shutil.copytree(source / "meta", temporary / "meta", dirs_exist_ok=True)
        _dump_json(temporary / "meta/stats.json", stats)

        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as parquet

        parquet_files = sorted((source / "data").rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files under {source / 'data'}")
        total_rows = 0
        action_sources = (
            ACTION_SOURCE_KEY,
            NAVIGATION_SOURCE_KEY,
            BASE_HEIGHT_SOURCE_KEY,
        )
        action_dimensions = (UPPER_BODY_DIM, NAVIGATION_DIM, BASE_HEIGHT_DIM)
        for parquet_path in parquet_files:
            table = parquet.read_table(parquet_path)
            missing = {STATE_KEY, *action_sources}.difference(table.column_names)
            if missing:
                raise ValueError(f"{parquet_path} is missing {sorted(missing)}")
            columns = [STATE_KEY, *action_sources]
            if IMAGE_KEY in table.column_names:
                columns.append(IMAGE_KEY)
            columns.extend(key for key in SYSTEM_FEATURE_KEYS if key in table.column_names)
            selected = table.select(columns)
            action_parts = []
            for name, dimension in zip(action_sources, action_dimensions, strict=True):
                values = np.asarray(selected[name].to_pylist(), dtype=np.float32)
                if values.ndim == 1 and dimension == 1:
                    values = values[:, None]
                if values.shape != (selected.num_rows, dimension):
                    raise ValueError(
                        f"{parquet_path}: {name} has shape {values.shape}; "
                        f"expected {(selected.num_rows, dimension)}"
                    )
                action_parts.append(values)
            action = np.concatenate(action_parts, axis=-1)
            action_column = pa.FixedSizeListArray.from_arrays(
                pa.array(action.reshape(-1), type=pa.float32()), ACTION_DIM
            )
            keep = [name for name in selected.column_names if name not in action_sources]
            selected = selected.select(keep).append_column(ACTION_KEY, action_column)
            destination = temporary / parquet_path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_table(selected, destination, compression="zstd")
            total_rows += selected.num_rows
        if total_rows != source_info.get("total_frames"):
            raise ValueError(
                f"Projected {total_rows} rows; metadata declares "
                f"{source_info.get('total_frames')}"
            )

        video_files = [
            path
            for path in sorted((source / "videos").rglob("*"))
            if path.is_file() and IMAGE_KEY in path.relative_to(source / "videos").parts
        ]
        if features[IMAGE_KEY]["dtype"] == "video" and not video_files:
            raise FileNotFoundError(f"No {IMAGE_KEY} videos under {source / 'videos'}")
        for video_path in video_files:
            _copy_file(video_path, temporary / video_path.relative_to(source))

        output_info = {**source_info, "features": features}
        if "total_videos" in output_info:
            output_info["total_videos"] = len(video_files)
        _dump_json(temporary / "meta/info.json", output_info)
        _dump_json(
            temporary / "meta/hmg_policy_contract.json",
            {
                "source": str(source),
                "model_features": [STATE_KEY, IMAGE_KEY, ACTION_KEY],
                "action_sources": list(action_sources),
                "rows": total_rows,
                "video_files": len(video_files),
            },
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_released_task(task: str, data_dir: Path) -> tuple[Path, list[str]]:
    """Download and prepare all eight public shards for one benchmark task."""
    from huggingface_hub import snapshot_download

    data_dir = data_dir.expanduser().resolve()
    download_root = data_dir / "source"
    v3_root = data_dir / "v3"
    projected_root = data_dir / "projected"
    repo_ids = [f"local/hmg_{task}_shard_{index:03d}" for index in range(8)]
    allow_patterns = [f"datasets/{task}/shard_{index:03d}/**" for index in range(8)]
    snapshot_download(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=allow_patterns,
        local_dir=download_root,
    )
    for index, repo_id in enumerate(repo_ids):
        source = download_root / "datasets" / task / f"shard_{index:03d}"
        v3_dataset = v3_root / repo_id
        projected = projected_root / repo_id
        if projected.exists():
            validate_dataset(projected)
            continue
        if not (v3_dataset / "meta/info.json").is_file():
            v3_dataset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, v3_dataset)
        info = _load_json(v3_dataset / "meta/info.json")
        if info.get("codebase_version") != "v3.0":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lerobot.datasets.v30.convert_dataset_v21_to_v30",
                    "--repo-id",
                    repo_id,
                    "--root",
                    str(v3_root),
                    "--push-to-hub",
                    "false",
                ],
                check=True,
            )
        project_dataset(v3_dataset, projected)
        validate_dataset(projected)
    return projected_root, repo_ids


def validate_dataset(root: Path) -> None:
    """Require the model-facing HMG state/image/WBC-goal schema."""
    info_path = root / "meta/info.json"
    stats_path = root / "meta/stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"Expected LeRobot v3 metadata under {root}")
    with info_path.open(encoding="utf-8") as handle:
        features = json.load(handle).get("features", {})
    with stats_path.open(encoding="utf-8") as handle:
        stats = json.load(handle)
    expected_shapes = {STATE_KEY: [43], ACTION_KEY: [35]}
    for key, shape in expected_shapes.items():
        if features.get(key, {}).get("shape") != shape:
            raise ValueError(f"{root}: {key} must have shape {shape}")
        if key not in stats:
            raise ValueError(f"{root}: statistics are missing {key}")
    if features.get(IMAGE_KEY, {}).get("dtype") not in {"image", "video"}:
        raise ValueError(f"{root}: missing {IMAGE_KEY} image/video feature")


def make_multi_dataset(cfg):
    """Enable LeRobot 0.4.4's existing multi-dataset for compatible shards."""
    import numpy as np
    import torch
    from lerobot.datasets.compute_stats import aggregate_stats as upstream_aggregate_stats
    from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
    from lerobot.datasets.transforms import ImageTransforms

    root = Path(cfg.dataset.root).expanduser().resolve()
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable
        else None
    )
    metas = [
        LeRobotDatasetMetadata(repo_id, root=root / repo_id, revision=cfg.dataset.revision)
        for repo_id in _MULTI_REPO_IDS
    ]
    reference = metas[0]
    for meta in metas[1:]:
        if meta.features != reference.features or meta.fps != reference.fps:
            raise ValueError(f"incompatible shards: {reference.repo_id}, {meta.repo_id}")
    delta_timestamps = resolve_delta_timestamps(cfg.policy, reference)

    def aggregate_stats_compat(stats_list):
        normalized = copy.deepcopy(stats_list)
        for dataset_stats in normalized:
            for feature, feature_stats in dataset_stats.items():
                count = feature_stats.get("count")
                if count is None:
                    continue
                flat = np.asarray(count).reshape(-1)
                if flat.size > 1:
                    if not np.all(flat == flat[0]):
                        raise ValueError(f"non-uniform count for {feature}")
                    feature_stats["count"] = flat[:1].copy()
        camera_keys = set(reference.camera_keys)
        aggregated = upstream_aggregate_stats(
            [
                {key: value for key, value in stats.items() if key not in camera_keys}
                for stats in normalized
            ]
        )
        for key in camera_keys:
            if key in normalized[0]:
                aggregated[key] = copy.deepcopy(normalized[0][key])
        return aggregated

    original_aggregate_stats = lerobot_dataset_module.aggregate_stats
    lerobot_dataset_module.aggregate_stats = aggregate_stats_compat
    try:
        dataset = MultiLeRobotDataset(
            _MULTI_REPO_IDS,
            root=root,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerances_s=dict.fromkeys(_MULTI_REPO_IDS, cfg.tolerance_s),
            video_backend=cfg.dataset.video_backend,
        )
    finally:
        lerobot_dataset_module.aggregate_stats = original_aggregate_stats

    meta = copy.copy(dataset._datasets[0].meta)
    meta.info = copy.deepcopy(meta.info)
    meta.info.update(
        total_episodes=dataset.num_episodes,
        total_frames=dataset.num_frames,
        splits={"train": f"0:{dataset.num_episodes}"},
    )
    meta.stats = dataset.stats
    episode_from, episode_to, frame_offset = [], [], 0
    for underlying in dataset._datasets:
        episode_from.extend(
            int(value) + frame_offset
            for value in underlying.meta.episodes["dataset_from_index"]
        )
        episode_to.extend(
            int(value) + frame_offset
            for value in underlying.meta.episodes["dataset_to_index"]
        )
        frame_offset += underlying.num_frames
    if frame_offset != dataset.num_frames:
        raise AssertionError((frame_offset, dataset.num_frames))
    if len(episode_from) != dataset.num_episodes:
        raise AssertionError((len(episode_from), dataset.num_episodes))
    meta.episodes = {
        "dataset_from_index": np.asarray(episode_from, dtype=np.int64),
        "dataset_to_index": np.asarray(episode_to, dtype=np.int64),
    }
    dataset.meta = meta
    dataset.episodes = None
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    print(
        "HMG_MULTI_DATASET "
        + json.dumps(
            {
                "repo_ids": _MULTI_REPO_IDS,
                "total_episodes": dataset.num_episodes,
                "total_frames": dataset.num_frames,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return dataset


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("~/.cache/humanoidmimicgen/policy_data"),
        help="download/preparation cache (default: %(default)s)",
    )
    parser.add_argument("--job-name")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    global _MULTI_REPO_IDS
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    provenance = require_lerobot_runtime()
    dataset_root, _MULTI_REPO_IDS = prepare_released_task(args.task, args.data_dir)
    dataset_repo_id = f"local/hmg_{args.task}_full8"
    job_name = args.job_name or f"hmg_{args.task}_long50"
    print("HMG_LEROBOT_UPSTREAM " + json.dumps(provenance, sort_keys=True), flush=True)
    print(
        "HMG_DATASETS "
        + json.dumps(
            {
                "source": DATASET_REPOSITORY,
                "revision": DATASET_REVISION,
                "task": args.task,
                "prepared_root": str(dataset_root),
                "repo_ids": _MULTI_REPO_IDS,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    import lerobot.scripts.lerobot_train as trainer

    trainer.make_dataset = make_multi_dataset
    sys.argv = [
        sys.argv[0],
        "--policy.type=diffusion",
        "--policy.push_to_hub=false",
        "--policy.n_obs_steps=1",
        "--policy.horizon=64",
        "--policy.n_action_steps=50",
        "--policy.drop_n_last_frames=49",
        "--policy.resize_shape=[256,256]",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        "--dataset.use_imagenet_stats=true",
        f"--output_dir={args.output_dir}",
        f"--job_name={job_name}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--batch_size={args.batch_size}",
        "--policy.device=cuda",
        f"--num_workers={args.num_workers}",
        "--wandb.enable=false",
    ]
    trainer.main()


if __name__ == "__main__":
    main()
