#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train the HMG 64/50 Diffusion Policy using upstream LeRobot.

Install the exact tested runtime first::

    pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@8fff0fde7c79f23a93d845d1a50e985de01f8b8a"

Inputs are LeRobot v3 datasets projected to ``observation.state`` (43),
``observation.images.ego_view``, and ``action`` (35). When DATASET_ROOT contains
multiple shards, they are discovered automatically from their metadata.
"""

from __future__ import annotations

import argparse
import copy
from importlib import metadata
import json
from pathlib import Path
import sys


LEROBOT_REPOSITORY = "https://github.com/huggingface/lerobot.git"
LEROBOT_VERSION = "0.4.4"
LEROBOT_COMMIT = "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
STATE_KEY = "observation.state"
IMAGE_KEY = "observation.images.ego_view"
ACTION_KEY = "action"
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


def discover_repo_ids(root: Path) -> list[str]:
    """Return deterministic repo IDs for every dataset nested under root."""
    root = root.expanduser().resolve()
    if (root / "meta/info.json").is_file():
        return []
    repo_ids = sorted(
        {
            info_path.parent.parent.relative_to(root).as_posix()
            for info_path in root.rglob("meta/info.json")
        }
    )
    if not repo_ids:
        raise FileNotFoundError(f"No LeRobot v3 datasets found under {root}")
    if len(repo_ids) == 1:
        raise ValueError(
            f"Found one nested dataset ({repo_ids[0]}). Pass its directory as "
            "DATASET_ROOT, or place all task shards under the supplied root."
        )
    return repo_ids


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
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-repo-id", default="local/hmg_dp_example")
    parser.add_argument("--job-name", default="hmg_dp_example")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    global _MULTI_REPO_IDS
    args = parse_args(argv)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    _MULTI_REPO_IDS = discover_repo_ids(args.dataset_root)
    roots = (
        [args.dataset_root / repo_id for repo_id in _MULTI_REPO_IDS]
        if _MULTI_REPO_IDS
        else [args.dataset_root]
    )
    for root in roots:
        validate_dataset(root)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    provenance = require_lerobot_runtime()
    print("HMG_LEROBOT_UPSTREAM " + json.dumps(provenance, sort_keys=True), flush=True)
    print(
        "HMG_DATASETS "
        + json.dumps(
            {
                "root": str(args.dataset_root.expanduser().resolve()),
                "repo_ids": _MULTI_REPO_IDS or [args.dataset_repo_id],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    import lerobot.scripts.lerobot_train as trainer

    if _MULTI_REPO_IDS:
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
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={args.dataset_root.expanduser().resolve()}",
        "--dataset.video_backend=pyav",
        "--dataset.use_imagenet_stats=true",
        f"--output_dir={args.output_dir.expanduser().resolve()}",
        f"--job_name={args.job_name}",
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
