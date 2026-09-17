# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small LeRobot dataset loader helpers used by WBC-goal replay."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import datasets
from datasets import load_dataset
import torch

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover - depends on installed LeRobot version.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


def hf_transform_to_torch_by_features(features, batch):
    """Keep dataset dtypes stable when Hugging Face batches are converted."""
    items_dict = dict(batch)
    for key, values in list(items_dict.items()):
        if key not in features:
            continue
        feature = features[key]
        if isinstance(feature, datasets.Sequence):
            dtype_str = feature.feature.dtype
        elif hasattr(feature, "dtype"):
            dtype_str = feature.dtype
        elif isinstance(feature, dict) and "dtype" in feature:
            dtype_str = feature["dtype"]
        else:
            continue
        dtype_mapping = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        if dtype_str in dtype_mapping:
            items_dict[key] = [torch.tensor(x, dtype=dtype_mapping[dtype_str]) for x in values]
    return items_dict


class TypedLeRobotDataset(LeRobotDataset):
    """LeRobotDataset variant that preserves numeric dtypes and can skip videos."""

    def __init__(self, load_video=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not load_video:
            video_keys = []
            for key, feature in self.meta.features.items():
                dtype = feature.get("dtype") if isinstance(feature, dict) else feature["dtype"]
                if dtype == "video":
                    video_keys.append(key)
            for key in video_keys:
                self.meta.features.pop(key)

    def load_hf_dataset(self) -> datasets.Dataset:
        if self.episodes is None:
            hf_dataset = load_dataset("parquet", data_dir=str(Path(self.root) / "data"), split="train")
        else:
            files = [str(Path(self.root) / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")
        hf_dataset.set_transform(partial(hf_transform_to_torch_by_features, hf_dataset.features))
        return hf_dataset

