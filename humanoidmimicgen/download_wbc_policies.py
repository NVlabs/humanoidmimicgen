# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the separately licensed lower-body policies used by WBC replay."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve


UPSTREAM_COMMIT = "4141c34280abb67c82e115342a8720f4a83d750d"
UPSTREAM_ROOT = (
    "https://github.com/NVlabs/GR00T-WholeBodyControl/raw/"
    f"{UPSTREAM_COMMIT}/decoupled_wbc/sim2mujoco/resources/robots/g1/policy"
)
POLICIES = {
    "stand.onnx": (
        f"{UPSTREAM_ROOT}/GR00T-WholeBodyControl-Balance.onnx",
        "f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18",
    ),
    "walk.onnx": (
        f"{UPSTREAM_ROOT}/GR00T-WholeBodyControl-Walk.onnx",
        "7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407",
    ),
}
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "wbc"
    / "external_dependencies"
    / "sim2mujoco"
    / "resources"
    / "robots"
    / "g1"
    / "policy"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_policies(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_sha256) in POLICIES.items():
        destination = output_dir / filename
        if destination.is_file() and file_sha256(destination) == expected_sha256:
            print(f"verified {destination}")
            continue

        temporary = destination.with_suffix(".onnx.download")
        print(f"downloading {url}")
        urlretrieve(url, temporary)
        actual_sha256 = file_sha256(temporary)
        if actual_sha256 != expected_sha256:
            temporary.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        temporary.replace(destination)
        print(f"saved {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Policy destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    download_policies(args.output_dir.expanduser().resolve())
    print("Licensed by NVIDIA Corporation under the NVIDIA Open Model License.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
