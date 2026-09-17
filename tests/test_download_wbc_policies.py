# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

from humanoidmimicgen import download_wbc_policies


def test_download_policies_saves_under_runtime_name(tmp_path, monkeypatch):
    content = b"policy"
    checksum = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        download_wbc_policies,
        "POLICIES",
        {"stand.onnx": ("https://example.test/balance.onnx", checksum)},
    )

    def fake_urlretrieve(url, destination):
        assert url == "https://example.test/balance.onnx"
        destination.write_bytes(content)

    monkeypatch.setattr(download_wbc_policies, "urlretrieve", fake_urlretrieve)

    download_wbc_policies.download_policies(tmp_path)

    assert (tmp_path / "stand.onnx").read_bytes() == content


def test_download_policies_keeps_verified_file(tmp_path, monkeypatch):
    content = b"policy"
    checksum = hashlib.sha256(content).hexdigest()
    destination = tmp_path / "walk.onnx"
    destination.write_bytes(content)
    monkeypatch.setattr(
        download_wbc_policies,
        "POLICIES",
        {"walk.onnx": ("https://example.test/walk.onnx", checksum)},
    )

    def fail_download(*args):
        raise AssertionError("verified files should not be downloaded again")

    monkeypatch.setattr(download_wbc_policies, "urlretrieve", fail_download)

    download_wbc_policies.download_policies(tmp_path)
