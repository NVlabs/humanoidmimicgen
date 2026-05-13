# HumanoidMimicGen

HumanoidMimicGen contains RoboCasa-based humanoid loco-manipulation
environment code prepared for an Apache License, Version 2.0 open source
release.

## Release Compliance

The planned release payload is the HumanoidMimicGen loco-manipulation
environment code under [robocasa/environments/locomanipulation](robocasa/environments/locomanipulation),
plus the minimal RoboCasa support modules under [robocasa/models](robocasa/models)
and [robocasa/utils](robocasa/utils) needed to import and register those
environments. Large local RoboCasa assets are intentionally excluded from
this repository snapshot. See:

- [LICENSE](LICENSE) for the Apache 2.0 license text.
- [NOTICE](NOTICE) for NVIDIA project notices.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party source and dependency notices.
- [CONTRIBUTING.md](CONTRIBUTING.md) for DCO language and the ongoing IP review process.
- [LICENSES/ROBOCASA-MIT.txt](LICENSES/ROBOCASA-MIT.txt) for the RoboCasa MIT license text.
- [PAPER_ENVS.md](PAPER_ENVS.md) for the scan-script-derived environment inventory.

# Overview

This repository packages humanoid manipulation environments built on top of
RoboCasa and robosuite. The included code focuses on simulation environment
definitions used by the HumanoidMimicGen paper scans and the small support
surface needed for import and registration. Datasets, model weights,
generated outputs, and large object/scene asset payloads are not included in
this snapshot.

# Getting Started

```bash
python -m pip install -e .
```

RoboCasa-compatible assets are required to instantiate most environments.
Before public distribution, confirm the asset distribution plan and update
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) with any asset licenses and
attributions that will be shipped.

# Requirements

- Python 3.10 or newer.
- MuJoCo and robosuite versions compatible with
  [pyproject.toml](pyproject.toml).
- `robosuite-models` is required for the G1 robot names used by the paper
  scan configurations.
- Runtime dependencies declared in [pyproject.toml](pyproject.toml).

# Usage

Importing `robocasa` registers the included loco-manipulation environments
with robosuite. Environment instantiation also requires compatible RoboCasa
MJCF/object assets at `robocasa/models/assets` or an equivalent asset root.

Trajectory datasets stored in robosuite / robomimic-style HDF5 files can be
inspected with the playback script:

```bash
python scripts/playback_trajectories.py /path/to/demo.hdf5 --mode state
```

State playback is the default because it restores the recorded MuJoCo states
directly. To replay open-loop actions and check for simulator drift:

```bash
python scripts/playback_trajectories.py /path/to/demo.hdf5 --mode action --check-drift
```

For headless video export, install the optional video extra and pass
`--video-path`:

```bash
python -m pip install -e ".[video]"
python scripts/playback_trajectories.py /path/to/demo.hdf5 --video-path playback.mp4 --no-render
```

Playback video export defaults to the `egoview` camera. Use `--camera` to
choose a different render camera. For example, `--camera 3pv` aliases to a
third-person `frontview` camera when the dataset XML uses standard RoboCasa
camera names. MuJoCo site markers, including gripper / IK debug indicators,
are hidden by default; pass `--show-sites` when those markers are useful for
debugging.

If the RoboCasa asset tree is outside this checkout, pass it explicitly:

```bash
python scripts/playback_trajectories.py /path/to/demo.hdf5 --assets-root /path/to/robocasa/models/assets
```

# Contribution Guidelines

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Third-party contributions must
include a Developer Certificate of Origin sign-off.

## Security

See [SECURITY.md](SECURITY.md). Do not file public issues for security reports.

# License

This project is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) for details. Third-party components retain their own
licenses as described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
