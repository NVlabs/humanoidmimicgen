# HumanoidMimicGen

HumanoidMimicGen contains RoboCasa-based humanoid loco-manipulation
environment code prepared for an Apache License, Version 2.0 open source
release.

## Release Compliance

The planned release payload is the HumanoidMimicGen loco-manipulation
environment code under [robocasa/environments/locomanipulation](robocasa/environments/locomanipulation),
plus the minimal RoboCasa support modules under [robocasa/models](robocasa/models)
and [robocasa/utils](robocasa/utils) needed to import and register those
environments. The repository also includes the RoboCasa asset subset under
[robocasa/models/assets](robocasa/models/assets) needed for the retained
loco-manipulation environments and playback rendering. See:

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
surface needed for import, registration, and rendering. Datasets, model
weights, generated outputs, and unrelated large RoboCasa asset payloads are
not included in this snapshot.

# Getting Started

```bash
git clone https://github.com/NVlabs/humanoidmimicgen.git
cd humanoidmimicgen
mamba create -n humanoidmimicgen python=3.12 -y
mamba activate humanoidmimicgen
python -m pip install -e ".[video]"
python -c "import robocasa; print('robocasa import OK')"
```

The retained RoboCasa loco-manipulation assets are included at
`robocasa/models/assets`, so the paper environments and trajectory playback
do not require an external asset root by default.

# Requirements

- Python 3.10 or newer.
- MuJoCo and robosuite versions compatible with
  [pyproject.toml](pyproject.toml).
- `robosuite-models` is required for the G1 robot names used by the paper
  scan configurations.
- Runtime dependencies declared in [pyproject.toml](pyproject.toml).

# Usage

Importing `robocasa` registers the included loco-manipulation environments
with robosuite. Environment instantiation uses the bundled assets at
`robocasa/models/assets` by default.

Trajectory datasets stored in robosuite / robomimic-style HDF5 files can be
inspected with the playback script:

```bash
python scripts/playback_trajectories.py /path/to/demo.hdf5
```

Action playback is the default. When recorded actions match the robosuite
environment action space, the script uses `env.step`. For G1 whole-body demos
whose `actions` are recorded 43-dof whole-body control outputs, the script uses
the robot-control `step(action)` adapter over the saved episode XML, applying
those controls to the MuJoCo actuators instead of setting every recorded state.
If you have a compatible robosuite controller configuration, pass
`--controller-config` with `--action-backend env` to replay through the
robosuite environment itself. To compare action replay against recorded states:

```bash
python scripts/playback_trajectories.py /path/to/demo.hdf5 --check-drift
```

State playback remains available for dataset inspection:

```bash
python scripts/playback_trajectories.py /path/to/demo.hdf5 --mode state
```

For headless video export, pass `--video-path`:

```bash
python scripts/playback_trajectories.py /path/to/demo.hdf5 \
  --video-path playback.mp4 \
  --width 320 \
  --height 240
```

Playback video export defaults to the `egoview` camera. Use `--camera` to
choose a different render camera. For example, `--camera 3pv` aliases to a
third-person `frontview` camera when the dataset XML uses standard RoboCasa
camera names. MuJoCo site markers, including gripper / IK debug indicators,
are hidden by default; pass `--show-sites` when those markers are useful for
debugging.

For G1 LeRobot datasets with saved WBC goals, use
[scripts/playback_wbc_goals.py](scripts/playback_wbc_goals.py). That path
replays the saved high-level WBC goal, including navigation command, through
the bundled HumanoidMimicGen whole-body-controller runtime. See
[docs/wbc_goal_replay.md](docs/wbc_goal_replay.md) for environment details and
example commands.

If you need to override the bundled asset tree, pass an alternate root
explicitly:

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
