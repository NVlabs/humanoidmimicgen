# HumanoidMimicGen

HumanoidMimicGen contains RoboCasa-based humanoid loco-manipulation
environment code prepared for an Apache License, Version 2.0 open source
release.

## Release Compliance

The planned release payload is the HumanoidMimicGen loco-manipulation
environment code under [robocasa/environments/locomanipulation](robocasa/environments/locomanipulation),
the vendored RoboCasa support tree under [robocasa](robocasa), and the
compatible vendored RoboSuite tree under [robosuite](robosuite). The repository
also includes the RoboCasa and RoboSuite assets needed for the retained
loco-manipulation environments and playback rendering. See:

- [LICENSE](LICENSE) for the Apache 2.0 license text.
- [NOTICE](NOTICE) for NVIDIA project notices.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party source and dependency notices.
- [CONTRIBUTING.md](CONTRIBUTING.md) for DCO language and the ongoing IP review process.
- [LICENSES/ROBOCASA-MIT.txt](LICENSES/ROBOCASA-MIT.txt) for the RoboCasa MIT license text.
- [PAPER_ENVS.md](PAPER_ENVS.md) for the scan-script-derived environment inventory.

# Overview

This repository packages humanoid manipulation environments built on top of
RoboCasa and RoboSuite. The included code focuses on simulation environment
definitions used by the HumanoidMimicGen paper scans and the small support
surface needed for import, registration, and rendering. Datasets, model
weights, and generated outputs are not included in this snapshot.

# Getting Started

For environment import and other non-WBC workflows:

```bash
git clone https://github.com/NVlabs/humanoidmimicgen.git
cd humanoidmimicgen
mamba create -n humanoidmimicgen python=3.12 -y
mamba activate humanoidmimicgen
python -m pip install -e .
python -c "import robocasa; print('robocasa import OK')"
```

The retained RoboCasa loco-manipulation assets are included at
`robocasa/models/assets`, so the paper environments do not require an external
asset root by default.

For G1 LeRobot WBC-goal replay, use the replay extra:

```bash
git clone https://github.com/NVlabs/humanoidmimicgen.git
cd humanoidmimicgen
mamba create -n humanoidmimicgen-wbc python=3.10 -y
mamba activate humanoidmimicgen-wbc
python -m pip install -e ".[wbc-replay]"
```

The WBC replay path uses the vendored `robocasa` and `robosuite` trees in this
repository. Put this repository on `PYTHONPATH` before running WBC replay:

```bash
export PYTHONPATH=/path/to/humanoidmimicgen
```

For the local GR00T workspace this is:

```bash
export PYTHONPATH=/home/linke/Projects/humanoidmimicgen
```

The LeRobot demo datasets may still live in the GR00T workspace; only the
runtime package imports are local to this repository.

To reproduce the 9-task action/WBC-goal MP4 benchmark from the local dataset
mirror:

```bash
DATA_ROOT=/home/linke/Projects/gr00t/groot/dexmg/collected_demo \
OUT=/tmp/hmg_wbc_goal_benchmark \
scripts/run_wbc_goal_benchmark.sh
```

# Requirements

- Python 3.10 or newer.
- MuJoCo `3.2.6`, as pinned in [pyproject.toml](pyproject.toml). Newer
  MuJoCo releases can reject some retained mesh assets during model compile.
- `robosuite-models` is required for the G1 robot names used by the paper
  scan configurations.
- Runtime dependencies declared in [pyproject.toml](pyproject.toml). Use the
  `wbc-replay` extra for G1 LeRobot WBC-goal replay.

# Usage

Importing `robocasa` registers the included loco-manipulation environments
with robosuite. Environment instantiation uses the bundled assets at
`robocasa/models/assets` by default.

For G1 LeRobot datasets, use
[scripts/playback_wbc_goals.py](scripts/playback_wbc_goals.py). By default it
matches GR00T add-skillgen `--use-actions --use-wbc-goals` behavior by replaying
stored WBC goals while the bundled HumanoidMimicGen whole-body-controller
runtime generates the stabilizing lower-body / leg actions. See
[docs/wbc_goal_replay.md](docs/wbc_goal_replay.md) for environment details and
example commands. This sim replay path does not require ROS or `rclpy`.

# Contribution Guidelines

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Third-party contributions must
include a Developer Certificate of Origin sign-off.

## Security

See [SECURITY.md](SECURITY.md). Do not file public issues for security reports.

# License

This project is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) for details. Third-party components retain their own
licenses as described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
