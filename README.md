# HumanoidMimicGen

HumanoidMimicGen packages humanoid loco-manipulation simulation
environments, G1 whole-body-controller replay utilities, and the RoboCasa /
RoboSuite support code needed to run them from this repository.

The repository is intentionally focused on simulation and replay. It does not
include training datasets, generated experiment outputs, or policy checkpoints
except for the small bundled G1 lower-body ONNX policies used by WBC replay.

## What Is Included

- Humanoid loco-manipulation task definitions in
  [robocasa/environments/locomanipulation](robocasa/environments/locomanipulation).
- MJCF scenes, objects, fixtures, and G1 robot assets under
  [robocasa/models/assets](robocasa/models/assets).
- A retained RoboCasa support layer for object placement, scene configs,
  task success checks, and G1 robot registration.
- A retained RoboSuite support layer for MuJoCo environments, robots,
  controllers, rendering, and model utilities.
- G1 WBC replay/runtime code under [humanoidmimicgen/wbc](humanoidmimicgen/wbc).
- Entry-point scripts under [scripts](scripts).

The core simulation environments live under the RoboCasa namespace because they
register with RoboSuite as RoboCasa environments. Importing `robocasa` registers
the retained humanoid loco-manipulation tasks.

## Repository Layout

| Path | Purpose |
| --- | --- |
| [robocasa/environments/locomanipulation](robocasa/environments/locomanipulation) | Environment classes for retained humanoid loco-manipulation tasks. |
| [robocasa/models](robocasa/models) | RoboCasa model helpers and bundled MJCF assets used by the retained tasks. |
| [robocasa/utils/scene](robocasa/utils/scene) | Scene object placement, scene config, and success criteria helpers. |
| [robosuite](robosuite) | Vendored RoboSuite runtime pieces required by the retained tasks and WBC replay. |
| [humanoidmimicgen/wbc](humanoidmimicgen/wbc) | G1 WBC policy/runtime code plus robot and policy assets. |
| [humanoidmimicgen/wbc_goal_playback.py](humanoidmimicgen/wbc_goal_playback.py) | LeRobot dataset playback through stored WBC goals. |
| [scripts/demo_random_action.py](scripts/demo_random_action.py) | Random upper-body action smoke test in a WBC RoboCasa env. |
| [scripts/playback_wbc_goals.py](scripts/playback_wbc_goals.py) | Replay one LeRobot dataset and write MP4 output. |
| [scripts/run_wbc_goal_benchmark.sh](scripts/run_wbc_goal_benchmark.sh) | Batch replay wrapper for the retained 9-task WBC benchmark. |
| [docs/environments.md](docs/environments.md) | Retained environment inventory. |
| [docs/wbc_goal_replay.md](docs/wbc_goal_replay.md) | WBC-goal replay details and examples. |

## Why RoboCasa And RoboSuite Are Vendored

The retained humanoid tasks depend on local RoboCasa and RoboSuite changes for
G1 robot registration, active-base control, controller configs, MJCF assets,
and replay-compatible MuJoCo behavior. Keeping those trees in this repository
makes the environment import and WBC replay paths reproducible without requiring
a matching upstream checkout.

Only the source surface needed by the retained loco-manipulation and WBC replay
paths is intended to remain here.

## Install

Use Python 3.10 for WBC replay. The base environment import path supports
Python 3.10 or newer, but the replay dependencies are pinned around the tested
Python 3.10 stack.

```bash
git clone https://github.com/NVlabs/humanoidmimicgen.git
cd humanoidmimicgen
mamba create -n humanoidmimicgen python=3.10 -y
mamba activate humanoidmimicgen
python -m pip install -e ".[wbc-replay]"
```

For environment import only, without LeRobot / ONNXRuntime / Torch replay
dependencies, install the base package:

```bash
python -m pip install -e .
```

Editable install is the preferred setup. The top-level scripts also add the
checkout root to `sys.path` when run from source, so manual path exports are
not required for the documented scripts.

## Quick Checks

Confirm environment registration:

```bash
python - <<'PY'
import robocasa

print("registered envs:")
for name in robocasa.RETAINED_LOCOMANIPULATION_ENVIRONMENTS:
    print(" ", name)
PY
```

Run a short random-action WBC smoke:

```bash
python scripts/demo_random_action.py \
  --task LMPushButton \
  --steps 200 \
  --seed 0 \
  --video-path /tmp/hmg_random_action.mp4 \
  --navigate-cmd 0 0 0 \
  --arm-mode random \
  --lower-body-mode policy
```

Replay a LeRobot dataset through stored WBC goals:

```bash
python scripts/playback_wbc_goals.py \
  /path/to/G1_LMPushButton_dataset \
  --num-episodes 1 \
  --video-path /tmp/pushbutton_wbc_goal.mp4 \
  --lowres-video-path /tmp/pushbutton_wbc_goal_320w.mp4
```

Run the retained 9-task benchmark if the local LeRobot datasets are available:

```bash
DATA_ROOT=/path/to/collected_demo \
OUT=/tmp/hmg_wbc_goal_benchmark \
EPISODES=all \
MAKE_GRID=1 \
scripts/run_wbc_goal_benchmark.sh
```

## Requirements

- MuJoCo `3.2.6`, pinned in [pyproject.toml](pyproject.toml). Newer MuJoCo
  versions can reject retained mesh assets during model compilation.
- `robosuite-models` for robot models referenced by the retained configs.
- The `wbc-replay` extra for LeRobot dataset playback and random-action WBC
  smoke tests.

For headless Linux rendering, set:

```bash
export MUJOCO_GL=egl
```

## Documentation

- [docs/environments.md](docs/environments.md): retained environment names and
  their source modules.
- [docs/wbc_goal_replay.md](docs/wbc_goal_replay.md): replay data format,
  script usage, and expected output artifacts.

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Third-party contributions must
include a Developer Certificate of Origin sign-off.

## Security

See [SECURITY.md](SECURITY.md). Do not file public issues for security reports.

## License

This project is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) for details. Third-party components retain their own
licenses as described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
