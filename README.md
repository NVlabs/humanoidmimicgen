# HumanoidMimicGen

HumanoidMimicGen packages humanoid loco-manipulation simulation
environments, G1 whole-body-controller replay utilities, and the RoboSuite
compatibility code needed to run them from this repository.

## Paper And Website

- Project website: [humanoidmimicgen.github.io](https://humanoidmimicgen.github.io/)
- arXiv paper: [HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning](https://arxiv.org/abs/2605.27724)

The repository is intentionally focused on simulation and replay. It does not
include training datasets, generated experiment outputs, or policy checkpoints.
The optional G1 lower-body ONNX policies used by WBC replay are downloaded
separately from GR00T Whole-Body Control.

If you use HumanoidMimicGen in your work, please cite:

```bibtex
@misc{lin2026humanoidmimicgendatagenerationlocomanipulation,
  title={HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning},
  author={Kevin Lin and Ajay Mandlekar and Caelan Reed Garrett and Nikita Chernyadev and Yu Fang and Runyu Ding and Yuqi Xie and Justin Tran and Linxi Fan and Yuke Zhu},
  year={2026},
  eprint={2605.27724},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2605.27724},
}
```

## What Is Included

- Humanoid loco-manipulation task definitions in
  [humanoidmimicgen/locomanipulation/envs](humanoidmimicgen/locomanipulation/envs).
- Retained MJCF arena, object, and G1 robot assets under
  [humanoidmimicgen/locomanipulation/models/assets](humanoidmimicgen/locomanipulation/models/assets).
- A retained loco-manipulation support layer for object placement, scene configs,
  task success checks, and G1 robot registration.
- A small compatibility layer that adapts public `robosuite==1.5.1` for the
  retained humanoid tasks and WBC replay path.
- G1 WBC replay/runtime code under [humanoidmimicgen/wbc](humanoidmimicgen/wbc).
- Entry-point scripts under [scripts](scripts).

The core simulation environments subclass RoboSuite's manipulation environment
base and register directly with RoboSuite. Import
`humanoidmimicgen.locomanipulation` to register the retained humanoid
loco-manipulation tasks.

## Repository Layout

| Path | Purpose |
| --- | --- |
| [humanoidmimicgen/locomanipulation/envs](humanoidmimicgen/locomanipulation/envs) | Environment classes for retained humanoid loco-manipulation tasks. |
| [humanoidmimicgen/locomanipulation/models](humanoidmimicgen/locomanipulation/models) | Model helpers and bundled MJCF assets used by the retained tasks. |
| [humanoidmimicgen/locomanipulation/utils/scene](humanoidmimicgen/locomanipulation/utils/scene) | Scene object placement, scene config, and success criteria helpers. |
| [humanoidmimicgen/locomanipulation/utils/robosuite_compat.py](humanoidmimicgen/locomanipulation/utils/robosuite_compat.py) | Import-time patches that keep public RoboSuite compatible with retained G1 replay configs. |
| [humanoidmimicgen/wbc](humanoidmimicgen/wbc) | G1 WBC policy/runtime code plus robot assets and configs. |
| [humanoidmimicgen/wbc_goal_playback.py](humanoidmimicgen/wbc_goal_playback.py) | LeRobot dataset playback through stored WBC goals. |
| [scripts/demo_random_action.py](scripts/demo_random_action.py) | Random upper-body action smoke test in a WBC loco-manipulation env. |
| [scripts/playback_wbc_goals.py](scripts/playback_wbc_goals.py) | Replay one LeRobot dataset and write MP4 output. |
| [scripts/run_wbc_goal_benchmark.sh](scripts/run_wbc_goal_benchmark.sh) | Batch replay wrapper for the retained 9-task WBC benchmark. |
| [docs/environments.md](docs/environments.md) | Retained environment inventory. |
| [docs/wbc_goal_replay.md](docs/wbc_goal_replay.md) | WBC-goal replay details and examples. |

## Why This Layer Exists

The retained humanoid tasks depend on local code for G1 robot registration,
controller configs, MJCF assets, object placement, and task-success predicates.
Public `robosuite==1.5.1` provides the core MuJoCo runtime, while
`humanoidmimicgen.locomanipulation.utils.robosuite_compat` installs only the WBC/JPos,
NullBase, and sensor cleanup compatibility patches exercised by the retained G1
replay configs.

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
python -m humanoidmimicgen.download_wbc_policies
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
import humanoidmimicgen.locomanipulation as locomanip

print("registered envs:")
for name in locomanip.RETAINED_LOCOMANIPULATION_ENVIRONMENTS:
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
MAKE_GRID=1 \
scripts/run_wbc_goal_benchmark.sh
```

This is the full 17-episode playback regression and should be run before
merging packaging, asset, replay, or repository-structure changes. Compare the
`TOTAL` row in `success_summary.tsv` with the last accepted `main` result to
catch task-predicate success degradation. A lower total should block the merge
until the affected per-task logs are inspected or rerun, because the predicates
are drift-sensitive near the end of some episodes. Set `EPISODES=1` only for a
faster one-demo-per-task smoke.

## Requirements

- MuJoCo `3.2.6`, pinned in [pyproject.toml](pyproject.toml). Newer MuJoCo
  versions can reject retained mesh assets during model compilation.
- `robosuite==1.5.1`, used as the public RoboSuite runtime.
- `robosuite-models` for robot models referenced by the retained configs.
- The `wbc-replay` extra for LeRobot dataset playback and random-action WBC
  smoke tests.
- The separately downloaded lower-body policies. See
  [docs/wbc_goal_replay.md](docs/wbc_goal_replay.md#lower-body-policy-files) for
  the downloader, pinned source-to-runtime rename mapping, checksums, and model
  license.

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

NVIDIA-authored source code is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for details. Optional lower-body ONNX model weights are
not distributed with this repository; the separately downloaded weights are
licensed under the [NVIDIA Open Model
License](LICENSES/NVIDIA-OPEN-MODEL-LICENSE.txt). Third-party components retain
their own licenses as described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
