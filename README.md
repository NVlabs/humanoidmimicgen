<div align="center">

  <img src="docs/assets/readme_teaser.gif" alt="HumanoidMimicGen G1 loco-manipulation tasks" width="100%">

</div>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-76B900.svg)](LICENSE)
[![Project Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://humanoidmimicgen.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-2605.27724-b31b1b.svg)](https://arxiv.org/abs/2605.27724)

</div>

---

# HumanoidMimicGen

HumanoidMimicGen ([CoRL 2026](https://2026.corl.org/)) generates humanoid
loco-manipulation data by adapting contact-rich whole-body skills from source
demonstrations to new scenes. This release packages the project's nine G1
simulation environments, MuJoCo assets, whole-body controller (WBC), and
physics-correct recorded state/action playback.

## Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Common Examples](#common-examples)
- [G1 Loco-Manipulation Benchmark](#g1-loco-manipulation-benchmark)
- [Train and Evaluate a Policy](#train-and-evaluate-a-policy)
- [Dataset Playback](#dataset-playback)
- [What's Included](#whats-included)
- [Citation](#citation)
- [License](#license)

## Requirements

- Linux and Python 3.10
- An NVIDIA GPU and CUDA only for WBC execution or EGL rendering; recorded
  playback without rendering can run CPU-only
- `ffmpeg` only when generating a downscaled replay video
- External datasets for playback

MuJoCo is pinned to `3.2.6` and RoboSuite to `1.5.1`. Datasets, generated
outputs, and optional lower-body ONNX weights are not stored in this repository.

## Quick Start

```bash
git clone https://github.com/NVlabs/humanoidmimicgen.git
cd humanoidmimicgen

mamba create -n humanoidmimicgen python=3.10 -y
mamba activate humanoidmimicgen
python -m pip install -e ".[wbc-replay]"

# Required only for WBC-goal or random-action execution.
python -m humanoidmimicgen.download_wbc_policies

# Headless Linux rendering.
export MUJOCO_GL=egl
```

## Common Examples

### Run random actions

Smoke-test an environment and its WBC without a dataset:

```bash
python scripts/demo_random_action.py \
  --task LMPushButton \
  --steps 1000 \
  --seed 0
```

Rendering is opt-in:

```bash
python scripts/demo_random_action.py \
  --task LMPushButton \
  --steps 1000 \
  --seed 0 \
  --video-path /tmp/push_button_random.mp4
```

### Replay human source demonstrations

The public
[nine-task human source-demo replay episodes](https://huggingface.co/datasets/linkenv/humanoidmimicgen-g1-source-demo-replay)
contain one successful human-operated episode per task, including each
episode's exact MuJoCo XML scene:

```bash
hf download linkenv/humanoidmimicgen-g1-source-demo-replay \
  --repo-type dataset \
  --local-dir /path/to/hmg_source_demo_replay
```

```bash
python scripts/playback_dataset.py \
  /path/to/hmg_source_demo_replay/datasets/02_push_button/demo.hdf5 \
  --action-source state \
  --require-task-success \
  --video-path /tmp/push_button_source_demo.mp4
```

The `state` mode restores every recorded simulator state so the original human
trajectory can be visualized faithfully. It does not validate action or
physics fidelity.

### Validate recorded-action physics

For exact action-driven regression, use the public
[nine-task recorded-action replay episodes](https://huggingface.co/datasets/linkenv/humanoidmimicgen-g1-nine-task-action-replay).
The bundle contains one validated seed-0 checkpoint-generated policy rollout
for every environment shipped here, under `datasets/<task>`:

```bash
hf download linkenv/humanoidmimicgen-g1-nine-task-action-replay \
  --repo-type dataset \
  --local-dir /path/to/hmg_nine_task_replay
```

```bash
python scripts/playback_dataset.py \
  /path/to/hmg_nine_task_replay/datasets/02_push_button \
  --action-source recorded \
  --num-episodes 1 \
  --state-atol 0
```

Replace `02_push_button` with any of the nine task directories listed in the
bundle. These replay episodes are for exact physics regression; they are not
policy-training datasets and make no benchmark-performance claim.

Add `--video-path /tmp/push_button_replay.mp4` when you also want an MP4.
The default `recorded` mode applies the dataset's low-level joint actions and
checks every resulting MuJoCo state. Use `--action-source wbc-goal` separately
to test regenerating those actions from the higher-level WBC goals.

## G1 Loco-Manipulation Benchmark

### Nine industrial humanoid tasks.

Object lifting, pushing, placing, obstacle navigation, and shelf interaction on
a humanoid robot.

<table>
  <tr>
    <td width="33%" valign="top"><img src="docs/assets/benchmark_tasks/box-lift-floor.jpg" alt="Box Lift Floor task snapshots" width="100%"><br><strong>Box Lift Floor</strong><br>Grasp the box from the floor and lift to a target height.</td>
    <td width="33%" valign="top"><img src="docs/assets/benchmark_tasks/push-button.jpg" alt="Push Button task snapshots" width="100%"><br><strong>Push Button</strong><br>Approach an industrial panel and press its button.</td>
    <td width="33%" valign="top"><img src="docs/assets/benchmark_tasks/box-lift.jpg" alt="Box Lift task snapshots" width="100%"><br><strong>Box Lift</strong><br>Approach the table, grasp the box, and lift to a target height.</td>
  </tr>
  <tr>
    <td valign="top"><img src="docs/assets/benchmark_tasks/push-shelf-forward.jpg" alt="Push Shelf Forward task snapshots" width="100%"><br><strong>Push Shelf Forward</strong><br>Push the shelving cart into a marked target zone.</td>
    <td valign="top"><img src="docs/assets/benchmark_tasks/drill-lift.jpg" alt="Drill Lift task snapshots" width="100%"><br><strong>Drill Lift</strong><br>Approach a table, grasp a drill, and lift to a target height.</td>
    <td valign="top"><img src="docs/assets/benchmark_tasks/drill-pnp.jpg" alt="Drill PnP task snapshots" width="100%"><br><strong>Drill PnP</strong><br>Pick a drill from one table and place it on a second table.</td>
  </tr>
  <tr>
    <td valign="top"><img src="docs/assets/benchmark_tasks/box-table-to-shelf.jpg" alt="Box Table To Shelf task snapshots" width="100%"><br><strong>Box Table To Shelf</strong><br>Transfer a box from a table into a shelf.</td>
    <td valign="top"><img src="docs/assets/benchmark_tasks/pick-drill-from-holder.jpg" alt="Pick Drill From Holder task snapshots" width="100%"><br><strong>Pick Drill From Holder</strong><br>Walk to the holder, grasp the drill, and lift it.</td>
    <td valign="top"><img src="docs/assets/benchmark_tasks/drill-lift-obstacle.jpg" alt="Obstacle-Aware Pick Drill task snapshots" width="100%"><br><strong>Obstacle-Aware Pick Drill</strong><br>Navigate around a blocking shelf, then grasp and lift the drill.</td>
  </tr>
</table>

The public [HumanoidMimicGen G1 Loco-Manipulation Benchmark training dataset](https://huggingface.co/datasets/linkenv/humanoidmimicgen-g1-benchmark)
contains about 9K demonstrations across these nine tasks, generated with the
HumanoidMimicGen data-generation algorithm and organized into eight LeRobot
shards per task.

## Train and Evaluate a Policy

The sample policy training and evaluation scripts provide a minimal end-to-end
example using the upstream LeRobot Diffusion Policy at the exact tested
revision:

```bash
python -m pip install --upgrade \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
```

Choose one task from the public benchmark. The training script downloads all
eight shards for that task from
[`linkenv/humanoidmimicgen-g1-benchmark`](https://huggingface.co/datasets/linkenv/humanoidmimicgen-g1-benchmark),
converts them to LeRobot v3, and projects them to the model's 43-dimensional
`observation.state`, `observation.images.ego_view`, and 35-dimensional WBC-goal
`action` contract. Downloads and prepared shards are cached under
`~/.cache/humanoidmimicgen/policy_data` by default.

For example, train Task06 using all eight released shards:

```bash
python scripts/train_policy_example.py \
  --task 06_drill_pnp \
  --output-dir ./outputs/task06_dp_long50
```

There is no dataset path or Hugging Face repo ID to discover. The script uses
the pinned public dataset above and creates the local LeRobot IDs automatically
(for example, `local/hmg_06_drill_pnp_shard_000`). Use `--data-dir` only to
place the download/preparation cache somewhere other than its default location.

This trains the validated 64-step prediction-horizon, 50-step action-chunk
configuration for 20K updates by default. Evaluate a checkpoint over 20 native
episodes (seeds 0 through 19) with:

```bash
python scripts/evaluate_policy_example.py \
  --checkpoint-dir ./outputs/task06_dp_long50/checkpoints/020000 \
  --task 06_drill_pnp \
  --num-episodes 20 \
  --seed 0 \
  --output /tmp/task06_dp_long50_eval.json
```

Use `--help` on either script for the complete task and runtime options.

## Dataset Playback

`scripts/playback_dataset.py` accepts a LeRobot dataset root or source-demo
HDF5 file and reads the environment identity from its metadata.

| Mode | Behavior | Use |
|---|---|---|
| `recorded` | Restores the initial state, applies recorded actions, and checks each resulting state. | Physics replay (default) |
| `wbc-goal` | Regenerates low-level actions from recorded WBC goals. | WBC determinism |
| `state` | Restores every recorded state without stepping physics. | Visualization only |

Playback fails on state divergence by default (`atol=1e-5`; use
`--state-atol 0` for exact equality). Add `--video-path` for an MP4 or
`--require-task-success` to require the task predicate.

> [!IMPORTANT]
> The published replay episodes are validated with this runtime. Other
> HMG-generated demos may not include the MuJoCo XML or a record of where fixed
> scene objects were placed. In those cases, restoring `states[0]` alone may not
> reproduce the original scene; faithful replay requires that missing scene
> information.

## What's Included

| Component | Description |
|---|---|
| [`humanoidmimicgen/locomanipulation`](humanoidmimicgen/locomanipulation) | Nine environments, scene helpers, success predicates, and MJCF assets. |
| [`humanoidmimicgen/wbc`](humanoidmimicgen/wbc) | G1 WBC implementation, robot description, and configs. |
| [`scripts/demo_random_action.py`](scripts/demo_random_action.py) | Random-action environment and WBC smoke test, with optional rendering. |
| [`scripts/train_policy_example.py`](scripts/train_policy_example.py) | Sample policy training entrypoint with automatic shard discovery and the validated LeRobot 64/50 configuration. |
| [`scripts/evaluate_policy_example.py`](scripts/evaluate_policy_example.py) | Sample native HMG policy evaluation entrypoint for the resulting checkpoints. |
| [`scripts/playback_dataset.py`](scripts/playback_dataset.py) | Dataset playback CLI with optional MP4 rendering. |

## Citation

If you find HumanoidMimicGen useful in your research, please cite:

```bibtex
@inproceedings{lin2026humanoidmimicgen,
  title={HumanoidMimicGen Data Generation for Loco-Manipulation via Whole-Body Planning and Adaptation},
  author={Kevin Lin and Ajay Mandlekar and Caelan Reed Garrett and Nikita Chernyadev and Yu Fang and Runyu Ding and Yuqi Xie and Justin Tran and Linxi Fan and Yuke Zhu},
  booktitle={10th Annual Conference on Robot Learning},
  year={2026},
  url={https://openreview.net/forum?id=9kXz22fI7a}
}
```

## License

NVIDIA-authored source is licensed under Apache-2.0; see [LICENSE](LICENSE).
RoboCasa-derived source and assets, RoboSuite, RoboSuite Models, and Unitree G1
robot assets retain their respective licenses under [LICENSES](LICENSES).
Downloaded lower-body ONNX weights use the NVIDIA Open Model License.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source and README-media
provenance, attributions, dependency-license references, and distribution
notices.

## Contributing and Security

See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes and
[SECURITY.md](SECURITY.md) for private security-reporting instructions.
