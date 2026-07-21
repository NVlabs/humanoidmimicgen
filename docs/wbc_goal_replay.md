# WBC-Goal Replay

WBC-goal replay renders G1 LeRobot episodes by resetting the simulator to the
recorded initial MuJoCo state, replaying stored upper-body WBC goals, and using
the bundled WBC runtime to generate stabilizing lower-body actions.

This path is simulation-only. It does not require ROS or `rclpy`.

## Data Format

Use [scripts/playback_wbc_goals.py](../scripts/playback_wbc_goals.py) for a
LeRobot dataset root containing:

- `meta/episodes.jsonl`
- `observation.sim.mujoco_state`
- `action.eef`
- `teleop.navigate_command`
- `teleop.base_height_command`
- `observation.sim.target_upper_body_pose`

If the command receives a `demo.hdf5` path, it uses that file's parent directory
as the dataset root.

## Install

```bash
mamba create -n humanoidmimicgen python=3.10 -y
mamba activate humanoidmimicgen
python -m pip install -e ".[wbc-replay]"
```

MuJoCo is pinned to `3.2.6`. Keep that pin for replay; newer MuJoCo versions
can reject retained mesh assets during model compilation.

For headless Linux rendering:

```bash
export MUJOCO_GL=egl
```

## Lower-Body Policy Files

The repository already includes the two lower-body ONNX policies required by
WBC replay, so a normal checkout does not need an additional download. If those
files are missing or need to be restored, download the byte-identical public
GR00T Whole-Body Control models and save them under the local runtime names:

```bash
POLICY_DIR=humanoidmimicgen/wbc/external_dependencies/sim2mujoco/resources/robots/g1/policy
MODEL_BASE=https://github.com/NVlabs/GR00T-WholeBodyControl/raw/4141c34280abb67c82e115342a8720f4a83d750d/decoupled_wbc/sim2mujoco/resources/robots/g1/policy

mkdir -p "$POLICY_DIR"
curl -fL "$MODEL_BASE/GR00T-WholeBodyControl-Balance.onnx" \
  -o "$POLICY_DIR/stand.onnx"
curl -fL "$MODEL_BASE/GR00T-WholeBodyControl-Walk.onnx" \
  -o "$POLICY_DIR/walk.onnx"

sha256sum -c <<'EOF'
f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18  humanoidmimicgen/wbc/external_dependencies/sim2mujoco/resources/robots/g1/policy/stand.onnx
7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407  humanoidmimicgen/wbc/external_dependencies/sim2mujoco/resources/robots/g1/policy/walk.onnx
EOF
```

The model weights are licensed under the
[NVIDIA Open Model License](../LICENSES/NVIDIA-OPEN-MODEL-LICENSE.txt). See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for pinned provenance links
and required attribution.

## Replay One Dataset

```bash
python scripts/playback_wbc_goals.py \
  /path/to/G1_LMPushButton_dataset \
  --num-episodes 1 \
  --video-path /tmp/pushbutton_wbc_goal.mp4 \
  --lowres-video-path /tmp/pushbutton_wbc_goal_320w.mp4 \
  --lowres-width 320
```

Useful flags:

- `--num-episodes N`: replay the first `N` complete episodes.
- `--debug`: replay only the first 20 steps.
- `--strict`: exit nonzero if state replay diverges. Leave this off when the
  goal is MP4 generation.

The script writes a raw MP4. If `--lowres-video-path` is provided, it also
invokes `ffmpeg` to write a downscaled MP4.

## Random-Action Smoke

Use [scripts/demo_random_action.py](../scripts/demo_random_action.py) to build a
WBC loco-manipulation environment and run random upper-body actions:

```bash
python scripts/demo_random_action.py \
  --task LMPushButton \
  --steps 1000 \
  --seed 0 \
  --video-path /tmp/hmg_random_action.mp4 \
  --navigate-cmd 0 0 0 \
  --arm-mode random \
  --lower-body-mode policy
```

## Batch Benchmark

Use [scripts/run_wbc_goal_benchmark.sh](../scripts/run_wbc_goal_benchmark.sh)
when the retained LeRobot datasets are available locally:

```bash
DATA_ROOT=/path/to/collected_demo \
OUT=/tmp/hmg_wbc_goal_benchmark \
MAKE_GRID=1 \
scripts/run_wbc_goal_benchmark.sh
```

The wrapper defaults to the full 17-episode regression across the retained
9-task benchmark set. It writes per-task raw and low-resolution MP4s, records
`.log`, `.rc`, and `.ffprobe` files, writes `summary.txt` and
`success_summary.tsv`, and optionally creates a grid preview MP4.

Check the `TOTAL` row in `success_summary.tsv` against the last accepted
`main` run before merging packaging, asset, replay, or repository-structure
changes. `EPISODES=1` replays only `demo_1` for each task when a faster smoke is
enough. A lower total should block the merge until the affected per-task logs
are inspected or rerun, because the task predicates are drift-sensitive near
the end of some episodes. `EPISODES=all` replays every episode present in each
retained dataset and is the default.

## Interpreting Success

The replay driver reports two notions of success:

- Process success: the dataset process exits `rc=0` and writes the MP4 outputs.
- Task-predicate success: the environment task success predicate becomes true
  during replay, and whether it is true on the final frame.

Task predicates can be sensitive to small replay drift. Use process success and
MP4 generation as the reproduction smoke check; use task-predicate summaries as
diagnostics.
