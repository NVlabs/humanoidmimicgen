# WBC-Goal Replay

This is the minimal path that reproduced the useful full-episode G1 replay
videos in GR00T: replay stored `wbc_goal` records through the whole-body
controller instead of applying the saved joint action vector directly.

## What This Replays

Use `scripts/playback_wbc_goals.py` for LeRobot-style datasets that contain:

- `meta/episodes.jsonl`
- saved simulator state under `observation.sim.mujoco_state`
- saved WBC goal fields reconstructed by GR00T's dataset loader:
  - `action.eef`
  - `teleop.navigate_command`
  - `teleop.base_height_command`
  - `observation.sim.target_upper_body_pose`

The wrapper sets the GR00T playback config to:

```text
use_actions=True
use_wbc_goals=True
use_teleop_cmd=False
save_video=True
enable_offscreen=True
enable_onscreen=False
MUJOCO_GL=egl
```

This is a qualitative/video replay path. It can render valid episodes even
when strict state equality still reports drift.

## Environment

The public HumanoidMimicGen repo includes the environment subset, but this
WBC-goal replay also needs GR00T's controller stack and WBC checkpoints. Use a
GR00T checkout that has the G1 controller dependencies and LFS assets:

```bash
git clone git@github.com:NVlabs/gr00t.git /path/to/gr00t
cd /path/to/gr00t
git lfs pull --include="groot/control/robot_model/model_data/**,groot/control/wbc_checkpoints/**,external_dependencies/**"
uv sync --extra dev
source .venv/bin/activate
```

Install this repo into the same environment:

```bash
cd /path/to/humanoidmimicgen
python -m pip install -e ".[video]"
```

For headless Linux rendering:

```bash
export MUJOCO_GL=egl
```

## Run One Dataset

Pass the LeRobot dataset root. If you pass a `demo.hdf5` path, the wrapper uses
its parent as the dataset root.

```bash
python scripts/playback_wbc_goals.py \
  /path/to/gr00t/groot/dexmg/collected_demo/G1_LMPushButton_20260129_234556 \
  --groot-root /path/to/gr00t \
  --video-path /tmp/pushbutton_wbc_goal_raw.mp4 \
  --lowres-video-path /tmp/pushbutton_wbc_goal_320w.mp4 \
  --lowres-width 320
```

Use `--ci-test` for a 20-step smoke test. Use `--strict` only when you want the
process to exit nonzero on state drift; leave it off for video generation.

## Batch Used For The Successful Videos

The successful full-episode render batch used these datasets:

```text
outputs/G1_LMDrillLiftObstacleDT_20260417_093010
outputs/G1_LMPickDrillFromHolderStandingEasy_20260408_144715
groot/dexmg/collected_demo/G1_LMPushButton_20260129_234556
groot/dexmg/collected_demo/G1_LMDrillPnP90_20260129_231446
groot/dexmg/collected_demo/G1_LMBoxLift_20260129_230924
groot/dexmg/collected_demo/G1_LMPnPBottleToBinStatic
```

The observed result for that batch was:

```text
valid MP4 render rate: 6/6
strict state replay success: 0/6
```

So the WBC-goal path is currently the best reproduction path for full-episode
videos, but strict replay correctness is not solved by this wrapper.

## Stitch Outputs

After rendering individual videos, create a low-resolution grid with ffmpeg:

```bash
ffmpeg -y \
  -i task1_320w.mp4 -i task2_320w.mp4 -i task3_320w.mp4 \
  -i task4_320w.mp4 -i task5_320w.mp4 -i task6_320w.mp4 \
  -filter_complex "[0:v][1:v][2:v]hstack=3[top];[3:v][4:v][5:v]hstack=3[bot];[top][bot]vstack=2[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  /tmp/wbc_goal_grid.mp4
```
