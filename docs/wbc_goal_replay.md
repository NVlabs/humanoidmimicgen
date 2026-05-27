# WBC-Goal Replay

This is the minimal path that reproduced useful full-episode G1 replay videos:
replay stored `wbc_goal` records through the whole-body controller instead of
applying the saved joint action vector directly.

## What This Replays

Use `scripts/playback_wbc_goals.py` for LeRobot-style datasets that contain:

- `meta/episodes.jsonl`
- saved simulator state under `observation.sim.mujoco_state`
- saved WBC goal fields reconstructed by HumanoidMimicGen's dataset loader:
  - `action.eef`
  - `teleop.navigate_command`
  - `teleop.base_height_command`
  - `observation.sim.target_upper_body_pose`

The HumanoidMimicGen replay module sets the local WBC playback config to:

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

HumanoidMimicGen owns the WBC-goal replay driver, dataset reader, WBC runtime
wrapper, G1 robot model assets, and bundled `stand.onnx` / `walk.onnx` policy
files. This documented path is sim-only and does not require ROS or `rclpy`.

Create the replay environment with:

```bash
mamba create -n humanoidmimicgen-wbc python=3.10 -y
mamba activate humanoidmimicgen-wbc
python -m pip install -e ".[wbc-replay]"
```

The WBC replay path imports `robocasa.wrappers.ik_wrapper`, which is not part
of this trimmed release snapshot. Put a full RoboCasa checkout, a compatible
RoboSuite checkout, and this repository on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/full/robocasa:/path/to/compatible/robosuite:/path/to/humanoidmimicgen
```

For the local GR00T workspace this is:

```bash
export PYTHONPATH=/home/linke/Projects/gr00t/groot/dexmg/grootrobocasa:/home/linke/Projects/gr00t/groot/dexmg/grootrobosuite:/home/linke/humanoidmimicgen
```

Run the replay command from outside this repository, such as `/tmp`, so the
trimmed bundled `robocasa` package does not shadow the full RoboCasa checkout.

For headless Linux rendering, use EGL:

```bash
export MUJOCO_GL=egl
```

## Run One Dataset

Pass the LeRobot dataset root. If you pass a `demo.hdf5` path, the script uses
its parent as the dataset root.

```bash
python scripts/playback_wbc_goals.py \
  /path/to/collected_demo/G1_LMPushButton_20260129_234556 \
  --video-path /tmp/pushbutton_wbc_goal_raw.mp4 \
  --lowres-video-path /tmp/pushbutton_wbc_goal_320w.mp4 \
  --lowres-width 320
```

To render exactly one complete episode, add `--num-episodes 1`. The default is
to render every episode in the dataset. Use `--ci-test` only for a 20-step
smoke test. Use `--strict` only when you want the process to exit nonzero on
state drift; leave it off for video generation.

The command shape used in the local GR00T workspace is:

```bash
cd /tmp
MUJOCO_GL=egl mamba run -n humanoidmimicgen-wbc env \
  PYTHONPATH=/home/linke/Projects/gr00t/groot/dexmg/grootrobocasa:/home/linke/Projects/gr00t/groot/dexmg/grootrobosuite:/home/linke/humanoidmimicgen \
  python /home/linke/humanoidmimicgen/scripts/playback_wbc_goals.py \
  /home/linke/Projects/gr00t/groot/dexmg/collected_demo/G1_LMDrillPnP90_20260129_231446 \
  --num-episodes 1 \
  --video-path /tmp/drillpnp90_full_ep1_wbc_goal_raw.mp4 \
  --lowres-video-path /tmp/drillpnp90_full_ep1_wbc_goal_320w.mp4 \
  --lowres-width 320
```

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
videos, but strict replay correctness is not solved by this replay path.

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
