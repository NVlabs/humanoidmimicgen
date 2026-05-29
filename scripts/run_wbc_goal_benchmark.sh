#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-/home/linke/Projects/gr00t/groot/dexmg/collected_demo}
OUT=${OUT:-/tmp/hmg_wbc_goal_benchmark}
ENV_NAME=${ENV_NAME:-humanoidmimicgen-wbc}
EPISODES=${EPISODES:-1}
LOWRES_WIDTH=${LOWRES_WIDTH:-320}
MAKE_GRID=${MAKE_GRID:-1}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT"/summary.txt "$OUT"/grid_inputs.txt

SCRIPT="$REPO/scripts/playback_wbc_goals.py"

cat > "$OUT/expected_demo1_steps.txt" <<'EOF'
01_box_lift_floor 952
02_push_button 563
03_box_lift 452
04_push_shelf_forward 1285
05_drill_lift 452
06_drill_pnp 859
07_box_table_to_shelf 689
08_pick_drill_from_holder 495
09_obstacle_aware_pick_drill 952
EOF

run_one() {
  local label="$1"
  local dataset_rel="$2"
  local dataset="$DATA_ROOT/$dataset_rel"
  local video="$OUT/${label}_hmg_actions_demo1.mp4"
  local lowres="$OUT/${label}_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4"
  local log="$OUT/${label}.log"

  if [[ ! -e "$dataset" ]]; then
    echo "missing dataset: $dataset" | tee "$OUT/${label}.status"
    return 2
  fi

  echo "START $label $(date -Is)" | tee "$OUT/${label}.status"
  rm -f "$video" "$lowres" "$OUT/${label}.rc" "$OUT/${label}.ffprobe"

  set +e
  episode_args=()
  if [[ "$EPISODES" != "all" ]]; then
    episode_args=(--num-episodes "$EPISODES")
  fi

  mamba run -n "$ENV_NAME" env PYTHONPATH="$REPO" python "$SCRIPT" \
    "$dataset" \
    "${episode_args[@]}" \
    --video-path "$video" \
    --lowres-video-path "$lowres" \
    --lowres-width "$LOWRES_WIDTH" \
    >"$log" 2>&1
  local rc=$?
  set -e

  echo "$rc" > "$OUT/${label}.rc"
  if [[ -f "$video" ]]; then
    ffprobe -v error -select_streams v:0 \
      -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
      -of default=noprint_wrappers=1 \
      "$video" > "$OUT/${label}.ffprobe" 2>&1 || true
  fi
  if [[ -f "$lowres" ]]; then
    printf "file '%s'\n" "$lowres" >> "$OUT/grid_inputs.txt"
  fi

  {
    echo "END $label rc=$rc $(date -Is)"
    grep -Ei "task success|success_steps|first_success|final_success|Playback encountered" "$log" || true
  } | tee -a "$OUT/${label}.status"
}

run_one 01_box_lift_floor benchmark_dec_7_mid_conservative/G1_LMBoxLiftFloor/demo.hdf5
run_one 02_push_button G1_LMPushButton_20260129_234556/demo.hdf5
run_one 03_box_lift G1_LMBoxLift_20260129_230924/demo.hdf5
run_one 04_push_shelf_forward benchmark_dec_7_mid_conservative/G1_LMPushShelfForward/demo.hdf5
run_one 05_drill_lift G1_LMDrillLift_20260129_231155/demo.hdf5
run_one 06_drill_pnp G1_LMDrillPnP90_20260129_231446/demo.hdf5
run_one 07_box_table_to_shelf new_src_demos_jan_21/G1_LMBoxTableToShelfStaticIndustrial_Again/demo.hdf5
run_one 08_pick_drill_from_holder G1_LMPickDrillFromHolderStandingEasyFar_20260416_090757/demo.hdf5
run_one 09_obstacle_aware_pick_drill G1_LMDrillLiftObstacleDT_20260417_093010/demo.hdf5

{
  echo "Output: $OUT"
  echo "Repo: $REPO"
  echo "Data root: $DATA_ROOT"
  echo
  for rc_file in "$OUT"/*.rc; do
    label=$(basename "$rc_file" .rc)
    printf "%s rc=%s\n" "$label" "$(cat "$rc_file")"
    grep -Ei "task success|success_steps|first_success|final_success|Playback encountered" "$OUT/${label}.log" || true
    [[ -f "$OUT/${label}.ffprobe" ]] && cat "$OUT/${label}.ffprobe"
    echo
  done
} > "$OUT/summary.txt"

if [[ "$MAKE_GRID" == "1" ]]; then
  ffmpeg -y \
    -i "$OUT/01_box_lift_floor_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4" \
    -i "$OUT/02_push_button_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4" \
    -i "$OUT/03_box_lift_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4" \
    -i "$OUT/04_push_shelf_forward_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4" \
    -i "$OUT/05_drill_lift_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4" \
    -i "$OUT/06_drill_pnp_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4" \
    -i "$OUT/07_box_table_to_shelf_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4" \
    -i "$OUT/08_pick_drill_from_holder_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4" \
    -i "$OUT/09_obstacle_aware_pick_drill_hmg_actions_demo1_${LOWRES_WIDTH}w.mp4" \
    -filter_complex "[0:v][1:v][2:v]hstack=3[row0];[3:v][4:v][5:v]hstack=3[row1];[6:v][7:v][8:v]hstack=3[row2];[row0][row1][row2]vstack=3[v]" \
    -map "[v]" -shortest -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
    "$OUT/wbc_goal_benchmark_grid_${LOWRES_WIDTH}w.mp4" \
    >> "$OUT/grid.log" 2>&1 || true
fi

date -Is > "$OUT/DONE"
