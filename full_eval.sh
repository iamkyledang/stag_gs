#!/usr/bin/env bash
#
# LazyGS benchmark driver (bash version of full_eval.py):
# train -> render (test split) -> metrics, run on NeRF-DS first, then HyperNeRF (interp).
#
# Usage:
#   ./full_eval.sh [output_path] [iteration]
#
# Defaults write results under ./output/<dataset>/<scene> and use iteration 20000,
# matching the checkpoint saved by train.py's --save_iterations.

set -e

OUTPUT_PATH="${1:-./output}"
ITERATION="${2:-20000}"
NERF_DS_ROOT="datasets/nerf_ds"
HYPERNERF_ROOT="datasets/hypernerf_interp"

# scene name -> path relative to the dataset root (the folder that holds dataset.json)
declare -A HYPERNERF_SCENES=(
  [aleks-teapot]="interp_aleks-teapot/aleks-teapot"
  [chickchicken]="interp_chickchicken/chickchicken"
  [cut-lemon]="interp_cut-lemon/cut-lemon1"
  [hand]="interp_hand/hand1-dense-v2"
  [slice-banana]="interp_slice-banana/slice-banana"
  [torchocolate]="interp_torchocolate/torchocolate"
)

# lists every immediate subfolder of $1 (each one a scene dir containing dataset.json)
discover_scenes () {
  local root="$1"
  for dir in "$root"/*/; do
    basename "$dir"
  done
}

train_scenes_all () {
  local root="$1" tag="$2"
  while IFS= read -r scene; do
    local source="$root/$scene"
    local model="$OUTPUT_PATH/$tag/$scene"
    python train.py -s "$source" -m "$model" --eval --load2gpu_on_the_fly --quiet \
      --iterations "$ITERATION" --test_iterations "$ITERATION" --save_iterations "$ITERATION"
  done < <(discover_scenes "$root")
}

render_scenes_all () {
  local root="$1" tag="$2"
  while IFS= read -r scene; do
    local source="$root/$scene"
    local model="$OUTPUT_PATH/$tag/$scene"
    python render.py -s "$source" -m "$model" --iteration "$ITERATION" --mode render --skip_train --quiet
  done < <(discover_scenes "$root")
}

metrics_for_all () {
  local root="$1" tag="$2"
  local models=()
  while IFS= read -r scene; do
    models+=("$OUTPUT_PATH/$tag/$scene")
  done < <(discover_scenes "$root")
  python metrics.py -m "${models[@]}"
}

train_scenes () {
  local root="$1" tag="$2"
  local -n scenes_ref="$3"
  for scene in "${!scenes_ref[@]}"; do
    local source="$root/${scenes_ref[$scene]}"
    local model="$OUTPUT_PATH/$tag/$scene"
    python train.py -s "$source" -m "$model" --eval --load2gpu_on_the_fly --quiet \
      --iterations "$ITERATION" --test_iterations "$ITERATION" --save_iterations "$ITERATION"
  done
}

render_scenes () {
  local root="$1" tag="$2"
  local -n scenes_ref="$3"
  for scene in "${!scenes_ref[@]}"; do
    local source="$root/${scenes_ref[$scene]}"
    local model="$OUTPUT_PATH/$tag/$scene"
    python render.py -s "$source" -m "$model" --iteration "$ITERATION" --mode render --skip_train --quiet
  done
}

metrics_for () {
  local tag="$1"
  local -n scenes_ref="$2"
  local models=()
  for scene in "${!scenes_ref[@]}"; do
    models+=("$OUTPUT_PATH/$tag/$scene")
  done
  python metrics.py -m "${models[@]}"
}

# ---------------- NeRF-DS (all scenes under $NERF_DS_ROOT) ----------------
train_scenes_all "$NERF_DS_ROOT" "nerf_ds"
render_scenes_all "$NERF_DS_ROOT" "nerf_ds"
metrics_for_all "$NERF_DS_ROOT" "nerf_ds"

# ---------------- HyperNeRF (interp) ----------------
# disabled for now
# train_scenes "$HYPERNERF_ROOT" "hypernerf" HYPERNERF_SCENES
# render_scenes "$HYPERNERF_ROOT" "hypernerf" HYPERNERF_SCENES
# metrics_for "hypernerf" HYPERNERF_SCENES