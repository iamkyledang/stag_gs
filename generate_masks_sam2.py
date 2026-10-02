#
# Offline, one-scene-at-a-time mask generator for datasets that ship no dynamic-object mask
# (HyperNeRF-interp; NeRF-DS already has one under mask/{ratio}x/*.png.png and does not need this).
#
# You annotate the FIRST frame once (click on the moving object), SAM2's video predictor propagates
# that mask through the whole sequence, and the result is written in the exact layout
# scene/mask/{ratio}x/{frame_id}.png.png that scene/dataset_readers.py already knows how to read --
# so training picks the mask up automatically once it exists, no other change needed.
#
# This script is intentionally standalone: `sam2` is NOT a dependency of the main training env
# (train.py / render.py never import it). Install it yourself when you're ready to segment a scene:
#   pip install "git+https://github.com/facebookresearch/sam2.git"
#   (+ download a checkpoint, e.g. sam2.1_hiera_large.pt, and its matching model config yaml --
#    see https://github.com/facebookresearch/sam2#getting-started)
#
# Usage:
#   python generate_masks_sam2.py -s E:\motion_GS\hypernerf_interp\interp_chickchicken\chickchicken \
#       --sam2_checkpoint path\to\sam2.1_hiera_large.pt --sam2_config sam2.1_hiera_l.yaml
#
# Add --dry_run to only pop the annotation window and print the clicked points (no sam2 needed) --
# useful to sanity-check the click UI before installing/downloading anything.
#
# NOTE: SAM2's Python API has changed across releases; the propagation call in _run_sam2_propagation()
# may need small adjustments to match whatever sam2 version you end up installing.
#

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


def infer_ratio(scene_path):
    """Mirror scene/dataset_readers.py:readNerfiesCameras's branch logic so we read/write the same
    rgb/{ratio}x <-> mask/{ratio}x resolution the training-time loader expects."""
    with open(os.path.join(scene_path, "dataset.json")) as f:
        dataset_json = json.load(f)
    # dataset_readers.py keys the branch off the PARENT folder name (e.g. "interp_chickchicken"),
    # not the leaf scene folder ("chickchicken") -- match that exactly or ratio/rgb-folder picks diverge.
    name = os.path.basename(os.path.dirname(os.path.normpath(scene_path)))
    has_split = "train_ids" in dataset_json and "val_ids" in dataset_json
    if name.startswith("vrig"):
        return 0.25, dataset_json["train_ids"] + dataset_json["val_ids"]
    if name.startswith("NeRF") or (has_split and not name.startswith("interp")):
        return 1.0, dataset_json["train_ids"] + dataset_json["val_ids"]
    if name.startswith("interp"):
        # single continuous video: train/val here are an interleaved time-subsample of the SAME camera,
        # not a second viewpoint, so segment the union in chronological (dataset.json) order.
        return 0.5, dataset_json["ids"]
    return 0.5, dataset_json["ids"][::4]


def pick_first_frame_points(image_path):
    """Pop the first frame; left-click the dynamic object (as many points as you like), close the
    window (or press Enter) when done. Returns (points [N,2] float, labels [N] int, all label=1)."""
    import matplotlib.pyplot as plt

    img = np.array(Image.open(image_path))
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Left-click the dynamic object (>=1 pt), then close this window / press Enter")
    pts = fig.ginput(n=-1, timeout=0)
    plt.close(fig)
    if len(pts) == 0:
        raise RuntimeError("No points were clicked -- at least one prompt point is required.")
    points = np.array(pts, dtype=np.float32)
    labels = np.ones((points.shape[0],), dtype=np.int32)
    return points, labels


def stage_frames(image_paths, staging_dir):
    """SAM2's video predictor expects a directory of frames named as zero-padded ints; stage symlinks
    (falls back to copies on platforms/filesystems without symlink permission, e.g. some Windows setups)
    in `ids` order so frame index == position in `image_paths`."""
    os.makedirs(staging_dir, exist_ok=True)
    ext = os.path.splitext(image_paths[0])[1]
    for i, src in enumerate(image_paths):
        dst = os.path.join(staging_dir, f"{i:05d}{ext}")
        if os.path.exists(dst):
            continue
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copyfile(src, dst)


def _run_sam2_propagation(staging_dir, points, labels, checkpoint, model_cfg):
    """Isolated so the fragile part (sam2's API surface) is easy to patch without touching the rest
    of the script. Returns {frame_idx: bool mask [H,W]}."""
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as e:
        raise ImportError(
            "sam2 is not installed. Run this script again after:\n"
            "  pip install \"git+https://github.com/facebookresearch/sam2.git\"\n"
            "and downloading a checkpoint + matching model config (see module docstring)."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    masks = {}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        state = predictor.init_state(video_path=staging_dir)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                        points=points, labels=labels)
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks[frame_idx] = (mask_logits[0, 0] > 0.0).cpu().numpy()
    return masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_path", "-s", required=True, type=str, help="scene folder (has dataset.json etc.)")
    p.add_argument("--sam2_checkpoint", type=str, default=None)
    p.add_argument("--sam2_config", type=str, default=None)
    p.add_argument("--dry_run", action="store_true", help="only run the click UI, skip SAM2 + mask writing")
    args = p.parse_args()

    ratio, ids = infer_ratio(args.source_path)
    res_dir = f"{int(1 / ratio)}x"
    rgb_dir = os.path.join(args.source_path, "rgb", res_dir)
    image_paths = [os.path.join(rgb_dir, f"{i}.png") for i in ids]
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frame(s) missing, e.g. {missing[0]}")
    print(f"[masks] {len(ids)} frames at rgb/{res_dir}")

    points, labels = pick_first_frame_points(image_paths[0])
    print(f"[masks] {len(points)} prompt point(s) on frame 0 ({ids[0]})")
    if args.dry_run:
        print("[masks] --dry_run set: stopping before SAM2. Points:", points.tolist())
        return

    if not args.sam2_checkpoint or not args.sam2_config:
        raise ValueError("--sam2_checkpoint and --sam2_config are required unless --dry_run is set")

    staging_dir = os.path.join(args.source_path, ".sam2_frames_tmp")
    stage_frames(image_paths, staging_dir)
    try:
        masks = _run_sam2_propagation(staging_dir, points, labels, args.sam2_checkpoint, args.sam2_config)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    out_dir = os.path.join(args.source_path, "mask", res_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i, frame_id in enumerate(ids):
        m = masks.get(i)
        if m is None:
            print(f"[masks] WARNING: no propagated mask for frame {i} ({frame_id}), skipping")
            continue
        Image.fromarray((m.astype(np.uint8) * 255)).save(os.path.join(out_dir, f"{frame_id}.png.png"))
    print(f"[masks] wrote {len(masks)}/{len(ids)} masks to {out_dir}")


if __name__ == "__main__":
    main()
