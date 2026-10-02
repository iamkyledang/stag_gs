# Stage A: mask-splatting infra (dynamic/static segmentation prior)

Context: diagnosis of the STAG vs. Deformable-MLP PSNR gap on `as_novel_view` pointed at
per-Gaussian free motion producing floaters (low-opacity Gaussians drifting into
train-camera-unsupervised regions) and a `select_dynamic` score that's meaningless per-Gaussian
(100% of Gaussians end up "dynamic"). Agreed direction: segment Gaussians into dynamic-object vs.
static-background first, then (Stage B) give the dynamic object multiple control nodes instead of
per-Gaussian anchors, and (Stage C) redefine the change-score denominator at the object level.
This entry covers Stage A only: the segmentation signal itself, staged so it's testable in
isolation before Stage B/C touch the motion model.

## Decisions confirmed with the user
- Segmentation granularity: **binary fg (dynamic) / bg (static)**, not multi-instance. Matches the
  masks NeRF-DS already ships and the single-subject HyperNeRF-interp scenes.
- Mask-splatting mechanism: the CUDA rasterizer (`submodules/depth-diff-gaussian-rasterization`)
  only returns the final composited image + depth + radii, not per-Gaussian per-pixel blend
  weights, so a true non-differentiable weighted-vote splat isn't possible without patching the
  CUDA kernel (declined). Implemented instead as a **learned per-Gaussian scalar** (`_dyn_logit`,
  like opacity) rasterized through the same `colors_precomp` path used for RGB and supervised by a
  BCE loss against the ground-truth mask every iteration. Gradients pull each Gaussian's logit
  toward the label of the pixels it contributes to -- functionally a learned splat.
- Mask supervision starts at **`warm_up`** (not iteration 1), once canonical geometry has settled
  and `motion.active` is true, so it converges well before `dyn_select_iter` (Stage B/C will hook
  into this).
- HyperNeRF has no shipped masks. Built a standalone `generate_masks_sam2.py` scaffold (manual
  first-frame click + SAM2 video-predictor propagation) that the user will run themselves later;
  NOT wired into training and NOT exercised against a real SAM2 install in this session (no `sam2`
  package available here) -- see Known risks below.

## Changes
- `scene/dataset_readers.py`: `CameraInfo` gained an optional `mask` field (np.array or None).
  `readNerfiesCameras` now looks for `{scene}/mask/{ratio}x/{frame_id}.png.png` (same folder/naming
  convention as the rgb path it already builds) and loads it if present; logs once per scene if the
  `mask/` folder is absent (expected for HyperNeRF until `generate_masks_sam2.py` is run). No effect
  on Colmap/Blender/DTU/Neu3D/Dynamic-360 readers (field defaults to `None`).
- `utils/camera_utils.py` (`loadCam`): resizes+binarizes (`>0.5`) the mask to the same resolution as
  the training image; passed to `Camera`.
- `scene/cameras.py`: `Camera` stores `self.mask` (`[1,H,W]` float tensor or `None`);
  `load2device` moves it with everything else.
- `scene/gaussian_model.py`: new per-Gaussian parameter `_dyn_logit` (`get_dyn_prob = sigmoid(...)`),
  initialized to 0 (neutral, prob 0.5) in `create_from_pcd`. Threaded through
  `training_setup` (own optimizer group, `dyn_logit_lr`), `densify_and_split`/`densify_and_clone`/
  `densification_postfix`, `prune_points` (generic, no extra code needed there beyond the assignment),
  and `save_ply`/`load_ply` (`construct_list_of_attributes` gained `dyn_logit`; `load_ply` defaults
  to zeros if the property is missing, so old checkpoints -- e.g. the existing `as_novel_view` run --
  still load).
- `gaussian_renderer/__init__.py`: new `render_dyn_mask(...)`, a second rasterization pass with
  `colors_precomp = pc.get_dyn_prob.expand(-1, 3)` (3-channel colour path re-used for a scalar,
  since `colors_precomp` is fixed at 3 channels and mutually exclusive with `shs`), `bg=0` so
  uncovered pixels read as "static". Same deformed means/scales/rotations as `render()`
  (same clamp on scale as the existing `render()` negative-scale fix).
- `train.py`: after the RGB loss, if `motion.active` and `viewpoint_cam.mask is not None`, renders
  the predicted mask and adds `lazy.lambda_mask * BCE(pred, gt_mask)` to the total loss.
- `arguments/__init__.py` / `configs.json`: new `optimization.dyn_logit_lr` (0.05) and
  `lazy.lambda_mask` (0.1).
- `generate_masks_sam2.py` (new, repo root): offline scaffold for HyperNeRF-interp scenes --
  `infer_ratio()` mirrors `dataset_readers.py`'s branch logic (keyed off the *parent* folder name,
  e.g. `interp_chickchicken`, not the leaf scene folder) to pick the right `rgb/{ratio}x` and write
  to the matching `mask/{ratio}x/{frame_id}.png.png`; treats `interp_*` scenes as **one continuous
  video** (train/val there are a time-interleaved split of the same camera, not two viewpoints,
  unlike NeRF-DS) so segmentation runs once per scene; `--dry_run` exercises the click UI without
  requiring `sam2` to be installed.

## Verification
- Smoke-tested on `as_novel_view` (real masks): 40 iterations, `warm_up=5`,
  densify/clone/split forced every 5 iters from iter 1 -- exercised mask loading, the mask BCE loss,
  `_dyn_logit` through clone+split+prune, and ply save/load, with no errors.
  `dyn_logit` moved from init (mean 0, std 0) to mean 0.30 / std 0.67 / no NaNs after 40 iters,
  confirming the BCE gradient reaches it.
- `infer_ratio()` verified against `hypernerf_interp/interp_chickchicken/chickchicken`: returns
  ratio 0.5 / 456 ids, and `rgb/2x/` exists as expected (this is the folder the original bug --
  using the leaf folder name instead of the parent -- would have missed).
- Did not run a full 20k-iteration training in this session (out of scope for Stage A validation).

## Known risks / open items for next session
- `generate_masks_sam2.py`'s SAM2 call sequence (`build_sam2_video_predictor`,
  `add_new_points_or_box`, `propagate_in_video`) matches the API as of the sam2 repo's published
  examples but was **not exercised against a real install** (no `sam2` package in this env) --
  expect to need small signature fixes once you actually run it.
- `render_dyn_mask` doubles the rasterizer calls per iteration once `motion.active` (extra pass any
  iteration the current camera has a mask). Not a problem at NeRF-DS/HyperNeRF-interp scale, but
  flagging as a perf cost of the differentiable-splat approach vs. a hypothetical fused CUDA path.
- Two near-duplicate NeRF-DS mask folders exist on disk (`mask/` and `resized_mask/`, ~2% pixel
  disagreement); the loader uses `mask/`. `resized_mask/` is unused -- flag if you'd rather use it.
- Stage A only produces the per-Gaussian dynamic probability. It does **not** yet change
  `select_dynamic`, the anchor/control-node structure, or the change-score denominator -- that's
  Stage B (multi-node per dynamic object, density-scaled voxel sampling reusing `_select_controls`)
  and Stage C (denominator = object bounding-radius), both still to do, plus the background handling
  decision (fully static, no anchors) that depends on Stage B's grouping existing.
