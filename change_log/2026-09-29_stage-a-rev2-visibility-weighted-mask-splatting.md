# Stage A (rev. 2): replace learned `_dyn_logit` with classical visibility-weighted mask splatting

Supersedes the mask-splatting mechanism in `2026-09-29_stage-a-mask-splatting.md`. Everything in
that entry about mask *loading* (dataset reader, camera, `generate_masks_sam2.py`) is unchanged and
still applies; only the per-Gaussian dynamic-probability mechanism is replaced.

## Why
The learned scalar was a workaround for the CUDA rasterizer not exposing per-Gaussian blend
weights. User authorized modifying the rasterizer, so the proper formulation is now implemented:
for every training frame with a mask, each Gaussian $i$ accumulates

$$\text{fg}_i \mathrel{+}= \sum_{p} w_{ip}\,\text{label}_p, \qquad
  \text{w}_i \mathrel{+}= \sum_{p} w_{ip}, \qquad w_{ip} = \alpha_{ip}\,T_{ip}$$

where $w_{ip}$ is exactly the compositing weight the forward pass already computes for colour.
`dyn_prob_i = fg_i / w_i` is then a true visibility-weighted vote: no learning, no loss weight, no
learning rate, and it is a *statistic of the render* rather than a parameter that can drift.

## Decisions confirmed with the user
- **Fused into the existing forward pass** (not a separate accumulation entry point): zero extra
  rasterization cost; the previous approach ran a second full pass every iteration.
- **EMA over iterations**, `mask_vote_decay = 0.999` (~1000-iteration memory, ≈ one pass over the
  846 NeRF-DS frames), so votes track Gaussians as they move / densify instead of going stale.
- **Never-seen Gaussians → prob 0 (static)**: `prob = fg / max(w, 1e-8)`. A Gaussian with ~0
  accumulated visibility was never rendered by any training camera; those are precisely the
  off-frustum "parking" floaters identified in the diagnosis, and freezing them is the intended
  behaviour.
- **Accumulate from iteration 1** (revised from the earlier "from warm_up": votes don't affect
  geometry, more data is strictly better, and EMA decay makes early votes fade anyway).
- Votes are **persisted in the .ply** (`dyn_fg`, `dyn_w`) and **children inherit the parent's
  votes** on clone/split.

## Changes — CUDA rasterizer (`submodules/depth-diff-gaussian-rasterization`, a git submodule)
- `cuda_rasterizer/forward.cu` / `forward.h`: `renderCUDA` / `FORWARD::render` take three new
  optional pointers `pix_label` (per-pixel float label, `[H*W]`), `accum_fg`, `accum_w` (per-Gaussian
  `[P]`). Inside the blending loop, right where colour is composited, `atomicAdd(&accum_w[id], w)` and
  `atomicAdd(&accum_fg[id], w * label)` when `accum_w != nullptr`. Same atomics pattern the backward
  pass already uses for per-Gaussian gradient accumulation. The colour/depth path is untouched
  (factored `w = alpha*T` into a local, numerically identical).
- `cuda_rasterizer/rasterizer.h` / `rasterizer_impl.cu`: `Rasterizer::forward` forwards the three
  pointers; added as **trailing defaulted params** so other internal callers stay source-compatible.
- `rasterize_points.h` / `.cu`: `RasterizeGaussiansCUDA` takes an extra `pix_label` tensor
  (empty tensor = disabled) and returns two extra `[P]` float tensors (`accum_fg`, `accum_w`, zero-
  sized when disabled). Validates `pix_label.numel() == H*W`.
- `diff_gaussian_rasterization/__init__.py`: `_RasterizeGaussians.forward/backward` plumb the extra
  input/outputs; vote tensors are `mark_non_differentiable`; backward returns an extra `None` for
  `pix_label`. `GaussianRasterizer.forward(..., pix_label=None)` now returns **5** values
  `(color, radii, depth, accum_fg, accum_w)` instead of 3 — the only caller in this repo
  (`gaussian_renderer.render`) is updated. **Any other code calling the rasterizer directly must be
  updated for the new arity** (Deformable-3D-Gaussians has its own separate copy and is unaffected).
- Rebuilt and reinstalled into the `stag_gs` conda env (see Build notes).

## Changes — Python
- `scene/gaussian_model.py`: `_dyn_logit` parameter + its optimizer group **removed**. Replaced by
  non-learned buffers `dyn_fg`, `dyn_w` (`[N]`, like `max_radii2D`): zero-init in `create_from_pcd`;
  `get_dyn_prob = clamp(dyn_fg / max(dyn_w, 1e-8), 0, 1)`; `accumulate_dyn_votes(fg, w, decay)` does
  the EMA merge; `prune_points` masks them; `densification_postfix` gathers the parent's votes via
  `parent_idx` *before* the tensors grow and appends them (unlike the gradient stats, which reset);
  `save_ply`/`load_ply` persist `dyn_fg`/`dyn_w` (missing in old checkpoints → zeros, so e.g. the
  existing `output/as_novel_view` run still loads).
- `gaussian_renderer/__init__.py`: `render_dyn_mask` **deleted**. `render(..., pix_label=None)`
  passes the mask through and returns `"dyn_votes_fg"` / `"dyn_votes_w"` in its dict.
- `train.py`: no more BCE loss / second pass. The main `render()` call gets
  `pix_label=viewpoint_cam.mask`; inside the existing `no_grad` block (before densification, since
  the vote tensors are sized for the pre-densify point set) `gaussians.accumulate_dyn_votes(...)`.
- `arguments/__init__.py`, `configs.json`: `optimization.dyn_logit_lr` and `lazy.lambda_mask`
  **removed**; `lazy.mask_vote_decay = 0.999` added.

## Verification
- **Kernel-level test** (synthetic 3000-Gaussian scene, temp script, not committed):
  - RGB/depth output bit-identical with `pix_label=None`, all-ones label, and half-plane label.
  - `sum_i accum_w = 3823.4727` vs. per-pixel alpha summed over the image `3823.4727` (exact);
    `sum_i accum_fg = 2020.7478` vs. alpha summed over the labelled half `2020.7476`.
  - Half-plane label: Gaussians clearly on the labelled side get fg-fraction 1.000, other side 0.000.
  - Gradients w.r.t. means/colours/opacity still flow with voting enabled; vote outputs carry no grad.
  - Mismatched label size raises a clear error.
- **End-to-end** on `as_novel_view`: 300 iters, `warm_up=50`, forced densify every 20 iters —
  clean run, save OK. Saved votes: 99.9 % of Gaussians seen, prob distribution mean 0.68 with 27 %
  confidently `>0.9` and 0.2 % `<0.1` after only 300 iters (fg-heavy because the NeRF-DS masks label
  ~80 % of the frame as dynamic). Both the new ply and the legacy `as_novel_view` ply reload.
- Found `fg > w` for 77/20258 Gaussians by at most **2.1e-7 relative**: float32 atomic summation
  order differs between the two buffers. Benign; `get_dyn_prob` clamps to `[0,1]`.

## Build notes (Windows, this machine)
The system-wide CUDA on PATH is **13.2** but torch is `1.13.1+cu116`; the conda env ships its own
`cuda-nvcc 11.6` + headers/libs. MSVC 14.44 is rejected by CUDA 11.6's host-compiler check; the
installed **14.29** toolset works. Rebuild recipe (PowerShell):
```
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29 && set'  # import into session
$env:CUDA_HOME = $env:CUDA_PATH = 'C:\Users\dangm\.conda\envs\stag_gs'
$env:PATH = 'C:\Users\dangm\.conda\envs\stag_gs\bin;' + $env:PATH; $env:DISTUTILS_USE_SDK = '1'
python -m pip install --no-build-isolation --no-deps --force-reinstall ./submodules/depth-diff-gaussian-rasterization
```

## Known risks / open items
- The rasterizer change lives in the **submodule's working tree** (7 files, +89/−24); it is not
  committed there and the parent repo's submodule pointer still references upstream `d595eac`. A
  fresh clone + `git submodule update` would silently lose it. Decide whether to fork/commit the
  submodule or vendor it.
- Atomic contention: 256 threads/tile can hit the same Gaussian; cost is a fraction of the backward
  pass (which does the same) and was not measurable in the smoke run, but not profiled at 20k iters.
- The NeRF-DS masks are fg-heavy (~80 % of pixels), so `dyn_prob` will skew high scene-wide; the
  Stage B/C object grouping should threshold on `dyn_prob` *and* weight by `dyn_w` (confidence),
  not treat `dyn_prob` alone as ground truth.
- Stage B (multi control-nodes per dynamic object) and Stage C (object bounding-radius change-score
  denominator) still to do; they now consume `gaussians.get_dyn_prob` / `gaussians.dyn_w`.
