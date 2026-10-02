# motion_GS — Architecture & Engineering Specification

> Generated as a snapshot for external analysis (fed into an LLM for review). Reflects the state of
> the workspace as of 2026-10-01. Three related codebases live side by side; `stag_gs` is the actively
> developed one and the primary subject of this document.

## 1. Workspace layout

```
motion_GS/
  Deformable-3D-Gaussians/   # unmodified-ish upstream baseline (MLP-based deformation field)
  stag_gs/                   # this project's research code, formerly named "LazyGS"
  dnerf/                     # D-NeRF synthetic dataset (blender-style, per-frame transforms.json)
  nerf_ds/                   # NeRF-DS real-capture dataset (dynamic specular objects, has masks)
  hypernerf_interp/          # HyperNeRF "interp_*" scenes (single continuous video, no masks yet)
```

Both `Deformable-3D-Gaussians` and `stag_gs` are full, independent training/rendering pipelines built
on top of the original 3D Gaussian Splatting (3DGS) codebase (Inria/GRAPHDECO), each vendoring its own
copy of the CUDA rasterizer and `simple-knn` as git submodules. `stag_gs` started as a fork of the
former (`LazyGS`) and has since diverged substantially, including a **patched CUDA rasterizer**.

---

## 2. Baseline: `Deformable-3D-Gaussians`

Standard 3DGS (canonical point cloud: xyz, SH color, opacity, scale, rotation) plus a small MLP
"deform net" that maps `(xyz, t) -> (d_xyz, d_rotation, d_scaling)`.

- **Interface contract**: `DeformModel.step(xyz[N,3] detached, t[N,1]) -> (d_xyz, d_rotation[N,4], d_scaling[N,3])`.
  `d_xyz` is `[N,3]`, or `[N,4,4]` homogeneous when `--is_6dof`. Called from `train.py`, `render.py`,
  `train_gui.py`.
- **Compositing** in `gaussian_renderer/__init__.py` is purely **additive**:
  `means = xyz + d_xyz`; `scales = exp(_scaling) + d_scaling`; `rotations = normalize(_rotation) + d_rotation`
  (quaternion is **not** re-normalized post-addition — the CUDA rasterizer's normalization step is
  commented out). The `compute_cov3D_python` code path bypasses deformation entirely (bug/limitation).
- SH view-direction uses the **canonical** xyz, not the deformed one. Opacity and SH coefficients are
  never deformed (opacity deformation was added later, only in `stag_gs`).
- The deform net is a pure field `f(x,t)` with **no per-Gaussian learned parameters** — densification
  and pruning in `GaussianModel` are untouched by the deform model. (This is the key structural
  difference from `stag_gs`, which *does* need densify/prune to carry motion-model state.)
- **Training schedule**: `warm_up = 3000` static iterations (deform outputs forced to zero) before the
  deform net is used; AST-style time noise (`randn() * (1/num_frames) * smooth_term(iter)`) is added to
  the query time for non-blender (real-capture) datasets; separate Adam optimizer for the deform net
  (`lr = position_lr_init * 5`, exponential decay over `deform_lr_max_steps = 40000`). Loss is
  `L1 + 0.2 * (1 - SSIM)` only (no extra regularizers).
- **Time id (`fid`) convention**, shared with `stag_gs`: blender uses `frame['time']` directly (already
  `[0,1]`); NeRF-DS/HyperNeRF (`nerfies`) uses `time_id / max_time`; COLMAP-style sequences use
  `int(name) / (num_frames - 1)`.

### Baseline bugs found & fixed (this workspace's copy)
- **Submodules empty after clone** — needed `git submodule update --init --recursive`.
- **Windows path bug** in `scene/dataset_readers.py: readNerfiesCameras` — used
  `path.split('/')[-2]`, which breaks on Windows backslash paths (`IndexError`), and a case-sensitive
  `'NeRF'` prefix check that fails against the lowercase `nerf_ds` folder name. Fixed with
  `os.path.normpath(path).split(os.sep)` plus a `.lower()` comparison.
- **`fetchPly()` assumed normals always present.** NeRF-DS's supplied `points3d.ply` has no
  `nx/ny/nz`, raising a `KeyError` silently swallowed by a bare `except` in the caller, yielding
  `pcd=None` and crashing later in `create_from_pcd`. Fixed by defaulting normals to zero when absent.
- **Negative/zero Gaussian scale crash.** `scales = pc.get_scaling + d_scaling` can go `<= 0` because
  `d_scaling` (MLP output) is unbounded while `get_scaling = exp(_scaling) > 0` always. A non-positive
  scale produces an invalid covariance, crashing the CUDA rasterizer ("Storage size calculation
  overflowed"). Fixed with `torch.clamp_min(pc.get_scaling + d_scaling, 1e-6)`.
- **Test-time CUDA OOM.** `training_report` concatenated every test image into one giant GPU tensor
  before computing L1/PSNR once at the end — OOMs on large NeRF-DS test sets. Fixed by accumulating
  scalar per-image sums via `.item()`, `del`-ing tensors each iteration, and `torch.cuda.empty_cache()`
  after each validation config. This changes `test_psnr`/`cur_psnr` from a tensor to a plain float at
  the call site.
- **`torchvision` import crash on Python 3.7.1`** (`typing.OrderedDict` was only added in 3.7.2, and
  torchvision's `maxvit.py` needs it). Fixed with a `typing.OrderedDict` shim before the import in
  `render.py`/`metrics.py`.
- **Device-mismatch crash in `render.py`'s interpolation modes.** `loadCam` builds cameras with
  `data_device='cpu'` when `load2gpu_on_the_fly=True` (carried over from the training `cfg_args`).
  `render_set` correctly calls `view.load2device()` per frame, but all of the `interpolate_time/view/
  all/poses/view_original` functions (used by `--mode time/view/all/pose/original`) accepted the same
  flag (`load2gpt_on_the_fly`, typo in the upstream fork) but never called `load2device()` — an
  upstream bug. Produced "Expected all tensors to be on the same device" because
  `world_view_transform` was force-`.cuda()`'d while `projection_matrix` stayed on CPU. Fixed by adding
  `if load2gpt_on_the_fly: view.load2device()` once per interpolate function.

---

## 3. `stag_gs` (formerly `LazyGS`) — core design

### 3.1 Motivating idea ("framework.tex")

Replace the MLP deform field with a **lazy-evaluated, sparse control-node** motion representation:
instead of asking a neural net "what is the motion of point x at time t?", store motion *samples*
("anchors") at a small number of discrete times and a small number of discrete spatial control nodes,
and reconstruct the motion of any Gaussian at any query time via local-linear interpolation/
extrapolation — done **through** the query operator so training gradients reach every anchor used in
a given render (not a two-stage fit-then-render pipeline).

High-level query algorithm (`LazyMotionModel.deltas_at(t)` and friends, in `scene/lazy_motion.py`):

1. Map continuous query time `t` to a **window of `W` anchor intervals** around `t` at stride `A`
   (frames), i.e. `W` anchors on each side.
2. At each anchor, compute per-control-node **motion rates** `u ∈ R^10`:
   `(v [velocity, 3], omega [so(3) angular rate, 3], eta [log-scale rate, 3], zeta [opacity-logit rate, 1])`.
3. For each query point, find its **K spatial nearest neighbours** among control nodes and fit a local
   linear gradient `J` via weighted least squares (`local_linear_fit`, Tikhonov-regularized).
4. **Bidirectional Taylor transport**: extrapolate the anchor's rate forward/backward in time
   (1st order = velocity only when `W=1`/`transport_order=1`; 2nd order adds acceleration) to the exact
   query time, from both the preceding and following anchors in the window.
5. **Confidence-weighted consensus**: combine the forward/backward/multi-anchor estimates, weighted by
   time-proximity and (optionally) the WLS fit residual (`consensus_sigma`, `no_fit_confidence`).
6. Output: `(d_xyz, d_rotation, d_scaling, d_opacity)` per dynamic Gaussian, fed into
   `gaussian_renderer.render()` exactly like the baseline's MLP deltas.

### 3.2 Representation evolution (what's actually implemented today)

The design went through several iterations in-session (see `stag_gs/change_log/*.md` for full detail);
**the current (Tier 1) state is**:

- Every Gaussian is either **frozen-static** (never moves, no motion parameters at all) or **dynamic**.
  Static/dynamic split comes from a learned-free, differentiable **visibility-weighted mask vote**
  (Stage A rev. 2 — see §3.3), decided **once** at `dyn_select_iter` (one-shot background freeze).
- Of the dynamic set, only a small fixed-size subset (**control nodes**, `num_nodes`, default 512,
  chosen via density-scaled voxel sampling, `_select_controls`) carry **independently optimised**
  per-anchor deltas (`_dx`, `_drot`, `_dscale`, `_dopacity`, shaped `[M anchors, Nctrl, .]`).
- Every other dynamic (non-control) Gaussian's motion is **derived**, every query, from its nearest
  control nodes via the same local-linear extrapolation machinery (`_predict_from_controls`) — it is
  never an independent free parameter. This makes "floaters" (Gaussians drifting unsupervised into
  novel-view-only regions) structurally impossible for non-control points.
- All KNN/neighbour graphs (`build_knn`, the control↔non-control prediction graph `_fill_graphs`) are
  restricted to the **dynamic object's own index set** (`free_idx`/`dyn_idx`), never the whole point
  cloud — this was a deliberate fix for an earlier cross-object contamination bug (see §4).
- `object_radius` (95th-percentile distance from the dynamic set's centroid, computed once at freeze)
  replaces per-Gaussian scale as the denominator for the **change-score** used by storage
  sparsification (Stage C).

### 3.3 Dynamic/static segmentation ("mask-vote splatting", Stage A)

- NeRF-DS ships per-frame binary masks (`mask/{ratio}x/{id}.png.png`); HyperNeRF scenes do not — a
  standalone `generate_masks_sam2.py` (manual first-frame click + SAM2 video-predictor propagation) is
  provided but **not wired into training**, and is untested against a real SAM2 install.
- **Mechanism (current, rev. 2)**: the CUDA rasterizer forward pass was patched
  (`submodules/depth-diff-gaussian-rasterization`, working-tree change, **not committed in the
  submodule**) to optionally accept a per-pixel label (`pix_label`) and atomically accumulate, per
  Gaussian, the same compositing weight already used for colour:
  `accum_w += alpha*T`, `accum_fg += alpha*T*label`. This is a true visibility-weighted vote, not a
  learned parameter — `get_dyn_prob = clamp(dyn_fg / max(dyn_w, 1e-8), 0, 1)`.
  `GaussianRasterizer.forward(..., pix_label=None)` now returns **5** values
  `(color, radii, depth, accum_fg, accum_w)` (was 3).
- `dyn_fg`/`dyn_w` are **non-learned buffers** on `GaussianModel` (like `max_radii2D`): EMA-updated
  every iteration from iteration 1 (`mask_vote_decay = 0.999`, ≈1000-iteration memory), inherited by
  children on densify/clone/split, persisted in the `.ply`. A never-rendered Gaussian defaults to
  `prob = 0` (static) — intentionally, since those are exactly the off-frustum "parking" floaters the
  whole investigation started from.
- An earlier revision (Stage A rev. 1) used a *learned* per-Gaussian scalar (`_dyn_logit`) supervised
  by BCE loss against the mask (a second rasterization pass). This was **fully replaced** by the
  visibility-weighted vote above (zero extra rasterizer cost, no learning rate/loss-weight to tune,
  not a parameter that can drift).
- **Mask polarity bug (critical, found+fixed)**: `loadCam`'s binarization assumed white=dynamic, but
  NeRF-DS's real convention is inverted (white ≈ 80% of pixels = static background; black ≈ 20% = the
  moving subject). This silently existed since rev. 1 but only caused visible damage once Tier 1
  started thresholding `dyn_prob` for a hard freeze decision — it froze the true moving object and
  kept a diffuse chunk of background as "dynamic" (`object_radius` blew up to 0.54 vs scene extent
  0.234). Fixed to `(mask_pixel < 0.5)`.

### 3.4 Segmentation-driven control-node architecture (Stage B/C, "Tier 1")

Implemented, decisions locked before implementation:
- Background freeze is **one-shot**, rule `dyn_prob > 0.5 & dyn_w > dyn_w_min`, fixed for the rest of
  training (no periodic re-evaluation).
- Control-node budget is an **absolute count** (`num_nodes`, default 512), not a fraction of the
  dynamic set.
- Kept the existing local-linear-fit/MLS math; declined an alternative Embedded-Deformation-
  Graph/LBS formulation.
- Temporal visibility (opacity) is a first-class 10th channel now, not deferred.

Key `LazyMotionModel` surface (post-Tier-1):
- `freeze_and_select_controls(tau_w, num_nodes)` replaces the old per-Gaussian `select_dynamic` (which
  was circular — it needed motion to already exist to decide who gets motion).
- `is_dyn` (bool `[N]`), `is_ctrl` (bool `[N]`, subset of `is_dyn`), `ctrl_idx`/`free_idx` properties
  (`free_idx == ctrl_idx` post-freeze, `== dyn_idx` pre-freeze — the single switch that makes most of
  the class phase-agnostic).
- `on_densify`/`on_prune` key off `is_ctrl` once set, so only control-node children/removals touch the
  free-parameter tensors.
- Rate/field vectors extended **9-D → 10-D** (added `zeta`, opacity-logit rate).
  `gaussian_renderer.render()` gained `d_opacity=0.0`, applied as `sigmoid(pc._opacity + d_opacity)`.
- `motion.step`/`deltas_at` now return a **5-tuple** `(d_xyz, d_rotation, d_scaling, d_opacity, info)`
  (was 4) — updated at all call sites (`train.py` main loop + `training_report`, `render.py`'s 7 call
  sites: `render_set` ×2, `interpolate_time/view/all/poses/view_original`).
- **Storage simplified**: the old two-tier control + non-control-residual encoding is gone — only
  control-node records are ever persisted (`motion.npz` stores `ctrl_idx_global`); non-control values
  are always re-derived at load time via `_predict_from_controls`.
- `LazyParams` changes: removed `eps_dyn`, `control_frac`; added `dyn_w_min` (10.0), `num_nodes` (512),
  `kappa_opacity` (2.0), `lambda_opacity` (0.1), `motion_opacity_lr` (0.05).
- **Not safe on HyperNeRF yet** — without masks, `dyn_prob` is uniformly ~0 everywhere and
  `freeze_and_select_controls` would freeze the entire point cloud.

### 3.5 "Tier 0" fixes (applied before Tier 1, to keep effects attributable)

1. **Anchor learning-rate schedule rescoping.** `update_learning_rate(iteration)` fed the *global*
   iteration into a decay curve built once over `[0, opt.iterations]`; anchors created later by
   `refine_anchors()` inherited an already-decayed rate designed for a run they never saw the start of.
   Fixed with `_rebuild_schedule(ref_iter)`, rebuilt at `warm_up` and at every `refine_anchors(iteration)`
   call (now requires the `iteration` argument); decay floor raised from ~1% to 10% of init
   (`motion_lr_final`).
2. **Temporal-acceleration regularizer rescaling.** The translational smoothness term normalised
   second differences of `dx` by each Gaussian's *own* scale (~1e-3 of scene extent) instead of a
   scene-scale quantity, inflating the effective loss weight by ~1e6×. Fixed to normalise by
   `self.extent`. `lambda_rigid`/`lambda_scale` were deliberately left unchanged (their over-strength
   is tied to Stage B's graph fix and Tier 1's real temporal-visibility channel, respectively).
3. **AST-style time jitter**, ported from the Deformable-3D-Gaussians MLP baseline: query time gets
   `randn() * (1/num_frames) * smooth_term(iteration)` (annealed ~half a frame → ~0), applied
   unconditionally (this project has no blender/is_blender distinction) only in the training loop, not
   at eval/render time.

### 3.6 Loss function (current)

```
loss = (1 - lambda_dssim) * L1(render, gt) + lambda_dssim * (1 - SSIM(render, gt))
     + lambda_rigid    * reg['rigid']
     + lambda_temporal * reg['temporal']
     + lambda_scale    * reg['scale']
     + lambda_opacity  * reg['opacity']
```
`reg` comes from `LazyMotionModel.regularization(info)`, computed over the **free (control) point set
only** (rigid/ARAP-style term, temporal 2nd-difference smoothness, scale-delta penalty, opacity-delta
penalty). No BCE/mask loss term anymore (removed with Stage A rev. 2 — the mask vote is now a
non-differentiable statistic, not a supervised prediction).

### 3.7 Key hyperparameters (`stag_gs/configs.json`, single source of truth; CLI flags override)

| Group | Key params |
|---|---|
| `model` | `sh_degree=3`, `load2gpu_on_the_fly=false` |
| `optimization` | `iterations=20000`, `warm_up=3000`, standard 3DGS lr/densify schedule |
| `lazy` (representation) | `anchor_stride=8` (A), `temporal_window=2` (W), `knn_k=16` (K), `transport_order=2`, `mls_beta=0.5`, `consensus_sigma=1.0`, `wls_eps=0.01`, `coarse_levels=3` |
| `lazy` (schedule) | `anchor_refine_iters=[5000,7000,9000]`, `dyn_select_iter=11000`, `dyn_w_min=10.0`, `num_nodes=512`, `knn_update_interval=1000` |
| `lazy` (segmentation) | `mask_vote_decay=0.999` |
| `lazy` (storage) | `epsilon=0.05`, `score_kappa_rot=0.2`, `score_kappa_scale=0.2`, `kappa_opacity=2.0` |
| `lazy` (regularizers) | `lambda_rigid=1.0`, `lambda_temporal=0.05`, `lambda_scale=1.0`, `lambda_opacity=0.1` |
| `lazy` (motion lr) | `motion_lr_init=0.0008`, `motion_lr_final=0.00008`, `motion_rot_lr=0.001`, `motion_scale_lr=0.001`, `motion_opacity_lr=0.05` |

### 3.8 Training loop structure (`stag_gs/train.py`)

```
for iteration in 1..opt.iterations:
    if iteration == warm_up: motion.setup(gaussians, opt)      # motion params created here
    if motion.active:
        if iteration in anchor_refine_iters: motion.refine_anchors(iteration)
        if iteration == dyn_select_iter: motion.freeze_and_select_controls(dyn_w_min, num_nodes)
        if iteration % knn_update_interval == 0: motion.knn_dirty = True

    pick random train camera, compute ast_noise, d_xyz/d_rot/d_scale/d_opacity = motion.deltas_at(t)
    render(..., pix_label=cam.mask, d_opacity=d_opacity)         # single fused pass: RGB + mask vote
    loss = L1/SSIM + motion.regularization(info) terms
    loss.backward()
    gaussians.accumulate_dyn_votes(votes_fg, votes_w, mask_vote_decay)   # before densify (sizing)
    training_report(...)                                          # periodic eval, scalar-accumulated
    densify_and_prune / reset_opacity (standard 3DGS schedule; on_densify/on_prune hooks keep motion
        parameters in sync with the point cloud)
    optimizer.step() for both gaussians.optimizer and motion.optimizer
```

### 3.9 Datasets supported (`scene/dataset_readers.py`)

`sceneLoadTypeCallbacks` dispatches on folder signature: COLMAP (`readColmapCameras`), Blender/D-NeRF
synthetic (`readCamerasFromTransforms`), DTU (`readDTUCameras`), Nerfies/HyperNeRF/NeRF-DS
(`readNerfiesCameras`), and a `.npy`-camera variant (`readCamerasFromNpy`). The dataset-variant branch
inside `readNerfiesCameras` (vrig / NeRF-DS / interp / hypernerf) keys off the **parent** folder name
(`os.path.dirname`), not the leaf scene folder — e.g. for `hypernerf_interp/interp_chickchicken/
chickchicken`, the branch name is `interp_chickchicken`, not `chickchicken`.
`hypernerf_interp`'s `interp_*` scenes are a **time-interleaved train/val split of a single
camera/video** (not two separate viewpoints like NeRF-DS), so segmentation treats them as one
continuous sequence.

---

## 4. Chronological record of major changes / bugs in `stag_gs`

1. **Stage A rev. 1** — learned `_dyn_logit` + BCE mask loss (superseded).
2. **Stage A rev. 2** — classical visibility-weighted mask-vote splatting fused into the CUDA forward
   pass (current mechanism, see §3.3). Requires a patched rasterizer submodule.
3. **Tier 0** — anchor-lr rescoping, temporal-regularizer rescaling, AST time-noise (see §3.5).
4. **Tier 1 / Stage B+C** — one-shot background freeze, control-node promotion, opacity channel,
   per-object KNN graphs, simplified storage (see §3.4). Initially **not smoke-tested**; a first real
   20k run regressed hard due to the mask-polarity bug (§3.3), which was found and fixed; a
   post-fix retrain is analyzed in §5.

### Deep, hard-won debugging lessons
- **Training is not run-to-run reproducible**, even with an identical seed (`safe_state` seeds
  `random`/`numpy`/`torch` to 0). Verified directly: two 500-iteration runs with byte-identical
  config/code/seed produced Gaussian counts differing by ~4% (26009 vs 24978) and very different
  xyz/opacity statistics. **Root cause**: `backward.cu` uses `atomicAdd` for per-Gaussian gradient
  accumulation across pixels/tiles — GPU warp-scheduling-dependent float summation order introduces
  tiny per-iteration gradient noise, which the chaotic densify/prune/split feedback loop amplifies
  into visibly different final states over many iterations. **Practical implication**: a few
  hundredths of PSNR / thousandths of SSIM-LPIPS difference between two "identical" 20k runs, with
  mixed sign across metrics, is within this noise floor — not evidence of a real regression. Trust
  only larger, consistently-signed deltas, or average multiple seeds.
- Always **sanity-check a mask/label's polarity by rendering it**, not by trusting a code comment's
  assumption — the mask-inversion bug above existed silently for an entire development phase before
  it had any visible effect.
- The CUDA rasterizer patch lives only in the **submodule's working tree** (uncommitted); a fresh
  clone + `git submodule update` would silently lose it. Needs to be forked/committed or vendored.

---

## 5. Current empirical status / open investigation

- **Baseline comparison (pre-Tier-1 diagnosis)**: on `nerf_ds/as_novel_view` at 20k iterations,
  stag_gs scored 24.80 dB test PSNR vs. the Deformable-MLP baseline's 26.36 dB, despite near-identical
  train-view PSNR (35.4 vs 35.9) — i.e. the gap is **novel-view generalization**, not fitting quality.
  Root cause identified as free per-anchor motion producing floaters (low-opacity Gaussians moved into
  regions unsupervised by any training camera but visible from the test camera). This is what motivated
  Stage A/B/C.
- Canonical point counts in that diagnosis: stag_gs 167k Gaussians (43% with opacity < 0.05) vs. MLP
  baseline's 95k.
- `select_dynamic`'s original per-Gaussian normalisation (by the Gaussian's own size, ~0.002, vs. scene
  extent 0.234) made ~100% of Gaussians register as "dynamic" — the sparsity mechanism never actually
  engaged. Fine (stride-8, 107-anchor) anchors only received gradient on ~4% of iterations
  (`p ≈ 2WA/T`) with an already-decayed learning rate — this motivated Tier 0's lr-rescoping fix.
- **Post-mask-fix Tier-1 retrain (2026-10-01)**: full 20k retrain with the mask-polarity fix gave
  `as_novel_view` test PSNR mean 24.10 (barely moved from the pre-fix 24.04), std 3.17 (up from a
  historical ~1.79). Decomposing per-frame PSNR against the Deformable-MLP baseline on identical frame
  names: **98/846 test frames (11.6%)** have stag_gs PSNR < 20 dB (mean 17.26) clustered in 3 evenly
  spaced, contiguous blocks (~30-35 frames each, spaced ~185-190 frames apart — suggestive of a
  periodic/oscillating novel-view camera path hitting the same hard viewing angle repeatedly); the MLP
  baseline gets 25.35 dB mean on those exact frames with no visible artifact. On the remaining 88.4% of
  frames the gap shrinks to a much more plausible ~1.5 dB (25.00 vs 26.49) — closer to an "architecture
  expressiveness" gap than a structural failure.
  - Confirmed via side-by-side render (frame 00729, the worst) that the failure is a **static/
    canonical reconstruction defect** (a black under-reconstructed blob + overall blur), **not a
    motion-model defect**: rendering with all deltas zeroed reproduces the identical artifact.
  - Total Gaussian count after this run: N=120412, dyn=47803 (39.7%), ctrl=608 — notably **lower**
    than earlier pre-Tier-1 runs (~163k-167k), consistent with a possible densification shortfall.
  - **Leading unconfirmed hypothesis**: Tier 1's regularizer/mask-vote terms compete with the
    photometric loss during early/mid training, leaving geometric coverage holes that only surface
    from specific novel-view angles — but this has not been isolated from ordinary run-to-run
    nondeterminism (no backup of the pre-Tier-1 `per_view.json` survived for a direct before/after
    comparison; output directories are overwritten each run).
  - **Next diagnostic steps (not yet done)**: keep a per-run backup of `per_view.json`; inspect the
    Gaussian-count growth curve / opacity-reset timing around `dyn_select_iter` for interaction with
    `densify_and_prune`/`reset_opacity`; rerun with `motion.active=False` (pure static 3DGS) to check
    whether the same 3 periodic holes appear (would prove it's not Tier-1-specific).

---

## 6. Environment / build notes

- `stag_gs` conda env: Python 3.7.1, `C:\Users\dangm\.conda\envs\stag_gs\python.exe`. Needs a
  `typing.OrderedDict = collections.OrderedDict` shim before importing `torchvision` (added to
  `render.py`/`metrics.py`).
- Windows CUDA rebuild recipe for the patched rasterizer submodule (system CUDA on `PATH` is 13.2, but
  torch is `cu116` — must use the env's own `cuda-nvcc 11.6`):
  ```
  cmd /c '"...\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29 && set'   # import into session; MSVC 14.44 rejected by CUDA 11.6, 14.29 works
  $env:CUDA_HOME = $env:CUDA_PATH = 'C:\Users\dangm\.conda\envs\stag_gs'
  $env:PATH = 'C:\Users\dangm\.conda\envs\stag_gs\bin;' + $env:PATH
  $env:DISTUTILS_USE_SDK = '1'
  python -m pip install --no-build-isolation --no-deps --force-reinstall ./submodules/depth-diff-gaussian-rasterization
  ```
- Both baselines (`Deformable-3D-Gaussians`, `stag_gs`) are evaluated with identical metric code
  (`metrics.py`, SSIM/PSNR, LPIPS net=`vgg`), same 20k-iteration budget, same NeRF-DS reader, for
  apples-to-apples comparison.

---

## 7. Suggested angles for external review

- Is the local-linear-fit / WLS Jacobian extrapolation (vs. a learned MLP, or an Embedded-Deformation-
  Graph/LBS formulation that was explicitly declined) the right tradeoff for generalizing to novel
  camera views, given the observed gap is specifically a novel-view problem?
- Is the one-shot, hard (non-differentiable) background freeze at `dyn_select_iter` too early/brittle
  compared to a soft or periodically-re-evaluated split?
- Interaction between `num_nodes=512` (untuned), `dyn_w_min=10.0` (untuned guess), and the observed
  densification shortfall (120k vs 167k Gaussians) post-Tier-1 — is control-node sparsity suppressing
  legitimate static-region densification somehow, or is this unrelated?
- Whether the CUDA-level `atomicAdd` nondeterminism (acknowledged upstream 3DGS limitation) is
  significant enough here to warrant deterministic reduction, given how it confounds every
  before/after architecture comparison in this project.
- The uncommitted CUDA submodule patch is a reproducibility/deployment risk independent of the
  modeling questions above.
