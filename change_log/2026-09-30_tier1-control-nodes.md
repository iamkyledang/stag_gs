# Tier 1: control-node motion (Stage B/C) -- freeze background, promote controls to the live representation

Implements the plan agreed at the end of the previous session, with the design resolved through your
answers to the pre-implementation questions (see below). This is the largest architectural change so
far; **not yet smoke-tested in this session** (you chose to skip the test run) -- please review before
training. Everything under "Verification" describes what I *would* run to validate this; none of it
has been executed yet.

## Decisions locked in before implementation
- **Spatial model**: keep the existing local-linear-fit / MLS machinery (declined the alternative
  Embedded-Deformation-Graph/LBS proposal) -- just fix the neighbourhood it operates over.
- **Temporal visibility**: include now (not deferred) -- a 10th per-anchor channel (opacity logit
  delta), not a separate follow-up.
- **Background freeze**: one-shot, at `dyn_select_iter`, rule `dyn_prob > 0.5 & dyn_w > dyn_w_min`,
  fixed for the rest of training (no periodic re-evaluation).
- **Node budget**: absolute target count (`num_nodes`, default 512), not a fraction of the dynamic set.
- **Stage C rescope**: mask-voting now performs the static/dynamic split, so "object-bounding-radius
  denominator" moves to the (still-existing) storage epsilon-encoding threshold instead.
- **Anchor stride A**: left unchanged this pass, to isolate the architecture change.

## The core idea
Previously *every* dynamic Gaussian carried its own free per-anchor delta (`_dx`/`_drot`/`_dscale`
shaped `[M, Nd, 3]`, Nd up to ~160k), spatially corrected via a linear Jacobian fit over a KNN graph
built across the *entire* point cloud (confirmed contaminated by static-background neighbours at
object boundaries). This is replaced by:

1. **One-shot background freeze** (`LazyMotionModel.freeze_and_select_controls`, called at
   `dyn_select_iter` instead of the old `select_dynamic`): Gaussians the mask-vote splat (Stage A)
   never confidently called dynamic (`get_dyn_prob<=0.5` or `dyn_w<=dyn_w_min`) are dropped from
   `is_dyn` permanently. No more per-Gaussian change-score circularity (the old mechanism needed
   motion to already exist to decide who gets motion).
2. **Control-node selection**, from the survivors, via the existing `_select_controls` voxel search
   now targeting an absolute `num_nodes` (was `control_frac * Nd`). `object_radius` (95th-percentile
   distance from the dynamic set's centroid) is computed once here too -- this is Stage C's
   replacement denominator.
3. **Free parameters collapse to control nodes only**: `self._dx/_drot/_dscale/_dopacity` go from
   `[M, Nd, .]` to `[M, Nctrl, .]` (e.g. potentially ~160k -> 512). Every other dynamic Gaussian's
   delta is *derived*, every query, via `_predict_from_controls` (the local-linear control->non-control
   extrapolation that already existed as a save-time compression trick) -- never independently
   optimised. A "floater" (a Gaussian drifting off on its own into an unsupervised region) is now
   structurally impossible for non-control Gaussians: they can only move as an extrapolation of
   nearby controls.
4. **Every neighbour graph is now restricted to the dynamic object**: `build_knn()` builds over
   `free_idx` (controls post-freeze, all-dynamic pre-freeze) instead of the whole cloud; the
   control<->non-control prediction graph (`_fill_graphs`) is likewise built only over `dyn_idx`. This
   is the fix for the cross-object KNN contamination found while planning Tier 0.
5. **Temporal visibility**: the rate/field vectors grow from 9-D `(v, omega, eta)` to 10-D
   `(v, omega, eta, zeta)` where `zeta` is the opacity-logit rate; `gaussian_renderer.render()` gained
   `d_opacity` applied as `sigmoid(pc._opacity + d_opacity)`.

## Code changes

### `scene/lazy_motion.py` (near-total rewrite)
- New state: `is_ctrl` (`[N]` bool, subset of `is_dyn`), `object_radius`, `_free_xyz` (canonical xyz of
  whatever `free_idx` currently is), `_pred_graph` (the live control<->non-control graph), `_dopacity`
  parameter alongside `_dx/_drot/_dscale`.
- New properties: `ctrl_idx`, `free_idx` (= `ctrl_idx` post-freeze, `dyn_idx` pre-freeze -- this is the
  single switch that makes most of the file phase-agnostic).
- `setup()`: unchanged in spirit (dense per-Gaussian phase from `warm_up`), now also allocates
  `_dopacity`.
- `_setup_optimizer`/`_rebuild_schedule`/`_params`: extended with a `dopacity` group
  (`motion_opacity_lr`, same 10%-floor rescoped schedule as Tier 0's other groups).
- `on_densify`/`on_prune`: now key off `is_ctrl` (when set) instead of `is_dyn`, so children/removals
  of **control** Gaussians correctly grow/shrink the free-parameter tensors; children of non-control
  dynamic or frozen-static Gaussians need no free-parameter bookkeeping (nothing to inherit).
- `build_knn()`: restricted to `free_idx`; post-freeze also builds and caches the non-control
  prediction graph (`_pred_graph`) via the existing `_fill_graphs`.
- `_mls_correct`: simplified -- no more global scatter/gather dance, since the graph is already
  restricted to the free set (`field[nbr]` directly, `nbr` already local indices). Confidence grouping
  extended from 3 to 4 groups (v, omega, eta, opacity).
- `_anchor_deltas`/`_rates`/`_transport`/`query`: extended 9-D -> 10-D throughout for opacity.
- `deltas_at()`: now assembles the final per-Gaussian result from **two** sources -- control results
  straight from `query()`, non-control results derived via `_predict_from_controls` from the live
  `_pred_graph` -- merged by `is_ctrl` mask into the full `dyn_idx`-ordered arrays before scattering
  into the `[N]`-sized renderer-facing tensors. Returns `d_opacity` alongside the existing three.
- `regularization()`: rewritten to operate directly over the free (control) point set (no more
  `full_dx` global-scatter workaround -- unnecessary now that the graph only spans Nctrl points);
  extended with an opacity acceleration term and a new `loss_opacity` (paired with `lambda_opacity`).
- `_score_denominator()`/`change_score()`: denominator is now the single `object_radius` scalar (Stage
  C), not each Gaussian's own ~1e-3-extent scale; `change_score` extended with the opacity component
  (`kappa_opacity`).
- `select_dynamic` **removed**, replaced by `freeze_and_select_controls(tau_w, num_nodes)`.
- `_select_controls`: retargeted from `frac * Nd` to an absolute `target_n`.
- Storage layer (`_encode`/`save`/`_report_storage`/`load`/`_set_sparse`/`_decoded_anchor`/
  `_dense_from_sparse`/`resparsify`) **simplified**: the old two-tier control+non-control-residual
  encoding is gone. Non-control Gaussians no longer have an independent value to residual-correct --
  they were always exactly "predicted from controls," nothing more -- so only control-node records are
  ever stored; non-control values are always re-derived at load time via the same
  `_predict_from_controls` path used live during training. `motion.npz` now stores `ctrl_idx_global`
  (which Gaussians are controls) alongside `dyn_idx`; the old `is_ctrl`/`res_*` fields are gone.

### `gaussian_renderer/__init__.py`
- `render(..., d_opacity=0.0)`: `opacity = torch.sigmoid(pc._opacity + d_opacity)` (was
  `pc.get_opacity`); defaults to the canonical value when `d_opacity` is the default float 0.0.

### `train.py`
- `motion_deltas()` returns `(d_xyz, d_rotation, d_scaling, d_opacity, info)` (5-tuple, was 4).
- `motion.select_dynamic(lazy.eps_dyn)` -> `motion.freeze_and_select_controls(lazy.dyn_w_min, lazy.num_nodes)`.
- Main loop and the validation loop in `training_report` both thread `d_opacity` into their `render()`
  calls.
- `ema_reg` / progress-bar postfix gained an `opacity` entry; postfix also shows live control count
  once `is_ctrl` exists.
- Loss combination gained `+ lazy.lambda_opacity * reg['opacity']`.

### `render.py`
- All 7 `motion.step(...)` call sites (`render_set` x2, `interpolate_time/view/all/poses/view_original`)
  updated to unpack the new 4-tuple and pass `d_opacity` into their `render()` calls.

### `arguments/__init__.py` / `configs.json` (`LazyParams`, `lazy` section)
- Removed: `eps_dyn` (dead -- replaced by mask-driven freeze), `control_frac` (dead -- replaced by
  `num_nodes`).
- Added: `dyn_w_min=10.0` (freeze visibility-confidence floor), `num_nodes=512` (control-node target),
  `kappa_opacity=2.0`, `lambda_opacity=0.1`, `motion_opacity_lr=0.05`.

## Known risks / not yet done
- **Not smoke-tested in this session.** Before training for real, run a tiny forced config exercising:
  warm_up -> a couple of `anchor_refine_iters` (dense phase) -> `dyn_select_iter` (freeze + control
  selection) -> densify/prune of both control and non-control Gaussians post-freeze -> save -> reload
  via `render.py`. This is the single most important next step.
- `dyn_w_min=10.0` default is a guess based on the Stage-A diagnostic's `dyn_w` distribution from an
  *unfrozen* (100%-dynamic) run; may need tuning once real freeze statistics are available.
- `num_nodes=512` is an untuned starting point.
- Anchor stride `A` intentionally left unchanged -- per the agreed plan, revisit once this is validated
  (nodes are far cheaper than per-Gaussian anchors, so a finer stride is now much more affordable).
- The regularizer weights `lambda_rigid`/`lambda_scale` were deliberately left at their old values in
  Tier 0 specifically because their over-strength was diagnosed as tied to this Stage B change (a
  cross-object-contaminated graph, now fixed) and to real temporal-visibility (now added) -- worth
  revisiting now that both are in place, but not changed in this pass to keep it isolated.
- HyperNeRF scenes still have no mask (Stage A's `generate_masks_sam2.py` prerequisite, not run) --
  `freeze_and_select_controls` on an unmasked scene would see `get_dyn_prob` uniformly ~0 everywhere
  (never voted on) and freeze the entire point cloud. Do not run Tier 1 on HyperNeRF until masks exist.
