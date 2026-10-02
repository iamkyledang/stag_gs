# Tier 0: anchor lr rescoping, temporal-regularizer rescaling, AST time-noise

Three isolated bug fixes agreed as "do first, before Stage B" so their effect is attributable
separately from the bigger multi-node architecture change.

## 1. Anchor lr schedule rescoped + floor raised (`scene/lazy_motion.py`)
- Bug: `update_learning_rate(iteration)` fed the *global* iteration into a decay closure built once
  at `motion.setup()` over `[0, opt.iterations]`. Anchors created at `warm_up` (3000) already start
  partway down the curve; anchors added later by `refine_anchors()` (coarse-to-fine, e.g. iters
  5000/7000/9000) inherit whatever lr the *original* schedule says at that iteration, decaying
  further from there -- fine anchors that only exist post-9000 were getting a schedule designed for
  a full 20000-iteration run they never see the start of.
- Fix: new `_rebuild_schedule(ref_iter)` builds the exponential decay over
  `[ref_iter, opt.iterations]` instead, called from `_setup_optimizer` (`ref_iter=opt.warm_up`) and
  from `refine_anchors(iteration)` (new required-by-caller `iteration` param) every time anchors are
  refined. `update_learning_rate` now evaluates the schedule at `iteration - self._lr_ref_iter`.
  `train.py`'s `motion.refine_anchors()` call site updated to `motion.refine_anchors(iteration)`.
- Also raised the decay floor from ~1% of init to 10% (`motion_lr_final`: 0.000008 -> 0.00008 in
  `configs.json` / `arguments/__init__.py`; drot/dscale floor ratio `*0.01` -> `*0.1` in
  `_rebuild_schedule`), since fine anchors only get gradient a few % of iterations (see prior
  diagnosis: `p ~= 2WA/T ~= 4%`) and need headroom left late in training, not a near-zero rate.
- Verified numerically (last refine at 9000, `opt.iterations=20000`): old schedule gives lr=3.2e-5 at
  iter 14000 / 8.0e-6 at iter 20000; new schedule gives 2.8e-4 (~9x) / 8.0e-5 (10x) at the same
  iterations.

## 2. Temporal-acceleration regularizer rescaled (`scene/lazy_motion.py: regularization()`)
- Bug: the translational term of the temporal-smoothness loss normalised second differences of
  `dx` by `_score_denominator(dyn_idx)` -- each Gaussian's *own* scale (~1e-3 of scene extent).
  Dividing a real, extent-scale acceleration by a ~1000x-smaller number inflates the squared loss by
  ~1e6, making `lambda_temporal=0.05` on this term far stronger in practice than its name suggests,
  and pushing anchors toward near-zero acceleration (over-smoothed / frozen motion) regardless of the
  actual scene.
- Fix: normalise by `self.extent` (scene-scale) instead, so the loss is in units of "extent per
  anchor-interval^2", consistent with what the translation term `dx` itself is measured in.
- `lambda_rigid` and `lambda_scale` intentionally left untouched: their over-strength is tied to
  Stage B (per-object KNN graph) and Tier 1 (real temporal-visibility deltas) respectively, per the
  agreed plan -- changing them now would confound Stage B's before/after comparison.

## 3. AST-style time jitter (`train.py`)
- The Deformable-3D-Gaussians MLP baseline perturbs the query time by
  `randn() * (1/num_frames) * smooth_term(iteration)` on non-blender (real-capture) data, annealed
  from ~half a frame early in training to ~0 by the end -- this was identified as present in the MLP
  and absent in stag_gs.
- Added: `time_interval = 1/T` and `ast_smooth = get_linear_noise_func(0.1, 1e-15, lr_delay_mult=0.01,
  max_steps=opt.iterations)` computed once in `training()`; `motion_deltas(motion, fid, t_noise=0.0)`
  gained a `t_noise` param; the main training loop computes
  `ast_noise = randn() * time_interval * ast_smooth(iteration)` (only while `motion.active`) and
  passes it through. `training_report`'s validation calls and `render.py` are untouched (default
  `t_noise=0.0`), matching how the MLP baseline only perturbs time during the training loop, never at
  eval/render time.
- stag_gs's `ModelParams` has no `is_blender` flag (only NeRF-DS/HyperNeRF are in scope for this
  project), so the noise is applied unconditionally rather than gated like the MLP's `is_blender`
  check.

## Verification
- Smoke-tested on `as_novel_view`: 30 iterations, `warm_up=5`, forced `anchor_refine_iters=[10,15,20]`
  (three refine events, exercising `_rebuild_schedule` from `refine_anchors` each time), forced
  densify/prune every 5 iters, forced `dyn_select_iter=22`. Clean run, no exceptions, saved
  successfully.
- lr-schedule fix verified numerically in isolation (see above table).
- Did not re-run a full 20k-iteration comparison in this session (per the established noise-floor
  finding, a single run isn't a reliable signal anyway -- deferred to averaging multiple seeds once
  Stage B also lands, so all of Tier 0 + Stage B can be evaluated together against multiple-seed
  baselines).

## Next
Stage B/C (multi control-nodes per dynamic object using `gaussians.get_dyn_prob`/`dyn_w`, background
freeze, per-object KNN graph -- also fixes the cross-object KNN contamination noted while planning
Tier 0) is still pending, not started in this session.
