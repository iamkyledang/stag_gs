# Stage 4: learned per-Gaussian dynamic/static gate (replaces one-shot freeze_background)

Completes the 4-stage rollout from `change_log/2026-10-05_stage1-temporal-basis-fit.md` and
`.../2026-10-05_stage2-3-twist-transport-residual-beta.md`.

## What changed (`scene/sparse_anchor_motion.py`, `train.py`, `arguments/__init__.py`, `configs.json`)
- Removed `freeze_background(tau_w)` entirely (the only method that ever shrank `is_dyn`). `is_dyn`
  now stays all-True for the rest of training once `setup()` creates it -- it only ever changes in
  lockstep with the *normal* Gaussian point-cloud lifecycle (`on_densify`/`on_prune`, called from the
  same hooks as `gaussians.on_densify`/`on_prune`), never shrinks on its own. Every Gaussian keeps its
  motion parameters (anchors, beta, gate) permanently, per the user's decision -- this trades away the
  old memory/compute savings from structurally deleting "confidently static" Gaussians' parameters.
- Removed the `train.py` call site (`if iteration == sparse_anchor.dyn_select_iter:
  motion.freeze_background(...)`) and the now-dead `dyn_select_iter`/`dyn_w_min` args
  (`arguments/__init__.py`, `configs.json`).
- New learned parameter `self._gate` (same `[1, N, 1]`-shaped pattern as `self._beta` from Stage 3, so
  it rides the existing `_params()`/`on_densify`/`on_prune`/checkpoint machinery unchanged): `g_i =
  sigmoid(gate_logit_i)`, own Adam group (`motion_gate_lr=0.005`), trained end-to-end **purely by the
  photometric + regularization loss** -- no BCE/mask supervision, per the user's explicit choice.
  Initialised **high** (`motion_gate_init=0.98`), not low like beta: starting near 1 reproduces the
  old pre-freeze behaviour (motion fully applied everywhere) so early training isn't starved of
  gradient for genuinely moving Gaussians through a near-zero multiplicative gate; the gate only
  learns to suppress motion where that actually reduces the loss, rather than needing to first "turn
  on" before anything can be learned.
- Applied once, at the very end of `deltas_at` (the rich anchor/basis-fit/twist-transport/beta-blend
  pipeline inside `query()`/`_tangent()` is completely unaware of the gate):
  - translation: `d_xyz = g_i * x_hat` (plain multiply, `x_hat` is already a translation offset).
  - log-scale: `d_scale = exp(l_can + g_i * l_hat) - exp(l_can)` (plain multiply in log-space).
  - rotation: `q_gated = slerp(identity, q_hat, g_i)` -- **not** a naive multiply (meaningless for
    quaternions). Since `q_hat` is itself the accumulated rotation *delta* (relative to canonical,
    built via `so3_exp` of accumulated rotation vectors upstream), slerping from the identity
    quaternion to `q_hat` by fraction `g_i` is exactly "scale the rotation vector by `g_i`" --
    the geometrically correct analogue of the plain multiplies used for translation/scale. Reuses the
    existing `quat_slerp` (already handles the shortest-arc sign flip).
- `_score_denominator()` (storage change-score denominator) now always falls back to `self.extent`
  (`self.object_radius` is never set anymore, since nothing calls `freeze_background`) -- required no
  code change, this fallback already existed (`max(self.object_radius or self.extent, 1e-6)`).
- `save()`/`load()`/`capture()`/`restore()` updated to persist `_gate` alongside `_beta` (same
  dense-array pattern, not anchor-indexed so not part of the sparsified `_encode`/`_decode` path).
- Updated stale docstrings/comments referencing the old mask-vote-driven hard freeze and the
  since-removed cubic-MLS "same-object" KNN scoping (`build_knn`'s graph is now over the whole point
  cloud; cross-object/background contamination is handled *softly* via Stage 2's `w_motion` instead
  of a hard per-segment graph boundary -- noted explicitly as a deliberate emergent synergy with
  Stage 2, not an oversight).

## Deliberately left unchanged (flagged, not resolved)
- **Mask-vote splatting infra is now vestigial.** `dyn_fg`/`dyn_w`/`get_dyn_prob` (`GaussianModel`),
  the CUDA rasterizer's `pix_label` forward-pass patch (`submodules/depth-diff-gaussian-rasterization`,
  uncommitted working-tree change), and `train.py`'s `gaussians.accumulate_dyn_votes(...)` call still
  run every iteration (so mask datasets are still loaded, the extra rasterizer pass still executes),
  but **nothing consumes the result anymore** (the only consumer, `freeze_background`, is gone).
  Deliberately *not* ripped out this pass: full removal would mean touching the CUDA submodule,
  `render()`'s signature, and `scene/dataset_readers.py`'s mask loading -- a materially riskier,
  unrelated change not required for Stage 4 to work correctly, and not explicitly requested. If the
  user wants to reclaim that compute later, this is a contained, well-scoped follow-up.
- No sparsity-promoting regularizer was added on `g_i` (e.g. an L1 penalty pushing static Gaussians'
  gate toward 0). Not requested, and would be a speculative addition without a demonstrated need --
  the user's selected option was "purely" photometric+regularization-loss-trained. Left as a candidate
  follow-up if floaters/background drift reappear without the old hard freeze's structural guarantee
  (there is no longer any explicit mechanism forcing truly-static Gaussians' contribution to exactly
  zero, only gradient pressure through the existing losses).

## Verification
- End-to-end smoke test (fake point cloud, no real Scene/rasterizer, mirrors the project's standing
  practice for this file): `setup` (confirmed gate initializes at sigmoid=0.98) -> `deltas_at` over 6
  query times with `regularization` + `backward` + `optimizer.step()`/`update_learning_rate` for each
  -> `on_densify` (20 new children) -> `on_prune` (10 removed) -> asserted `is_dyn` stays all-True and
  sized to the current point count (310) after both -> `save` -> `load` (fresh instance, correct gate
  shape) -> `capture`/`restore` (fresh instance via checkpoint path, correct gate shape). All shapes
  correct, no errors.
- NOT yet validated with a real end-to-end training run on actual data (needs a real Scene/
  rasterizer) -- strongly recommended before trusting any quality numbers, per this project's
  established practice (see repo memory: training is not run-to-run reproducible even with identical
  seeds, so compare against the Deformable-MLP baseline with multiple seeds / large consistently-
  signed deltas only).

## All 4 stages: summary of the full diff surface
`scene/sparse_anchor_motion.py` (most of the rewrite), `arguments/__init__.py` (hyperparameter
renames/additions: `transport_order` removed, `temporal_basis_order`/`motion_beta_init`/
`motion_beta_lr`/`motion_gate_init`/`motion_gate_lr` added, `dyn_select_iter`/`dyn_w_min` removed),
`configs.json` (mirrors the above), `train.py` (one call-site removal, no other changes --
`render.py`'s `motion.step()` interface and `gaussian_renderer.render()` were never touched).
