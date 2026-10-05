# Stages 2-3: Lie-consistent local twist transport + trained residual blend beta_i

Continues `change_log/2026-10-05_stage1-temporal-basis-fit.md` (4-stage rollout, decisions locked via
`vscode_askQuestions`).

## Stage 2: spatial twist transport (`scene/sparse_anchor_motion.py`)
- Removed the degree-3 cubic-MLS machinery entirely (`_CUBIC_EXPONENTS`, `_cubic_basis`,
  `cubic_mls_fit`, `_cubic_mls_fit_chunk`) -- nothing else referenced it.
- `_mls_correct` replaced by `_twist_correct(field, a_sig)`: each KNN neighbour `j` proposes
  `v_{j->i} = v_j + omega_j x (x_i - x_j)`, `omega_{j->i} = omega_j`, `eta_{j->i} = eta_j` (no
  lever-arm correction on eta, as specified). Applied identically to both the value (`field[:, :9]`)
  and slope/acceleration (`field[:, 9:]`) halves of the 18-dim field -- valid because `x_i, x_j` are
  fixed canonical positions, so differentiating the transport formula w.r.t. time commutes through
  the constant lever arm (`d/dt[v_j + omega_j x (xi-xj)] = dv_j/dt + domega_j/dt x (xi-xj)`).
- Weights `w_ij = w_geo_ij * w_motion_ij`. `w_geo` reuses the existing adaptive kernel
  (`utils/knn_utils.knn_weights`, bandwidth = per-query mean neighbour distance). `w_motion` is new:
  same adaptive-kernel function applied to `||a_i - a_j||` instead of spatial distance, where `a_i` is
  the per-Gaussian motion signature -- **decision: velocity+omega Legendre coefficients only** (not
  eta, not higher per-coefficient-order weighting beyond what's naturally in the fitted `C` tensor),
  `a_sig = C[..., 0:6].permute(1,0,2).reshape(Ndyn, -1)` built in `_tangent` directly from Stage 1's
  basis-fit coefficients (this is the "temporal dictionary gives every Gaussian coefficients a_i"
  connection from the user's notes). No new hyperparameter needed for `sigma_a` since `knn_weights`'s
  adaptive-bandwidth convention is reused as-is.
- Confidence (`conf`, feeds the existing F/B consensus blend in `query()`) is now the weighted spread
  of the *transported proposals* around their aggregate (analogous to the old MLS fit residual, same
  grouping/formula downstream), instead of a polynomial-fit residual.
- Verified: (1) direct unit test of the aggregation formula against a known ground-truth rigid body
  `v(x) = v0 + omega0 x (x-x0)` recovers `v_true` to ~1e-7 (expected -- every neighbour transports to
  the exact same value when they truly share one rigid motion, independent of weights). (2) Invoked
  `_twist_correct` directly with random data: correct output shapes, confidence in a sane (0,1) range,
  gradients reach both `field` and `a_sig`.

## Stage 3: trained residual blend beta_i (`scene/sparse_anchor_motion.py`, `arguments/__init__.py`, `configs.json`)
- New learnable parameter `self._beta` (note: internal attribute is `_beta`, not `_beta_logit` --
  required so it matches the `'_' + group_name` convention `_replace_params` uses to write back
  swapped tensors; see bug below), one scalar per dynamic Gaussian, stored with a dummy leading
  `M=1` axis (shape `[1, Ndyn, 1]`) purely so it rides the *existing* `(M, Ndyn, D)`-shaped
  densify/prune/replace machinery (`_params()`, `on_densify`, `on_prune`) completely unchanged.
  `beta_i = sigmoid(logit_i)`, logit initialised so `beta_i` starts at `motion_beta_init=0.02`
  (near 0: trust each Gaussian's own anchor motion first, per the user's decision), own Adam group
  (`motion_beta_lr=0.005`) with the same rescoped exponential-decay schedule as `dx`/`drot`/`dscale`.
- `_replace_params` changed to skip any param group *absent* from the `new` dict (`if name not in
  new: continue`) instead of requiring every group -- needed because `refine_anchors` only swaps the
  anchor-indexed `dx`/`drot`/`dscale` (beta has no anchor/time axis, nothing to refine).
- Applied in `_tangent`, right after `_twist_correct`:
  `field = field + sigmoid(beta_logit) * (field_local - field)` -- the user's
  `u_final = u_self + beta_i*(u_local - u_self)`, extended to also blend the slope/acceleration half
  with the same per-Gaussian beta (not specified in the user's math, which only covers `u=(v,omega,
  eta)`; chosen for consistency rather than leaving the second-order term unblended).
- Persistence: `capture()`/`restore()` (checkpoint path) carry `_beta` like the other params.
  `save()`/`load()` (sparsified inference checkpoint path) store/restore it as a plain dense array
  (`beta_logit` key in the `.npz`) since it isn't anchor-indexed and doesn't go through the
  change-score sparsification (`_encode`).
- **Bug found + fixed during verification**: initially named the attribute `self._beta_logit` while
  registering it in `_params()` as `('beta', self._beta_logit)`. `_replace_params`/`on_densify`/
  `on_prune`/`freeze_background` all write back via `setattr(self, '_' + name, p)` =
  `setattr(self, '_beta', p)` -- silently creating a *new*, never-read `self._beta` attribute while
  `self._beta_logit` (what `_tangent` actually reads) stayed stale at the pre-freeze/pre-densify/
  pre-prune shape. First smoke test caught this immediately as a shape-mismatch crash in `_tangent`
  right after `freeze_background` (300 vs 152). Fixed by renaming the attribute to `self._beta`
  everywhere (matches the `dx`/`drot`/`dscale` <-> `_dx`/`_drot`/`_dscale` convention already used by
  every other motion parameter). **Lesson**: any new per-Gaussian parameter added to `_params()` must
  use the exact same name for the dict key and the `self._<name>` attribute, or every lifecycle hook
  that relies on the generic `setattr(self, '_' + name, p)` pattern will silently desync.
- Verified end-to-end (fake point cloud, no real Scene/rasterizer): `setup` -> `deltas_at` (several
  `t`) -> `regularization` + `backward` + `optimizer.step()` for 3 iterations -> `freeze_background`
  (beta correctly sliced down from 300 to 152) -> `deltas_at` post-freeze -> `on_densify` (beta grows
  152->159) -> `on_prune` (159->155) -> `save`/`load` round-trip (beta shape/values preserved) ->
  `capture`/`restore` round-trip. All shapes correct, no errors, `sigmoid(beta_logit)` stayed near its
  0.02 init after only a few optimizer steps (expected).

## Remaining stage (not yet implemented)
4. **Dynamic/static gating**: replace the one-shot hard `freeze_background` with a learned
   per-Gaussian logit, trained end-to-end purely by the photometric + regularization loss (no BCE/
   mask supervision), continuously gating `d_xyz`/`d_rotation`/`d_scaling`. Every Gaussian keeps
   motion parameters permanently (no structural pruning of "static" Gaussians anymore). Still open:
   fate of the mask-vote splatting infra (`dyn_fg`/`dyn_w`/`get_dyn_prob`, CUDA `pix_label` patch) and
   how `object_radius` (storage change-score denominator) is defined without a crisp frozen
   dynamic-object subset -- see the Stage 1 changelog for the full note. NOT yet validated with a
   real end-to-end training run on actual data (needs a real Scene/rasterizer) -- recommend a short
   smoke run before trusting quality numbers, per this project's established practice.
