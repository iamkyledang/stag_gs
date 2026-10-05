# Stage 1 (of 4): temporal transport becomes a Legendre-basis fit, not constant acceleration

User-requested rewrite (notes.txt, "5/10/2026" entry). Four related changes were scoped via
`vscode_askQuestions` before implementation; this is Stage 1 of the agreed incremental rollout
(temporal basis -> spatial twist transport -> residual beta blend -> dynamic p_i_dyn gating),
each stage tested in isolation before the next.

## What changed (`scene/sparse_anchor_motion.py`)
- `_wls_line_fit` (2-point weighted line fit: `value(tau) ~= v + a*(tau-t0)`, i.e. "constant
  acceleration" once plugged into `_transport`'s quadratic Taylor step) replaced by
  `_wls_basis_fit(taus, weights, values, t0, order)`: a genuine weighted least-squares solve
  (`torch.linalg.solve` on a tiny `(order+1)x(order+1)` matrix, shared across all Gaussians since the
  sample times/weights are the same for every query point) against a degree-`order` **Legendre**
  basis in the normalized offset `xi=(tau-t0)/s` (`s` = window half-width). Returns the full
  coefficient tensor `C [order+1, Ndyn, 9]` (a real "linear algebra basis" fit, as requested) instead
  of just `(v, a)`.
- New module-level helpers `_legendre_values(x, order)` / `_legendre_derivatives(x, order)` (Bonnet's
  recursion + derivative recursion, plain Python floats since the basis values at the sample/query
  points are shared scalars, not per-Gaussian).
- `_tangent` evaluates the fitted polynomial's **value and derivative at xi=0** (i.e. at `t_star`)
  from `C` to reconstruct the same `(value, slope)` pair `_transport`/`_mls_correct` already expect
  (18-dim field unchanged) -- this stage deliberately keeps the spatial-MLS smoothing and the
  quadratic Taylor transport integrator untouched, so its effect is isolated to "how good is the
  (v,a) estimate at the anchor", not "how is position reconstructed from it" (that's Stage 2+).
  Order degrades gracefully (down to 0 = just the weighted mean) when the window has fewer samples
  than `order+1` (sequence edges) -- same graceful-degradation behavior as the old single-sample
  special case.
- Renamed hyperparameter: `transport_order` (1/2, "velocity only" / "velocity+acceleration") ->
  `temporal_basis_order` (int, default 3 = cubic fit of the rate series). `self.order` ->
  `self.basis_order`. Updated `arguments/__init__.py`, `configs.json`, and the `save()`/`load()`
  checkpoint meta dict (`'order'` key -> `'basis_order'`, restored into `args.temporal_basis_order`).

## Verification
- Numerically confirmed `_wls_basis_fit(..., order=1)` reduces to the old `_wls_line_fit` output
  (`v`/`a` match to ~1e-6) -- order=1 is mathematically the same 2-point line fit, just solved via
  the general Legendre machinery instead of the closed-form formula.
- Confirmed order=3 fit runs, produces `[4, Ndyn, 9]` coefficients, gradients reach the input rate
  tensors (`loss.backward()` on `C.sum()`), and the single-sample edge case still returns a
  `[1, Ndyn, 9]` (order forced down to 0) coefficient tensor.
- NOT yet validated with a real end-to-end training run (needs the full Stage 1-4 rollout or at
  least a standalone smoke test with a real Scene/rasterizer before trusting quality numbers).

## Remaining stages (agreed via vscode_askQuestions on 2026-10-05, not yet implemented)
1. ~~Temporal basis~~ DONE (this entry).
2. **Spatial: Lie-consistent twist transport.** Replace `_mls_correct` (cubic MLS regression of
   absolute rate values) with: each KNN neighbour `j` proposes `v_j + omega_j x (x_i-x_j)`,
   `omega_j`, `eta_j` (no lever-arm correction on eta); aggregate with
   `w_ij = w_geo_ij * w_motion_ij` (`w_geo` already implemented as `knn_utils.knn_weights`;
   `w_motion = exp(-||a_i-a_j||^2 / 2 sigma_a^2)` is new, built from the fitted coefficients --
   **decision: a_i = velocity+omega coefficients only**, not eta, not the full Legendre coefficient
   set).
3. **Residual blend.** `u_i^final = u_i^self + beta_i * (u_i^local - u_i^self)`. Decisions: beta_i is
   a **single scalar per dynamic Gaussian** (not per-channel), parametrized `sigmoid(logit_i)`,
   **initialized near 0** (trust self/independent anchor motion first, neighbour info phases in only
   where it helps the loss), own Adam param group + densify/prune/checkpoint hooks like `_dx` etc.
4. **Dynamic/static gating.** Replace the one-shot hard freeze (`freeze_background` at
   `dyn_select_iter`) with a **learned per-Gaussian logit**, trained end-to-end purely by the
   photometric + regularization loss (no BCE/mask supervision), continuously gating
   `d_xyz/d_rotation/d_scaling` (sigmoid output in [0,1], not a hard threshold). Decision: **every
   Gaussian keeps motion parameters permanently** (no structural pruning of "static" Gaussians'
   anchor params anymore -- simpler, trades away the old memory/compute savings from freezing
   background). Still needs a decision on: fate of the existing mask-vote splatting infra
   (`dyn_fg`/`dyn_w`/`get_dyn_prob`, CUDA `pix_label` patch) now that nothing consumes it for a
   freeze decision, and how `object_radius` (storage change-score denominator) is defined without a
   crisp frozen dynamic-object subset -- to be resolved when Stage 4 is implemented.
