"""Sparse-Anchor Dynamic 3D Gaussians: sparse per-Gaussian anchors + query-time motion interpolation.

Revised design (see change_log/):
  * canonical 3DGS + per-anchor deltas (dx, so(3) log-rotation, dlog-scale) at anchors of stride A, for
    every Gaussian (all Gaussians keep motion parameters permanently now -- see the learned per-Gaussian
    gate below -- no control-node subsampling, no one-shot background freeze).
  * query t -> window of W anchor intervals -> motion rates u in R^9 -> per-Gaussian Legendre-basis
    fit of u(tau) (replaces the old 2-point "constant acceleration" line fit, see _wls_basis_fit) ->
    Lie-consistent local twist transport over the K=48 nearest neighbours (motion-aware KNN weights,
    see _twist_correct) -> trained residual blend u_self + beta_i*(u_local-u_self) -> bidirectional
    (forward/backward) Taylor transport -> confidence-weighted consensus.
  * a learned per-Gaussian gate g_i = sigmoid(gate_logit_i), trained end-to-end purely by the
    photometric + regularization loss (no mask/BCE supervision), continuously scales how much motion
    is applied (see deltas_at) -- replaces the old one-shot `freeze_background` hard dynamic/static
    split; the dynamic/static boundary can now move in either direction throughout training.
  * opacity is never deformed: only (d_xyz, d_rotation, d_scaling) are produced.

Training is done *through* the query operator (gradients reach every anchor in the window), so the
anchors are optimised for what the interpolant renders, not for what they store.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from utils.general_utils import get_expon_lr_func
from utils.knn_utils import knn_graph, knn_weights
from utils.lie_utils import quat_conj, quat_mul, quat_normalize, quat_rotate, quat_slerp, so3_exp, so3_log
from utils.system_utils import searchForMaxIteration


def _legendre_values(x, order):
    """[P_0(x), ..., P_order(x)] (physicists' Legendre polynomials) via Bonnet's recursion. x: float."""
    P = [1.0, x]
    for n in range(1, order):
        P.append(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1))
    return P[:order + 1]


def _legendre_derivatives(x, order):
    """[P_0'(x), ..., P_order'(x)] via P'_{n+1} = (n+1) P_n + x P'_n."""
    P = _legendre_values(x, order)
    Pp = [0.0, 1.0]
    for n in range(1, order):
        Pp.append((n + 1) * P[n] + x * Pp[n])
    return Pp[:order + 1]


def _wls_basis_fit(taus, weights, values, t0, order):
    """Weighted least-squares fit of the rate series `values(tau)` (python list, len n) against a
    degree-`order` Legendre basis in the normalized offset xi = (tau - t0) / s (s = window half-width),
    replacing the old 2-point (v, a) "constant acceleration" line fit with a genuine small linear-
    algebra solve for `order + 1` per-Gaussian coefficients -- a basis/tensor fit instead of a fixed
    functional-form assumption. Falls back to a lower order (down to 0, i.e. just the mean) when fewer
    than order + 1 samples are available (window edges). Returns (coeffs [order+1, ..., D], s).
    """
    n = len(taus)
    order = max(min(order, n - 1), 0)
    if order == 0:
        return values[0].unsqueeze(0), 1.0
    s = max(max(abs(tt - t0) for tt in taus), 1e-6)
    xi = [(tt - t0) / s for tt in taus]
    Phi = torch.tensor([_legendre_values(x, order) for x in xi], dtype=values[0].dtype, device=values[0].device)
    w = torch.tensor(weights, dtype=values[0].dtype, device=values[0].device)
    wPhi = w.unsqueeze(-1) * Phi                                    # [n, order+1]
    M = wPhi.t() @ Phi                                              # [order+1, order+1]
    M = M + 1e-6 * torch.eye(order + 1, device=M.device, dtype=M.dtype)
    V = torch.stack(values, dim=0)                                  # [n, ..., D]
    rhs = wPhi.t() @ V.reshape(n, -1)                               # [order+1, prod(...)*D]
    C = torch.linalg.solve(M, rhs).reshape((order + 1,) + V.shape[1:])
    return C, s


class SparseAnchorMotionModel:
    """Per-Gaussian sparse anchor storage + query-time reconstruction of (x, r, s) for every
    Gaussian. Every Gaussian keeps its own independently-optimised anchor deltas permanently (no
    control-node subsampling / prediction tier, no one-shot background freeze); a learned per-Gaussian
    gate (see deltas_at) continuously controls how much of the reconstructed motion is actually
    applied, so the dynamic/static boundary is trained rather than hard-coded."""

    def __init__(self, args, num_train_frames, scene_extent):
        self.args = args
        self.A = int(args.anchor_stride)
        self.W = int(args.temporal_window)
        self.K = int(args.knn_k)
        self.basis_order = int(args.temporal_basis_order)
        self.sigma_c = float(args.consensus_sigma)
        self.use_confidence = not args.no_fit_confidence
        self.T = max(int(num_train_frames), 2)
        self.extent = float(scene_extent)
        self.level = int(args.coarse_levels)
        self.kappa_rot = float(args.score_kappa_rot)
        self.kappa_scale = float(args.score_kappa_scale)
        self.wls_eps = float(args.wls_eps)

        self.gaussians = None
        self.anchor_times = None      # [M] float tensor (cuda)
        self._dx = self._drot = self._dscale = None   # [M, Ndyn, .]
        self.is_dyn = None             # [N] bool: always all-True now (kept in lockstep with normal
                                        # Gaussian densify/prune); dynamic/static is the learned gate now
        self.object_radius = None      # unused (no more one-shot freeze); _score_denominator falls back to extent
        self._dyn_xyz = None           # canonical xyz of the dynamic (anchor-carrying) points [Ndyn,3]
        self.nbr_idx = self.nbr_w = None
        self.knn_dirty = True
        self.optimizer = None
        self.training = True
        self.sparse = None            # decoded lazily at inference
        self._anchor_cache = {}
        self._rate_cache = {}
        self._tangent_cache = {}

    # ------------------------------------------------------------------ anchors / bookkeeping
    @property
    def active(self):
        return self.anchor_times is not None

    @property
    def num_anchors(self):
        return 0 if self.anchor_times is None else int(self.anchor_times.numel())

    @property
    def dyn_idx(self):
        return torch.nonzero(self.is_dyn, as_tuple=False).squeeze(1)

    def _anchor_times_for_level(self, level):
        stride = self.A * (2 ** level)
        M = int(math.ceil((self.T - 1) / float(stride))) + 1
        return torch.linspace(0.0, 1.0, max(M, 2), device='cuda')

    def setup(self, gaussians, opt):
        """Called once at the end of the static warm-up: every Gaussian starts dynamic with zero deltas.
        Dynamic/static is now a continuously trained per-Gaussian gate (self._gate, see deltas_at), not
        a one-shot freeze -- every Gaussian keeps its motion parameters for the rest of training."""
        self.gaussians = gaussians
        N = gaussians.get_xyz.shape[0]
        self.anchor_times = self._anchor_times_for_level(self.level)
        M = self.num_anchors
        self.is_dyn = torch.ones(N, dtype=torch.bool, device='cuda')
        self._dx = nn.Parameter(torch.zeros(M, N, 3, device='cuda'))
        self._drot = nn.Parameter(torch.zeros(M, N, 3, device='cuda'))
        self._dscale = nn.Parameter(torch.zeros(M, N, 3, device='cuda'))
        # residual-blend weight beta_i = sigmoid(logit_i), one scalar per Gaussian (not per-anchor --
        # stored with a dummy leading "M=1" axis purely so it can ride the existing (M,Ndyn,D)-shaped
        # densify/prune/replace machinery in _params()/on_densify/on_prune unchanged). Initialised near
        # 0 (trust each Gaussian's own anchor motion first; neighbour twist-transport info phases in
        # only where it helps the loss).
        beta0 = float(self.args.motion_beta_init)
        beta_logit0 = math.log(beta0 / (1.0 - beta0))
        self._beta = nn.Parameter(torch.full((1, N, 1), beta_logit0, device='cuda'))
        # dynamic/static gate g_i = sigmoid(logit_i), trained end-to-end purely by the photometric +
        # regularization loss (no mask/BCE supervision) -- replaces the one-shot freeze_background split.
        # Initialised high (motion fully applied by default, matching the old pre-freeze behaviour) so
        # early training isn't starved of gradient for genuinely moving Gaussians; the gate only learns
        # to suppress motion where that actually reduces the loss.
        gate0 = float(self.args.motion_gate_init)
        gate_logit0 = math.log(gate0 / (1.0 - gate0))
        self._gate = nn.Parameter(torch.full((1, N, 1), gate_logit0, device='cuda'))
        self.knn_dirty = True
        self._setup_optimizer(opt)
        print("[SparseAnchor] motion enabled: {} anchors (stride {}), {} Gaussians, W={}, K={}".format(
            M, self.A * 2 ** self.level, N, self.W, self.K))

    def capture(self):
        """Full training-state snapshot (dense anchor params + Adam moments + freeze state), for
        train.py --checkpoint_iterations/--start_checkpoint. Unlike save()/load() (lossy, sparsified,
        inference-only), this preserves the live dense tensors bit-for-bit, so it works correctly at
        any iteration motion is active, frozen or not (is_dyn may still be all-True pre-freeze)."""
        return {
            'level': self.level, 'anchor_times': self.anchor_times, 'is_dyn': self.is_dyn,
            'object_radius': self.object_radius, 'dx': self._dx, 'drot': self._drot, 'dscale': self._dscale,
            'beta_logit': self._beta, 'gate_logit': self._gate,
            'optimizer': self.optimizer.state_dict(), 'lr_ref_iter': self._lr_ref_iter,
        }

    def restore(self, gaussians, state, opt):
        self.gaussians = gaussians
        self.level = state['level']
        self.anchor_times = state['anchor_times']
        self.is_dyn = state['is_dyn']
        self.object_radius = state['object_radius']
        self._dx = nn.Parameter(state['dx'])
        self._drot = nn.Parameter(state['drot'])
        self._dscale = nn.Parameter(state['dscale'])
        self._beta = nn.Parameter(state['beta_logit'])
        self._gate = nn.Parameter(state['gate_logit'])
        self.knn_dirty = True
        self.training = True
        self._setup_optimizer(opt)
        self._rebuild_schedule(state['lr_ref_iter'])
        self.optimizer.load_state_dict(state['optimizer'])

    def _setup_optimizer(self, opt):
        a = self.args
        groups = [
            {'params': [self._dx], 'lr': a.motion_lr_init, 'name': 'dx'},
            {'params': [self._drot], 'lr': a.motion_rot_lr, 'name': 'drot'},
            {'params': [self._dscale], 'lr': a.motion_scale_lr, 'name': 'dscale'},
            {'params': [self._beta], 'lr': a.motion_beta_lr, 'name': 'beta'},
            {'params': [self._gate], 'lr': a.motion_gate_lr, 'name': 'gate'},
        ]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self._opt = opt
        self._rebuild_schedule(opt.warm_up)

    def _rebuild_schedule(self, ref_iter):
        """(Re)build the anchor lr decay to span [ref_iter, opt.iterations] rather than [0, opt.iterations]:
        anchors created at warm_up or refined mid-training (coarse-to-fine) would otherwise inherit an
        already-decayed rate and stop learning long before the run ends. Called from _setup_optimizer and
        refine_anchors. Floor is 10% of init (was ~1%) since fine anchors only get gradient a few % of
        iterations and need headroom left late in training."""
        a = self.args
        self._lr_ref_iter = ref_iter
        max_steps = max(self._opt.iterations - ref_iter, 1)
        self._sched = {
            'dx': get_expon_lr_func(a.motion_lr_init, a.motion_lr_init * 0.1, max_steps=max_steps),
            'drot': get_expon_lr_func(a.motion_rot_lr, a.motion_rot_lr * 0.1, max_steps=max_steps),
            'dscale': get_expon_lr_func(a.motion_scale_lr, a.motion_scale_lr * 0.1, max_steps=max_steps),
            'beta': get_expon_lr_func(a.motion_beta_lr, a.motion_beta_lr * 0.1, max_steps=max_steps),
            'gate': get_expon_lr_func(a.motion_gate_lr, a.motion_gate_lr * 0.1, max_steps=max_steps),
        }

    def update_learning_rate(self, iteration):
        if self.optimizer is None:
            return
        t = max(iteration - self._lr_ref_iter, 0)
        for g in self.optimizer.param_groups:
            g['lr'] = self._sched[g['name']](t)

    def _params(self):
        return (('dx', self._dx), ('drot', self._drot), ('dscale', self._dscale),
                ('beta', self._beta), ('gate', self._gate))

    def _replace_params(self, new, keep_state_fn=None):
        """Swap parameter tensors (new: dict name->tensor; groups absent from `new` are left untouched --
        e.g. refine_anchors only swaps the anchor-indexed dx/drot/dscale, not the per-Gaussian beta).
        Optimizer moments are rebuilt by keep_state_fn(name, exp_avg, exp_avg_sq)->(exp_avg, exp_avg_sq)
        or reset to zero."""
        for g in self.optimizer.param_groups:
            name = g['name']
            if name not in new:
                continue
            old = g['params'][0]
            state = self.optimizer.state.pop(old, None)
            p = nn.Parameter(new[name].detach().contiguous().requires_grad_(True))
            g['params'][0] = p
            if state is not None and keep_state_fn is not None:
                ea, es = keep_state_fn(name, state['exp_avg'], state['exp_avg_sq'])
                state['exp_avg'], state['exp_avg_sq'] = ea, es
                self.optimizer.state[p] = state
            setattr(self, '_' + name, p)
        self._clear_caches()

    def _clear_caches(self):
        self._anchor_cache.clear()
        self._rate_cache.clear()
        self._tangent_cache.clear()

    # ------------------------------------------------------------------ densification hooks
    def on_densify(self, parent_idx):
        """Children inherit their parent's dynamic status and (if dynamic) its anchor deltas."""
        if self.is_dyn is None:
            return
        parent_idx = parent_idx.to(self.is_dyn.device)
        child_dyn = self.is_dyn[parent_idx]
        g2l = torch.cumsum(self.is_dyn.long(), 0) - 1
        loc = g2l[parent_idx[child_dyn]]
        new = {name: torch.cat((p.data, p.data[:, loc]), dim=1) for name, p in self._params()}
        n_new = loc.numel()
        self._replace_params(new, lambda n, ea, es: (
            torch.cat((ea, ea.new_zeros(ea.shape[0], n_new, ea.shape[2])), 1),
            torch.cat((es, es.new_zeros(es.shape[0], n_new, es.shape[2])), 1)))
        self.is_dyn = torch.cat((self.is_dyn, child_dyn))
        self.knn_dirty = True

    def on_prune(self, remove_mask):
        if self.is_dyn is None:
            return
        keep = ~remove_mask.to(self.is_dyn.device)
        keep_local = keep[self.is_dyn]
        new = {name: p.data[:, keep_local] for name, p in self._params()}
        self._replace_params(new, lambda n, ea, es: (ea[:, keep_local], es[:, keep_local]))
        self.is_dyn = self.is_dyn[keep]
        self.knn_dirty = True

    # ------------------------------------------------------------------ KNN graph (K=48, whole point cloud)
    def build_knn(self, force=False):
        if not (self.knn_dirty or force):
            return
        dyn_idx = self.dyn_idx
        self._dyn_xyz = self.gaussians.get_xyz.detach()[dyn_idx]
        # self-graph over every Gaussian (no more segmentation-scoped subset): cross-object/background
        # contamination is now handled softly by _twist_correct's motion-aware weight w_motion instead
        # of a hard per-object graph boundary (two spatially-close but motion-different Gaussians --
        # e.g. a moving object next to static background -- get a weak edge via w_motion, not zero).
        k = min(self.K, self._dyn_xyz.shape[0] - 1)
        self.nbr_idx, dist = knn_graph(self._dyn_xyz, k)
        self.nbr_w = knn_weights(dist)
        self.knn_dirty = False
        self._tangent_cache.clear()

    # ------------------------------------------------------------------ anchor states (Sec. 4)
    def _anchor_deltas(self, ids):
        """ids: python list of anchor indices -> (dx [n,Ndyn,3], q [n,Ndyn,4], dl [n,Ndyn,3])."""
        if self._dx is not None:
            idx = torch.tensor(ids, device='cuda', dtype=torch.long)
            dx, drot, dl = self._dx[idx], self._drot[idx], self._dscale[idx]
        else:
            outs = [self._decoded_anchor(a) for a in ids]
            dx = torch.stack([o[0] for o in outs]); drot = torch.stack([o[1] for o in outs])
            dl = torch.stack([o[2] for o in outs])
        return dx, so3_exp(drot), dl

    def _rates(self, k, dx, q, dl, dt):
        """Motion rate u_k in R^9 (v, omega, eta) between consecutive anchors."""
        if not self.training and k in self._rate_cache:
            return self._rate_cache[k]
        v = (dx[1] - dx[0]) / dt
        om = so3_log(quat_mul(q[1], quat_conj(q[0]))) / dt
        eta = (dl[1] - dl[0]) / dt
        u = torch.cat((v, om, eta), dim=-1)
        if not self.training:
            self._rate_cache[k] = u
        return u


    # ------------------------------------------------------------------ query operator (Sec. 3-8)
    def _locate(self, t):
        times = self.anchor_times
        M = times.numel()
        m = int(torch.searchsorted(times, torch.tensor([t], device=times.device), right=True).item()) - 1
        m = max(0, min(m, M - 2))
        a0, a1 = times[m].item(), times[m + 1].item()
        s = 0.0 if a1 <= a0 else min(max((t - a0) / (a1 - a0), 0.0), 1.0)
        return m, s, a0, a1

    def _tangent(self, side, m):
        """Velocity/acceleration field at anchor a_m (side 'F') or a_{m+1} (side 'B') from a Legendre-basis
        fit (order self.basis_order) of the W nearest interval rates on that side, then Lie-consistent
        local twist transport over the dynamic object's own KNN graph (K=48). Returns (field [Ndyn,18],
        confidence [Ndyn,1])."""
        key = (side, m)
        if not self.training and key in self._tangent_cache:
            return self._tangent_cache[key]
        M = self.num_anchors
        times = self.anchor_times
        if side == 'F':
            ks = [k for k in range(m - self.W + 1, m + 1) if 0 <= k <= M - 2]
            t_star = times[m].item()
        else:
            ks = [k for k in range(m, m + self.W) if 0 <= k <= M - 2]
            t_star = times[m + 1].item()
        anchors = list(range(ks[0], ks[-1] + 2))
        dx, q, dl = self._anchor_deltas(anchors)
        us, taus, ws = [], [], []
        for k in ks:
            i = k - anchors[0]
            dt = (times[k + 1] - times[k]).item()
            us.append(self._rates(k, dx[i:i + 2], q[i:i + 2], dl[i:i + 2], dt))
            taus.append(0.5 * (times[k] + times[k + 1]).item())
            ws.append(1.0 / (1.0 + abs(k - m)))
        # Legendre-basis fit of u(tau) (order up to self.basis_order, degraded near window edges where
        # fewer samples are available) -- replaces the old 2-point (v, a) constant-acceleration line fit.
        # The fitted coefficients are then evaluated (value + derivative) at tau=t_star for the existing
        # Taylor-transport step below, so this is a drop-in quality upgrade of that estimate.
        C, s = _wls_basis_fit(taus, ws, us, t_star, self.basis_order)
        order = C.shape[0] - 1
        P0, Pp0 = _legendre_values(0.0, order), _legendre_derivatives(0.0, order)
        v = sum(P0[p] * C[p] for p in range(order + 1))
        a = sum(Pp0[p] * C[p] for p in range(order + 1)) / s
        field = torch.cat((v, a), dim=-1)                                     # [Ndyn, 18]
        conf = torch.ones(field.shape[0], 1, device=field.device)
        if self.nbr_idx is not None:
            # motion signature a_i for the motion-aware KNN weight: velocity+omega Legendre coefficients
            # only (decision: eta/scale-rate and higher-channel info excluded from the identity signature).
            a_sig = C[..., 0:6].permute(1, 0, 2).reshape(C.shape[1], -1)
            field_local, conf = self._twist_correct(field, a_sig)
            # residual blend: u_final = u_self + beta_i * (u_local - u_self), beta_i = sigmoid(logit_i)
            # trained per-Gaussian (see setup()); applied to both the value and slope/acceleration
            # halves of `field` with the same beta_i (one trust level per Gaussian).
            beta = torch.sigmoid(self._beta[0])                               # [Ndyn, 1]
            field = field + beta * (field_local - field)
        out = (field, conf)
        if not self.training:
            self._tangent_cache[key] = out
        return out

    def _twist_correct(self, field, a_sig):
        """Lie-consistent local twist transport (replaces the old cubic-MLS spatial smoothing). Each KNN
        neighbour j proposes a transported motion at i: v_{j->i} = v_j + omega_j x (x_i - x_j),
        omega_{j->i} = omega_j, eta_{j->i} = eta_j -- the same formula is applied to both the value and
        the slope/acceleration halves of `field` (valid since x_i, x_j are fixed canonical positions, so
        differentiating the transport formula w.r.t. time commutes through the constant lever arm).
        Proposals are aggregated with w_ij = w_geo_ij * w_motion_ij: w_geo is the existing adaptive
        spatial kernel (`knn_weights`, same bandwidth convention as before); w_motion is a matching
        adaptive kernel over the per-Gaussian temporal-basis coefficients `a_sig` (velocity+omega only),
        so two spatially-close Gaussians whose fitted trajectories disagree get a weak edge."""
        xyz = self._dyn_xyz
        nbr = self.nbr_idx                                                    # [Ndyn,K] local indices
        lever = xyz.unsqueeze(1) - xyz[nbr]                                   # [Ndyn,K,3] (x_i - x_j)
        w_geo = self.nbr_w
        a_dist = (a_sig[nbr] - a_sig.unsqueeze(1)).norm(dim=-1)
        w_motion = knn_weights(a_dist)
        w = w_geo * w_motion
        wsum = w.sum(1, keepdim=True).clamp_min(1e-8)

        field_nbr = field[nbr]                                                # [Ndyn,K,18]
        halves = []
        value_prop = None
        for h0 in (0, 9):
            v_j, om_j, eta_j = field_nbr[..., h0:h0 + 3], field_nbr[..., h0 + 3:h0 + 6], field_nbr[..., h0 + 6:h0 + 9]
            v_t = v_j + torch.linalg.cross(om_j, lever, dim=-1)
            prop = torch.cat((v_t, om_j, eta_j), dim=-1)                      # [Ndyn,K,9]
            if h0 == 0:
                value_prop = prop
            halves.append((w.unsqueeze(-1) * prop).sum(1) / wsum)
        field = torch.cat(halves, dim=-1)                                     # [Ndyn,18]

        conf = torch.ones(field.shape[0], 1, device=field.device)
        if self.use_confidence:
            res = value_prop - halves[0].unsqueeze(1)                        # disagreement of proposals
            msr = (w.unsqueeze(-1) * res.detach() ** 2).sum(1) / wsum         # [Ndyn,9]
            r = 0.0
            groups = ((0, 3), (3, 6), (6, 9))   # v, omega, eta
            for g0, g1 in groups:
                grp = msr[:, g0:g1].sum(-1)
                r = r + grp / (grp.mean() + 1e-12)
            conf = (1.0 / (1.0 + r / (len(groups) * self.sigma_c))).unsqueeze(-1)
        return field, conf

    @staticmethod
    def _transport(dx0, q0, dl0, field, dt):
        """Taylor transport of duration dt (signed) with velocity/acceleration field [Ndyn,18] (Sec. 7)."""
        step = field[:, :9] * dt + 0.5 * field[:, 9:] * dt * dt
        x = dx0 + step[:, 0:3]
        q = quat_mul(so3_exp(step[:, 3:6]), q0)
        l = dl0 + step[:, 6:9]
        return x, q, l

    def query(self, t):
        """Reconstruct (dx_hat, q_hat, dl_hat) for the dynamic point set at normalized time t (Sec. 7-8)."""
        self.build_knn()
        t = float(min(max(t, 0.0), 1.0))
        m, s, a0, a1 = self._locate(t)
        dx, q, dl = self._anchor_deltas([m, m + 1])
        fF, cF = self._tangent('F', m)
        fB, cB = self._tangent('B', m)
        xF, qF, lF = self._transport(dx[0], q[0], dl[0], fF, (t - a0))
        xB, qB, lB = self._transport(dx[1], q[1], dl[1], fB, (t - a1))
        tau = 1.0 - s
        lam = tau * cF / (tau * cF + (1.0 - tau) * cB).clamp_min(1e-12)      # [Ndyn,1]
        x_hat = lam * xF + (1.0 - lam) * xB
        q_hat = quat_slerp(qB, qF, lam)
        l_hat = lam * lF + (1.0 - lam) * lB
        info = {'m': m, 's': s, 'anchors': list(range(max(0, m - self.W + 1), min(self.num_anchors, m + self.W + 1)))}
        return x_hat, q_hat, l_hat, info

    def deltas_at(self, t):
        """Deltas for all N Gaussians in the additive convention expected by gaussian_renderer.render:
        (d_xyz, d_rotation, d_scaling, info). Opacity is never deformed. A learned per-Gaussian gate
        g_i = sigmoid(gate_logit_i) (see setup()) continuously scales how much of the reconstructed
        motion is actually applied -- rotation is gated by slerping from identity to q_hat by g_i
        (the geometrically correct analogue of scaling a rotation vector), translation/log-scale by a
        plain multiply."""
        gs = self.gaussians
        N = gs.get_xyz.shape[0]
        x_hat, q_hat, l_hat, info = self.query(t)
        dyn_idx = self.dyn_idx
        r_c = quat_normalize(gs._rotation[dyn_idx])
        l_can = gs._scaling[dyn_idx]
        g = torch.sigmoid(self._gate[0])                                     # [Ndyn, 1]
        id_q = torch.zeros_like(q_hat)
        id_q[:, 0] = 1.0
        q_gated = quat_slerp(id_q, q_hat, g)
        d_xyz = torch.zeros(N, 3, device='cuda').index_put((dyn_idx,), g * x_hat)
        d_rot = torch.zeros(N, 4, device='cuda').index_put((dyn_idx,), quat_mul(q_gated, r_c) - r_c)
        d_scale = torch.zeros(N, 3, device='cuda').index_put((dyn_idx,), torch.exp(l_can + g * l_hat) - torch.exp(l_can))
        return d_xyz, d_rot, d_scale, info

    def step(self, xyz, time_input):
        """Per-frame interface used by render.py: (d_xyz, d_rotation, d_scaling)."""
        t = float(time_input.reshape(-1)[0].item()) if torch.is_tensor(time_input) else float(time_input)
        d_xyz, d_rot, d_scale, _ = self.deltas_at(t)
        return d_xyz, d_rot, d_scale

    # ------------------------------------------------------------------ regularisation
    def regularization(self, info, max_points=120000):
        """ARAP local rigidity (bracketing anchors), temporal acceleration (window +-1), scale L2 --
        all over the dynamic point set only."""
        xyz = self._dyn_xyz
        Nf = xyz.shape[0]
        M = self.num_anchors
        anchors = info['anchors']
        aid = torch.tensor([info['m'], info['m'] + 1], device='cuda', dtype=torch.long)
        dx, drot = self._dx[aid], self._drot[aid]                              # [2,Nf,3]

        # --- local rigidity (canonical neighbourhood rotated by q_i must match deformed neighbourhood)
        sub = torch.arange(Nf, device='cuda')
        dx_i, drot_i = dx, drot
        if Nf > max_points:
            sub = torch.randperm(Nf, device='cuda')[:max_points]
            dx_i, drot_i = dx[:, sub], drot[:, sub]
        nbr = self.nbr_idx[sub]                                               # [S,K] local indices into xyz/dx
        w = self.nbr_w[sub]
        rel_c = xyz[nbr] - xyz[sub].unsqueeze(1)                              # [S,K,3]
        dx_j = dx[:, nbr]                                                     # [2,S,K,3]
        rel_a = rel_c.unsqueeze(0) + dx_j - dx_i.unsqueeze(2)
        q = so3_exp(drot_i)                                                   # [2,S,4]
        rot_rel = quat_rotate(q.unsqueeze(2), rel_c.unsqueeze(0))
        num = (w.unsqueeze(0).unsqueeze(-1) * (rot_rel - rel_a) ** 2).sum((-1, -2))
        den = (w * (rel_c ** 2).sum(-1)).sum(-1).unsqueeze(0) + 1e-12
        loss_rigid = (num / den).mean()

        # --- temporal acceleration (second differences across consecutive anchors, scene-scale normalised:
        # a Gaussian's own size is ~1e-3 of extent, which turned a real acceleration into a ~100x-inflated
        # loss and effectively froze fine anchors; extent gives units consistent with actual motion magnitude)
        lo, hi = max(0, anchors[0] - 1), min(M - 1, anchors[-1] + 1)
        if hi - lo >= 2:
            ids = torch.arange(lo, hi + 1, device='cuda')
            denom = max(self.extent, 1e-6)
            acc_x = (self._dx[ids][2:] - 2 * self._dx[ids][1:-1] + self._dx[ids][:-2]) / denom
            acc_r = (self._drot[ids][2:] - 2 * self._drot[ids][1:-1] + self._drot[ids][:-2]) / self.kappa_rot
            acc_s = (self._dscale[ids][2:] - 2 * self._dscale[ids][1:-1] + self._dscale[ids][:-2]) / self.kappa_scale
            loss_temporal = (acc_x ** 2).sum(-1).mean() + (acc_r ** 2).sum(-1).mean() + (acc_s ** 2).sum(-1).mean()
        else:
            loss_temporal = torch.zeros((), device='cuda')

        aidx = torch.tensor(anchors, device='cuda', dtype=torch.long)
        loss_scale = (self._dscale[aidx] ** 2).mean()
        return {'rigid': loss_rigid, 'temporal': loss_temporal, 'scale': loss_scale}

    # ------------------------------------------------------------------ change score (storage sparsification)
    def _score_denominator(self):
        """Denominator for the storage change-score. `object_radius` is never set anymore (no more
        one-shot background freeze, see setup()/deltas_at()'s learned gate instead), so this always
        falls back to the scene-level `extent` -- every Gaussian is a candidate dynamic Gaussian now,
        there is no single frozen "dynamic object" subset to compute a tighter bounding radius from."""
        return max(self.object_radius or self.extent, 1e-6)

    def change_score(self, dx, drot, dl, denom):
        """Normalised change score d: translation relative to the object's bounding radius, rotation and
        log-scale relative to their kappa units. Shapes [..., N, D_component]."""
        d_x = dx.norm(dim=-1) / denom
        d_r = drot.norm(dim=-1) / self.kappa_rot
        d_s = dl.abs().amax(dim=-1) / self.kappa_scale
        return torch.max(torch.max(d_x, d_r), d_s)

    # ------------------------------------------------------------------ coarse-to-fine anchors
    @torch.no_grad()
    def refine_anchors(self, iteration=None):
        if self.level <= 0:
            return
        self.level -= 1
        old_t = self.anchor_times
        new_t = self._anchor_times_for_level(self.level)
        M_old = old_t.numel()
        m = torch.clamp(torch.searchsorted(old_t, new_t, right=True) - 1, 0, M_old - 2)
        dt = (old_t[m + 1] - old_t[m]).clamp_min(1e-12)
        s = ((new_t - old_t[m]) / dt).clamp(0, 1).view(-1, 1, 1)
        q0, q1 = so3_exp(self._drot.data[m]), so3_exp(self._drot.data[m + 1])
        new = {
            'dx': (1 - s) * self._dx.data[m] + s * self._dx.data[m + 1],
            'drot': so3_log(quat_slerp(q0, q1, s)),
            'dscale': (1 - s) * self._dscale.data[m] + s * self._dscale.data[m + 1],
        }
        self.anchor_times = new_t
        self._replace_params(new)   # fresh Adam moments: shape changed along the anchor axis
        if iteration is not None:
            self._rebuild_schedule(iteration)
        print("[SparseAnchor] anchors refined: {} -> {} (stride {})".format(M_old, new_t.numel(), self.A * 2 ** self.level))

    # ------------------------------------------------------------------ sparse storage
    @torch.no_grad()
    def _encode(self, dense, eps):
        """dense [M,Ndyn,9] -> sparse per-dynamic-Gaussian records (missing == exactly zero)."""
        M, Nd, _ = dense.shape
        denom = self._score_denominator()
        rec_idx, rec_val, rec_cnt = [], [], []
        for a in range(M):
            D = dense[a]
            keep = self.change_score(D[:, 0:3], D[:, 3:6], D[:, 6:9], denom) > eps
            ic = torch.nonzero(keep).squeeze(1)
            rec_idx.append(ic.int().cpu()); rec_val.append(D[ic].float().cpu()); rec_cnt.append(ic.numel())
        return {
            'rec_counts': np.array(rec_cnt, dtype=np.int32), 'rec_idx': torch.cat(rec_idx).numpy(),
            'rec_val': torch.cat(rec_val).numpy(),
        }

    def _dense_from_params(self):
        return torch.cat((self._dx.data, self._drot.data, self._dscale.data), dim=-1)

    @torch.no_grad()
    def save(self, model_path, iteration, eps=None):
        eps = self.args.epsilon if eps is None else eps
        out_dir = os.path.join(model_path, "motion", "iteration_{}".format(iteration))
        os.makedirs(out_dir, exist_ok=True)
        dense = self._dense_from_params() if self._dx is not None else self._dense_from_sparse()
        S = self._encode(dense, eps)
        meta = {
            'A': self.A, 'W': self.W, 'K': self.K, 'basis_order': self.basis_order, 'sigma_c': self.sigma_c,
            'use_confidence': self.use_confidence, 'epsilon': eps,
            'kappa_rot': self.kappa_rot, 'kappa_scale': self.kappa_scale,
            'wls_eps': self.wls_eps, 'extent': self.extent, 'object_radius': self.object_radius,
            'T': self.T, 'N': int(self.is_dyn.shape[0]), 'level': self.level,
        }
        np.savez(os.path.join(out_dir, "motion.npz"),
                 anchor_times=self.anchor_times.cpu().numpy().astype(np.float32),
                 dyn_idx=self.dyn_idx.cpu().numpy().astype(np.int32),
                 beta_logit=self._beta[0].detach().cpu().numpy().astype(np.float32),
                 gate_logit=self._gate[0].detach().cpu().numpy().astype(np.float32),
                 meta=json.dumps(meta), **{
                     k: (v.numpy() if torch.is_tensor(v) else v) for k, v in S.items()})
        self._report_storage(S, dense.shape, out_dir)

    def _report_storage(self, S, dense_shape, out_dir):
        M, Nd, _ = dense_shape
        n_rec = int(S['rec_counts'].sum())
        sparse_bytes = n_rec * (4 + 9 * 4) + M * 4 * 3
        dense_bytes = M * Nd * 9 * 4
        file_bytes = os.path.getsize(os.path.join(out_dir, "motion.npz"))
        print("[SparseAnchor] storage: {} anchors x {} dynamic Gaussians, records kept {}/{} ({:.1f}%): "
              "sparse {:.2f} MB vs dense-fp32 {:.2f} MB (file {:.2f} MB)".format(
                  M, Nd, n_rec, M * Nd, 100.0 * n_rec / max(1, M * Nd), sparse_bytes / 2 ** 20,
                  dense_bytes / 2 ** 20, file_bytes / 2 ** 20))

    # ------------------------------------------------------------------ loading / sparse decode
    @classmethod
    def load(cls, model_path, gaussians, args, iteration=-1, override=None):
        """Load a sparse motion checkpoint. `override` may set W / K / epsilon for query-time ablations."""
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(model_path, "motion"))
        path = os.path.join(model_path, "motion", "iteration_{}".format(iteration), "motion.npz")
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        for k in ('A', 'W', 'K', 'basis_order', 'sigma_c'):
            setattr(args, {'A': 'anchor_stride', 'W': 'temporal_window', 'K': 'knn_k',
                           'basis_order': 'temporal_basis_order', 'sigma_c': 'consensus_sigma'}[k], meta[k])
        args.no_fit_confidence = not meta['use_confidence']
        args.score_kappa_rot, args.score_kappa_scale = meta['kappa_rot'], meta['kappa_scale']
        args.wls_eps = meta['wls_eps']
        args.epsilon, args.coarse_levels = meta['epsilon'], meta['level']
        override = override or {}
        if override.get('W') is not None:
            args.temporal_window = override['W']
        if override.get('K') is not None:
            args.knn_k = override['K']
        self = cls(args, meta['T'], meta['extent'])
        self.gaussians = gaussians
        self.training = False
        self.level = meta['level']
        self.object_radius = meta.get('object_radius', meta['extent'])
        N = gaussians.get_xyz.shape[0]
        assert N == meta['N'], "point cloud ({}) and motion checkpoint ({}) disagree".format(N, meta['N'])
        self.anchor_times = torch.from_numpy(z['anchor_times']).float().cuda()
        self.is_dyn = torch.zeros(N, dtype=torch.bool, device='cuda')
        self.is_dyn[torch.from_numpy(z['dyn_idx'].astype(np.int64)).cuda()] = True
        self._beta = torch.from_numpy(z['beta_logit']).float().cuda().unsqueeze(0)
        self._gate = torch.from_numpy(z['gate_logit']).float().cuda().unsqueeze(0)
        self._set_sparse({k: z[k] for k in ('rec_counts', 'rec_idx', 'rec_val')}, fill_k=meta['K'])
        n_rec = int(z['rec_counts'].sum())
        print("[SparseAnchor] loaded {}: {} anchors, {} dynamic / {} Gaussians, {} sparse records".format(
            path, self.num_anchors, int(self.is_dyn.sum()), N, n_rec))
        if override.get('epsilon') is not None and override['epsilon'] != meta['epsilon']:
            self.resparsify(override['epsilon'])
        return self

    def _set_sparse(self, S, fill_k):
        dev = 'cuda'
        dyn_idx = self.dyn_idx
        xyz_dyn = self.gaussians.get_xyz.detach()[dyn_idx]
        co = np.concatenate(([0], np.cumsum(S['rec_counts'])))
        k = min(fill_k, xyz_dyn.shape[0] - 1)
        nbr_idx, dist = knn_graph(xyz_dyn, k)
        nbr_w = knn_weights(dist)
        self.sparse = {
            'off': co,
            'rec_idx': torch.as_tensor(np.asarray(S['rec_idx']).astype(np.int64)).to(dev),
            'rec_val': torch.as_tensor(np.asarray(S['rec_val']).astype(np.float32)).to(dev),
        }
        self.nbr_idx, self.nbr_w = nbr_idx, nbr_w
        self._dyn_xyz = xyz_dyn
        self.knn_dirty = False
        self._clear_caches()

    @torch.no_grad()
    def _decoded_anchor(self, a):
        """Reconstruct dynamic-Gaussian deltas of anchor a from its sparse records (missing == zero)."""
        if a in self._anchor_cache:
            return self._anchor_cache[a]
        S = self.sparse
        Nd = int(self.is_dyn.sum())
        D = torch.zeros(Nd, 9, device='cuda')
        c0, c1 = S['off'][a], S['off'][a + 1]
        D[S['rec_idx'][c0:c1]] = S['rec_val'][c0:c1]
        out = (D[:, 0:3], D[:, 3:6], D[:, 6:9])
        self._anchor_cache[a] = out
        return out

    def _dense_from_sparse(self):
        return torch.stack([torch.cat(self._decoded_anchor(a), dim=-1) for a in range(self.num_anchors)])

    @torch.no_grad()
    def resparsify(self, eps):
        """Re-encode the (decoded) dynamic-Gaussian anchors with a different epsilon; used for storage ablations."""
        dense = self._dense_from_sparse()
        S = self._encode(dense, eps)
        self.args.epsilon = eps
        self._set_sparse(S, fill_k=self.K)
        M, Nd = dense.shape[0], dense.shape[1]
        n_rec = int(S['rec_counts'].sum())
        print("[SparseAnchor] re-sparsified with epsilon={}: records {}/{} ({:.1f}%), ~{:.2f} MB".format(
            eps, n_rec, M * Nd, 100.0 * n_rec / max(1, M * Nd), (n_rec * 40 + Nd * 4) / 2 ** 20))
