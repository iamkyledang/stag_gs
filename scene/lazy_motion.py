"""Lazy-Evaluated Dynamic 3D Gaussians: sparse per-Gaussian anchors + query-time motion interpolation.

Revised design (see change_log/):
  * canonical 3DGS + per-anchor deltas (dx, so(3) log-rotation, dlog-scale) at anchors of stride A, for
    every dynamic Gaussian (segmentation-based dynamic/static split, no control-node subsampling):
    each dynamic Gaussian always keeps its own independently-optimised anchor deltas.
  * query t -> window of W anchor intervals -> motion rates u in R^9 -> degree-3 cubic MLS smoothing
    over the K=48 nearest neighbours *within the same (dynamic) object* -> bidirectional (forward/
    backward) Taylor transport -> confidence-weighted consensus.
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


# --------------------------------------------------------------------------------------------------
# Degree-3 cubic moving-least-squares motion model
# --------------------------------------------------------------------------------------------------
# Exponents (ex, ey, ez) of the 20 monomials of a trivariate polynomial basis up to total degree 3.
_CUBIC_EXPONENTS = [
    (0, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1),
    (3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1),
]


def _cubic_basis(xi):
    """xi [...,3] normalized local coordinates -> Phi [...,20] monomial basis, total degree <= 3."""
    x, y, z = xi[..., 0], xi[..., 1], xi[..., 2]
    terms = []
    for ex, ey, ez in _CUBIC_EXPONENTS:
        t = torch.ones_like(x)
        if ex:
            t = t * x ** ex
        if ey:
            t = t * y ** ey
        if ez:
            t = t * z ** ez
        terms.append(t)
    return torch.stack(terms, dim=-1)


def cubic_mls_fit(dx, w, field_nbr, eps_rel=1e-2, chunk_size=8192):
    """Degree-3 moving-least-squares fit of `field_nbr` (values sampled at K neighbours, [Q,K,D]) as a
    cubic trivariate polynomial of the neighbours' *normalized* local coordinates `dx` ([Q,K,3],
    already divided by each query point's own local bandwidth), weighted by `w` [Q,K]. Returns the
    MLS-smoothed field value at the query point itself (the polynomial's constant/0th-order term,
    i.e. evaluated at the normalized origin) and the per-neighbour fit residual (used for confidence).

    Processed in chunks along Q: the batched 20x20 solve and its K-neighbour expansions scale with
    Q*K*D and can exhaust GPU memory for large dynamic point sets otherwise (differentiable either way).
    """
    Q = dx.shape[0]
    if Q <= chunk_size:
        return _cubic_mls_fit_chunk(dx, w, field_nbr, eps_rel)
    centers, residuals = [], []
    for lo in range(0, Q, chunk_size):
        hi = min(lo + chunk_size, Q)
        c, r = _cubic_mls_fit_chunk(dx[lo:hi], w[lo:hi], field_nbr[lo:hi], eps_rel)
        centers.append(c)
        residuals.append(r)
    return torch.cat(centers, dim=0), torch.cat(residuals, dim=0)


def _cubic_mls_fit_chunk(dx, w, field_nbr, eps_rel):
    Phi = _cubic_basis(dx)                                          # [Q,K,20]
    wPhi = w.unsqueeze(-1) * Phi
    cov = torch.einsum('qkb,qkc->qbc', wPhi, Phi)                   # [Q,20,20]
    reg = eps_rel * cov.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    cov = cov + reg.view(-1, 1, 1) * torch.eye(Phi.shape[-1], device=dx.device, dtype=dx.dtype)
    rhs = torch.einsum('qkb,qkd->qbd', wPhi, field_nbr)             # [Q,20,D]
    C = torch.linalg.solve(cov, rhs)                                # [Q,20,D]
    field_center = C[:, 0, :]                                       # constant term = value at xi=0
    pred_nbr = torch.einsum('qkb,qbd->qkd', Phi, C)
    res = field_nbr - pred_nbr
    return field_center, res


def _wls_line_fit(taus, weights, values, t0):
    """Per-element weighted linear fit  value(tau) ~= v + a (tau - t0)  over a few time samples.

    taus, weights: python lists (len n); values [n, ...]. Returns (v, a) with the shape of values[0].
    """
    if len(taus) == 1:
        return values[0], torch.zeros_like(values[0])
    x = torch.tensor([tt - t0 for tt in taus], dtype=values.dtype, device=values.device)
    om = torch.tensor(weights, dtype=values.dtype, device=values.device)
    shape = (-1,) + (1,) * (values.dim() - 1)
    s0, s1, s2 = om.sum(), (om * x).sum(), (om * x * x).sum()
    u0 = (om.view(shape) * values).sum(0)
    u1 = ((om * x).view(shape) * values).sum(0)
    den = (s0 * s2 - s1 * s1).clamp_min(1e-12)
    a = (s0 * u1 - s1 * u0) / den
    v = (u0 - a * s1) / s0
    return v, a


class LazyMotionModel:
    """Per-Gaussian sparse anchor storage + lazy query-time reconstruction of (x, r, s) for every
    dynamic Gaussian. Every Gaussian the mask-vote segmentation calls dynamic keeps its own
    independently-optimised anchor deltas (no control-node subsampling / prediction tier); the only
    thing segmentation does is scope the KNN graph (build_knn) and the cubic-MLS smoothing to the
    dynamic object's own points, so a per-Gaussian motion never gets smoothed against unrelated static
    background neighbours."""

    def __init__(self, args, num_train_frames, scene_extent):
        self.args = args
        self.A = int(args.anchor_stride)
        self.W = int(args.temporal_window)
        self.K = int(args.knn_k)
        self.order = int(args.transport_order)
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
        self.is_dyn = None             # [N] bool: dynamic (post-freeze) vs frozen-static
        self.object_radius = None      # robust bounding radius of the dynamic object, set at freeze
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
        Background freeze happens later, see freeze_background."""
        self.gaussians = gaussians
        N = gaussians.get_xyz.shape[0]
        self.anchor_times = self._anchor_times_for_level(self.level)
        M = self.num_anchors
        self.is_dyn = torch.ones(N, dtype=torch.bool, device='cuda')
        self._dx = nn.Parameter(torch.zeros(M, N, 3, device='cuda'))
        self._drot = nn.Parameter(torch.zeros(M, N, 3, device='cuda'))
        self._dscale = nn.Parameter(torch.zeros(M, N, 3, device='cuda'))
        self.knn_dirty = True
        self._setup_optimizer(opt)
        print("[Lazy] motion enabled: {} anchors (stride {}), {} Gaussians, W={}, K={}".format(
            M, self.A * 2 ** self.level, N, self.W, self.K))

    def capture(self):
        """Full training-state snapshot (dense anchor params + Adam moments + freeze state), for
        train.py --checkpoint_iterations/--start_checkpoint. Unlike save()/load() (lossy, sparsified,
        inference-only), this preserves the live dense tensors bit-for-bit, so it works correctly at
        any iteration motion is active, frozen or not (is_dyn may still be all-True pre-freeze)."""
        return {
            'level': self.level, 'anchor_times': self.anchor_times, 'is_dyn': self.is_dyn,
            'object_radius': self.object_radius, 'dx': self._dx, 'drot': self._drot, 'dscale': self._dscale,
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
        }

    def update_learning_rate(self, iteration):
        if self.optimizer is None:
            return
        t = max(iteration - self._lr_ref_iter, 0)
        for g in self.optimizer.param_groups:
            g['lr'] = self._sched[g['name']](t)

    def _params(self):
        return (('dx', self._dx), ('drot', self._drot), ('dscale', self._dscale))

    def _replace_params(self, new, keep_state_fn=None):
        """Swap the four parameter tensors (new: dict name->tensor). Optimizer moments are rebuilt
        by keep_state_fn(name, exp_avg, exp_avg_sq)->(exp_avg, exp_avg_sq) or reset to zero."""
        for g in self.optimizer.param_groups:
            name = g['name']
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

    # ------------------------------------------------------------------ KNN graph (same-object scope, K=48)
    def build_knn(self, force=False):
        if not (self.knn_dirty or force):
            return
        dyn_idx = self.dyn_idx
        self._dyn_xyz = self.gaussians.get_xyz.detach()[dyn_idx]
        # single self-graph over the dynamic object's own points only (no cross-object contamination,
        # no control-node tier): every dynamic Gaussian's own motion is smoothed against its own object.
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
        """Velocity/acceleration field at anchor a_m (side 'F') or a_{m+1} (side 'B') from the W nearest
        intervals on that side, then degree-3 cubic-MLS-smoothed over the dynamic object's own KNN
        graph (K=48). Returns (field [Ndyn,18], confidence [Ndyn,1])."""
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
        v, a = _wls_line_fit(taus, ws, torch.stack(us), t_star)
        if self.order < 2:
            a = torch.zeros_like(a)
        field = torch.cat((v, a), dim=-1)                                     # [Ndyn, 18]
        conf = torch.ones(field.shape[0], 1, device=field.device)
        if self.nbr_idx is not None:
            field, conf = self._mls_correct(field)
        out = (field, conf)
        if not self.training:
            self._tangent_cache[key] = out
        return out

    def _mls_correct(self, field):
        """Degree-3 cubic MLS smoothing of the rate/acceleration field over the dynamic object's own
        KNN graph (K=48), using local coordinates normalized by each query point's own neighbour-
        distance bandwidth. The polynomial's 0th-order term directly replaces `field` (no residual/
        beta blend): it is a robust local estimate of the field at the query point itself, derived
        only from same-object neighbours."""
        xyz = self._dyn_xyz
        nbr = self.nbr_idx                                                    # [Ndyn,K] local indices
        w = self.nbr_w
        dx = xyz[nbr] - xyz.unsqueeze(1)                                      # [Ndyn,K,3]
        bandwidth = dx.norm(dim=-1).mean(dim=1, keepdim=True).clamp_min(1e-8)
        dx_n = dx / bandwidth.unsqueeze(-1)                                   # normalized local coordinates
        field_nbr = field[nbr]                                               # [Ndyn,K,18]
        field, res = cubic_mls_fit(dx_n, w, field_nbr, self.wls_eps)
        conf = torch.ones(field.shape[0], 1, device=field.device)
        if self.use_confidence:
            wsum = w.sum(1, keepdim=True).clamp_min(1e-8)
            msr = (w.unsqueeze(-1) * res[..., :9].detach() ** 2).sum(1) / wsum   # [Ndyn,9]
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
        (d_xyz, d_rotation, d_scaling, info). Opacity is never deformed."""
        gs = self.gaussians
        N = gs.get_xyz.shape[0]
        x_hat, q_hat, l_hat, info = self.query(t)
        dyn_idx = self.dyn_idx
        r_c = quat_normalize(gs._rotation[dyn_idx])
        l_can = gs._scaling[dyn_idx]
        d_xyz = torch.zeros(N, 3, device='cuda').index_put((dyn_idx,), x_hat)
        d_rot = torch.zeros(N, 4, device='cuda').index_put((dyn_idx,), quat_mul(q_hat, r_c) - r_c)
        d_scale = torch.zeros(N, 3, device='cuda').index_put((dyn_idx,), torch.exp(l_can + l_hat) - torch.exp(l_can))
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
        """Object-bounding-radius denominator (was each Gaussian's own ~1e-3-extent scale, which made
        every motion look huge and defeated the dynamic-set/storage thresholds): a single scalar
        describing the dynamic object's own spatial extent, set once at freeze time."""
        return max(self.object_radius or self.extent, 1e-6)

    def change_score(self, dx, drot, dl, denom):
        """Normalised change score d: translation relative to the object's bounding radius, rotation and
        log-scale relative to their kappa units. Shapes [..., N, D_component]."""
        d_x = dx.norm(dim=-1) / denom
        d_r = drot.norm(dim=-1) / self.kappa_rot
        d_s = dl.abs().amax(dim=-1) / self.kappa_scale
        return torch.max(torch.max(d_x, d_r), d_s)

    # ------------------------------------------------------------------ background freeze (segmentation)
    @torch.no_grad()
    def freeze_background(self, tau_w):
        """One-shot: freeze every Gaussian the mask-vote splat never confidently called dynamic
        (get_dyn_prob<=0.5 or not-yet-visible-enough, dyn_w<=tau_w). Every surviving dynamic Gaussian
        keeps its own independently-optimised anchor deltas (no control-node subsampling) -- the
        segmentation mask only scopes the KNN graph/cubic-MLS smoothing to the dynamic object itself."""
        gs = self.gaussians
        dyn_idx = self.dyn_idx
        keep_local = (gs.get_dyn_prob[dyn_idx] > 0.5) & (gs.dyn_w[dyn_idx] > tau_w)
        N = self.is_dyn.shape[0]
        keep_full = torch.zeros(N, dtype=torch.bool, device='cuda')
        keep_full[dyn_idx[keep_local]] = True

        new = {name: p.data[:, keep_local] for name, p in self._params()}
        self._replace_params(new, lambda n, ea, es: (ea[:, keep_local], es[:, keep_local]))
        n_before = int(self.is_dyn.sum())
        self.is_dyn = keep_full
        print("[Lazy] background freeze: {} -> {} dynamic Gaussians ({:.1f}%), mask-vote driven".format(
            n_before, int(keep_full.sum()), 100.0 * keep_full.float().mean().item()))

        dyn_idx = self.dyn_idx   # refreshed post-freeze
        xyz_dyn = gs.get_xyz.detach()[dyn_idx]
        centroid = xyz_dyn.mean(0)
        self.object_radius = float((xyz_dyn - centroid).norm(dim=-1).quantile(0.95).clamp_min(1e-6))
        self.knn_dirty = True
        print("[Lazy] dynamic object: {} Gaussians, object_radius={:.4f}".format(
            int(self.is_dyn.sum()), self.object_radius))

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
        print("[Lazy] anchors refined: {} -> {} (stride {})".format(M_old, new_t.numel(), self.A * 2 ** self.level))

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
            rec_idx.append(ic.int().cpu()); rec_val.append(D[ic].half().cpu()); rec_cnt.append(ic.numel())
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
            'A': self.A, 'W': self.W, 'K': self.K, 'order': self.order, 'sigma_c': self.sigma_c,
            'use_confidence': self.use_confidence, 'epsilon': eps,
            'kappa_rot': self.kappa_rot, 'kappa_scale': self.kappa_scale,
            'wls_eps': self.wls_eps, 'extent': self.extent, 'object_radius': self.object_radius,
            'T': self.T, 'N': int(self.is_dyn.shape[0]), 'level': self.level,
        }
        np.savez(os.path.join(out_dir, "motion.npz"),
                 anchor_times=self.anchor_times.cpu().numpy().astype(np.float32),
                 dyn_idx=self.dyn_idx.cpu().numpy().astype(np.int32),
                 meta=json.dumps(meta), **{
                     k: (v.numpy() if torch.is_tensor(v) else v) for k, v in S.items()})
        self._report_storage(S, dense.shape, out_dir)

    def _report_storage(self, S, dense_shape, out_dir):
        M, Nd, _ = dense_shape
        n_rec = int(S['rec_counts'].sum())
        sparse_bytes = n_rec * (4 + 9 * 2) + M * 4 * 3
        dense_bytes = M * Nd * 9 * 2
        file_bytes = os.path.getsize(os.path.join(out_dir, "motion.npz"))
        print("[Lazy] storage: {} anchors x {} dynamic Gaussians, records kept {}/{} ({:.1f}%): "
              "sparse {:.2f} MB vs dense-fp16 {:.2f} MB (file {:.2f} MB)".format(
                  M, Nd, n_rec, M * Nd, 100.0 * n_rec / max(1, M * Nd), sparse_bytes / 2 ** 20,
                  dense_bytes / 2 ** 20, file_bytes / 2 ** 20))

    # ------------------------------------------------------------------ loading / lazy decode
    @classmethod
    def load(cls, model_path, gaussians, args, iteration=-1, override=None):
        """Load a sparse motion checkpoint. `override` may set W / K / epsilon for query-time ablations."""
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(model_path, "motion"))
        path = os.path.join(model_path, "motion", "iteration_{}".format(iteration), "motion.npz")
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        for k in ('A', 'W', 'K', 'order', 'sigma_c'):
            setattr(args, {'A': 'anchor_stride', 'W': 'temporal_window', 'K': 'knn_k', 'order': 'transport_order',
                           'sigma_c': 'consensus_sigma'}[k], meta[k])
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
        self._set_sparse({k: z[k] for k in ('rec_counts', 'rec_idx', 'rec_val')}, fill_k=meta['K'])
        n_rec = int(z['rec_counts'].sum())
        print("[Lazy] loaded {}: {} anchors, {} dynamic / {} Gaussians, {} sparse records".format(
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
        """Lazily reconstruct dynamic-Gaussian deltas of anchor a from its sparse records (missing == zero)."""
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
        print("[Lazy] re-sparsified with epsilon={}: records {}/{} ({:.1f}%), ~{:.2f} MB".format(
            eps, n_rec, M * Nd, 100.0 * n_rec / max(1, M * Nd), (n_rec * 22 + Nd * 4) / 2 ** 20))
