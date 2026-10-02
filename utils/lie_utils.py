"""SO(3) / unit-quaternion helpers. Quaternions are (w, x, y, z), matching GaussianModel._rotation.

All functions are batched over leading dims and are differentiable everywhere, including at
the identity (zero rotation vector), which is where anchor deltas are initialised.
"""
import torch

_EPS2 = 1e-16


def quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_conj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), dim=-1)


def quat_rotate(q, v):
    """Rotate vectors v[...,3] by unit quaternions q[...,4] (broadcastable)."""
    qv = q[..., 1:]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.linalg.cross(qv, t, dim=-1)


def so3_exp(w):
    """Rotation vector [...,3] -> unit quaternion [...,4]."""
    theta = torch.sqrt((w * w).sum(-1, keepdim=True) + _EPS2)
    half = 0.5 * theta
    return torch.cat((torch.cos(half), w * (torch.sin(half) / theta)), dim=-1)


def so3_log(q):
    """Unit quaternion [...,4] -> rotation vector [...,3] (shortest arc, |theta| <= pi)."""
    q = quat_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    sin_half = torch.sqrt((v * v).sum(-1, keepdim=True) + _EPS2)
    theta = 2.0 * torch.atan2(sin_half, q[..., :1])
    return v * (theta / sin_half)


def quat_slerp(q0, q1, s):
    """Geodesic interpolation from q0 (s=0) to q1 (s=1); s is [...,1]."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
    rel = quat_mul(quat_conj(q0), q1)
    return quat_mul(q0, so3_exp(s * so3_log(rel)))
