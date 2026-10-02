"""K-nearest-neighbour graph over Gaussian centres (CPU cKDTree build, GPU tensors out)."""
import numpy as np
import torch
from scipy.spatial import cKDTree


def knn_graph(xyz, k, query_xyz=None, exclude_self=True):
    """Return (idx [Q,k] long, dist [Q,k] float) of the k nearest points of `xyz` for each query.

    If `query_xyz` is None the queries are `xyz` itself and the trivial self-match is removed.
    """
    pts = xyz.detach().cpu().numpy().astype(np.float64)
    tree = cKDTree(pts)
    q = pts if query_xyz is None else query_xyz.detach().cpu().numpy().astype(np.float64)
    kk = k + 1 if (query_xyz is None and exclude_self) else k
    kk = min(kk, pts.shape[0])
    dist, idx = tree.query(q, k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    if query_xyz is None and exclude_self:
        dist, idx = dist[:, 1:], idx[:, 1:]
    dev = xyz.device
    return (torch.from_numpy(np.ascontiguousarray(idx)).long().to(dev),
            torch.from_numpy(np.ascontiguousarray(dist)).float().to(dev))


def knn_weights(dist):
    """Adaptive Gaussian weights w_ij = exp(-d^2 / (2 sigma_i^2)), sigma_i = mean neighbour distance."""
    sigma = dist.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.exp(-0.5 * (dist / sigma) ** 2)
