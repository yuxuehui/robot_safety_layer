"""Signed distance functions: numpy for placement / evaluation metrics, torch for the differentiable cost.
Identical formulas in both, so 'clearance' means the same thing in the guidance and in the metrics.
torch is imported lazily so that numpy-only scripts can use this module without it."""
import numpy as np


def sdf_cylinder_np(p, center, radius, half_height):
    """Signed distance from points p (..., 3) to a z-aligned solid cylinder."""
    q = p - center
    dr = np.linalg.norm(q[..., :2], axis=-1) - radius
    dz = np.abs(q[..., 2]) - half_height
    outside = np.linalg.norm(np.stack([np.maximum(dr, 0), np.maximum(dz, 0)], -1), axis=-1)
    inside = np.minimum(np.maximum(dr, dz), 0)
    return outside + inside


def sdf_cylinder_torch(p, center, radius, half_height):
    import torch

    q = p - center
    dr = torch.linalg.norm(q[..., :2], dim=-1) - radius
    dz = q[..., 2].abs() - half_height
    outside = torch.linalg.norm(torch.stack([dr.clamp(min=0), dz.clamp(min=0)], -1), dim=-1)
    inside = torch.maximum(dr, dz).clamp(max=0)
    return outside + inside


def sdf_capsule_np(p, a, b, r):
    """Signed distance from points p (..., 3) to a capsule with axis a->b and radius r."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab + 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1) - r


def sdf_capsule_torch(p, a, b, r):
    """Signed distance from p (..., 3) to a capsule with axis a -> b (broadcastable (..., 3)) and radius r."""
    import torch

    ab = b - a
    t = (((p - a) * ab).sum(-1) / ((ab * ab).sum(-1) + 1e-12)).clamp(0.0, 1.0)
    return torch.linalg.norm(p - (a + t[..., None] * ab), dim=-1) - r
