"""Gated learned correction vector (line 10 of Algorithm 1:  v <- v + lam * g_hat + gate(ctx) * A(u)).

When the Supervisor's gate opens (the policy's own intent has collapsed and the end effector is not making progress),
the sampler learns a world-frame displacement offset u (m per control step) for this episode and adds it to the flow
at every Euler step through the action map A (the same mechanism as the deadlock escape, whose fixed lift it
generalises). u is fitted per policy call in closed form, CAR-vector style: B candidate offsets u_b = u + sigma * xi_b
are rolled out through the guided flow in one batch (common noise), scored by a recovery reward, and u is updated to
their softmax-weighted mean (EMA over chunks). The candidates' end points are produced by the policy's own velocity
field, so the correction pulls towards chunks the policy can generate (on-manifold), not towards an arbitrary direction.

Recovery reward of candidate b (chunk x_b, executed prefix ends at p_end):
    r_b = - w_J * J(x_b) / d_safe^2            barrier cost: the correction must stay safe
          - w_obj * dist_xy(p_end, nearest task object)   move back towards the task (objects still on the table)
          + w_move * |p_end - p_0|             leave the stall region
          - w_reg * |u_b|^2 / sigma^2          regulariser
"""
import dataclasses

import numpy as np
import torch

from .cost import chunk_eef_positions, obstacle_costs


@dataclasses.dataclass
class RecoveryConfig:
    B: int = 16  # candidate offsets per gated policy call (one batched flow integration)
    sigma: float = 0.03  # m per control step: exploration of the offset (the flow absorbs a large part of it)
    tau: float = 0.5  # softmax temperature on the standardised reward (r - mean) / std
    w_J: float = 1.0
    w_clr: float = 1.0  # penalty for predicted minimum clearance below d_safe (keeps the correction away from obstacles)
    w_obj: float = 1.0
    w_move: float = 0.5
    w_reg: float = 0.05
    beta: float = 0.5  # EMA weight of the per-call closed-form solution
    hold: int = 1  # chunks the correction stays applied after the gate closes
    decay: float = 0.7  # u decay per chunk while the gate is closed
    u_max: float = 0.03  # m per control step, cap on |u|


def recovery_reward(xs, cands, ctx, T, cfg: RecoveryConfig):
    """xs (B, H, D) candidate chunks, cands (B, 3) offsets -> reward (B,), diagnostics dict."""
    P = chunk_eef_positions(xs, ctx, T)  # (B, H', 3)
    e = min(ctx.exec_steps, P.shape[1]) - 1
    p_end = P[:, e]
    costs, clear = obstacle_costs(xs, ctx, T)  # (K, B), (B,)
    J = costs.sum(0) if costs.shape[0] else torch.zeros(xs.shape[0], device=xs.device)
    near = torch.relu(ctx.d_safe - clear) / ctx.d_safe if costs.shape[0] else torch.zeros_like(J)
    move = (p_end - T["p0"]).norm(dim=-1)
    if ctx.obj_xy is not None and len(ctx.obj_xy):
        obj = torch.as_tensor(np.asarray(ctx.obj_xy, dtype=float), dtype=torch.float32, device=xs.device)  # (M, 2)
        dobj = torch.cdist(p_end[:, :2], obj).min(dim=1).values
    else:
        dobj = torch.zeros_like(move)
    reg = (cands**2).sum(-1) / cfg.sigma**2
    r = -cfg.w_J * J / ctx.d_safe**2 - cfg.w_clr * near - cfg.w_obj * dobj + cfg.w_move * move - cfg.w_reg * reg
    # how much of the commanded offset the flow actually realises: |p_end_b - p_end_0| / (|u_b - u_0| * executed steps)
    du = (cands[1:] - cands[:1]).norm(dim=-1)
    dp = (p_end[1:] - p_end[:1]).norm(dim=-1)
    gain = float((dp / (du * (e + 1) + 1e-9))[du > 1e-6].mean()) if (du > 1e-6).any() else 0.0
    return r, {"rec_J_mean": float(J.mean()), "rec_dobj_mean": float(dobj.mean()), "rec_move_mean": float(move.mean()),
               "rec_end_spread_cm": 100 * float(p_end.std(dim=0).norm()), "rec_gain": gain}
