"""Template: a runtime constraint of your own, used instead of the built-in obstacle cost.

cost_fn(a_hat, ctx, T) -> (J, violation)
  a_hat      (B, H, D) clean-chunk estimate in the policy's normalised action space (differentiable)
  ctx        ChunkCostContext: exec_steps, margins, d_safe, cbf_vref, cost_type ... (built by the Supervisor)
  T          dict of tensors for this chunk (action map, robot spheres, obstacles, ...)
  J          (B,) cost per sample; its gradient w.r.t. a_hat drives the push
  violation  (B,) metres; the push magnitude is min(1, violation / ctx.cbf_vref)  (same rule as the built-in barrier)
"""
import torch

import safeguide as sg
from safeguide.core.cost import chunk_eef_positions, sphere_positions, step_weights


def keep_above_table(a_hat, ctx, T, z_min=0.02):
    """End effector must stay above z_min (world frame)."""
    p = chunk_eef_positions(a_hat, ctx, T)  # (B, H, 3) end-effector path implied by the chunk
    viol = torch.relu(z_min - p[..., 2])  # (B, H)
    w = step_weights(ctx, p.shape[1], p.device)  # 1 on the executed steps, 0.3 on the tail
    return (w * viol**2).sum(1), viol.amax(1)


def stay_inside_box(a_hat, ctx, T, lo=(-0.3, -0.4, 0.0), hi=(0.3, 0.4, 0.5)):
    """Every robot sphere must stay inside an axis-aligned workspace box (uses the whole sphere model)."""
    p = chunk_eef_positions(a_hat, ctx, T)
    x = sphere_positions(p, T)  # (B, H, P, 3) sphere centres along the chunk
    lo_t, hi_t = torch.as_tensor(lo, device=x.device), torch.as_tensor(hi, device=x.device)
    viol = torch.relu(lo_t - x).amax(-1) + torch.relu(x - hi_t).amax(-1)  # (B, H, P)
    w = step_weights(ctx, x.shape[1], x.device)
    return (w * (viol**2).sum(-1)).sum(1), viol.amax((1, 2))


def combined(a_hat, ctx, T):
    """Add the built-in obstacle cost to a custom one: J = J_obstacles + J_custom, violation = max of both."""
    costs_k, clear = sg.obstacle_costs(a_hat, ctx, T)  # (K, B), (B,)
    j2, v2 = keep_above_table(a_hat, ctx, T)
    return costs_k.sum(0) + j2, torch.maximum(torch.relu(ctx.d_safe - clear), v2)


# layer = sg.SafeGuide(adapter, robot, scene, cost_fn=keep_above_table).install()
