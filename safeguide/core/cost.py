"""Chunk-level obstacle cost, all in the world frame (metres).

The cost is independent of the policy: the only model-specific piece is the ActionMap, which turns the policy's
normalised action chunk into end-effector positions (and maps a world-frame displacement back into normalised
action space for the deadlock escape). Everything else works on robot collision spheres, static cylinders and
moving capsules.

Cost types
  hinge:  sum_h w_h relu(margin_p - clearance_{h,p})^2
  cbf:    discrete-time exponential barrier anchored at the measured chunk-start clearance c0:
          floor_{h,p} = min(d_safe, margin_p + (1-gamma)^(h+1) relu(c0_p - margin_p)),  viol = relu(floor - clearance)
          far spheres may approach at a rate proportional to their clearance, spheres sitting still, moving
          tangentially or away are not penalised, and the margin itself is never crossed.
  energy: long-range energy (sigma sqrt(pi)/2) erfc(clearance/sigma), used by the CAR baseline.
"""
import dataclasses

import numpy as np
import torch

from .geometry import sdf_capsule_torch, sdf_cylinder_torch


def _f(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class ActionMap:
    """Policy action space <-> end-effector motion. One subclass per action convention (delta-EEF, joint targets, ...).

    to_torch(device)                    tensors this map needs, merged into the per-chunk dict T
    eef_positions(a_norm, T)            (B, H, D) normalised actions -> (B, H, 3) EEF positions after each control step
    world_offset_to_action(off, T)      (3,) world displacement per control step -> the normalised-action offset on the
                                        position dims that realises it (deadlock escape)
    position_dims                       slice of the action vector that moves the EEF position
    """

    position_dims = slice(0, 3)

    def to_torch(self, device):
        raise NotImplementedError

    def eef_positions(self, a_norm, T):
        raise NotImplementedError

    def world_offset_to_action(self, off_world, T):
        raise NotImplementedError


@dataclasses.dataclass
class DeltaEEFMap(ActionMap):
    """Delta-EEF position actions (robosuite OSC_POSE style) normalised per dimension to [-1, 1] by two per-dim
    reference vectors: openpi's quantile Unnormalize with (q01, q99), or GR00T's min/max normalisation with
    (min, max). a = (a_norm + 1)/2 (hi - lo) + lo, clipped to the controller's input range [-1, 1], then
    dp = G a with the calibrated gain G (m per unit action per control step; fitted on baseline rollouts)."""

    q01: np.ndarray  # (D,) lower reference per action dim (only the first 3 are used)
    q99: np.ndarray  # (D,) upper reference
    G: np.ndarray  # (3, 3) action -> EEF displacement gain
    n_valid: int | None = None  # chunk steps that carry real actions (GR00T pads the chunk to max_action_horizon)

    def to_torch(self, device):
        return {"q01": _f(self.q01, device), "q99": _f(self.q99, device), "G": _f(self.G, device)}

    def eef_positions(self, a_norm, T):
        if self.n_valid is not None:
            a_norm = a_norm[..., : self.n_valid, :]
        a = (a_norm[..., :3].float() + 1.0) / 2.0 * (T["q99"][:3] - T["q01"][:3] + 1e-6) + T["q01"][:3]
        a = a.clamp(-1.0, 1.0)
        dp = a @ T["G"].T
        return T["p0"] + torch.cumsum(dp, dim=-2)

    def world_offset_to_action(self, off_world, T):
        """(3,) or (B, 3) world displacement per control step -> normalised-action offset(s) of the same shape."""
        off = torch.as_tensor(off_world, dtype=torch.float32, device=T["G"].device)
        # 3x3 solve on the CPU: cuSOLVER handle creation fails on a nearly full GPU
        if off.ndim == 2:
            return torch.linalg.solve(T["G"].cpu(), off.T.cpu()).T.to(off.device) * 2.0 / (T["q99"][:3] - T["q01"][:3] + 1e-6)
        return torch.linalg.solve(T["G"].cpu(), off.cpu()).to(off.device) * 2.0 / (T["q99"][:3] - T["q01"][:3] + 1e-6)


@dataclasses.dataclass
class ChunkCostContext:
    """Everything the cost needs for one action chunk, all in world frame (metres)."""

    p0: np.ndarray  # (3,) EEF (grip site) position at chunk start
    offsets: np.ndarray  # (P, 3) robot sphere centres minus p0 (gripper, and arm links if modelled)
    radii: np.ndarray  # (P,)
    alphas: np.ndarray  # (P,) sphere moves by alpha * EEF displacement (1 = rigid with the hand)
    obs_centers: np.ndarray  # (K, 3) pillar centres
    obs_radii: np.ndarray  # (K,)
    obs_half_heights: np.ndarray  # (K,)
    q01: np.ndarray = None  # legacy: action quantiles used by openpi Unnormalize (prefer action_map)
    q99: np.ndarray = None
    G: np.ndarray = None  # legacy: action->dp gain (prefer action_map)
    action_map: ActionMap = None  # policy action space -> EEF motion
    d_safe: float = 0.03
    exec_steps: int = 5  # replan_steps: the part of the chunk that is actually executed
    tail_weight: float = 0.3  # weight of the non-executed tail of the chunk
    cost_type: str = "hinge"  # "hinge": relu(margin - clearance)^2 ; "energy": long-range energy ; "cbf": barrier
    energy_sigma: float = 0.05  # m, length scale of the energy
    margins: np.ndarray = None  # (P,) per-sphere margin; None -> d_safe for every sphere
    cbf_gamma: float = 0.3  # cbf: allowed clearance decay per control step (discrete-time exponential CBF)
    cbf_vref: float = 0.01  # cbf: violation (m) at which the push reaches full (unit-RMS) size
    obj_xy: np.ndarray = None  # (M, 2) task-object xy (steer: single-pillar tie -> detour away from nearest object)
    sphere_groups: list = None  # per-sphere label (grip/hand/hand_w/link5/link6/link7/obj) for attribution logs
    motion: np.ndarray = None  # (P, 3, 3) per-sphere linear motion d p_sphere = M_p d p_ee (None -> alphas * d p_ee)
    # moving capsule obstacles (human hand): axis end points at chunk steps 0 (now) .. H (predicted), radii
    cap_a: np.ndarray = None  # (K2, H+1, 3)
    cap_b: np.ndarray = None  # (K2, H+1, 3)
    cap_r: np.ndarray = None  # (K2,)

    def map(self) -> ActionMap:
        if self.action_map is not None:
            return self.action_map
        return DeltaEEFMap(self.q01, self.q99, self.G if self.G is not None else 0.05 * np.eye(3))

    def to_torch(self, device):
        f = lambda x: _f(x, device)
        T = {k: f(getattr(self, k)) for k in
             ("p0", "offsets", "radii", "alphas", "obs_centers", "obs_radii", "obs_half_heights")}
        T.update(self.map().to_torch(device))
        T["margins"] = f(self.margins if self.margins is not None else np.full(len(self.radii), self.d_safe))
        if self.motion is not None:
            T["motion"] = f(self.motion)
        if self.cap_a is not None:
            T["cap_a"], T["cap_b"], T["cap_r"] = f(self.cap_a), f(self.cap_b), f(self.cap_r)
        return T


def chunk_eef_positions(a_norm, ctx, T):
    """a_norm: (B, H, D) normalized actions -> (B, H, 3) EEF positions after each step."""
    return ctx.map().eef_positions(a_norm, T)


def sphere_positions(p, T):
    """(B, H, 3) EEF positions -> (B, H, P, 3) sphere centres: p0 + offset + M_p (p_h - p0), with M_p the
    per-sphere Jacobian-based motion matrix when available, else alpha_p * I."""
    dp = p[..., None, :] - T["p0"]  # (B, H, 1, 3)
    if "motion" in T:
        return T["p0"] + T["offsets"] + torch.einsum("pij,bhj->bhpi", T["motion"], p - T["p0"])
    return T["p0"] + T["offsets"] + T["alphas"][:, None] * dp


def sphere_clearance(pts, T):
    """(K, B, H, P): SDF to obstacle k minus sphere radius. Pillars are static cylinders; capsule obstacles
    (moving hand) use their predicted pose at chunk step h (cap_* index h+1; index 0 is the current pose)."""
    out = [sdf_cylinder_torch(pts, T["obs_centers"][k], T["obs_radii"][k], T["obs_half_heights"][k]) - T["radii"]
           for k in range(T["obs_centers"].shape[0])]
    if "cap_a" in T:
        H = pts.shape[-3]
        for k in range(T["cap_a"].shape[0]):
            a = T["cap_a"][k, 1:H + 1][None, :, None, :]  # (1, H, 1, 3)
            b = T["cap_b"][k, 1:H + 1][None, :, None, :]
            out.append(sdf_capsule_torch(pts, a, b, T["cap_r"][k]) - T["radii"])
    return torch.stack(out)


def start_clearance(T):
    """(K, P): clearance of each sphere at the chunk start (the measured configuration)."""
    pts0 = T["p0"] + T["offsets"]
    out = [sdf_cylinder_torch(pts0, T["obs_centers"][k], T["obs_radii"][k], T["obs_half_heights"][k]) - T["radii"]
           for k in range(T["obs_centers"].shape[0])]
    if "cap_a" in T:
        for k in range(T["cap_a"].shape[0]):
            out.append(sdf_capsule_torch(pts0, T["cap_a"][k, 0], T["cap_b"][k, 0], T["cap_r"][k]) - T["radii"])
    return torch.stack(out)


def step_weights(ctx, H, device):
    w = torch.full((H,), ctx.tail_weight, device=device)
    w[: ctx.exec_steps] = 1.0
    return w


def costs_from_clearance(clr, ctx, T):
    """clr (K, B, H, P) -> per-obstacle costs (K, B) and per-sample max violation (B,) (0 for energy)."""
    K, B, H, P = clr.shape
    w = step_weights(ctx, H, clr.device)[None, None, :, None]
    if ctx.cost_type == "energy":
        s_ = ctx.energy_sigma
        U = (s_ * np.sqrt(np.pi) / 2.0) * torch.special.erfc(clr / s_)
        return (w * U).sum(dim=(-1, -2)), torch.zeros(B, device=clr.device)
    m = T["margins"]  # (P,)
    if ctx.cost_type == "cbf":
        c0 = start_clearance(T)  # (K, P)
        decay = (1.0 - ctx.cbf_gamma) ** torch.arange(1, H + 1, device=clr.device, dtype=clr.dtype)  # (H,)
        floor = (m + decay[:, None] * torch.relu(c0[:, None, :] - m)).clamp(max=ctx.d_safe)  # (K, H, P)
        viol = torch.relu(floor[:, None] - clr)
    else:
        viol = torch.relu(m - clr)
    return (w * viol**2).sum(dim=(-1, -2)), viol.amax(dim=(0, 2, 3))


def obstacle_costs_full(a_norm, ctx, T):
    """costs (K, B), min clearance (B,), max violation (B,), clearance (K, B, H, P)."""
    p = chunk_eef_positions(a_norm, ctx, T)
    clr = sphere_clearance(sphere_positions(p, T), T)
    costs, vmax = costs_from_clearance(clr, ctx, T)
    return costs, clr.amin(dim=(0, 2, 3)), vmax, clr


def obstacle_costs(a_norm, ctx, T):
    """Per-obstacle costs (K, B) and per-sample min clearance over all obstacles (B,)."""
    costs, clear, _, _ = obstacle_costs_full(a_norm, ctx, T)
    return costs, clear


def guidance_magnitude(clear, ctx, viol_max=None):
    """Per-sample push magnitude multiplier. hinge: 1 (unit-RMS push). energy: the reference energy at the
    closest point exp(-clear^2/sigma^2). cbf: proportional to the violation, min(1, viol_max / cbf_vref),
    so a 1 mm violation gets a 10x smaller push than a 1 cm one (minimal intervention)."""
    if ctx.cost_type == "energy":
        return torch.exp(-clear.clamp(min=0.0) ** 2 / ctx.energy_sigma**2)
    if ctx.cost_type == "cbf" and viol_max is not None:
        return (viol_max / ctx.cbf_vref).clamp(max=1.0)
    return torch.ones_like(clear)


def robot_clearance_per_step(a_norm, ctx, T):
    """(K, B, H): min over robot spheres of (SDF - radius) per obstacle and chunk step."""
    p = chunk_eef_positions(a_norm, ctx, T)
    return sphere_clearance(sphere_positions(p, T), T).amin(dim=-1)


SPHERE_GROUPS = ["grip"] * 5 + ["hand"] + ["hand_w"] * 2 + ["link5"] * 5 + ["link6"] * 5 + ["link7"] * 5


def obstacle_cost(a_norm, ctx, T):
    """Total cost over obstacles (summed over batch) and per-sample min clearance (B,)."""
    costs, clear = obstacle_costs(a_norm, ctx, T)
    return costs.sum(), clear
