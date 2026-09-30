"""Model-agnostic G1 guidance of a flow-matching action sampler (the configuration called "inference-time guidance").

At every Euler step of the flow:
    a_hat = clean-chunk estimate from (x_t, t, v)          (FlowSpec.a_hat)
    g     = d cost(a_hat) / d a_hat                        (world-frame CBF cost, no gradient through the model)
    g     <- g / RMS(g) * magnitude(violation)             (unit-RMS push, scaled by how much the barrier is violated)
    v     <- v + sign * lam * g                            (moves x_{t+dt} = x_t + dt v downhill in the cost)
The adapter supplies the conditioning and the velocity call; the Guide never touches model internals.
"""
import dataclasses
import time

import numpy as np
import torch

from .cost import (chunk_eef_positions, ChunkCostContext, costs_from_clearance, guidance_magnitude, obstacle_cost,
                   obstacle_costs_full, sphere_clearance, SPHERE_GROUPS, sphere_positions, start_clearance)


@dataclasses.dataclass
class GuideConfig:
    mode: str = "g1"  # "g1": guided | "none": plain sampling through the same integrator (paired baselines)
    scale: float = 1.0  # lam: push size in unit-RMS velocity units
    grad_through_model: bool = False  # True: d cost / d x_t through the velocity network (needs autograd through it)
    normalize_grad: bool = True  # rescale g to unit RMS per chunk (then scale is in "velocity units")
    t_min: float = 0.0  # only guide while the noise level is above t_min
    schedule: str = "const"  # "const" | "linear" (lam * progress: stronger near the data end)
    trigger_exec_only: bool = False  # ablation: guide only if an EXECUTED step (h < exec_steps) violates

    @classmethod
    def sota(cls, **kw):
        """The configuration evaluated as 'inference-time guidance' (G1, unit push, gradient w.r.t. a_hat only)."""
        return cls(mode="g1", scale=1.0, grad_through_model=False, normalize_grad=True, **kw)


class Guide:
    """Guided sampler. Set `ctx` (ChunkCostContext) and optionally `escape_off` before every policy call; read
    `last_log` afterwards. `adapter.install(guide)` hooks it into the policy's own inference path."""

    def __init__(self, adapter, cfg: GuideConfig | None = None, cost_fn=None):
        self.adapter = adapter
        self.cfg = cfg or GuideConfig()
        # Optional user-defined constraint: cost_fn(a_hat, ctx, T) -> (J (B,) differentiable, violation (B,) in metres).
        # None = the built-in obstacle cost (cylinders + capsules of the Scene). The violation scales the push through
        # min(1, violation / ctx.cbf_vref) exactly like the built-in barrier.
        self.cost_fn = cost_fn
        self.ctx: ChunkCostContext | None = None  # set by the Supervisor / evaluator before each policy call
        self.escape_off = None  # deadlock escape: world-frame displacement offset per control step (or None)
        self.last_log = {}
        self._a1 = None  # policy's unguided intent for this chunk (a_hat at the first flow step)
        self._guided = []

    def reset_episode(self, seed=None):
        self.escape_off = None
        self.last_log = {}

    # ------------------------------------------------------------------ one policy call
    def run(self, observation, noise=None, num_steps=None, device=None):
        """Sample one action chunk (B, H, D) in the policy's normalised action space."""
        ad, cfg, ctx = self.adapter, self.cfg, self.ctx
        t_start = time.perf_counter()
        num_steps = ad.default_num_steps if num_steps is None else num_steps
        if noise is None:
            noise = ad.sample_noise(ad.batch_size(observation), device)
        dev = noise.device if device is None else device
        with torch.no_grad():
            cond = ad.prepare(observation, dev)
        T = ctx.to_torch(dev) if ctx is not None else None
        self._a1, self._guided = None, []
        x_t, costs, gnorms = self.sample(cond, noise, num_steps, ctx if cfg.mode == "g1" else None, T)
        log = {"sample_ms": (time.perf_counter() - t_start) * 1e3, "cost_per_step": costs, "gnorm_per_step": gnorms}
        if T is not None:
            with torch.no_grad():
                c_final, clear = obstacle_cost(x_t, ctx, T)
            log["cost_final"] = float(c_final)
            log["pred_min_clearance"] = float(clear.min())
            if "cap_a" in T and T["obs_centers"].shape[0] > 0:  # both tasks active: report the two parts
                from ..tasks import split_costs
                with torch.no_grad():
                    parts = split_costs(x_t, ctx, T)
                log["cost_static"], log["cost_dynamic"] = float(parts["static"]), float(parts["dynamic"])
            if self._a1 is not None:
                log.update(self._attribution(x_t, ctx, T))
            log["guided_first"] = self._guided[0] if self._guided else None
        self.last_log = log
        return x_t

    # ------------------------------------------------------------------ guided Euler integration
    def sample(self, cond, noise, num_steps, ctx, T, record=True):
        """Euler integration of the flow; G1 guidance applied when ctx is given. Returns (x, costs, gnorms).
        record=False skips the intent / guided-flag diagnostics."""
        ad, cfg, flow = self.adapter, self.cfg, self.adapter.flow
        device = noise.device
        bsize = noise.shape[0]
        dt = flow.dt(num_steps)
        sign = flow.descent_sign
        x_t = noise
        costs, gnorms = [], []
        for i in range(num_steps):
            t = flow.time(i, num_steps)
            tt = torch.full((bsize,), t, dtype=torch.float32, device=device)
            guide = ctx is not None and cfg.scale > 0 and flow.noise_level(t) > cfg.t_min
            mag = None
            if guide and cfg.grad_through_model:
                with torch.enable_grad():
                    x = x_t.detach().requires_grad_(True)
                    v = ad.velocity(cond, x, tt)
                    a_hat = flow.a_hat(x, t, v)
                    costs_k, clear_hat, vmax, _ = obstacle_costs_full(a_hat, ctx, T)
                    c = costs_k.sum()
                    g = torch.autograd.grad(c, x)[0] if c.requires_grad and c.item() > 0 else torch.zeros_like(x)
                v = v.detach()
                mag = guidance_magnitude(clear_hat.detach(), ctx, vmax.detach()).view(-1, 1, 1)
                if i == 0 and record:
                    self._a1 = flow.a_hat(x_t, t, v).detach()
            else:
                with torch.no_grad():
                    v = ad.velocity(cond, x_t, tt)
                if i == 0 and record:
                    self._a1 = flow.a_hat(x_t, t, v).detach()  # the policy's unguided intent for this chunk
                g = torch.zeros_like(x_t)
                if guide and self.cost_fn is not None:  # user-defined constraint (see class docstring)
                    with torch.enable_grad():
                        a_hat = flow.a_hat(x_t, t, v).detach().requires_grad_(True)
                        J, viol = self.cost_fn(a_hat, ctx, T)
                        c = J.sum()
                        if c.item() > 0:
                            g = torch.autograd.grad(c, a_hat)[0]
                    mag = (viol.detach().reshape(-1) / ctx.cbf_vref).clamp(max=1.0).view(-1, 1, 1)
                elif guide:
                    with torch.enable_grad():
                        a_hat = flow.a_hat(x_t, t, v).detach().requires_grad_(True)
                        costs_k, clear_hat, vmax, clr = obstacle_costs_full(a_hat, ctx, T)
                        c = costs_k.sum()
                        trig = True
                        if cfg.trigger_exec_only:  # ignore violations that exist only in the unexecuted tail
                            c_exec, _ = costs_from_clearance(clr[:, :, : ctx.exec_steps], ctx, T)
                            trig = float(c_exec.sum()) > 0
                        if trig and c.item() > 0:
                            g = torch.autograd.grad(c, a_hat)[0]
                    mag = guidance_magnitude(clear_hat.detach(), ctx, vmax.detach()).view(-1, 1, 1)
            v = self._apply_escape(v, T, sign)
            if guide:
                costs.append(float(c.item()))
                if record:
                    self._guided.append(bool(float(g.detach().abs().sum()) > 0))
                if bsize > 1:  # batched call: normalise each sample's gradient on its own
                    gn_b = g.float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
                    gnorms.append(float(gn_b.mean()))
                    if cfg.normalize_grad:
                        g = torch.where(gn_b > 0, g / (gn_b + 1e-20), torch.zeros_like(g))
                else:
                    gn = g.float().pow(2).mean().sqrt()
                    gnorms.append(float(gn))
                    if cfg.normalize_grad and gn > 0:
                        g = g / gn
                g = g * mag.to(g.dtype)
                lam = cfg.scale * (flow.progress(t) if cfg.schedule == "linear" else 1.0)
                if sign < 0:
                    lam = -lam
                v = v + lam * g.to(v.dtype)
            x_t = x_t + dt * v
        return x_t, costs, gnorms

    def _apply_escape(self, v, T, sign):
        """The Supervisor's deadlock escape: a world-frame displacement offset per control step (self.escape_off, (3,)),
        applied at every flow step so x moves by +|dt| * offset per step. Uses self.ctx (set even when the guidance
        itself is off)."""
        if T is None or self.ctx is None or self.escape_off is None:
            return v
        amap = self.ctx.map()
        pd = amap.position_dims
        v = v.clone()
        off_a = amap.world_offset_to_action(self.escape_off, T)
        v[..., pd] = v[..., pd] - (sign * off_a).to(v.dtype)
        return v

    # ------------------------------------------------------------------ diagnostics
    def _attribution(self, x_final, ctx, T):
        """What triggers guidance on this chunk, measured on the policy's unguided intent a1 with the reference
        d_safe hinge (comparable across methods): first violating step h_star, executed/tail violation, binding
        sphere group and its chunk-start clearance, and how the executed motion was changed (push_cm;
        brake_frac > 0 = pushed backwards along the intended direction)."""
        with torch.no_grad():
            p1 = chunk_eef_positions(self._a1, ctx, T)
            clr = sphere_clearance(sphere_positions(p1, T), T)[:, 0]  # (K, H, P), batch element 0
            viol = torch.relu(ctx.d_safe - clr)
            E = ctx.exec_steps
            out = {"viol_exec_cm": 100 * float(viol[:, :E].max()),
                   "viol_tail_cm": 100 * float(viol[:, E:].max()) if viol.shape[1] > E else 0.0}
            hv = (viol.amax(dim=(0, 2)) > 0).nonzero()
            out["h_star"] = int(hv[0]) if len(hv) else -1
            if out["h_star"] >= 0:
                Kc, Hc, Pc = viol.shape
                idx = int(torch.argmax(viol.reshape(-1)))
                kb, hb, pb = idx // (Hc * Pc), (idx // Pc) % Hc, idx % Pc
                grp = ctx.sphere_groups or SPHERE_GROUPS
                out["bind_group"] = grp[pb] if pb < len(grp) else f"s{pb}"
                out["bind_h"] = hb
                out["c0_bind_cm"] = 100 * float(start_clearance(T)[kb, pb])
            pf = chunk_eef_positions(x_final, ctx, T)
            delta = pf[0, E - 1, :2] - p1[0, E - 1, :2]
            D = p1[0, -1, :2] - T["p0"][:2]
            out["push_cm"] = 100 * float(delta.norm())
            out["intent_cm"] = 100 * float(D.norm())
            out["brake_frac"] = (float(-(delta @ (D / D.norm())) / delta.norm())
                                 if float(delta.norm()) > 1e-4 and float(D.norm()) > 0.02 else None)
        return out
