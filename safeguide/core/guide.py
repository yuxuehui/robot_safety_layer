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
                   obstacle_costs, obstacle_costs_full, sphere_clearance, SPHERE_GROUPS, sphere_positions,
                   start_clearance)
from .recovery import RecoveryConfig, recovery_reward


@dataclasses.dataclass
class GuideConfig:
    mode: str = "g1"  # "g1": guided | "none": plain sampling through the same integrator (paired baselines)
    scale: float = 1.0  # lam: push size in unit-RMS velocity units
    grad_through_model: bool = False  # True: d cost / d x_t through the velocity network (needs autograd through it)
    normalize_grad: bool = True  # rescale g to unit RMS per chunk (then scale is in "velocity units")
    t_min: float = 0.0  # only guide while the noise level is above t_min
    schedule: str = "const"  # "const" | "linear" (lam * progress: stronger near the data end)
    trigger_exec_only: bool = False  # ablation: guide only if an EXECUTED step (h < exec_steps) violates
    # "map back to the manifold": on-manifold first. onmanifold_n > 0: draw N unguided policy samples (one batched
    # integration); if any satisfies the barrier (J = 0) execute one of those (sample 0 if safe, else the safe sample
    # closest to sample 0's path) with NO push; otherwise push the safest sample with G1 and, if project_tau > 0,
    # re-noise the result to that level and denoise again with the policy's flow (restart projection).
    onmanifold_n: int = 0
    project_tau: float = 0.0

    @classmethod
    def sota(cls, **kw):
        """The configuration evaluated as 'inference-time guidance' (G1, unit push, gradient w.r.t. a_hat only)."""
        return cls(mode="g1", scale=1.0, grad_through_model=False, normalize_grad=True, **kw)


class Guide:
    """Guided sampler. Set `ctx` (ChunkCostContext) and optionally `escape_off` before every policy call; read
    `last_log` afterwards. `adapter.install(guide)` hooks it into the policy's own inference path."""

    def __init__(self, adapter, cfg: GuideConfig | None = None, recovery: RecoveryConfig | None = None, cost_fn=None):
        self.adapter = adapter
        self.cfg = cfg or GuideConfig()
        # Optional user-defined constraint: cost_fn(a_hat, ctx, T) -> (J (B,) differentiable, violation (B,) in metres).
        # None = the built-in obstacle cost (cylinders + capsules of the Scene). The violation scales the push through
        # min(1, violation / ctx.cbf_vref) exactly like the built-in barrier.
        self.cost_fn = cost_fn
        self.ctx: ChunkCostContext | None = None  # set by the Supervisor / evaluator before each policy call
        self.escape_off = None  # deadlock escape: world-frame displacement offset per control step (or None)
        self.recovery = recovery  # gated learned correction vector (Algorithm 1, line 10: gate * A(u)); None = off
        self.gate = False  # set by the Supervisor / evaluator before each policy call
        self.u = None  # learned world-frame offset per control step (m), per episode
        self.external_u = None  # offset chosen outside (simulator-branching lookahead in the evaluator); overrides _recover
        self._rec_off = None  # offset applied during the current integration (u while gated / held)
        self._rec_hold = 0
        self._rec_gen = None
        self.last_log = {}
        self._a1 = None  # policy's unguided intent for this chunk (a_hat at the first flow step)
        self._guided = []

    def reset_episode(self, seed=None):
        self.escape_off = None
        self.gate, self.u, self._rec_off, self._rec_hold, self.external_u = False, None, None, 0, None
        self.last_log = {}
        if self.recovery is not None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._rec_gen = torch.Generator(device=dev)
            self._rec_gen.manual_seed(int(seed if seed is not None else 0) + 7919)

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
        extra = {}
        if self.recovery is not None and T is not None:
            extra = self._recover(cond, noise, num_steps, ctx, T)  # sets self._rec_off for this integration
        if cfg.onmanifold_n > 0 and T is not None and cfg.mode == "g1":
            x_t, costs, gnorms, onm = self._onmanifold(cond, noise, num_steps, ctx, T)
            extra = extra | onm
        else:
            x_t, costs, gnorms = self.sample(cond, noise, num_steps, ctx if cfg.mode == "g1" else None, T)
        log = {"sample_ms": (time.perf_counter() - t_start) * 1e3, "cost_per_step": costs, "gnorm_per_step": gnorms} | extra
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
    def sample(self, cond, noise, num_steps, ctx, T, record=True, level=1.0):
        """Euler integration of the flow; G1 guidance applied when ctx is given. Returns (x, costs, gnorms).
        record=False for candidate rollouts (does not touch the intent / guided-flag diagnostics).
        level < 1: `noise` is a partially noised chunk at that noise level and the integration starts there."""
        ad, cfg, flow = self.adapter, self.cfg, self.adapter.flow
        device = noise.device
        bsize = noise.shape[0]
        dt = flow.dt(num_steps) if level == 1.0 else flow.dt_from(level, num_steps)
        sign = flow.descent_sign
        x_t = noise
        costs, gnorms = [], []
        for i in range(num_steps):
            t = flow.time(i, num_steps) if level == 1.0 else flow.time_from(level, i, num_steps)
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
                if bsize > 1:  # candidate batch: normalise each sample's gradient on its own
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
        """World-frame displacement offsets per control step, applied at every flow step so x moves by +|dt| * offset
        per step: the Supervisor's deadlock escape (self.escape_off, (3,)) and / or the learned recovery vector
        (self._rec_off, (3,) or (B, 3) for candidate batches). Uses self.ctx (set even when the guidance itself is off)."""
        if T is None or self.ctx is None or (self.escape_off is None and self._rec_off is None):
            return v
        amap = self.ctx.map()
        pd = amap.position_dims
        v = v.clone()
        if self.escape_off is not None:
            off_a = amap.world_offset_to_action(self.escape_off, T)
            v[..., pd] = v[..., pd] - (sign * off_a).to(v.dtype)
        if self._rec_off is not None:
            off_r = amap.world_offset_to_action(self._rec_off, T)
            if off_r.ndim == 2:
                off_r = off_r[:, None, :]
            v[..., pd] = v[..., pd] - (sign * off_r).to(v.dtype)
        return v

    # ------------------------------------------------------------------ on-manifold first (+ restart projection)
    def _onmanifold(self, cond, noise, num_steps, ctx, T):
        cfg, ad = self.cfg, self.adapter
        dev = noise.device
        N = cfg.onmanifold_n
        noises = torch.cat([noise] + [ad.sample_noise(1, dev) for _ in range(N - 1)], dim=0)
        cond_b = ad.expand_cond(cond, N)
        with torch.no_grad():
            xs, _, _ = self.sample(cond_b, noises, num_steps, None, T, record=False)  # N pure policy samples
            costs_k, clear = obstacle_costs(xs, ctx, T)
            J = costs_k.sum(0) if costs_k.shape[0] else torch.zeros(N, device=dev)
            P = chunk_eef_positions(xs, ctx, T)
        e = min(ctx.exec_steps, P.shape[1])
        safe = J <= 0
        self._a1 = xs[0:1].detach()  # the policy's own intent = sample 0
        log = {"onm_safe_frac": float(safe.float().mean()), "onm_J_min": float(J.min()), "onm_pushed": False}
        if bool(safe.any()):
            if bool(safe[0]):
                b = 0
            else:
                dev_b = ((P[:, :e] - P[0:1, :e]) ** 2).sum(-1).mean(-1)
                dev_b = torch.where(safe, dev_b, torch.full_like(dev_b, float("inf")))
                b = int(dev_b.argmin())
            log["onm_pick"] = b
            self._guided = [False]
            return xs[b:b + 1], [0.0], [], log
        # no safe policy sample: guide the safest one, then project back with a restart
        b = int(J.argmin())
        log["onm_pick"], log["onm_pushed"] = b, True
        x, costs, gnorms = self.sample(cond, noises[b:b + 1], num_steps, ctx, T)
        if cfg.project_tau > 0:
            eps = ad.sample_noise(1, dev)
            x_tau = ad.flow.renoise(x, eps, cfg.project_tau)
            n2 = max(1, int(round(cfg.project_tau * num_steps)))
            with torch.no_grad():
                c_before, _ = obstacle_cost(x, ctx, T)
            x, c2, g2 = self.sample(cond, x_tau, n2, ctx, T, record=False, level=cfg.project_tau)
            costs, gnorms = costs + c2, gnorms + g2
            log["onm_proj_steps"] = n2
            log["onm_cost_before_proj"] = float(c_before)
        return x, costs, gnorms, log

    # ------------------------------------------------------------------ candidate chunks for an external (lookahead) search
    def candidate_chunks(self, cond, noise, num_steps, ctx, T, u0=None, B=8, sigma=0.03, u_max=0.03, cands_in=None):
        """B candidate offsets around u0 (or the given offsets cands_in (B, 3)) and their guided chunks (B, H, D)."""
        dev = noise.device
        if self._rec_gen is None:
            self.reset_episode(0)
        if cands_in is not None:
            cands = torch.as_tensor(np.asarray(cands_in, dtype=np.float32), dtype=torch.float32, device=dev)
            B = cands.shape[0]
        else:
            u0 = torch.as_tensor(u0 if u0 is not None else np.zeros(3), dtype=torch.float32, device=dev)
            xi = torch.randn(B, 3, generator=self._rec_gen, device=dev)
            xi[0] = 0.0
            xi[1] = -u0 / sigma
            cands = u0 + sigma * xi
        n = cands.norm(dim=-1, keepdim=True)
        cands = torch.where(n > u_max, cands * (u_max / (n + 1e-12)), cands)
        cond_b = self.adapter.expand_cond(cond, B)
        noise_b = noise.expand(B, *noise.shape[1:]).clone()
        self._rec_off = cands
        with torch.no_grad():
            xs, _, _ = self.sample(cond_b, noise_b, num_steps, ctx if self.cfg.mode == "g1" else None, T, record=False)
        self._rec_off = None
        return cands, xs

    # ------------------------------------------------------------------ gated learned correction vector
    def _recover(self, cond, noise, num_steps, ctx, T):
        """Update the per-episode offset u when the gate is open (closed-form reward-weighted mean over B guided
        candidate rollouts, EMA across chunks) and decide the offset applied to this call. Returns log entries."""
        cfg, ad = self.recovery, self.adapter
        dev = noise.device
        log = {"rec_gate": bool(self.gate)}
        if self.external_u is not None:
            self._rec_off = np.asarray(self.external_u, dtype=float)
            log["rec_applied_cm"] = 100 * float(np.linalg.norm(self._rec_off))
            log["rec_external"] = True
            return log
        if self.gate:
            if self._rec_gen is None:
                self.reset_episode(0)
            u0 = torch.as_tensor(self.u if self.u is not None else np.zeros(3), dtype=torch.float32, device=dev)
            B = cfg.B
            xi = torch.randn(B, 3, generator=self._rec_gen, device=dev)
            xi[0] = 0.0  # keep the current u
            xi[1] = -u0 / cfg.sigma  # and the uncorrected chunk
            cands = u0 + cfg.sigma * xi
            n = cands.norm(dim=-1, keepdim=True)
            cands = torch.where(n > cfg.u_max, cands * (cfg.u_max / (n + 1e-12)), cands)
            cond_b = ad.expand_cond(cond, B)
            noise_b = noise.expand(B, *noise.shape[1:]).clone()  # common random numbers across candidates
            self._rec_off = cands
            with torch.no_grad():
                xs, _, _ = self.sample(cond_b, noise_b, num_steps, ctx if self.cfg.mode == "g1" else None, T, record=False)
                r, diag = recovery_reward(xs, cands, ctx, T, cfg)
                r_n = (r - r.mean()) / (r.std() + 1e-8)  # standardised: the weights do not depend on the reward scale
                w = torch.softmax(r_n / cfg.tau, dim=0)
                u_star = (w[:, None] * cands).sum(0)
            u_new = (1.0 - cfg.beta) * u0 + cfg.beta * u_star
            self.u = u_new.detach().cpu().numpy().astype(float)
            self._rec_hold = cfg.hold
            log.update(diag | {"rec_u_cm": 100 * float(np.linalg.norm(self.u)), "rec_u": (100 * self.u).round(2).tolist(),
                               "rec_ess": float(1.0 / (w**2).sum()), "rec_r_best": float(r.max()), "rec_r_u0": float(r[0]),
                               "rec_r_zero": float(r[1])})
        else:
            if self._rec_hold > 0:
                self._rec_hold -= 1
            elif self.u is not None:
                self.u = self.u * cfg.decay
                if np.linalg.norm(self.u) < 1e-3:
                    self.u = None
        self._rec_off = None if self.u is None or (not self.gate and self._rec_hold <= 0) else self.u
        log["rec_applied_cm"] = 0.0 if self._rec_off is None else 100 * float(np.linalg.norm(self._rec_off))
        return log

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
