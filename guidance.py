"""Research sampler for pi0.5 (openpi PyTorch): the G1 integrator plus the baselines / variants evaluated in the
method comparison (steer, trajectory optimal control, MPPI, CAR, best-of-N, projection). The deployable
configuration ("inference-time guidance") lives in the model-agnostic `safeguide` package, whose cost primitives
are imported here so both share one definition of clearance and cost.

pi0/pi0.5 flow convention: t=1 is noise, t=0 is data, x_t = t*eps + (1-t)*a, v = eps - a,
Euler with dt = -1/N. Hence at any step the predicted clean chunk is  a_hat = x_t - t*v_t.

G1 "lookahead" guidance (image-experiment analogue of gcovA): at every flow step,
  g = d cost(a_hat) / d x_t                      (through the action expert, or through a_hat only)
  v <- v + lam_t * g      so that  x_{t+dt} = x_t + dt*v  moves along -g  (dt < 0).

The cost lives in *executed-action* space: un-normalize exactly like openpi's quantile
Unnormalize, clip to the env's [-1, 1], integrate EEF positions with the calibrated gain G
(fitted on baseline rollouts: ~0.012 m per unit action per 20 Hz step; robosuite's nominal
output_max 0.05 overestimates the motion ~4x), attach gripper collision spheres
(offsets measured from the sim at chunk start) and penalise  relu(d_safe - SDF)^2.
"""
import dataclasses
import time

import numpy as np
import torch

from safeguide.core.cost import (SPHERE_GROUPS, ChunkCostContext, chunk_eef_positions, costs_from_clearance,  # noqa: F401
                                 guidance_magnitude, obstacle_cost, obstacle_costs, obstacle_costs_full,
                                 robot_clearance_per_step, sphere_clearance, sphere_positions, start_clearance,
                                 step_weights)
from safeguide.core.geometry import sdf_capsule_torch, sdf_cylinder_torch  # noqa: F401


def _smootherstep(x):
    x = x.clamp(0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


class CarResidual(torch.nn.Module):
    """g_psi(x_pos, t): velocity residual on the chunk's position dims, (B, H, 3) -> (B, H, 3).

    Same role as UNetVectorResidual in the CAR ManiSkill code, whose input is the absolute waypoint
    positions x[..., :3] (same frame as the obstacles) and t. Here x_t is a normalised delta-action,
    so the input is the chunk's implied EEF waypoints relative to the pillars' mean centre (/0.1 m),
    making g_psi a spatial field tied to the static obstacles as in the reference; the output stays a
    velocity residual in normalised action space. Output layer zero-initialised, so before any training
    step CAR reduces exactly to g^approx (= G1 with the a_hat gradient).
    """

    def __init__(self, horizon, hidden=256, n_freq=8):
        super().__init__()
        self.horizon = horizon
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * np.pi)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(horizon * 3 + 2 * n_freq, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, horizon * 3),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_pos, t):
        B = x_pos.shape[0]
        ang = t.view(B, 1).float() * self.freqs
        h = torch.cat([x_pos.reshape(B, -1).float(), torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.net(h).view(B, self.horizon, 3)


@dataclasses.dataclass
class GuidanceConfig:
    mode: str = "g1"  # "g1" (in-the-loop guidance) | "bestofn" | "project" (post-hoc) | "none"
    scale: float = 1.0  # lam
    grad_through_model: bool = True  # True: d cost/d x_t through v (full Jacobian); False: d cost/d a_hat
    normalize_grad: bool = True  # rescale g to unit RMS per chunk (then scale is in "velocity units")
    t_min: float = 0.0  # only guide while t > t_min
    schedule: str = "const"  # "const" | "linear" (lam * (1 - t)): stronger near the data end
    n_samples: int = 8  # bestofn: number of unguided samples to choose from
    project_steps: int = 20  # project: gradient steps on the finished chunk
    project_lr: float = 0.05  # project: step size in normalized-action units (per unit-RMS gradient)
    # "oc": trajectory-level optimal control over per-flow-step controls (G2)
    oc_iters: int = 10  # max optimisation iterations per chunk (each = 10 fwd + 10 bwd through the action expert)
    oc_lr: float = 0.05  # step on u per unit-RMS gradient (normalized-action units)
    oc_reg: float = 0.0  # lambda in  J = Phi(x_final)/d_safe^2 + lambda * sum_k ||u_k||^2
    oc_decay: float = 1.0  # u <- decay*u - lr*g  (image-code OC used 0.995)
    oc_skip_if_safe: bool = True  # leave chunks whose unguided sample is already collision-free untouched
    # "car": g^car = g^approx + w(conflict) * g_psi   (CAR guidance, ManiSkill reference config)
    car_batch: int = 64  # online_batch_size: guided rollouts per policy call for the g_psi update
    car_train_steps: int = 1  # online_train_steps per policy call
    car_lr: float = 1e-3  # online_lr (Adam)
    car_thr: float = 0.15  # conflict_threshold
    car_temp: float = 0.1  # conflict_temperature
    car_reward_temp: float = 1.0  # reward_temp: w_b = softmax(r1 / tau)
    car_w_obs: float = 1.0  # obstacle_reward_weight
    car_obs_scale: float = 0.5  # |energy_scales| per obstacle (reference config: [-0.5, -0.5])
    car_obs_sigma: float = 0.03  # m; energy = exp(-clearance^2 / sigma^2), bounded in [0, 1] like the reference
    car_w_goal: float = 10.0  # goal_reward_weight (goal = end point of the unguided chunk)
    car_goal_sigma: float = 0.05  # m (reference 0.5 in its own scene scale; chunk end points differ by cm here)
    car_corr_scale: float = 1.0  # learned_correction_scale
    car_hidden: int = 256
    car_zero_thr: float = 0.0  # pillar counts in the conflict score only if its energy exp(-c^2/s^2) >= thr
                               # (reference zero_gradient_threshold; 0 = any non-zero gradient)
    # "steer, don't brake" (G1 a_hat branch): tail-only braking pushes become sideways detours, side latched
    steer: bool = False
    steer_release: float = 0.10  # m past the pillar (along the latched direction) before the latch is released
    steer_forget: float = 0.35  # m away from the pillar -> release
    steer_probe: float = 0.03  # m lateral probe used to pick the side
    trigger_exec_only: bool = False  # ablation: guide only if an EXECUTED step (h < exec_steps) violates
    # CAR-v2 (defaults reproduce the reference-faithful CAR exactly)
    car_conflict: str = "pillars"  # "pillars" (reference) | "progress" | "both" (max of the two gates)
    car_prog_thr: float = 0.75  # progress gate threshold on kappa = (1 - cos(obstacle push, intent))/2
    car_explore: float = 0.0  # m per control step: lateral exploration of the training rollouts (xi ~ U(-1,1))
    car_reward: str = "ref"  # "ref" (obstacle energy + goal) | "progress" (energy + progress - lateral deviation)
    car_w_obs2: float = 5.0  # progress reward: obstacle energy weight
    car_w_prog: float = 2.0  # progress reward: forward progress along the intent
    car_w_lat: float = 0.5  # progress reward: lateral deviation penalty (l / 0.15 m)^2
    car_param: str = "mlp"  # "mlp": reference g_psi network, retrained per call | "vector": one constant action-space
    # correction vector per episode; the reward-weighted matching loss then has the closed-form minimiser (weighted
    # mean residual over gated steps), applied as an EMA across chunks so the correction accumulates over the episode
    car_vec_beta: float = 0.5  # EMA weight of the per-chunk closed-form solution
    # "mppi": sampling-based (gradient-free) trajectory optimisation of a low-dim steering control of the flow
    mppi_samples: int = 64  # rollouts per iteration (one batched flow integration)
    mppi_iters: int = 2
    mppi_sigma: float = 0.01  # m per control step: std of the steering-offset samples
    mppi_lambda: float = 0.5  # temperature of the MPPI weights exp(-(S - S_min) / lambda)
    mppi_w_obs: float = 1.0  # weight of the obstacle cost / scale (20: residual mm-level violations matter)
    mppi_w_prog: float = 1.0  # progress term (fraction of the intended displacement achieved)
    mppi_w_ctrl: float = 0.05  # control cost on |c|^2 / sigma^2 (0.25 over-regularised: 4/4 smoke contacts)
    mppi_w_dev: float = 0.0  # step-weighted squared EEF-path deviation from the unguided chunk / d_safe^2 (0.01-0.05 hurt in smoke)
    mppi_ramp: bool = True  # control = constant offset + linear ramp over the chunk (6 dims) vs constant (3 dims)
    mppi_persist: float = 0.0  # per-episode correction vector: u_ep <- (1-b) u_ep + b * c_mean (world frame); 0 = off
    mppi_persist_decay: float = 0.8  # u_ep decay on chunks where MPPI does not engage (unguided chunk already safe)


class GuidedSampler:
    """Drop-in replacement for policy._sample_actions (openpi Policy, PyTorch model)."""

    def __init__(self, model, cfg: GuidanceConfig):
        self.model = model
        self.cfg = cfg
        if (cfg.steer or cfg.trigger_exec_only) and (cfg.grad_through_model or cfg.mode != "g1"):
            raise ValueError("--steer/--trigger_exec_only are implemented in the G1 a_hat branch only: "
                             "use --guidance g1 --grad_through_model 0")
        self.ctx: ChunkCostContext | None = None  # set by the evaluator before each infer()
        self.last_log = {}
        self._car_d = None
        self.latch = {}  # steer: pillar k -> (side s, latched direction d_xy, pillar xy, radius)
        self._a1 = None  # policy's unguided intent for this chunk (a_hat at the first flow step)
        self._steer_stats = {}

    def _prefix(self, observation):
        m = self.model
        images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = m._prepare_attention_masks_4d(prefix_att_2d)
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = m.paligemma_with_expert.forward(
            attention_mask=att_4d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True,
        )
        return state, prefix_pad_masks, past_key_values

    def __call__(self, device, observation, noise=None, num_steps=10):
        m, cfg, ctx = self.model, self.cfg, self.ctx
        t_start = time.perf_counter()
        bsize = observation.state.shape[0]
        shape = (bsize, m.config.action_horizon, m.config.action_dim)
        if noise is None:
            noise = m.sample_noise(shape, device)
        with torch.no_grad():
            state, prefix_pad_masks, pkv = self._prefix(observation)
        T = ctx.to_torch(device) if ctx is not None else None
        extra = {}
        self._a1, self._steer_stats, self._guided = None, {}, []
        if cfg.steer and ctx is not None and self.latch:
            p0 = np.asarray(ctx.p0)[:2]
            for k in list(self.latch):
                _, d_lat, ck, rk = self.latch[k]
                rel = p0 - ck
                if float(rel @ d_lat) > rk + cfg.steer_release or float(np.linalg.norm(rel)) > cfg.steer_forget:
                    del self.latch[k]

        if cfg.mode == "bestofn" and T is not None:
            # N unguided samples sharing the prefix KV cache; execute the lowest-cost one.
            best, best_c, cs = None, None, []
            for k in range(cfg.n_samples):
                nk = noise if k == 0 else m.sample_noise(shape, device)
                x, _, _ = self._integrate(nk, state, prefix_pad_masks, pkv, num_steps, None, None)
                with torch.no_grad():
                    c, _ = obstacle_cost(x, ctx, T)
                cs.append(float(c))
                if best is None or float(c) < best_c:
                    best, best_c = x, float(c)
            x_t, costs, gnorms = best, cs, []
            extra["chosen_cost"] = best_c
        elif cfg.mode == "oc" and T is not None:
            x_t, costs, gnorms, extra = self._optimal_control(noise, state, prefix_pad_masks, pkv, num_steps, ctx, T)
        elif cfg.mode == "car" and T is not None:
            x_t, costs, gnorms, extra = self._car(noise, state, prefix_pad_masks, pkv, num_steps, ctx, T)
        elif cfg.mode == "mppi" and T is not None:
            x_t, costs, gnorms, extra = self._mppi(noise, state, prefix_pad_masks, pkv, num_steps, ctx, T)
        else:
            guide_ctx = ctx if cfg.mode == "g1" else None
            x_t, costs, gnorms = self._integrate(noise, state, prefix_pad_masks, pkv, num_steps, guide_ctx, T)
            if cfg.mode == "project" and T is not None:
                # Post-hoc projection baseline: push the finished chunk out of the obstacle.
                a = x_t.detach().clone()
                for _ in range(cfg.project_steps):
                    with torch.enable_grad():
                        a.requires_grad_(True)
                        c, _ = obstacle_cost(a, ctx, T)
                        if c.item() == 0:
                            break
                        g = torch.autograd.grad(c, a)[0]
                    a = (a - cfg.project_lr * g / (g.pow(2).mean().sqrt() + 1e-12)).detach()
                x_t = a

        log = {"sample_ms": (time.perf_counter() - t_start) * 1e3, "cost_per_step": costs, "gnorm_per_step": gnorms} | extra
        if T is not None:
            with torch.no_grad():
                c_final, clear = obstacle_cost(x_t, ctx, T)
            log["cost_final"] = float(c_final)
            log["pred_min_clearance"] = float(clear.min())
            if self._a1 is not None:
                log.update(self._attribution(x_t, ctx, T))
            log.update(self._steer_stats)
            log["guided_first"] = self._guided[0] if self._guided else None
        self.last_log = log
        return x_t

    def _optimal_control(self, noise, state, prefix_pad_masks, pkv, num_steps, ctx, T):
        """G2: optimise per-step controls u_k over the whole flow trajectory (OC-Flow style).

        Controlled dynamics (same form as the image code's OC, x <- x + u before the velocity):
            y_k = x_k + u_k,   x_{k+1} = y_k + dt * v(y_k, t_k),   k = 0..N-1,  x_0 = noise (t=1)
        Objective:  J(u) = Phi(x_N) / d_safe^2 + oc_reg * sum_k ||u_k||^2
        dJ/du is the adjoint gradient; computed exactly by autograd through all N action-expert calls.
        Descent (J is minimised):  u <- oc_decay * u - oc_lr * g / rms(g).
        Returns the lowest-Phi iterate (L_best); stops at the first collision-free chunk.
        """
        m, cfg = self.model, self.cfg
        device, B = noise.device, noise.shape[0]
        dt = -1.0 / num_steps
        times = [torch.full((B,), 1.0 - k / num_steps, dtype=torch.float32, device=device) for k in range(num_steps)]

        def rollout(u):
            x = noise
            for k in range(num_steps):
                y = x + u[k]
                x = y + dt * m.denoise_step(state, prefix_pad_masks, pkv, y, times[k])
            return x

        u = torch.zeros((num_steps,) + tuple(noise.shape), device=device)
        costs, gnorms = [], []
        extra = {"oc_cost0": None, "oc_iters_used": 0, "oc_best_it": 0, "oc_u_rms": 0.0}
        best_x, best_c = None, float("inf")
        for it in range(cfg.oc_iters + 1):
            need_update = it < cfg.oc_iters
            u_var = u.detach().requires_grad_(need_update)
            with torch.set_grad_enabled(need_update):
                x = rollout(u_var)
                c, clr_min = obstacle_cost(x, ctx, T)
            cval = float(c)
            costs.append(cval)
            # energy is > 0 everywhere: "safe" means clearance >= d_safe; otherwise cost == 0
            safe = float(clr_min.min()) >= ctx.d_safe if ctx.cost_type == "energy" else cval == 0.0
            if it == 0:
                extra["oc_cost0"] = cval
                self._a1 = x.detach()  # OC's intent = its unguided (u = 0) sample
            if cval < best_c:
                best_x, best_c = x.detach(), cval
                extra["oc_best_it"] = it
                extra["oc_u_rms"] = float(u.pow(2).mean().sqrt())
            if safe or not need_update:
                break
            J = c / ctx.d_safe**2 + cfg.oc_reg * u_var.pow(2).sum()
            g = torch.autograd.grad(J, u_var)[0]
            gn = g.pow(2).mean().sqrt()
            gnorms.append(float(gn))
            if gn == 0:
                break
            u = (cfg.oc_decay * u - cfg.oc_lr * g / gn).detach()
            extra["oc_iters_used"] = it + 1
        extra["oc_best_cost"] = best_c
        return best_x, costs, gnorms, extra

    # ------------------------------------------------------------------ MPPI over a steering control of the flow
    def _mppi(self, noise, state, prefix_pad_masks, pkv, num_steps, ctx, T):
        """G2-MPPI: Model Predictive Path Integral control, used (like G2-OC) as a trajectory optimiser over the
        flow, but gradient-free and over a low-dimensional control. The control c (world-frame EEF displacement
        offset per control step in the (forward, lateral, up) frame of the unguided chunk: constant [+ linear ramp
        over the chunk], 3 or 6 dims) is mapped to normalised action space with the calibrated gain and subtracted
        from the velocity at every flow step, so each candidate chunk is still generated by pi0.5. The realised
        chunk shift is gain * offset with a model-dependent gain between 1/N (the policy undoes it) and 1 (it
        does not), so it is measured and logged (mppi_shift_*) rather than assumed.

        Per iteration: sample c_i = c_mean + sigma * eps_i (eps_0 = 0), roll out all flows in one batch from the
        SAME noise and score
            S_i = w_obs * cost_i / scale + w_prog * relu(1 - progress_i) + w_ctrl |c_i|^2 / sigma^2 + w_dev * dev_i
        (scale = d_safe^2, or the unguided cost c0 for the energy cost; progress = fraction of the unguided
        chunk's xy displacement achieved, overshoot not rewarded; dev = step-weighted squared deviation of the
        EEF path from the unguided one / d_safe^2), weights w_i ~ exp(-(S_i - min S) / lambda),
        c_mean <- sum w_i c_i. The rollout of the final c_mean is scored too and the executed chunk is the one with
        the lower S among it and the best sample seen (iteration 0 sample 0 is the unguided chunk). Selection is by
        S alone, not feasible-first: with the CBF cost the smoothed mean is often marginally infeasible yet far
        better overall, and a feasible-first rule executes large single-sample swerves that derail the policy.
        Chunks whose unguided sample is already safe are left untouched."""
        import copy

        m, cfg = self.model, self.cfg
        device, H = noise.device, noise.shape[1]
        dt = -1.0 / num_steps
        with torch.no_grad():
            x0, _, _ = self._integrate(noise, state, prefix_pad_masks, pkv, num_steps, None, None)
            c0, clr0 = obstacle_cost(x0, ctx, T)
        safe0 = float(clr0.min()) >= ctx.d_safe if ctx.cost_type == "energy" else float(c0) == 0.0
        extra = {"mppi_cost0": float(c0), "mppi_skipped": bool(safe0)}
        if getattr(self, "car_gen", None) is None:
            self.car_reset()
        if safe0:
            if cfg.mppi_persist > 0:
                self.ep_vec = self.ep_vec * cfg.mppi_persist_decay
                extra["mppi_ep_vec_cm"] = 100 * float(np.linalg.norm(self.ep_vec))
            return x0, [float(c0)], [], extra
        P0 = chunk_eef_positions(x0, ctx, T)  # (1, H, 3)
        D = P0[0, -1, :2] - T["p0"][:2]
        nD_raw = float(D.norm())
        has_intent = nD_raw >= 0.02  # below 2 cm the unguided chunk has no meaningful direction of travel
        d = D / nD_raw if has_intent else torch.tensor([1.0, 0.0], device=device)
        nD = max(nD_raw, 0.02)
        w_prog = cfg.mppi_w_prog if has_intent else 0.0
        cost_scale = max(float(c0), 1e-8) if ctx.cost_type == "energy" else ctx.d_safe**2
        sw = step_weights(ctx, H, device)
        # world basis (forward d, lateral n, up z) -> normalised-action directions per unit metre per step
        W = torch.stack([torch.stack([d[0], d[1], torch.zeros((), device=device)]),
                         torch.stack([-d[1], d[0], torch.zeros((), device=device)]),
                         torch.tensor([0.0, 0.0, 1.0], device=device)])  # (3, 3) rows = world directions
        # 3x3 solves on the CPU: cuSOLVER handle creation fails on a nearly full GPU
        A = torch.linalg.solve(T["G"].cpu(), W.T.cpu()).T.to(device) * 2.0 / (T["q99"][:3] - T["q01"][:3] + 1e-6)  # (3, 3) rows in action space
        dim = 6 if cfg.mppi_ramp else 3
        ramp = (torch.arange(H, device=device, dtype=torch.float32) + 1.0) / H  # (H,)

        def offsets(c):  # c (K, dim) -> (K, H, 3) action-space velocity offsets
            per_h = c[:, None, :3] + (c[:, None, 3:6] * ramp[None, :, None] if cfg.mppi_ramp else 0.0)  # world, (K,H,3)
            return per_h @ A  # world coefficients (forward, lateral, up) -> action space

        K = cfg.mppi_samples
        pkv_b = copy.deepcopy(pkv)
        pkv_b.batch_repeat_interleave(K)
        state_b = state.expand(K, *state.shape[1:])
        pad_b = prefix_pad_masks.expand(K, *prefix_pad_masks.shape[1:])

        def rollout(c, st, pad, kv):
            off = offsets(c)
            x = noise.expand(c.shape[0], *noise.shape[1:]).clone()
            for k in range(num_steps):
                t = 1.0 - k / num_steps
                with torch.no_grad():
                    v = m.denoise_step(st, pad, kv, x, torch.full((c.shape[0],), t, device=device))
                v = v.clone()
                v[..., :3] = v[..., :3] - off.to(v.dtype)
                x = x + dt * v
            return x

        def score(c, xs):  # -> S (B,), cost (B,), feasible (B,)
            ck, clr = obstacle_costs(xs, ctx, T)
            cost = ck.sum(0)
            P = chunk_eef_positions(xs, ctx, T)
            prog = ((P[:, -1, :2] - T["p0"][:2]) * d).sum(-1) / nD
            dev = (sw * ((P - P0) ** 2).sum(-1)).sum(-1) / ctx.d_safe**2
            S = (cfg.mppi_w_obs * cost / cost_scale + w_prog * torch.relu(1.0 - prog)
                 + cfg.mppi_w_ctrl * (c**2).sum(-1) / cfg.mppi_sigma**2 + cfg.mppi_w_dev * dev)
            feas = clr >= ctx.d_safe if ctx.cost_type == "energy" else cost == 0.0
            return S, cost, feas

        gen = self.car_gen  # per-episode generator: keeps the global noise stream aligned with the other arms
        c_mean = torch.zeros(1, dim, device=device)
        if cfg.mppi_persist > 0:
            # warm start from the per-episode correction vector (world frame -> this chunk's (fwd, lat, up) frame)
            ev = torch.as_tensor(self.ep_vec, dtype=torch.float32, device=device)
            c_mean[0, 0] = ev[0] * d[0] + ev[1] * d[1]
            c_mean[0, 1] = -ev[0] * d[1] + ev[1] * d[0]
            c_mean[0, 2] = ev[2]
            extra["mppi_warm_cm"] = 100 * float(c_mean[0, :3].norm())
        costs_log = [float(c0)]
        best = None  # (feasible, S, cost, x, c) of the lowest-S sample over all iterations
        for it in range(cfg.mppi_iters):
            eps = torch.randn(K, dim, generator=gen, device=device)
            eps[0] = 0.0  # always include the current mean
            if it == 0:
                eps[1] = -c_mean[0] / cfg.mppi_sigma  # and the unguided chunk (c = 0), even when warm-started
            c = c_mean + cfg.mppi_sigma * eps
            with torch.no_grad():
                xs = rollout(c, state_b, pad_b, pkv_b)
                S, cost, feas = score(c, xs)
                i = int(S.argmin())
                cand = (bool(feas[i]), float(S[i]), float(cost[i]), xs[i : i + 1].clone(), c[i : i + 1].clone())
                if best is None or cand[1] < best[1]:
                    best = cand
                w = torch.softmax(-(S - S.min()) / cfg.mppi_lambda, dim=0)
                c_mean = (w[:, None] * c).sum(0, keepdim=True)
            costs_log.append(float(cost.min()))
            extra[f"mppi_ess_{it}"] = float(1.0 / (w**2).sum())
            extra[f"mppi_feas_frac_{it}"] = float(feas.float().mean())
        del pkv_b
        with torch.no_grad():
            x_mean = rollout(c_mean, state, prefix_pad_masks, pkv)
            S_m, cost_m, feas_m = score(c_mean, x_mean)
        if cfg.mppi_persist > 0:
            cm = c_mean[0, :3].detach().cpu().numpy().astype(float)
            world = np.array([cm[0] * float(d[0]) - cm[1] * float(d[1]), cm[0] * float(d[1]) + cm[1] * float(d[0]), cm[2]])
            b = cfg.mppi_persist
            self.ep_vec = (1.0 - b) * self.ep_vec + b * world
            extra["mppi_ep_vec_cm"] = 100 * float(np.linalg.norm(self.ep_vec))
            extra["mppi_ep_vec_lat_cm"] = 100 * float(-self.ep_vec[0] * float(d[1]) + self.ep_vec[1] * float(d[0]))
        mean_wins = float(S_m[0]) <= best[1]
        x_final, c_exec = (x_mean, c_mean) if mean_wins else (best[3], best[4])
        with torch.no_grad():
            cf, clrf = obstacle_cost(x_final, ctx, T)
            Pf = chunk_eef_positions(x_final, ctx, T)
        costs_log.append(float(cf))
        n = torch.stack([-d[1], d[0]])
        dP = (Pf - P0)[0]  # (H, 3)
        e = ctx.exec_steps - 1
        extra.update({"mppi_exec": "mean" if mean_wins else "best", "mppi_S_mean": float(S_m[0]), "mppi_S_best": best[1],
                      "mppi_feasible": bool(feas_m[0]) if mean_wins else best[0],
                      "mppi_cost_final": float(cf), "mppi_clr_final_cm": 100 * float(clrf.min()),
                      "mppi_c_cm": 100 * float(c_exec[0, :3].norm()), "mppi_c_lat_cm": 100 * float(c_exec[0, 1]),
                      "mppi_cmean_cm": 100 * float(c_mean[0, :3].norm()),
                      "mppi_shift_lat_cm": 100 * float(dP[e, :2] @ n), "mppi_shift_fwd_cm": 100 * float(dP[e, :2] @ d),
                      "mppi_shift_exec_cm": 100 * float(dP[: ctx.exec_steps].norm(dim=-1).max()),
                      "mppi_shift_end_cm": 100 * float(dP[-1].norm()), "mppi_intent": bool(has_intent)})
        return x_final, costs_log, [], extra

    # ------------------------------------------------------------------ CAR guidance
    def car_reset(self, seed=0):
        """Call at the start of every episode: g_psi is trained online within one episode only.
        CAR's training rollouts use their own RNG so the executed-chunk noise stream stays aligned
        with the other methods run with the same torch seed."""
        self.car_net, self.car_opt, self.car_vec = None, None, None
        self.latch = {}  # steer side latches are per episode too
        self.ep_vec = np.zeros(3)  # MPPI per-episode correction vector (world xyz displacement offset per step)
        self.escape_off = None  # deadlock escape: world-frame displacement offset per control step added to every flow step
        self.car_gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        self.car_gen.manual_seed(int(seed) + 7919)

    def _car_feat(self, x, ctx, T):
        """g_psi input: implied EEF waypoints of chunk x relative to the pillars' mean centre, / 0.1 m."""
        with torch.no_grad():
            ref = T["obs_centers"].mean(0) if T["obs_centers"].shape[0] > 0 else T["cap_b"][:, 0].mean(0)
            return (chunk_eef_positions(x, ctx, T) - ref) / 0.1

    def _car_gate(self, grads, valid=None):
        """Conflict score (1 - cos)/2 between per-pillar guidance directions, per (b, h), averaged
        over valid pairs (both gradients non-zero), and its smootherstep gate (formula as in
        GCovGuidance._compute_conflict_score + _smootherstep_gate).
        Deviation (documented): the per-pillar directions are cost gradients at the endpoint estimate
        a_hat = x_t - t v (w.r.t. normalised actions), whereas the reference scores the ODE state x_t,
        which here is a noisy normalised delta-action with no spatial meaning at early t."""
        cfg = self.cfg
        B, H = grads[0].shape[:2]
        conflict = torch.zeros(B, H, device=grads[0].device)
        npairs = torch.zeros(B, H, device=grads[0].device)
        gp = [g[..., :3].float() for g in grads]
        nrm = [g.norm(dim=-1) for g in gp]
        for i in range(len(gp)):
            for j in range(i + 1, len(gp)):
                ok_i, ok_j = nrm[i] > 1e-12, nrm[j] > 1e-12
                if valid is not None:  # reference zero_gradient_threshold: ignore far-away pillars
                    ok_i, ok_j = ok_i & valid[i], ok_j & valid[j]
                valid_ij = (ok_i & ok_j).float()
                cos = (gp[i] * gp[j]).sum(-1) / (nrm[i] * nrm[j] + 1e-20)
                conflict = conflict + (1.0 - cos) / 2.0 * valid_ij
                npairs = npairs + valid_ij
        conflict = conflict / (npairs + 1e-8)
        gate = _smootherstep((conflict - (cfg.car_thr - cfg.car_temp)) / (2.0 * cfg.car_temp + 1e-8))
        return gate, conflict

    def _car_velocity(self, x_t, t, v, ctx, T, explore=None):
        """v + g^approx + gate * g_psi  for a batch.  g^approx = sum_k d cost_k / d a_hat at the
        predicted endpoint a_hat = x_t - t v (no Jacobian through the model, as in the reference),
        normalised per sample exactly like G1 (and scaled by the cost's magnitude law).
        gate: reference pillar-pair conflict ('pillars'), CAR-v2 avoidance-vs-progress conflict ('progress':
        the obstacle push points backwards w.r.t. the policy's intended direction), or their max ('both').
        explore: optional (B, 3) per-rollout velocity offset on the position dims (training rollouts only)."""
        cfg = self.cfg
        B, H = x_t.shape[0], x_t.shape[1]
        with torch.enable_grad():
            a_hat = (x_t - t * v).detach().requires_grad_(True)
            p = chunk_eef_positions(a_hat, ctx, T)
            pl = p.detach().requires_grad_(True)
            clr = sphere_clearance(sphere_positions(pl, T), T)
            costs, vmax = costs_from_clearance(clr, ctx, T)
            qs, grads = [], []
            for k in range(costs.shape[0]):
                ck = costs[k].sum()
                if ck.item() > 0:
                    q = torch.autograd.grad(ck, pl, retain_graph=True)[0]
                    grads.append(torch.autograd.grad(p, a_hat, grad_outputs=q, retain_graph=True)[0])
                else:
                    q = torch.zeros_like(pl)
                    grads.append(torch.zeros_like(a_hat))
                qs.append(q)
        clear_hat = clr.detach().amin(dim=(0, 2, 3))
        g = sum(grads)
        gn = g.float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        g = torch.where(gn > 0, g / (gn + 1e-20), torch.zeros_like(g))
        g = g * guidance_magnitude(clear_hat, ctx, vmax.detach()).view(-1, 1, 1)
        g_app = cfg.scale * g.to(v.dtype)
        v_new = v + g_app
        gate = torch.zeros(B, H, device=v.device)
        conflict = torch.zeros(B, H, device=v.device)
        if cfg.car_conflict in ("pillars", "both") and len(grads) > 1:
            valid = None
            if cfg.car_zero_thr > 0:
                sig = ctx.energy_sigma if ctx.cost_type == "energy" else cfg.car_obs_sigma
                valid = (cfg.car_obs_scale * torch.exp(-clr.detach().amin(dim=-1).clamp(min=0.0) ** 2 / sig**2)
                         >= cfg.car_zero_thr)  # reference: |scale_k| * energy >= zero_gradient_threshold
            gate, conflict = self._car_gate(grads, valid)
        if cfg.car_conflict in ("progress", "both") and self._car_d is not None:
            # push on ACTION h = -dcost/da_h, which aggregates every later waypoint (p = p0 + cumsum(G a));
            # map it to the world displacement it causes: dp = G diag((q99-q01)/2) da
            ga = sum(grads)[..., :3].detach().float()  # (B, H, 3)
            scale_a = (T["q99"][:3] - T["q01"][:3]) / 2.0
            qxy = ((ga * scale_a) @ T["G"].T)[..., :2]  # world-frame gradient per action step; the push is -qxy
            nq = qxy.norm(dim=-1)
            cos = (-qxy * self._car_d.view(1, 1, 2)).sum(-1) / (nq + 1e-20)
            kappa = (1.0 - cos) / 2.0  # 1 = push straight against the intended direction
            g_prog = _smootherstep((kappa - (cfg.car_prog_thr - cfg.car_temp)) / (2.0 * cfg.car_temp + 1e-8)) * (nq > 1e-12)
            gate = torch.maximum(gate, g_prog)
            conflict = torch.maximum(conflict, kappa * (nq > 1e-12))
        corr = torch.zeros_like(v)
        r = None
        if cfg.car_param == "vector":
            if self.car_vec is not None:
                r = self.car_vec.view(1, 1, 3).expand(B, H, 3)
        elif self.car_net is not None:
            tt = torch.full((B,), t, device=v.device)
            r = self.car_net(self._car_feat(x_t, ctx, T), tt)  # (B, H, 3)
        if r is not None:
            corr[..., :3] = (cfg.car_corr_scale * gate[..., None] * r).to(v.dtype)
            v_new = v_new + corr
        if explore is not None:
            v_new = v_new.clone()
            v_new[..., :3] = v_new[..., :3] - explore[:, None, :].to(v_new.dtype)  # x moves by +|dt|*explore
        self._car_last = {"g_app": g_app.detach(), "corr": corr.detach()}
        return v_new, gate, conflict, costs

    def _car(self, noise, state, prefix_pad_masks, pkv, num_steps, ctx, T):
        """CAR guidance (reference: CAR-guidance 3d_pc_robot_manipulation GCovGuidance, online_loss_type
        'gradient'). Per policy call: (1) online update of g_psi on car_batch guided rollouts by
        reward-weighted guidance matching restricted to conflict regions; (2) sample the executed
        chunk with v + g^approx + gate * g_psi.

        openpi time runs t: 1 (noise) -> 0 (data), x_t = t eps + (1-t) a, so the conditional
        velocity towards a rollout's end point a = x_final is  v_cond = (x_t - x_final) / t
        (reference, t: 0 -> 1:  (x1 - x_t) / (1 - t)).
        """
        import copy

        m, cfg = self.model, self.cfg
        device = noise.device
        dt = -1.0 / num_steps
        H = noise.shape[1]
        if getattr(self, "car_gen", None) is None:
            self.car_reset()
        if cfg.car_param == "mlp":
            if getattr(self, "car_net", None) is None:
                self.car_net = CarResidual(H, cfg.car_hidden).to(device)
            # fresh Adam on every policy call, as in the reference (train_model creates it per call)
            self.car_opt = torch.optim.Adam(self.car_net.parameters(), lr=cfg.car_lr)
        extra = {}

        # goal for the task-progress reward: end point of an unguided chunk (the policy's intent)
        with torch.no_grad():
            x_ref, _, _ = self._integrate(noise, state, prefix_pad_masks, pkv, num_steps, None, None)
            P_ref = chunk_eef_positions(x_ref, ctx, T)
            p_goal = P_ref[0, -1]
            D = P_ref[0, -1, :2] - T["p0"][:2]
            nD = float(D.norm())
            d_int = D / max(nD, 1e-9)
            n_int = torch.stack([-d_int[1], d_int[0]])
            self._car_d = d_int if nD >= 0.02 else None  # progress gate needs a meaningful intent
            # lateral unit direction mapped to normalised-action space: dp = G diag((q99-q01)/2) da
            n3 = torch.stack([n_int[0], n_int[1], torch.zeros((), device=device)])
            n_act = torch.linalg.solve(T["G"].cpu(), n3.cpu()).to(device) * 2.0 / (T["q99"][:3] - T["q01"][:3] + 1e-6)

        # (1) online g_psi update
        Bt = cfg.car_batch
        pkv_b = copy.deepcopy(pkv)
        pkv_b.batch_repeat_interleave(Bt)
        state_b = state.expand(Bt, *state.shape[1:])
        pad_b = prefix_pad_masks.expand(Bt, *prefix_pad_masks.shape[1:])
        for _ in range(cfg.car_train_steps):
            x = torch.randn(tuple([Bt] + list(noise.shape[1:])), generator=self.car_gen, device=device,
                            dtype=noise.dtype)
            explore = None
            if cfg.car_explore > 0:
                xi = torch.rand(Bt, generator=self.car_gen, device=device) * 2.0 - 1.0
                explore = xi[:, None] * cfg.car_explore * n_act[None, :]  # (Bt, 3)
            xs, vs, gates, ts, feats = [], [], [], [], []
            for k in range(num_steps):
                t = 1.0 - k / num_steps
                with torch.no_grad():
                    v = m.denoise_step(state_b, pad_b, pkv_b, x, torch.full((Bt,), t, device=device))
                    v_g, gate, _, _ = self._car_velocity(x, t, v, ctx, T, explore=explore)
                xs.append(x[..., :3].float()); vs.append(v[..., :3].float()); gates.append(gate); ts.append(t)
                feats.append(self._car_feat(x, ctx, T))
                x = (x + dt * v_g).detach()
            x_final = x
            with torch.no_grad():
                # reference terminal reward: -w_obs * sum_k |scale_k| * max_h energy_k(h) + w_goal * goal
                clear_kh = robot_clearance_per_step(x_final, ctx, T)  # (K, Bt, H)
                energy = torch.exp(-clear_kh.clamp(min=0.0) ** 2 / cfg.car_obs_sigma**2)
                r_obs = -(cfg.car_obs_scale * energy.amax(dim=-1)).sum(0)
                p_end = chunk_eef_positions(x_final, ctx, T)[:, -1]
                lat = ((p_end[:, :2] - p_goal[:2]) * n_int).sum(-1)  # lateral deviation from the unguided end point
                if cfg.car_reward == "progress":
                    prog = (((p_end[:, :2] - T["p0"][:2]) * d_int).sum(-1) / max(nD, 0.02)).clamp(-1.0, 1.0)
                    w_prog = cfg.car_w_prog if nD >= 0.02 else 0.0  # min(prog, 1): overshoot is not rewarded
                    r1 = (-cfg.car_w_obs2 * energy.amax(dim=-1).sum(0) + w_prog * prog
                          - cfg.car_w_lat * (lat / 0.15) ** 2)
                else:
                    r_goal = torch.exp(-((p_end - p_goal) ** 2).sum(-1) / cfg.car_goal_sigma**2)
                    r1 = cfg.car_w_obs * r_obs + cfg.car_w_goal * r_goal
                w_b = torch.softmax(r1 / cfg.car_reward_temp, dim=0)  # (Bt,)
                ess_i, lat_i = float(1.0 / (w_b**2).sum()), 100 * float(lat.std())
            gate_all = torch.stack(gates)  # (N, Bt, H)
            active = float((gate_all > 0.5).float().mean())
            extra.update({"car_active_ratio": active, "car_r1_max": float(r1.max()), "car_r1_mean": float(r1.mean())})
            if active < 1e-6:
                extra["car_loss"] = None
                continue
            X = torch.stack(xs)  # (N, Bt, H, 3)
            V = torch.stack(vs)
            Fx = torch.stack(feats)  # g_psi inputs at the stored states
            tvec = torch.tensor(ts, device=device).view(-1, 1, 1, 1)
            v_cond = (X - x_final[None, ..., :3].float()) / tvec
            N = X.shape[0]
            if cfg.car_param == "vector":
                # closed-form minimiser of the same reward-weighted matching loss for a constant vector, then EMA
                with torch.no_grad():
                    wgt = (gate_all[..., None] * w_b.view(1, Bt, 1, 1))  # (N, Bt, H, 1)
                    vec_star = (wgt * (v_cond - V)).sum(dim=(0, 1, 2)) / (wgt.sum() + 1e-8)  # (3,)
                    old_v = self.car_vec if self.car_vec is not None else torch.zeros_like(vec_star)
                    self.car_vec = (1.0 - cfg.car_vec_beta) * old_v + cfg.car_vec_beta * vec_star
                    pred = self.car_vec.view(1, 1, 1, 3) + V
                    loss = ((pred - v_cond) ** 2 * wgt).sum() / (wgt.sum() * 3 + 1e-8)
                extra["car_vec_norm"] = float(self.car_vec.norm())
                extra["car_vec_star_norm"] = float(vec_star.norm())
                # for visualisation: the vector in normalised action units and the world-frame displacement per control
                # step it implies at full gate (v <- v + r moves the clean action by -t r, so the world direction is -G r)
                extra["car_vec"] = [float(u) for u in self.car_vec]
                if isinstance(T, dict) and "G" in T:
                    Gm = T["G"].to(self.car_vec.device, self.car_vec.dtype)
                    raw = self.car_vec * (T["q99"][:3] - T["q01"][:3]).to(self.car_vec) / 2.0  # normalised -> raw delta units
                    extra["car_vec_world_cm"] = [float(u) for u in (-100.0 * Gm @ raw)]
            else:
                g_psi = self.car_net(Fx.reshape(N * Bt, H, 3), tvec.view(-1).repeat_interleave(Bt)).view(N, Bt, H, 3)
                pred = g_psi + V
                wgt = (gate_all[..., None] * w_b.view(1, Bt, 1, 1)).expand_as(pred)
                loss = ((pred - v_cond) ** 2 * wgt).sum() / (wgt.sum() + 1e-8)
                self.car_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.car_net.parameters(), max_norm=1.0)
                self.car_opt.step()
            extra["car_loss"] = float(loss)
            extra["car_ess"], extra["car_lat_spread_cm"] = ess_i, lat_i  # logged for trained steps only
        del pkv_b

        # (2) executed chunk
        x = noise
        costs_log, gate_log, gate_frac, ratios = [], [], [], []
        for k in range(num_steps):
            t = 1.0 - k / num_steps
            with torch.no_grad():
                v = m.denoise_step(state, prefix_pad_masks, pkv, x, torch.full((1,), t, device=device))
                v_g, gate, conflict, costs = self._car_velocity(x, t, v, ctx, T)
                v_g = self._apply_escape(v_g, T)
            costs_log.append(float(costs.sum()))
            gate_log.append(float(gate.max()))
            gate_frac.append(float((gate > 0.5).float().mean()))
            ga, co = self._car_last["g_app"], self._car_last["corr"]
            if float(ga.norm()) > 0 and float(gate.max()) > 0.5:  # only where the gate is open
                ratios.append(float(co.norm() / ga.norm()))
            x = x + dt * v_g
        extra.update({"car_gate_max": max(gate_log), "car_gate_steps": sum(g > 0.5 for g in gate_log),
                      "car_gate_frac": float(np.mean(gate_frac)),
                      "car_resid_ratio": float(np.median(ratios)) if ratios else None})
        return x, costs_log, [], extra

    # ------------------------------------------------------------------ steer, don't brake
    def _intent_frame(self, ctx, T):
        """Unit xy direction d of the policy's intended chunk (a1 end point - p0), lateral n = z x d,
        and a validity flag (intent must move >= 2 cm)."""
        P1 = chunk_eef_positions(self._a1, ctx, T)
        D = P1[:, -1, :2] - T["p0"][:2]
        nD = D.norm(dim=-1, keepdim=True)
        d = D / nD.clamp(min=1e-9)
        n = torch.stack([-d[:, 1], d[:, 0]], dim=-1)
        return d, n, nD.squeeze(-1) >= 0.02

    def _latch_side(self, k, p, ctx, T, d, n):
        """Side s in {+1, -1} (descent direction s*n) for pillar k, chosen once and latched in the world frame.
        Probe the whole chunk shifted +-steer_probe along n and take the cheaper side; near-tie (within 5%):
        gate -> towards the other pillar (into the gap), single pillar -> away from the nearest task object."""
        cfg = self.cfg
        with torch.no_grad():
            cs = []
            for sgn in (+1.0, -1.0):
                ps = p.clone()
                ps[..., :2] = ps[..., :2] + sgn * cfg.steer_probe * n[:, None, :]
                ck, _ = costs_from_clearance(sphere_clearance(sphere_positions(ps, T), T), ctx, T)
                cs.append(float(ck[k].sum()))
        c_p, c_m = cs
        ck_xy = T["obs_centers"][k, :2].cpu().numpy()
        n0 = n[0].cpu().numpy()
        if abs(c_p - c_m) > 0.05 * max(c_p, c_m) + 1e-12:
            s = 1.0 if c_p < c_m else -1.0
        elif T["obs_centers"].shape[0] > 1:
            j = 1 - k if T["obs_centers"].shape[0] == 2 else 0
            to_other = T["obs_centers"][j, :2].cpu().numpy() - ck_xy
            s = 1.0 if float(to_other @ n0) >= 0 else -1.0
        elif ctx.obj_xy is not None and len(ctx.obj_xy) > 0:
            o = np.asarray(ctx.obj_xy)[np.argmin(np.linalg.norm(np.asarray(ctx.obj_xy) - ck_xy, axis=1))]
            s = -1.0 if float((o - ck_xy) @ n0) > 0 else 1.0  # nearest object on the +n side -> detour to -n
        else:
            s = 1.0
        # store the side in the WORLD frame: side s relative to the lateral axis of the latched direction d
        self.latch[k] = (s, d[0].detach().cpu().numpy(), ck_xy, float(T["obs_radii"][k]))
        return s

    def _steer_grad(self, a_hat, ctx, T):
        """G1 gradient with 'steer, don't brake' (spec #1 / toy reference Steer):
        per pillar k, in the frame of its latched direction d_k (current intent if not yet latched):
        on unexecuted tail steps, a position-space push pointing backwards along d_k (braking) becomes a push
        of the same size towards the latched side; other tail pushes get their lateral part flipped to that
        side; executed steps keep the true gradient. Each pillar is mapped back to actions with its own exact
        VJP and the result is normalised by the pre-cancellation magnitude gn_pre = RMS(sum_k |g_k|), so two
        opposing pillar pushes cancel instead of being rescaled to full size."""
        cfg = self.cfg
        p = chunk_eef_positions(a_hat, ctx, T)
        pl = p.detach().requires_grad_(True)
        clr = sphere_clearance(sphere_positions(pl, T), T)
        costs, vmax = costs_from_clearance(clr, ctx, T)
        K, H = costs.shape[0], pl.shape[1]
        if cfg.trigger_exec_only:
            c_exec, _ = costs_from_clearance(clr.detach()[:, :, : ctx.exec_steps], ctx, T)
            if float(c_exec.sum()) == 0.0:
                self._steer_stats.setdefault("steer_brake_steps", 0)
                self._steer_stats["latches"] = len(self.latch)
                z = torch.zeros_like(a_hat)
                return z, costs.sum(), clr.amin(dim=(0, 2, 3)), vmax, torch.zeros((), device=a_hat.device)
        d_cur, _, ok = self._intent_frame(ctx, T)
        tail = torch.arange(H, device=pl.device) >= ctx.exec_steps
        gks, n_brake = [], 0
        for k in range(K):
            ck = costs[k].sum()
            if ck.item() <= 0:
                continue
            q = torch.autograd.grad(ck, pl, retain_graph=True)[0]  # (B, H, 3)
            if bool(ok.any()):
                if k in self.latch:
                    d_k = torch.as_tensor(self.latch[k][1], device=pl.device, dtype=pl.dtype).expand_as(d_cur)
                else:
                    d_k = d_cur
                n_k = torch.stack([-d_k[:, 1], d_k[:, 0]], dim=-1)
                qa = (q[..., :2] * d_k[:, None, :]).sum(-1)  # along-direction part (descent -q: qa > 0 brakes)
                ql = (q[..., :2] * n_k[:, None, :]).sum(-1)  # lateral part
                brake = (qa > 0) & tail[None, :] & ok[:, None]
                if k not in self.latch and bool(brake.any()):
                    self._latch_side(k, pl.detach(), ctx, T, d_k, n_k)
                if k in self.latch:
                    sk = self.latch[k][0]
                    mag = torch.sqrt(qa**2 + ql**2)
                    ql_new = torch.where(brake, -sk * mag, torch.where(tail[None, :] & (ql * sk > 0), -ql, ql))
                    qa_new = torch.where(brake, torch.zeros_like(qa), qa)
                    qxy = qa_new[..., None] * d_k[:, None, :] + ql_new[..., None] * n_k[:, None, :]
                    q = torch.cat([qxy, q[..., 2:]], dim=-1)
                    n_brake += int(brake.sum())
            gks.append(torch.autograd.grad(p, a_hat, grad_outputs=q, retain_graph=True)[0])
        self._steer_stats = {"steer_brake_steps": self._steer_stats.get("steer_brake_steps", 0) + n_brake,
                             "latches": len(self.latch),
                             "latch_sides": {int(k): self.latch[k][0] for k in self.latch}}
        if not gks:
            return torch.zeros_like(a_hat), costs.sum(), clr.amin(dim=(0, 2, 3)), vmax, torch.zeros((), device=a_hat.device)
        g = sum(gks)
        gn_pre = sum(gk.abs() for gk in gks).float().pow(2).mean().sqrt()
        return g, costs.sum(), clr.amin(dim=(0, 2, 3)), vmax, gn_pre

    def _attribution(self, x_final, ctx, T):
        """What triggers guidance on this chunk, measured on the policy's unguided intent a1 with the
        reference 3 cm hinge (comparable across arms): first violating step h_star, executed/tail violation,
        binding sphere group and its chunk-start clearance, and how the executed motion was changed
        (push_cm; brake_frac > 0 = pushed backwards along the intended direction)."""
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

    def _apply_escape(self, v, T):
        """Deadlock escape (eval-driven): shift the chunk by a fixed world-frame displacement offset per control
        step, applied at every flow step (x moves by +|dt| * offset per step)."""
        if T is None or getattr(self, "escape_off", None) is None:
            return v
        off = torch.as_tensor(self.escape_off, dtype=torch.float32, device=v.device)
        off_a = torch.linalg.solve(T["G"].cpu(), off.cpu()).to(v.device) * 2.0 / (T["q99"][:3] - T["q01"][:3] + 1e-6)
        v = v.clone()
        v[..., :3] = v[..., :3] - off_a.to(v.dtype)
        return v

    def _integrate(self, noise, state, prefix_pad_masks, pkv, num_steps, ctx, T):
        """Euler integration t: 1 -> 0; G1 guidance applied when ctx is given."""
        m, cfg = self.model, self.cfg
        device = noise.device
        bsize = noise.shape[0]
        dt = -1.0 / num_steps
        x_t = noise
        costs, gnorms = [], []
        for i in range(num_steps):
            t = 1.0 - i / num_steps
            tt = torch.full((bsize,), t, dtype=torch.float32, device=device)
            guide = ctx is not None and cfg.scale > 0 and t > cfg.t_min
            mag, gn_pre = None, None
            if guide and cfg.grad_through_model:
                with torch.enable_grad():
                    x = x_t.detach().requires_grad_(True)
                    v = m.denoise_step(state, prefix_pad_masks, pkv, x, tt)
                    a_hat = x - t * v
                    costs_k, clear_hat, vmax, _ = obstacle_costs_full(a_hat, ctx, T)
                    c = costs_k.sum()
                    g = torch.autograd.grad(c, x)[0] if c.requires_grad and c.item() > 0 else torch.zeros_like(x)
                v = v.detach()
                mag = guidance_magnitude(clear_hat.detach(), ctx, vmax.detach()).view(-1, 1, 1)
                if i == 0:
                    self._a1 = (x_t - t * v).detach()
            else:
                with torch.no_grad():
                    v = m.denoise_step(state, prefix_pad_masks, pkv, x_t, tt)
                if i == 0:
                    self._a1 = (x_t - t * v).detach()  # the policy's unguided intent for this chunk
                g = torch.zeros_like(x_t)
                if guide:
                    with torch.enable_grad():
                        a_hat = (x_t - t * v).detach().requires_grad_(True)
                        if cfg.steer:
                            g, c, clear_hat, vmax, gn_pre = self._steer_grad(a_hat, ctx, T)
                        else:
                            costs_k, clear_hat, vmax, clr = obstacle_costs_full(a_hat, ctx, T)
                            c = costs_k.sum()
                            trig = True
                            if cfg.trigger_exec_only:  # ignore violations that exist only in the unexecuted tail
                                c_exec, _ = costs_from_clearance(clr[:, :, : ctx.exec_steps], ctx, T)
                                trig = float(c_exec.sum()) > 0
                            if trig and c.item() > 0:
                                g = torch.autograd.grad(c, a_hat)[0]
                    mag = guidance_magnitude(clear_hat.detach(), ctx, vmax.detach()).view(-1, 1, 1)
            v = self._apply_escape(v, T)
            if guide:
                costs.append(float(c.item()))
                self._guided.append(bool(float(g.detach().abs().sum()) > 0))
                gn = g.float().pow(2).mean().sqrt()
                gnorms.append(float(gn))
                den = gn_pre if gn_pre is not None else gn  # steer: pre-cancellation magnitude
                if cfg.normalize_grad and den > 0:
                    g = g / den
                g = g * mag.to(g.dtype)
                lam = cfg.scale * ((1.0 - t) if cfg.schedule == "linear" else 1.0)
                v = v + lam * g.to(v.dtype)
            x_t = x_t + dt * v
        return x_t, costs, gnorms


def install_guidance(policy, cfg: GuidanceConfig):
    """Swap the (torch.compiled) sampler of an openpi PyTorch Policy for a GuidedSampler."""
    model = policy._model
    model.eval()
    sampler = GuidedSampler(model, cfg)
    policy._sample_actions = sampler
    return sampler

