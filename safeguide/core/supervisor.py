"""Per-chunk state machine around the guided sampler: builds the cost context from the robot and the scene and
runs the two episode-level heuristics of the 'inference-time guidance' configuration:

  arm-margin release   once the end effector is past a gate line (+ hysteresis, world-latched for the episode),
                       arm spheres switch to `arm_margin_post` (the arm may then brush the pillars it has passed)
  deadlock escape      guidance active on the last `escape_chunks` chunks and the EEF moved < `escape_min_disp`
                       -> lift the chunk by `escape_lift` per control step for `escape_len` chunks

Call order per policy call:   ctx = sup.before_chunk(gripper_closed);  guide.ctx = ctx; guide.escape_off = sup.escape_off
                              chunk = policy(...);  sup.after_chunk(guide.last_log)
"""
import dataclasses

import numpy as np

from .cost import ActionMap, ChunkCostContext


@dataclasses.dataclass
class SupervisorConfig:
    exec_steps: int = 5  # control steps executed per chunk (replan interval)
    d_safe: float = 0.03
    cost_type: str = "cbf"
    energy_sigma: float = 0.05
    margin_grip: float | None = 0.010  # per-sphere margins (None -> d_safe)
    margin_arm: float | None = 0.015
    margin_obj: float | None = None  # held-object spheres (None -> margin_grip)
    cbf_gamma: float = 0.3
    cbf_vref: float = 0.01
    include_arm: bool = True  # model forearm/wrist links, not only the gripper
    arm_motion: str = "jac"  # "jac": Jacobian-based sphere motion | "alpha": heuristic alpha * dEEF
    model_object: bool = True  # add the held task object's spheres to the robot model
    arm_release: bool = True  # post-gate arm-margin release
    arm_margin_post: float = 0.0
    release_hyst: float = 0.03  # m past the gate line before the release latches
    escape_chunks: int = 4  # 0 = off
    escape_min_disp: float = 0.02
    escape_lift: float = 0.015  # m per control step
    escape_len: int = 2  # chunks per escape
    # deadlock gate: "disp" = guidance active on ALL of the last escape_chunks chunks and net EEF displacement < escape_min_disp
    # (the pi0.5 configuration); "intent" = guidance active on ANY of them, the policy's own intent has collapsed
    # (median |a_hat end - p0| over those chunks < escape_intent) and the net displacement is < escape_min_disp:
    # catches the dithering stalls of GR00T that never freeze completely
    escape_mode: str = "disp"  # "disp" | "intent" | "window" (see _deadlocked)
    escape_intent: float = 0.04  # m per chunk
    escape_ratio: float = 2.0  # window mode: path length over the window >= ratio * net displacement (oscillation)
    escape_clear: float = 0.0  # intent mode: gate only when every obstacle is farther than this from the robot now (0 = ignore)
    recovery: bool = False  # True: the deadlock gate drives the learned correction vector (Guide.gate) instead of the fixed lift

    @classmethod
    def pillar_sota(cls, **kw):
        """Static obstacles: CBF + held object + Jacobian + release + escape (the evaluated 'inference-time guidance')."""
        return cls(**kw)

    @classmethod
    def hand_sota(cls, **kw):
        """Moving human hand: same cost, no gate line to release on, escape off (it lifted the gripper 40-50 cm
        while waiting for the hand to pass)."""
        return cls(arm_release=False, escape_chunks=0, **kw)


def clearance_now(ctx: ChunkCostContext):
    """Min over robot spheres and obstacles of the current clearance (m): cylinders and the capsules' current pose."""
    from .geometry import sdf_capsule_np, sdf_cylinder_np

    pts = ctx.p0 + ctx.offsets
    vals = []
    for k in range(len(ctx.obs_radii)):
        vals.append(float((sdf_cylinder_np(pts, ctx.obs_centers[k], ctx.obs_radii[k], ctx.obs_half_heights[k]) - ctx.radii).min()))
    if ctx.cap_a is not None:
        for k in range(len(ctx.cap_r)):
            vals.append(float((sdf_capsule_np(pts, ctx.cap_a[k, 0], ctx.cap_b[k, 0], ctx.cap_r[k]) - ctx.radii).min()))
    return min(vals) if vals else None


def build_ctx(robot, scene, action_map: ActionMap, cfg: SupervisorConfig, margin_arm=None, held=None, horizon=10):
    """Cost context for the chunk that starts now, from the live robot state and the scene."""
    pts, radii, alphas, groups, motion = robot.points(include_arm=cfg.include_arm, held=held, arm_motion=cfg.arm_motion)
    cyl_c, cyl_r, cyl_h = scene.cylinders()
    caps = scene.capsules(horizon)
    cap_a, cap_b, cap_r = caps if caps is not None else (None, None, None)
    p0 = robot.eef_pos()
    m_grip = cfg.d_safe if cfg.margin_grip is None else cfg.margin_grip
    m_arm_cfg = cfg.margin_arm if margin_arm is None else margin_arm
    m_arm = cfg.d_safe if m_arm_cfg is None else m_arm_cfg
    m_obj = m_grip if cfg.margin_obj is None else cfg.margin_obj
    margins = np.array([m_grip if g in ("grip", "hand", "hand_w") else m_obj if g == "obj" else m_arm for g in groups])
    return ChunkCostContext(
        p0=p0, offsets=pts - p0, radii=radii, alphas=alphas,
        obs_centers=cyl_c, obs_radii=cyl_r, obs_half_heights=cyl_h, action_map=action_map,
        d_safe=cfg.d_safe, exec_steps=cfg.exec_steps, cost_type=cfg.cost_type, energy_sigma=cfg.energy_sigma,
        margins=margins, cbf_gamma=cfg.cbf_gamma, cbf_vref=cfg.cbf_vref, sphere_groups=groups, motion=motion,
        cap_a=cap_a, cap_b=cap_b, cap_r=cap_r,
    )


class Supervisor:
    def __init__(self, robot, scene, action_map: ActionMap, cfg: SupervisorConfig | None = None, horizon=10):
        self.robot, self.scene, self.action_map = robot, scene, action_map
        self.cfg = cfg or SupervisorConfig()
        self.horizon = horizon
        self.reset()

    def reset(self, gate_line=None):
        """gate_line: (c_xy, d_xy, offset) - the EEF is 'past the gate' when (p_xy - c) . d > offset (+ hysteresis)."""
        self.gate = gate_line
        self.post_gate, self.post_gate_chunks = False, 0
        self.hist = []  # (EEF position at chunk start, guidance was active on that chunk)
        self.escape_left, self.escape_events, self.escape_off = 0, 0, None
        self.held_log = []
        self.recovery_gate = False
        self.ever_guided = False
        self._clear_now = None
        self._p_now = None

    def before_chunk(self, gripper_closed: bool) -> ChunkCostContext:
        cfg = self.cfg
        p_now = self.robot.eef_pos()
        if cfg.arm_release and self.gate is not None and not self.post_gate:
            c, d, off = self.gate
            if float((p_now[:2] - c) @ d) > off + cfg.release_hyst:
                self.post_gate = True  # latched in the world frame for the rest of the episode
        self.post_gate_chunks += int(self.post_gate)
        margin_arm_eff = cfg.arm_margin_post if (cfg.arm_release and self.post_gate) else cfg.margin_arm
        self.recovery_gate = False
        if cfg.escape_chunks > 0 and cfg.recovery:
            self.recovery_gate = len(self.hist) >= cfg.escape_chunks and self._deadlocked(p_now)
            self.escape_off = None
            if self.recovery_gate:
                self.escape_events += 1  # counted as gate openings in the results
        elif cfg.escape_chunks > 0:
            if self.escape_left > 0:
                self.escape_left -= 1
                if self.escape_left == 0:
                    self.escape_off = None
                    self.hist = []
            elif len(self.hist) >= cfg.escape_chunks and self._deadlocked(p_now):
                self.escape_off = np.array([0.0, 0.0, cfg.escape_lift])
                self.escape_left = cfg.escape_len
                self.escape_events += 1
        held = self.robot.held_object(gripper_closed) if cfg.model_object else None
        self.held_log.append(held)
        self._p_now = p_now
        ctx = build_ctx(self.robot, self.scene, self.action_map, cfg, margin_arm=margin_arm_eff, held=held,
                        horizon=self.horizon)
        self._clear_now = clearance_now(ctx)
        return ctx

    def _deadlocked(self, p_now):
        cfg = self.cfg
        recent = self.hist[-cfg.escape_chunks:]
        disp = float(np.linalg.norm(p_now - recent[0][0]))
        if cfg.escape_mode == "window":
            # dithering / fighting stall: over the last escape_chunks chunks the end effector made no net progress
            # (< escape_min_disp) while travelling at least escape_ratio times that distance, guidance was active at
            # some point in the episode [and every obstacle is farther than escape_clear now]
            pts = [h[0] for h in recent] + [p_now]
            path = float(sum(np.linalg.norm(pts[j + 1] - pts[j]) for j in range(len(pts) - 1)))
            far = cfg.escape_clear <= 0 or (self._clear_now is not None and self._clear_now > cfg.escape_clear)
            return self.ever_guided and far and disp < cfg.escape_min_disp and path >= cfg.escape_ratio * disp
        if cfg.escape_mode == "intent":
            # the constraint was active at some point but is NOT active now, yet the policy's own intent has collapsed
            # and the end effector is not making progress: a recovery problem, not a fight with the guidance
            intents = [h[2] for h in recent if len(h) > 2 and h[2] is not None]
            far = cfg.escape_clear <= 0 or (self._clear_now is not None and self._clear_now > cfg.escape_clear)
            return (self.ever_guided and not any(h[1] for h in recent) and disp < cfg.escape_min_disp and far
                    and bool(intents) and float(np.median(intents)) < cfg.escape_intent)
        return all(h[1] for h in recent) and disp < cfg.escape_min_disp

    def peek_ctx(self, gripper_closed: bool) -> ChunkCostContext:
        """Cost context for the CURRENT simulator state without updating any episode state (lookahead branches)."""
        cfg = self.cfg
        margin_arm_eff = cfg.arm_margin_post if (cfg.arm_release and self.post_gate) else cfg.margin_arm
        held = self.robot.held_object(gripper_closed) if cfg.model_object else None
        return build_ctx(self.robot, self.scene, self.action_map, cfg, margin_arm=margin_arm_eff, held=held, horizon=self.horizon)

    def after_chunk(self, last_log: dict):
        """Record whether guidance was active on the chunk just sampled and the policy's own intent (from Guide.last_log)."""
        guided = ((last_log.get("cost_per_step") or [0])[0] or 0) > 0
        self.ever_guided = self.ever_guided or guided
        intent = last_log.get("intent_cm")
        self.hist.append((self._p_now, guided, None if intent is None else float(intent) / 100.0))

    @property
    def held(self):
        return self.held_log[-1] if self.held_log else None
