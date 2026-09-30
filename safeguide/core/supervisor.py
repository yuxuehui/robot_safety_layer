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

    @classmethod
    def pillar_sota(cls, **kw):
        """Static obstacles: CBF + held object + Jacobian + release + escape (the evaluated 'inference-time guidance')."""
        return cls(**kw)

    @classmethod
    def hand_sota(cls, **kw):
        """Moving human hand: same cost, no gate line to release on, escape off (it lifted the gripper 40-50 cm
        while waiting for the hand to pass)."""
        return cls(arm_release=False, escape_chunks=0, **kw)


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
        if cfg.escape_chunks > 0:
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
        return build_ctx(self.robot, self.scene, self.action_map, cfg, margin_arm=margin_arm_eff, held=held,
                         horizon=self.horizon)

    def _deadlocked(self, p_now):
        """Guidance active on all of the last escape_chunks chunks and the end effector moved less than escape_min_disp."""
        cfg = self.cfg
        recent = self.hist[-cfg.escape_chunks:]
        disp = float(np.linalg.norm(p_now - recent[0][0]))
        return all(h[1] for h in recent) and disp < cfg.escape_min_disp

    def after_chunk(self, last_log: dict):
        """Record whether guidance was active on the chunk just sampled (from Guide.last_log)."""
        guided = ((last_log.get("cost_per_step") or [0])[0] or 0) > 0
        self.hist.append((self._p_now, guided))

    @property
    def held(self):
        return self.held_log[-1] if self.held_log else None
