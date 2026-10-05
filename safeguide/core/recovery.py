"""State-level recovery from steering-induced distribution shift ("state OOD").

After the safety layer has steered the arm around an obstacle, the frozen policy can end up in a state it never saw
during training: it then dithers (small, alternating intents, no net progress) and never completes the task. Action-
level corrections cannot fix this (the velocity field itself is wrong there), so the Supervisor instead

  1. scores every chunk-start state against the policy's own in-distribution behaviour (`ManifoldIndex`: nearest
     neighbour distance to the end-effector positions of the obstacle-free baseline rollouts of the same task, with
     matching gripper state),
  2. detects the dithering signature over a window of chunks (net displacement small while the travelled path is
     long, guidance no longer active), and
  3. rewinds: drives the end effector back along its own recorded path to the most recent chunk-start state that was
     in distribution (same gripper / held-object state), then hands control back to the policy. A second trigger
     rewinds to an earlier in-distribution state.

Rewind actions are raw delta-EEF commands a = G^-1 dp (clipped to the controller range), rotation held, gripper held.
"""
import dataclasses
import glob

import numpy as np


class ManifoldIndex:
    """Nearest-neighbour distance (m) from a state (EEF position, gripper closed?) to a set of reference trajectories."""

    def __init__(self, points, closed):
        self.points = np.asarray(points, float).reshape(-1, 3)
        self.closed = np.asarray(closed, bool).reshape(-1)
        self._by_flag = {f: self.points[self.closed == f] for f in (True, False)}

    @classmethod
    def from_trajectories(cls, trajs):
        """trajs: list of (eef_pos (T+1, 3), gripper_cmd (T,)) from eval_libero.py --out <baseline_dir>."""
        pts, cl = [], []
        for eef, grip in trajs:
            eef = np.asarray(eef, float); g = np.asarray(grip, float) > 0
            n = min(len(eef), len(g) + 1)
            pts.append(eef[:n]); cl.append(np.concatenate([g[: n - 1], g[n - 2: n - 1] if n > 1 else [False]]))
        return cls(np.concatenate(pts), np.concatenate(cl))

    @classmethod
    def from_baseline(cls, baseline_dir, suite, task_id):
        files = sorted(glob.glob(f"{baseline_dir}/{suite}/traj/t{task_id:02d}_e*.npz"))
        trajs = []
        for f in files:
            z = np.load(f)
            trajs.append((z["eef_pos"], z["actions"][:, 6]))
        return cls.from_trajectories(trajs) if trajs else None

    def distance(self, p, gripper_closed):
        ref = self._by_flag[bool(gripper_closed)]
        if len(ref) == 0:
            ref = self.points
        return float(np.sqrt(((ref - np.asarray(p, float)) ** 2).sum(-1)).min())

    def loo_distances(self, trajs):
        """Leave-one-trajectory-out nearest-neighbour distances (for choosing the tube radius)."""
        out = []
        for i, (eef, grip) in enumerate(trajs):
            idx = ManifoldIndex.from_trajectories([t for j, t in enumerate(trajs) if j != i])
            g = np.asarray(grip, float) > 0
            out += [idx.distance(p, g[min(k, len(g) - 1)]) for k, p in enumerate(np.asarray(eef))]
        return np.array(out)


@dataclasses.dataclass
class RewindConfig:
    window: int = 6  # chunks over which the dithering signature is evaluated
    net_thr: float = 0.08  # m: net EEF displacement over the window below this ...
    ratio: float = 2.0  # ... while the travelled path is at least ratio x the net displacement
    quiet: int = 2  # guidance inactive on the last `quiet` chunks (the detour is over; waiting for a hand is not OOD)
    ood_thr: float = 0.06  # m: nearest-neighbour distance above which a chunk-start state counts as out of distribution
    max_events: int = 2  # rewinds per episode
    step: float = 0.012  # m: max EEF displacement per rewind control step (~ one unit action)
    cooldown: int = 6  # chunks after a rewind before the trigger may fire again
    min_back: int = 2  # a rewind must go back at least this many chunks


class RewindRecovery:
    """Per-episode bookkeeping + the rewind planner. Owned by the Supervisor."""

    def __init__(self, cfg: RewindConfig, manifold: ManifoldIndex | None, G):
        self.cfg, self.manifold, self.G = cfg, manifold, np.asarray(G, float)
        self.reset()

    def reset(self):
        self.path = []  # (p_ee, gripper_closed) per control step, p_ee measured before the action was executed
        self.chunks = []  # per chunk start: dict(step, p, closed, held, ood, guided)
        self.events, self.steps_total, self.cooldown = 0, 0, 0
        self.last_target_step = None
        self.last_ood = None

    def note_step(self, p, gripper_closed):
        self.path.append((np.asarray(p, float).copy(), bool(gripper_closed)))

    def score(self, p, gripper_closed):
        self.last_ood = None if self.manifold is None else self.manifold.distance(p, gripper_closed)
        return self.last_ood

    def record_chunk(self, p, gripper_closed, held, guided):
        self.chunks.append({"step": len(self.path), "p": np.asarray(p, float).copy(), "closed": bool(gripper_closed),
                            "held": held, "ood": self.last_ood, "guided": bool(guided)})

    def dithering(self, p_now):
        cfg = self.cfg
        if len(self.chunks) < cfg.window:
            return False
        pts = [c["p"] for c in self.chunks[-cfg.window:]] + [np.asarray(p_now, float)]
        net = float(np.linalg.norm(pts[-1] - pts[0]))
        path = float(sum(np.linalg.norm(pts[j + 1] - pts[j]) for j in range(len(pts) - 1)))
        quiet = not any(c["guided"] for c in self.chunks[-cfg.quiet:]) if cfg.quiet > 0 else True
        ever = any(c["guided"] for c in self.chunks)
        return ever and quiet and net < cfg.net_thr and path >= cfg.ratio * net

    def target(self, gripper_closed, held):
        """Most recent in-distribution chunk start with the same gripper / held state; earlier than the previous target."""
        cfg = self.cfg
        limit = len(self.chunks) - cfg.min_back
        for c in reversed(self.chunks[:limit]):
            if self.last_target_step is not None and c["step"] >= self.last_target_step:
                continue
            if c["ood"] is not None and c["ood"] < cfg.ood_thr and c["closed"] == bool(gripper_closed) and c["held"] == held:
                return c
        return None

    def plan(self, target, gripper_closed):
        """Raw delta-EEF actions that retrace the recorded path from now back to the target chunk start."""
        cfg = self.cfg
        now = len(self.path)
        way = [p for p, _ in self.path[target["step"]:now]][::-1]  # from the current position back to the target
        if not way:
            return []
        pts = [way[0]]
        for p in way[1:]:  # keep waypoints at least `step` apart, split long jumps
            d = p - pts[-1]; n = float(np.linalg.norm(d))
            if n < cfg.step * 0.5:
                continue
            k = int(np.ceil(n / cfg.step))
            pts += [pts[-1] + d * (j / k) for j in range(1, k + 1)]
        Ginv = np.linalg.inv(self.G)
        g_cmd = 1.0 if gripper_closed else -1.0
        actions = []
        for j in range(1, len(pts)):
            a = Ginv @ (pts[j] - pts[j - 1])
            s = max(1.0, float(np.abs(a).max()))  # keep the direction, respect the controller range
            actions.append(np.array([*(a / s), 0.0, 0.0, 0.0, g_cmd], dtype=float))
        self.events += 1; self.steps_total += len(actions); self.cooldown = cfg.cooldown
        self.last_target_step = target["step"]
        return actions
