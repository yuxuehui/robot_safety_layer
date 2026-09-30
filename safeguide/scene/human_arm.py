"""Dynamic obstacle: a human arm tracked as (elbow, fingertip) once per control step.

Geometry used by the cost: two collinear capsules, forearm (elbow -> wrist, radius r_forearm) and palm + fingers
(wrist -> fingertip, radius r_palm), with the wrist at `forearm_len` from the elbow along the arm axis. The radii are
inflated by a human-safety margin plus a reaction-time allowance proportional to the fingertip speed:

    r_k(now) = r_k + margin + |v_tip| * tau

Poses over the chunk (steps 1..H) come from the tracked motion: constant-velocity extrapolation of the last two
observations ("cv"), the current pose held ("static"), or a caller-supplied oracle (sim ground truth; for analysis).
In the benchmark the observations come from the scripted HandMotion; on a robot they would come from a hand tracker.
"""
import numpy as np

from .base import Scene


class HumanArmTrack(Scene):
    def __init__(self, r_forearm=0.040, r_palm=0.045, forearm_len=0.24, dt=0.05, predict="cv", margin=0.0, tau=0.1,
                 oracle=None):
        self.r_forearm, self.r_palm, self.forearm_len, self.dt = r_forearm, r_palm, forearm_len, dt
        self.mode, self.margin, self.tau = predict, margin, tau
        self.oracle = oracle  # optional callable H -> (elbow (H, 3), tip (H, 3)) with the true future poses
        self.hist = []  # (elbow, tip) per control step, most recent last

    # ------------------------------------------------------------------ tracking
    def observe(self, elbow, tip):
        self.hist.append((np.asarray(elbow, dtype=float), np.asarray(tip, dtype=float)))

    def reset(self):
        self.hist = []

    def tip_speed(self):
        if len(self.hist) < 2:
            return 0.0
        return float(np.linalg.norm(self.hist[-1][1] - self.hist[-2][1]) / self.dt)

    def predict(self, H):
        """(H, 3) elbow and fingertip positions at control steps +1 .. +H."""
        elbow, tip = self.hist[-1]
        mode = self.mode
        if mode == "static" or len(self.hist) < 2 and mode == "cv":
            return np.repeat(elbow[None], H, 0), np.repeat(tip[None], H, 0)
        if mode == "cv":
            e0, t0 = self.hist[-2]
            ve, vt = (elbow - e0) / self.dt, (tip - t0) / self.dt
            k = np.arange(1, H + 1)[:, None] * self.dt
            return elbow + k * ve, tip + k * vt
        if mode == "oracle":
            if self.oracle is None:
                raise ValueError("oracle prediction needs an oracle callback")
            return self.oracle(H)
        raise ValueError(mode)

    # ------------------------------------------------------------------ Scene
    def inflation(self):
        return self.margin + self.tip_speed() * self.tau

    def capsules(self, horizon):
        if not self.hist:
            return None
        pe, pt = self.predict(horizon)
        e0, t0 = self.hist[-1]
        el = np.asarray(np.concatenate([e0[None], pe]), dtype=float)
        tp = np.asarray(np.concatenate([t0[None], pt]), dtype=float)
        u = tp - el
        u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
        wr = el + self.forearm_len * u
        cap_a, cap_b = np.stack([el, wr]), np.stack([wr, tp])
        cap_r = np.array([self.r_forearm, self.r_palm]) + float(self.inflation())
        return cap_a, cap_b, cap_r

    def clearance_now(self, pts, radii):
        """Measured clearance of robot spheres (pts (P, 3), radii (P,)) to the current (un-inflated) arm capsules."""
        from ..core.geometry import sdf_capsule_np

        e0, t0 = self.hist[-1]
        u = (t0 - e0) / (np.linalg.norm(t0 - e0) + 1e-9)
        wr = e0 + self.forearm_len * u
        return min(float((sdf_capsule_np(pts, a, b, r) - radii).min())
                   for a, b, r in ((e0, wr, self.r_forearm), (wr, t0, self.r_palm)))
