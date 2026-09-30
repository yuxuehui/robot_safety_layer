"""A moving human arm/hand as a kinematic obstacle in LIBERO scenes.

Geometry: a procedural human arm on one jointless body `oc_hand` (sleeved forearm, wrist, flat palm, four curled
fingers, thumb; elbow -> fingertips 43 cm). The guidance cost and the clearance metric use a conservative two-capsule
envelope of it (forearm r 4 cm, palm+fingers r 4.5 cm). The body pose is set every control step (model.body_pos/body_quat + forward), so it is
kinematic: it pushes nothing but contacts with it are detected on every physics substep like the pillars.
Collision capsules are group 0 / transparent; when `visible` the same capsules are drawn skin-coloured in group 1.

Motions are scripted per episode from the obstacle-free baseline path (same source as the pillar placement):
  sweep : the hand rests beside the table, then sweeps straight across the robot's approach path and back
          (triggered when the gripper comes within `trigger_dist` of the crossing point).
  reach : the hand reaches for the object the robot is about to grasp, hovers over it, and retracts
          (triggered when the gripper is within `trigger_dist` of that object and has not grasped yet).
The arm always points from a fixed "shoulder" point beside the table to the fingertip.

Predictions for the guidance cost (poses at the next H control steps):
  oracle : the scripted future (exact once the motion has been triggered; rest pose before that),
  cv     : constant-velocity extrapolation of the last observed fingertip/elbow motion,
  static : the current pose held.
"""
import dataclasses
import xml.etree.ElementTree as ET

import numpy as np

from safeguide.core.geometry import sdf_capsule_np  # noqa: F401
from safeguide.scene.human_arm import HumanArmTrack

HAND_BODY = "oc_hand"
# anthropometric-ish dimensions (m): forearm elbow->wrist, hand wrist->fingertip
FOREARM = 0.24
HAND_LEN = 0.19
L_ARM = FOREARM + HAND_LEN
R_FOREARM = 0.040  # coarse capsule radius used by the guidance cost / clearance metric
R_PALM = 0.045     # coarse capsule (palm + fingers) radius used by the guidance cost / clearance metric
SKIN = (0.87, 0.67, 0.53, 1.0)
SLEEVE = (0.22, 0.27, 0.42, 1.0)
DT = 0.05  # 20 Hz control


@dataclasses.dataclass
class HandSpec:
    visible: bool = True
    r_forearm: float = R_FOREARM
    r_palm: float = R_PALM


def _rot(axis, ang):
    axis = np.asarray(axis, dtype=float); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def hand_geoms():
    """Procedural human arm in the body frame: +x elbow -> fingertips, +y across the palm, +z = back of the hand.
    Returns [(name, kind, params, colour)] with kind 'capsule' (fromto a, b, radius), 'sphere' (pos, r) or
    'box' (pos, half-sizes, quat=None)."""
    g = []
    wrist = np.array([FOREARM, 0.0, 0.0])
    g.append(("forearm", "capsule", (np.array([0.0, 0.0, 0.0]), wrist - np.array([0.03, 0, 0]), 0.036), SLEEVE))
    g.append(("wrist", "sphere", (wrist, 0.030), SKIN))
    palm_len, palm_half_w, palm_half_t = 0.085, 0.040, 0.012
    palm_c = wrist + np.array([0.012 + palm_len / 2, 0.0, 0.0])
    g.append(("palm", "box", (palm_c, np.array([palm_len / 2, palm_half_w, palm_half_t])), SKIN))
    # four fingers: 3 phalanges each, slightly spread, curled downwards (relaxed reaching hand)
    knuckle_x = palm_c[0] + palm_len / 2
    for k, (y, spread, lens, r) in enumerate([(-0.030, -0.15, (0.036, 0.024, 0.020), 0.0080),
                                              (-0.010, -0.05, (0.042, 0.028, 0.022), 0.0085),
                                              (0.010, 0.05, (0.040, 0.026, 0.021), 0.0085),
                                              (0.030, 0.15, (0.032, 0.022, 0.018), 0.0075)]):
        p = np.array([knuckle_x, y, -0.002])
        pitch = 0.0
        for j, L in enumerate(lens):
            pitch += (0.30, 0.35, 0.30)[j]  # rad, curl per joint
            d = _rot([0, 0, 1], spread) @ _rot([0, 1, 0], pitch) @ np.array([1.0, 0.0, 0.0])
            q = p + L * d
            g.append((f"f{k}s{j}", "capsule", (p, q, r), SKIN))
            p = q
    # thumb: 2 phalanges from the side of the palm, angled outwards and down
    p = wrist + np.array([0.035, palm_half_w + 0.004, -0.004])
    pitch = 0.0
    for j, (L, yaw) in enumerate([(0.045, 0.95), (0.032, 0.55)]):
        pitch += 0.25
        d = _rot([0, 0, 1], yaw) @ _rot([0, 1, 0], pitch) @ np.array([1.0, 0.0, 0.0])
        q = p + L * d
        g.append((f"thumb{j}", "capsule", (p, q, 0.011), SKIN))
        p = q
    return g


def install_hand(env, spec: HandSpec):
    """Patch env so every (hard) reset builds the scene with the hand body, parked far away."""
    e = env.env
    orig_load_model = e._load_model

    def _load_model_with_hand():
        orig_load_model()
        body = ET.Element("body", name=HAND_BODY, pos="2 0 1.5")
        for name, kind, params, rgba in hand_geoms():
            common = {"name": f"{HAND_BODY}_{name}_col", "group": "0", "rgba": "0 0 0 0",
                      "contype": "1", "conaffinity": "1", "friction": "1 0.005 0.0001"}
            vis = {"name": f"{HAND_BODY}_{name}_vis", "group": "1", "rgba": " ".join(map(str, rgba)),
                   "contype": "0", "conaffinity": "0"}
            if kind == "capsule":
                a, b, r = params
                geo = {"type": "capsule", "fromto": " ".join(f"{v:.4f}" for v in (*a, *b)), "size": f"{r:.4f}"}
            elif kind == "sphere":
                c, r = params
                geo = {"type": "sphere", "pos": " ".join(f"{v:.4f}" for v in c), "size": f"{r:.4f}"}
            else:
                c, hs = params
                geo = {"type": "box", "pos": " ".join(f"{v:.4f}" for v in c), "size": " ".join(f"{v:.4f}" for v in hs)}
            ET.SubElement(body, "geom", **common, **geo)
            if spec.visible:
                ET.SubElement(body, "geom", **vis, **geo)
        e.model.worldbody.append(body)

    e._load_model = _load_model_with_hand
    env.oc_hand_spec = spec
    return env


def _quat_x_to(u):
    """Quaternion (w, x, y, z) rotating the +x axis onto unit vector u."""
    x = np.array([1.0, 0.0, 0.0])
    c = float(np.clip(np.dot(x, u), -1.0, 1.0))
    if c > 1 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about z
    axis = np.cross(x, u)
    axis /= np.linalg.norm(axis)
    ang = np.arccos(c)
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def _quat_from_xz(x, up=(0.0, 0.0, 1.0)):
    """Quaternion (w, x, y, z) of the frame whose +x is `x` and whose +z is `up` projected orthogonal to x."""
    x = np.asarray(x, dtype=float); x /= np.linalg.norm(x) + 1e-9
    z = np.asarray(up, dtype=float) - np.dot(up, x) * x
    if np.linalg.norm(z) < 1e-6:
        z = np.array([1.0, 0.0, 0.0]) - x[0] * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    return _quat_x_to(x)


def set_hand_pose(env, elbow, tip):
    e = env.env
    u = np.asarray(tip) - np.asarray(elbow)
    u = u / (np.linalg.norm(u) + 1e-9)
    bid = e.sim.model.body_name2id(HAND_BODY)
    e.sim.model.body_pos[bid] = np.asarray(elbow, dtype=float)
    e.sim.model.body_quat[bid] = _quat_from_xz(u)
    e.sim.forward()


def segments(elbow, tip, spec: HandSpec = HandSpec()):
    """[(a, b, r)] of the two capsules for an elbow/tip pose."""
    elbow, tip = np.asarray(elbow, dtype=float), np.asarray(tip, dtype=float)
    u = (tip - elbow) / (np.linalg.norm(tip - elbow) + 1e-9)
    wrist = elbow + FOREARM * u
    return [(elbow, wrist, spec.r_forearm), (wrist, tip, spec.r_palm)]


def hand_contacts(env, prefixes=None):
    """Names of geoms touching the hand's collision capsules (table excluded). With `prefixes`, only geoms whose
    name starts with one of them (e.g. robot links, gripper, task objects) count: the kinematic hand may pass
    through static scenery, which is not a collision of interest."""
    e = env.env
    m, d = e.sim.model, e.sim.data
    bid = m.body_name2id(HAND_BODY)
    cols = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g]}
    out = []
    for j in range(d.ncon):
        c = d.contact[j]
        for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if g in cols and other not in cols:
                name = m.geom_id2name(other) or f"geom{other}"
                if "table" in name:
                    continue
                if prefixes is None or name.startswith(tuple(prefixes)):
                    out.append(name)
    return out


def relevant_prefixes(env):
    """Geom-name prefixes whose contact with the hand counts as a safety violation: the robot and its gripper."""
    return ("robot0", "gripper0")


def points_clearance(pts, radii, elbow, tip, spec: HandSpec = HandSpec()):
    """min over spheres and capsules of (SDF - sphere radius)."""
    return min(float((sdf_capsule_np(pts, a, b, r) - radii).min()) for a, b, r in segments(elbow, tip, spec))


class HandMotion:
    """Scripted hand motion for one episode. Call update(eef_pos, step) once per control step BEFORE env.step;
    it returns (elbow, tip). predict(H, mode) returns the poses at the next H steps."""

    def __init__(self, kind, path, actions, objs_xyz, support_z, rng, speed=0.3, frac=0.4, trigger_dist=0.18,
                 side=None, dwell=1.5, rest_offset=0.45, shoulder_offset=0.80, obj_top=None):
        self.kind, self.speed = kind, float(speed * rng.uniform(0.85, 1.15))
        self.trigger_dist = float(trigger_dist * rng.uniform(0.85, 1.15))
        self.jitter_steps = int(rng.integers(-6, 7))  # +-0.3 s on the scheduled trigger
        xy = path[:, :2]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        grasp = next((k for k in range(len(actions)) if actions[k, 6] > 0), len(actions) - 1)
        # crossing point: fraction `frac` of the PRE-GRASP arc (the approach), path tangent / normal there
        s_g = s[min(grasp, len(s) - 1)]
        i = int(np.searchsorted(s, frac * s_g))
        i = int(np.clip(i, 3, len(path) - 4))
        d = xy[i + 3] - xy[i - 3]
        d = d / (np.linalg.norm(d) + 1e-9)
        n = np.array([-d[1], d[0]])
        self.side = float(side if side is not None else rng.choice([-1.0, 1.0]))
        self.c = np.array([xy[i, 0], xy[i, 1], max(path[i, 2], support_z + 0.13)])
        self.i_cross, self.grasp_step = i, int(grasp)
        z_arm = support_z + 0.22
        self.shoulder = np.array([*(xy[i] + self.side * shoulder_offset * n), z_arm])
        self.rest = np.array([*(xy[i] + self.side * rest_offset * n), self.c[2]])
        if kind == "sweep":
            far = np.array([*(xy[i] - self.side * 0.30 * n), self.c[2]])
            self.waypoints, self.dwells = [self.rest, far, self.rest], [0.0, 0.0, 0.0]
            self.target = self.c
        elif kind == "reach":
            g_pos = path[min(grasp, len(path) - 1)]
            names = list(objs_xyz)
            obj = min(names, key=lambda k: np.linalg.norm(objs_xyz[k][:2] - g_pos[:2]))
            o = np.asarray(objs_xyz[obj], dtype=float)
            top = float(obj_top(obj)) if obj_top is not None else o[2] + 0.04
            self.target = np.array([o[0], o[1], max(top + R_PALM + 0.03, support_z + 0.13)])
            self.target_name = obj
            self.waypoints, self.dwells = [self.rest, self.target, self.rest], [0.0, dwell, 0.0]
        else:
            raise ValueError(kind)
        # piecewise-linear schedule: (t_start, t_end, p_from, p_to)
        self.schedule, t = [], 0.0
        for k in range(len(self.waypoints) - 1):
            p0, p1 = self.waypoints[k], self.waypoints[k + 1]
            dur = float(np.linalg.norm(p1 - p0) / self.speed)
            self.schedule.append((t, t + dur, p0, p1)); t += dur
            if self.dwells[k + 1] > 0:
                self.schedule.append((t, t + self.dwells[k + 1], p1, p1)); t += self.dwells[k + 1]
        self.duration = t
        # scheduled trigger: the hand reaches the crossing / the object when the obstacle-free robot would be there
        travel = int(round(np.linalg.norm(self.waypoints[1] - self.waypoints[0]) / self.speed / DT))
        arrive = self.i_cross if kind == "sweep" else max(self.grasp_step - 10, 0)
        self.trigger_step = max(0, arrive - travel + self.jitter_steps)
        self.t_trigger = None
        self.hist = []  # (elbow, tip) per control step

    def key_tips(self):
        """Fingertip positions the motion visits (for checking that the arm does not pass through scenery)."""
        return [w for w in self.waypoints]

    def tip_speed(self):
        if len(self.hist) < 2:
            return 0.0
        return float(np.linalg.norm(self.hist[-1][1] - self.hist[-2][1]) / DT)

    def tip_at(self, tau):
        """Fingertip position tau seconds after the trigger (rest pose before the trigger / after the motion)."""
        if tau < 0:
            return self.rest.copy()
        for t0, t1, p0, p1 in self.schedule:
            if tau <= t1:
                a = 0.0 if t1 <= t0 else (tau - t0) / (t1 - t0)
                return p0 + a * (p1 - p0)
        return self.rest.copy()

    def pose_from_tip(self, tip):
        u = tip - self.shoulder
        u = u / (np.linalg.norm(u) + 1e-9)
        return tip - L_ARM * u, tip

    def update(self, eef, step, gripper_closed=False):
        if self.t_trigger is None and step >= self.trigger_step:
            self.t_trigger = step
        tau = -1.0 if self.t_trigger is None else (step - self.t_trigger) * DT
        elbow, tip = self.pose_from_tip(self.tip_at(tau))
        self.hist.append((elbow, tip))
        self.step = step
        return elbow, tip

    def predict(self, H, mode):
        """(H, 3) elbow and tip positions at control steps step+1 .. step+H."""
        elbow, tip = self.hist[-1]
        if mode == "static" or len(self.hist) < 2 and mode == "cv":
            return np.repeat(elbow[None], H, 0), np.repeat(tip[None], H, 0)
        if mode == "cv":
            e0, t0 = self.hist[-2]
            ve, vt = (elbow - e0) / DT, (tip - t0) / DT
            k = np.arange(1, H + 1)[:, None] * DT
            return elbow + k * ve, tip + k * vt
        if mode == "oracle":
            if self.t_trigger is None:
                return np.repeat(elbow[None], H, 0), np.repeat(tip[None], H, 0)
            out_e, out_t = [], []
            for h in range(1, H + 1):
                e_, t_ = self.pose_from_tip(self.tip_at((self.step + h - self.t_trigger) * DT))
                out_e.append(e_); out_t.append(t_)
            return np.array(out_e), np.array(out_t)
        raise ValueError(mode)

    def info(self):
        d = {"kind": self.kind, "speed": self.speed, "side": self.side, "crossing": self.c.round(4).tolist(),
             "rest": self.rest.round(4).tolist(), "duration_s": round(self.duration, 3), "trigger_step": self.t_trigger,
             "scheduled_trigger": self.trigger_step, "baseline_cross_step": self.i_cross, "baseline_grasp_step": self.grasp_step}
        if self.kind == "reach":
            d["target_object"] = self.target_name
        return d


class HandCapsuleScene(HumanArmTrack):
    """safeguide dynamic-obstacle scene over a scripted HandMotion: the HumanArmTrack (two-capsule envelope, prediction,
    speed-dependent inflation) whose observation history IS the script's pose history, with the scripted future as the
    oracle. Same numbers as the pre-package evaluator (verified bit for bit)."""

    def __init__(self, hm, spec: HandSpec, predict_mode="cv", margin_hand=0.0, margin_tau=0.1):
        super().__init__(r_forearm=spec.r_forearm, r_palm=spec.r_palm, forearm_len=FOREARM, dt=DT, predict=predict_mode,
                         margin=margin_hand, tau=margin_tau, oracle=lambda H: hm.predict(H, "oracle"))
        self.hm, self.spec = hm, spec
        self.hist = hm.hist  # shared list: HandMotion.update() appends the pose of every control step
