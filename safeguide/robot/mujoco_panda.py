"""Franka Panda in robosuite / MuJoCo (LIBERO): gripper + forearm/wrist sphere model, Jacobian motion model,
held task object detection. `env` is the LIBERO OffScreenRenderEnv (env.env = robosuite env)."""
import numpy as np

from .base import RobotModel

# ----------------------------------------------------------------------------- gripper geometry
# (body_or_site, radius) - spheres approximating the Panda gripper + hand.
GRIPPER_SPHERES = [
    ("site:gripper0_grip_site", 0.020),
    ("body:gripper0_finger_joint1_tip", 0.015),
    ("body:gripper0_finger_joint2_tip", 0.015),
    ("body:gripper0_leftfinger", 0.020),
    ("body:gripper0_rightfinger", 0.020),
    ("body:robot0_right_hand", 0.040),
]


def gripper_points(env):
    """(P, 3) world positions and (P,) radii of the gripper collision spheres."""
    e = env.env
    m, d = e.sim.model, e.sim.data
    pts = []
    for key, _ in GRIPPER_SPHERES:
        kind, name = key.split(":")
        if kind == "site":
            pts.append(d.site_xpos[m.site_name2id(name)].copy())
        else:
            pts.append(d.body_xpos[m.body_name2id(name)].copy())
    # extra points across the hand, along the finger-separation axis (hand is ~20 cm wide)
    hand = pts[-1]
    axis = pts[2] - pts[1]
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    pts += [hand + 0.07 * axis, hand - 0.07 * axis]
    radii = [r for _, r in GRIPPER_SPHERES] + [0.03, 0.03]
    return np.array(pts), np.array(radii)


# Arm links that actually hit the pillars once the gripper is guided (G1/G2 sweeps: link5 forearm,
# link6/link7 wrist). Each collision mesh is approximated by spheres along its longest axis.
ARM_GEOMS = ["robot0_link5_collision", "robot0_link6_collision", "robot0_link7_collision"]
ARM_RADIUS_SCALE = 1.0  # multiplies the mesh cross-section half-size; set by calibrate_arm_model.py
ARM_SPHERES_PER_LINK = 5


def arm_points(env, radius_scale=None, return_bodies=False):
    """(M, 3) sphere centres, (M,) radii, (M,) motion weights alpha of the arm-link sphere model.

    Within one action chunk a sphere is assumed to move by alpha * (EEF displacement):
    alpha = 1 for the wrist links (rigid with the hand), and for forearm spheres alpha is the
    position along the elbow (link4 origin) -> wrist (link6 origin) segment, since the elbow
    barely moves while the wrist follows the gripper.
    """
    e = env.env
    m, d = e.sim.model, e.sim.data
    rs = ARM_RADIUS_SCALE if radius_scale is None else radius_scale
    elbow = d.body_xpos[m.body_name2id("robot0_link4")].copy()
    wrist = d.body_xpos[m.body_name2id("robot0_link6")].copy()
    seg = wrist - elbow
    pts, radii, alphas, bodies = [], [], [], []
    for name in ARM_GEOMS:
        g = m.geom_name2id(name)
        body = int(m.geom_bodyid[g])
        c = d.geom_xpos[g].copy()
        R = d.geom_xmat[g].reshape(3, 3)
        hs = np.array(m.geom_size[g], dtype=float)  # mesh geoms: half-sizes of the fitted box
        ax = int(np.argmax(hs))
        cross = np.sort(np.delete(hs, ax))[-1]  # larger cross-section half-size
        r = rs * cross
        half_len = max(hs[ax] - r, 0.0)
        for s in np.linspace(-1.0, 1.0, ARM_SPHERES_PER_LINK):
            p = c + s * half_len * R[:, ax]
            if name == "robot0_link5_collision":
                a = float(np.clip(np.dot(p - elbow, seg) / (np.dot(seg, seg) + 1e-12), 0.0, 1.0))
            else:
                a = 1.0
            pts.append(p); radii.append(r); alphas.append(a); bodies.append(body)
    if return_bodies:
        return np.array(pts), np.array(radii), np.array(alphas), bodies
    return np.array(pts), np.array(radii), np.array(alphas)


def arm_motion_matrices(env, damping=1e-4):
    """(M, 3, 3) linear motion model of the arm-link spheres: d p_sphere = M_p d p_ee, from the sim Jacobians.

    The OSC controller realises a commanded EEF twist dx (position + rotation) with joint velocities
    dq = Jbar dx, Jbar = Minv J^T (J Minv J^T)^-1 (dynamically consistent inverse, nullspace posture ignored),
    so a sphere on link b moves by J_p dq. Guidance only perturbs the position part of dx, hence
    M_p = J_p[:, arm] Jbar[:, :3]. Replaces the heuristic alpha * d p_ee of arm_points."""
    import mujoco
    e = env.env
    sim = e.sim
    m, d = sim.model._model, sim.data._data
    r = e.robots[0]
    dof = [int(m.jnt_dofadr[m.joint(j).id]) for j in r.robot_model.joints]
    pts, _, _, bodies = arm_points(env, return_bodies=True)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, sid)
    J = np.concatenate([jp[:, dof], jr[:, dof]])  # (6, 7)
    Mfull = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, Mfull, d.qM)
    Minv = np.linalg.inv(Mfull[np.ix_(dof, dof)])
    lam = J @ Minv @ J.T
    Jbar = Minv @ J.T @ np.linalg.inv(lam + damping * np.eye(6))  # (7, 6)
    out = []
    for pnt, b in zip(pts, bodies):
        jpp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jpp, None, np.asarray(pnt, dtype=float), int(b))
        out.append(jpp[:, dof] @ Jbar[:, :3])
    return np.array(out)


# ----------------------------------------------------------------------------- held task object
def _collision_geoms(m, body_id):
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == body_id and (m.geom_contype[g] or m.geom_conaffinity[g])]


def _geom_spheres(m, d, g, max_spheres=3):
    """Spheres covering one box/cylinder/capsule/sphere geom: placed along its longest axis, radius = the
    half-diagonal of the cross-section (covers the box corners)."""
    c = d.geom_xpos[g].copy()
    R = d.geom_xmat[g].reshape(3, 3)
    typ = int(m.geom_type[g])
    size = np.array(m.geom_size[g], dtype=float)
    if typ == 2:  # sphere
        return [c], [float(size[0])]
    if typ in (3, 5):  # capsule / cylinder: (radius, half-length)
        hs = np.array([size[0], size[0], size[1]])
    else:  # box (6) or mesh bounding box (7)
        hs = size
    ax = int(np.argmax(hs))
    a, b = np.delete(hs, ax)
    r = float(np.sqrt(a * a + b * b))
    half_len = max(float(hs[ax]) - r, 0.0)
    n = int(np.clip(np.ceil(hs[ax] / max(r, 1e-3)), 1, max_spheres))
    ss = np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1)
    return [c + t * half_len * R[:, ax] for t in ss], [r] * n


OBJECT_MAX_SPHERES = 3


def object_points(env, name, max_spheres=None):
    """(P, 3) sphere centres and (P,) radii covering the collision geoms of task object `name`: the per-geom
    spheres are merged into at most max_spheres bounding spheres, one per slab along the world z axis
    (a bowl's rim -> one sphere, a bottle -> up to three stacked spheres)."""
    e = env.env
    m, d = e.sim.model, e.sim.data
    pts, radii = [], []
    for g in _collision_geoms(m, e.obj_body_id[name]):
        c, r = _geom_spheres(m, d, g)
        pts += c; radii += r
    pts, radii = np.array(pts), np.array(radii)
    if len(pts) == 0:
        return pts.reshape(0, 3), radii
    K = OBJECT_MAX_SPHERES if max_spheres is None else max_spheres
    z = pts[:, 2]
    extent = float((z + radii).max() - (z - radii).min())
    r_typ = float(np.median(radii))
    n = int(np.clip(np.ceil(extent / max(2.0 * r_typ, 1e-3)), 1, K))
    edges = np.quantile(z, np.linspace(0, 1, n + 1)) if n > 1 else None
    out_c, out_r = [], []
    for i in range(n):
        sel = np.ones(len(z), bool) if n == 1 else (z >= edges[i] - 1e-9) & (z <= edges[i + 1] + 1e-9)
        if not sel.any():
            continue
        c = pts[sel].mean(axis=0)
        out_c.append(c); out_r.append(float((np.linalg.norm(pts[sel] - c, axis=1) + radii[sel]).max()))
    return np.array(out_c), np.array(out_r)


def object_footprint(env, name):
    """(r_xy, half_height) of the object about its body origin, from its collision spheres (for placement)."""
    e = env.env
    o = e.sim.data.body_xpos[e.obj_body_id[name]]
    pts, radii = object_points(env, name)
    r_xy = float((np.linalg.norm(pts[:, :2] - o[:2], axis=1) + radii).max())
    hh = float((np.abs(pts[:, 2] - o[2]) + radii).max())
    return r_xy, hh


def held_object(env, gripper_closed):
    """Name of the task object carried by the gripper: the gripper command is 'close' and a finger geom is in
    contact with one of the object's geoms in the sim (None otherwise). With two candidates the one with more
    finger contacts wins."""
    if not gripper_closed:
        return None
    e = env.env
    m, d = e.sim.model, e.sim.data
    body2obj = {b: k for k, b in e.obj_body_id.items()}
    counts = {}
    for i in range(d.ncon):
        con = d.contact[i]
        for ga, gb in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
            na = m.geom_id2name(ga) or ""
            if "gripper0_finger" in na or "gripper0_right" in na or "gripper0_left" in na:
                k = body2obj.get(int(m.geom_bodyid[gb]))
                if k is not None:
                    counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get) if counts else None


def robot_points(env, include_arm=True, held=None, arm_motion="alpha"):
    """Gripper spheres (alpha = 1) plus, optionally, the arm-link spheres and the spheres of the held
    task object (rigid with the hand, alpha = 1). Also returns the per-sphere group labels."""
    pts, radii = gripper_points(env)
    alphas = np.ones(len(radii))
    groups = ["grip"] * 5 + ["hand"] + ["hand_w"] * 2
    motion = [np.eye(3)] * len(radii)  # per-sphere 3x3 motion matrix (used when arm_motion == "jac")
    if include_arm:
        ap, ar, aa = arm_points(env)
        pts, radii, alphas = np.concatenate([pts, ap]), np.concatenate([radii, ar]), np.concatenate([alphas, aa])
        groups += [g.split("_")[1] for g in ARM_GEOMS for _ in range(ARM_SPHERES_PER_LINK)]
        motion += list(arm_motion_matrices(env)) if arm_motion == "jac" else [a * np.eye(3) for a in aa]
    if held is not None:
        op, orad = object_points(env, held)
        if len(op):
            pts, radii = np.concatenate([pts, op]), np.concatenate([radii, orad])
            alphas = np.concatenate([alphas, np.ones(len(orad))])
            groups += ["obj"] * len(orad)
            motion += [np.eye(3)] * len(orad)
    return pts, radii, alphas, groups, (np.array(motion) if arm_motion == "jac" else None)


def eef_pos(env):
    e = env.env
    return e.sim.data.site_xpos[e.sim.model.site_name2id("gripper0_grip_site")].copy()


class MujocoPandaRobot(RobotModel):
    """RobotModel over a live LIBERO / robosuite Panda env."""

    def __init__(self, env):
        self.env = env

    def eef_pos(self):
        return eef_pos(self.env)

    def points(self, include_arm=True, held=None, arm_motion="jac"):
        return robot_points(self.env, include_arm=include_arm, held=held, arm_motion=arm_motion)

    def held_object(self, gripper_closed):
        return held_object(self.env, gripper_closed)
