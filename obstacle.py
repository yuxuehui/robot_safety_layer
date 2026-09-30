"""Static obstacle for LIBERO scenes + gripper collision geometry + SDF.

The obstacle is a vertical cylinder standing on the table, added as a body WITHOUT a
joint, so it adds nothing to qpos/qvel: LIBERO's stored init states (flattened sim
state) stay valid and set_init_state() works unchanged. It is injected at
_load_model() level because LIBERO envs use hard_reset=True (every env.reset()
rebuilds the MJCF from env.model); its xy position is then moved per episode through
model.body_pos without rebuilding.

The same cylinder SDF is used by the evaluator (clearance metric) and, in torch form,
by the guidance cost, so "distance" means the same thing in both places.
"""
import dataclasses
import xml.etree.ElementTree as ET

import numpy as np

from safeguide.core.geometry import sdf_cylinder_np, sdf_cylinder_torch  # noqa: F401
from safeguide.robot.mujoco_panda import (ARM_GEOMS, ARM_RADIUS_SCALE, ARM_SPHERES_PER_LINK, GRIPPER_SPHERES,  # noqa: F401
                                          OBJECT_MAX_SPHERES, arm_motion_matrices, arm_points, eef_pos,
                                          gripper_points, held_object, object_footprint, object_points,
                                          robot_points)
from safeguide.scene.base import Scene

BODY = "oc_obstacle"
COL_GEOM = "oc_obstacle_col"
VIS_GEOM = "oc_obstacle_vis"
TABLE_TOP_Z = 0.90  # LIBERO tabletop scenes: table_collision center z 0.875 + half-height 0.025


@dataclasses.dataclass
class ObstacleSpec:
    radius: float = 0.025
    height: float = 0.30  # standing on the table; tall enough that going over is not the easy way out
    visible: bool = True
    rgba: tuple = (0.85, 0.15, 0.15, 1.0)

    @property
    def half_height(self):
        return self.height / 2


def _names(i):
    """Body / collision / visual geom names of pillar i (pillar 0 keeps the original names)."""
    sfx = "" if i == 0 else f"_{i}"
    return BODY + sfx, COL_GEOM + sfx, VIS_GEOM + sfx


def install_obstacle(env, spec: ObstacleSpec):
    """Single pillar (kept for the one-obstacle experiments)."""
    return install_obstacles(env, [spec])


def install_obstacles(env, specs):
    """Patch env so every (hard) reset builds the scene with K pillars. Call before env.reset()."""
    e = env.env
    orig_load_model = e._load_model

    def _load_model_with_obstacles():
        orig_load_model()
        for i, spec in enumerate(specs):
            body_n, col_n, vis_n = _names(i)
            body = ET.Element("body", name=body_n, pos=f"{i} 0 {TABLE_TOP_Z + spec.half_height}")
            # Collision geom: group 0, fully transparent so it never shows up in camera images by itself.
            ET.SubElement(body, "geom", name=col_n, type="cylinder",
                          size=f"{spec.radius} {spec.half_height}", group="0", rgba="0 0 0 0",
                          contype="1", conaffinity="1", friction="1 0.005 0.0001")
            if spec.visible:
                ET.SubElement(body, "geom", name=vis_n, type="cylinder",
                              size=f"{spec.radius} {spec.half_height}", group="1",
                              rgba=" ".join(map(str, spec.rgba)), contype="0", conaffinity="0")
            e.model.worldbody.append(body)

    e._load_model = _load_model_with_obstacles
    env.oc_obstacle_specs = list(specs)
    env.oc_obstacle_spec = specs[0]
    return env


def support_z(env, xy):
    """Height of the static support surface (table / floor) below xy.

    LIBERO scenes differ (tabletop z=0.90, floor z=0 for libero_object, living-room table
    for libero_10), so cast a ray straight down and skip anything that is not static scenery:
    robot / gripper / mount geoms, movable objects (bodies with joints), and the obstacle.
    """
    import mujoco

    e = env.env
    m = getattr(e.sim.model, "_model", e.sim.model)
    d = getattr(e.sim.data, "_data", e.sim.data)
    obs_body = m.body(BODY).id if BODY in [m.body(i).name for i in range(m.nbody)] else -1
    z = 3.0
    geomid = np.array([-1], dtype=np.int32)
    for _ in range(20):
        dist = mujoco.mj_ray(m, d, np.array([xy[0], xy[1], z]), np.array([0.0, 0.0, -1.0]), None, 1, obs_body, geomid)
        if dist < 0:
            return 0.0  # nothing below: fall back to world floor height
        hit_z = z - dist
        g = int(geomid[0])
        body = int(m.geom_bodyid[g])
        name = (m.geom(g).name or "") + "|" + (m.body(body).name or "")
        root = int(m.body_rootid[body])
        movable = m.body_jntnum[body] > 0 or m.body_jntnum[root] > 0
        if name.startswith(("robot0", "gripper0", "mount0", COL_GEOM, VIS_GEOM)) or "|robot0" in name \
                or "|gripper0" in name or "|mount0" in name or f"|{BODY}" in name or movable:
            z = hit_z - 1e-3  # continue the ray below this geom
            continue
        return float(hit_z)
    return 0.0


def set_obstacle_xy(env, xy, idx=0):
    """Move pillar idx onto the support surface at xy (after env.reset()/set_init_state,
    since hard resets rebuild the model)."""
    e = env.env
    spec = env.oc_obstacle_specs[idx]
    zs = support_z(env, xy)
    bid = e.sim.model.body_name2id(_names(idx)[0])
    e.sim.model.body_pos[bid] = np.array([xy[0], xy[1], zs + spec.half_height])
    e.sim.forward()
    if idx == 0:
        env.oc_obstacle_base_z = zs


def obstacle_center(env, idx=0):
    e = env.env
    return e.sim.data.body_xpos[e.sim.model.body_name2id(_names(idx)[0])].copy()


def obstacle_centers(env):
    return np.array([obstacle_center(env, i) for i in range(len(env.oc_obstacle_specs))])


def obstacle_contacts_by_pillar(env):
    """[names of geoms touching pillar i (table excluded) for each pillar i]."""
    e = env.env
    m, d = e.sim.model, e.sim.data
    cols = [m.geom_name2id(_names(i)[1]) for i in range(len(env.oc_obstacle_specs))]
    hits = [[] for _ in cols]
    for j in range(d.ncon):
        c = d.contact[j]
        for i, col in enumerate(cols):
            if c.geom1 == col or c.geom2 == col:
                other = c.geom2 if c.geom1 == col else c.geom1
                name = m.geom_id2name(other) or f"geom{other}"
                if "table" not in name:
                    hits[i].append(name)
    return hits


def obstacle_contacts(env):
    """Names of geoms currently touching any pillar (table excluded)."""
    return [n for h in obstacle_contacts_by_pillar(env) for n in h]


# ----------------------------------------------------------------------------- clearance metrics
# (robot sphere model, Jacobian motion model and held-object detection live in safeguide.robot.mujoco_panda)
def object_clearance(env, name):
    """Min over pillars and the object's spheres of (SDF - radius)."""
    pts, radii = object_points(env, name)
    return min(float((sdf_cylinder_np(pts, obstacle_center(env, i), sp.radius, sp.half_height) - radii).min())
               for i, sp in enumerate(env.oc_obstacle_specs))


def arm_clearance(env):
    """Min over pillars and arm-link spheres of (SDF - radius); arm-only counterpart of gripper_clearance."""
    pts, radii, _ = arm_points(env)
    return min(float((sdf_cylinder_np(pts, obstacle_center(env, i), sp.radius, sp.half_height) - radii).min())
               for i, sp in enumerate(env.oc_obstacle_specs))


def gripper_clearance_per(env):
    """Per pillar: min over gripper spheres of (SDF to pillar - sphere radius). <0 = penetration."""
    pts, radii = gripper_points(env)
    return [float((sdf_cylinder_np(pts, obstacle_center(env, i), sp.radius, sp.half_height) - radii).min())
            for i, sp in enumerate(env.oc_obstacle_specs)]


def gripper_clearance(env):
    """Min over pillars and gripper spheres of (SDF - sphere radius). <0 means penetration."""
    return min(gripper_clearance_per(env))


class PillarScene(Scene):
    """safeguide Scene over the pillars installed in a LIBERO env (install_obstacles + set_obstacle_xy)."""

    def __init__(self, env):
        self.env = env

    def cylinders(self):
        specs = getattr(self.env, "oc_obstacle_specs", [])
        return (obstacle_centers(self.env) if specs else np.zeros((0, 3)),
                np.array([sp.radius for sp in specs]), np.array([sp.half_height for sp in specs]))
