"""Calibrate the arm-link sphere model against real MuJoCo contacts.

For a set of arm poses (recorded during unguided rollouts) and many pillar placements around the
forearm / wrist, compare  model clearance = min(SDF - radius) over the arm spheres  with whether
MuJoCo reports a pillar-arm contact after mj_forward. A good model has clearance <= 0 exactly when
there is contact. Reports, per radius scale: false-negative rate (contact but clearance > 0, the
dangerous case), false-positive rate, and the clearance distribution at contact.

Usage (openpi venv): python calibrate_arm_model.py
"""
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import obstacle as ob

rng = np.random.default_rng(0)
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
spec = ob.ObstacleSpec(radius=0.025, height=1.2)  # tall pillar so it reaches the forearm at any pose
records = []  # (contact_with_arm, {scale: clearance})
scales = [0.8, 0.9, 1.0, 1.1, 1.2]
for task_id in [0, 3, 5, 8]:
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
    ob.install_obstacles(env, [spec])
    for ep in range(3):
        env.reset()
        env.set_init_state(suite.get_task_init_states(task_id)[ep])
        # a few arm poses: random EEF motions from the start pose
        for pose in range(4):
            a = np.zeros(7); a[:3] = rng.uniform(-1, 1, 3) * np.array([1, 1, 0.6]); a[6] = -1
            for _ in range(int(rng.integers(5, 25))):
                env.step(a.tolist())
            e = env.env
            ap, _, _ = ob.arm_points(env, radius_scale=1.0)
            lo, hi = ap[:, :2].min(0) - 0.15, ap[:, :2].max(0) + 0.15
            qpos, qvel = e.sim.data.qpos.copy(), e.sim.data.qvel.copy()
            for _ in range(150):
                xy = rng.uniform(lo, hi)
                ob.set_obstacle_xy(env, xy)  # mj_forward inside -> contacts updated, arm not moved
                hits = ob.obstacle_contacts(env)
                arm_hit = any(h in ob.ARM_GEOMS for h in hits)
                other_hit = any(h.startswith(("gripper0", "robot0_link")) and h not in ob.ARM_GEOMS for h in hits)
                if other_hit:
                    continue  # calibrate on contacts with the modelled links only (link5/6/7)
                cl = {}
                for s in scales:
                    pts, radii, _ = ob.arm_points(env, radius_scale=s)
                    cl[s] = float((ob.sdf_cylinder_np(pts, ob.obstacle_center(env), spec.radius, spec.half_height) - radii).min())
                records.append((arm_hit, cl))
            e.sim.data.qpos[:], e.sim.data.qvel[:] = qpos, qvel
            e.sim.forward()
    env.close()

hit = np.array([r[0] for r in records])
print(f"{len(records)} placements, {hit.sum()} with link5/6/7 contact")
for s in scales:
    c = np.array([r[1][s] for r in records])
    fn = np.mean(c[hit] > 0) if hit.any() else np.nan  # contact but model says clear
    fp = np.mean(c[~hit] <= 0) if (~hit).any() else np.nan
    print(f"radius_scale {s:.1f}: false-negative {fn:.3f}  false-positive {fp:.3f}  "
          f"clearance at contact: median {100 * np.median(c[hit]):.1f} cm, max {100 * c[hit].max():.1f} cm")
