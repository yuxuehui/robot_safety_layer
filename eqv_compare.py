"""Equivalence check between two eval_obstacle run dirs (e.g. legacy sampler vs safeguide package, or a rerun vs an
archived run): per (task, episode) compares the result records and the executed action / EEF trajectories bit for bit.
Usage: python eqv_compare.py <runA> <runB> [suite]"""
import glob
import json
import pathlib
import sys

import numpy as np

a, b = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
suite = sys.argv[3] if len(sys.argv) > 3 else "libero_object"
KEYS = ["success", "steps", "collided_any", "collided_robot", "safe_success", "contact_steps", "min_clearance",
        "min_arm_clearance", "min_obj_clearance", "held_chunks", "post_gate_chunks", "escape_events", "n_chunks",
        "pred_min_clearance_mean", "obstacle_xy"]


def load(d):
    out = {}
    for f in glob.glob(str(d / suite / "results_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            out[(r["task_id"], r["episode"])] = r
    return out


ra, rb = load(a), load(b)
common = sorted(set(ra) & set(rb))
print(f"{len(ra)} eps in A, {len(rb)} in B, {len(common)} common")
n_ok = 0
for k in common:
    x, y = ra[k], rb[k]
    if ("skipped" in x) or ("skipped" in y):
        print(k, "skipped:", x.get("skipped"), y.get("skipped")); continue
    diffs = [kk for kk in KEYS if x.get(kk) != y.get(kk)]
    tag = f"t{k[0]:02d}_e{k[1]:02d}" + (f"_s{x['torch_seed']}" if "torch_seed" in x else "")
    fa, fb = a / suite / "traj" / f"{tag}.npz", b / suite / "traj" / f"{tag}.npz"
    traj = "no-traj"
    if fa.exists() and fb.exists():
        za, zb = np.load(fa), np.load(fb)
        same_act = za["actions"].shape == zb["actions"].shape and np.array_equal(za["actions"], zb["actions"])
        same_eef = za["eef_pos"].shape == zb["eef_pos"].shape and np.array_equal(za["eef_pos"], zb["eef_pos"])
        if same_act and same_eef:
            traj = "traj identical"
        else:
            n = min(len(za["actions"]), len(zb["actions"]))
            first = next((i for i in range(n) if not np.array_equal(za["actions"][i], zb["actions"][i])), n)
            traj = f"traj DIFFERS (len {len(za['actions'])} vs {len(zb['actions'])}, first differing step {first}, " \
                   f"max |da| {np.abs(za['actions'][:n] - zb['actions'][:n]).max():.2e})"
    ok = not diffs and traj == "traj identical"
    n_ok += ok
    print(k, "OK" if ok else "MISMATCH", traj, "" if not diffs else {kk: (x.get(kk), y.get(kk)) for kk in diffs})
print(f"identical: {n_ok}/{len(common)}")
