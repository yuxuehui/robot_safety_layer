"""Why do guided robots stall? Per stalled episode (no pillar contact, task not done):
- progress: furthest fraction of the obstacle-free baseline path (by arc length) the EEF got close to (<4 cm)
- passed: did the EEF get past the pillar(s) along the baseline path direction
- end distance: final EEF xy distance to the nearest pillar / the gate centre
- oscillation: in the last 100 steps, path length / net displacement (large = dithering in place)
- freeze: mean EEF speed in the last 100 steps (cm/step)
- grasp: gripper command at the end (>0 closed) -> was the object picked up
- time near pillars: fraction of steps with EEF within 12 cm (xy) of a pillar
Usage: python stall_diagnostics.py obs3_gate20_g1 obs3_gate17_g1 obs3_single_g1 ...
"""
import glob
import json
import sys

import numpy as np


def arclen(p):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def analyse(run):
    rows = []
    for f in glob.glob(f"runs/{run}/*/results_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if "skipped" in r or r["success"] or r["collided_any"]:
                continue
            tag = f"t{r['task_id']:02d}_e{r['episode']:02d}" + (f"_s{r['torch_seed']}" if r.get("torch_seed") is not None else "")
            z = np.load(f"runs/{run}/{r['suite']}/traj/{tag}.npz")
            base = np.load(f"runs/baseline/{r['suite']}/traj/t{r['task_id']:02d}_e{r['episode']:02d}.npz")["eef_pos"]
            eef, act = z["eef_pos"], z["actions"]
            pil = np.array(z["obstacle_xy"]).reshape(-1, 2)
            s = arclen(base) / max(arclen(base)[-1], 1e-9)
            d = np.linalg.norm(eef[:, None, :] - base[None, :, :], axis=-1)  # (T, Tb)
            reached = s[(d.min(axis=0) < 0.04)]
            progress = float(reached.max()) if len(reached) else 0.0
            # pillar location along the baseline path
            dp = np.linalg.norm(base[:, None, :2] - pil[None, :, :], axis=-1).min(axis=1)
            s_pillar = float(s[np.argmin(dp)])
            last = eef[-100:]
            seg = np.linalg.norm(np.diff(last, axis=0), axis=1)
            net = np.linalg.norm(last[-1] - last[0])
            near = np.linalg.norm(eef[:, None, :2] - pil[None, :, :], axis=-1).min(axis=1) < 0.12
            centre = pil.mean(axis=0)
            rows.append({
                "suite": r["suite"], "task": r["task_id"], "ep": r["episode"],
                "progress": progress, "s_pillar": s_pillar, "passed": progress > s_pillar + 0.05,
                "end_d_pillar_cm": 100 * float(np.linalg.norm(eef[-1, :2] - pil, axis=1).min()),
                "end_d_centre_cm": 100 * float(np.linalg.norm(eef[-1, :2] - centre)),
                "osc": float(seg.sum() / max(net, 1e-3)), "speed_cm": 100 * float(seg.mean()),
                "grasped": bool(act[-1, 6] > 0), "frac_near": float(near.mean()),
                "end_z_above_base_cm": 100 * float(eef[-1, 2] - base[-1, 2]),
            })
    return rows


def summary(run, rows):
    if not rows:
        print(f"{run}: no stalled episodes")
        return
    a = lambda k: np.array([r[k] for r in rows], dtype=float)
    print(f"\n{run}: {len(rows)} stalled episodes")
    print(f"  passed the pillar(s) along the path: {a('passed').mean():.0%}   progress median {np.median(a('progress')):.2f} "
          f"(pillar at {np.median(a('s_pillar')):.2f})")
    print(f"  grasped object at end: {a('grasped').mean():.0%}")
    print(f"  end distance to nearest pillar: median {np.median(a('end_d_pillar_cm')):.1f} cm; to gate centre {np.median(a('end_d_centre_cm')):.1f} cm")
    print(f"  last 100 steps: speed median {np.median(a('speed_cm')):.2f} cm/step, oscillation (path/net) median {np.median(a('osc')):.1f}")
    print(f"  fraction of steps within 12 cm of a pillar: median {np.median(a('frac_near')):.0%}")
    frz = np.mean((a('speed_cm') < 0.1))
    dith = np.mean((a('speed_cm') >= 0.1) & (a('osc') > 5))
    print(f"  -> frozen (speed<0.1 cm/step): {frz:.0%}; dithering (moving but osc>5): {dith:.0%}; "
          f"stuck near pillar (<12 cm at end): {np.mean(a('end_d_pillar_cm') < 12):.0%}")


if __name__ == "__main__":
    out = {}
    for run in sys.argv[1:]:
        rows = analyse(run)
        summary(run, rows)
        out[run] = rows
    json.dump(out, open("runs/stall_diagnostics.json", "w"), indent=1)
