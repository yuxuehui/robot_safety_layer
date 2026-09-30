"""Placement-only dry run (no policy): how many episodes get a valid pillar layout under the given placement
flags, and why the others are rejected. Also the baseline-path clearance distribution of the accepted ones.
Usage: python placement_stats.py --layout gate --gate_half_gap 0.17 --clean_gate 0.075 [--suites libero_spatial libero_object]
"""
import argparse
import collections
import pathlib

import numpy as np

import eval_obstacle as eo
import obstacle as ob
from libero.libero import benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object"])
    ap.add_argument("--layout", default="gate")
    ap.add_argument("--gate_half_gap", type=float, default=0.17)
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--frac2", type=float, default=0.65)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--min_obj_dist", type=float, default=0.10)
    ap.add_argument("--min_start_dist", type=float, default=0.10)
    ap.add_argument("--clean_gate", type=float, default=0.0)
    ap.add_argument("--clean_window", type=float, default=0.10)
    ap.add_argument("--radius", type=float, default=0.025)
    ap.add_argument("--height", type=float, default=0.30)
    ap.add_argument("--baseline_dir", default=str(pathlib.Path(__file__).resolve().parent.parent / "runs" / "baseline"))
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    spec = ob.ObstacleSpec(radius=args.radius, height=args.height, visible=False)
    reasons, clears, held, fracs, best_rej = collections.Counter(), [], collections.Counter(), collections.Counter(), []
    for suite_name in args.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        n_ok = 0
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            env = eo.make_env(task, 7)
            ob.install_obstacles(env, [spec] * (1 if args.layout == "single" else 2))
            for ep in range(args.episodes):
                base = pathlib.Path(args.baseline_dir) / suite_name / "traj" / f"t{task_id:02d}_e{ep:02d}.npz"
                env.reset()
                env.set_init_state(init_states[ep])
                if not base.exists():
                    reasons["no_baseline_traj"] += 1
                    continue
                bz = np.load(base)
                xys, info = eo.place_layout(env, bz["eef_pos"], eo.object_xy(env), spec, args, actions=bz["actions"])
                if xys is None:
                    reasons[info.split(":best_clean")[0]] += 1
                    if ":best_clean=" in info:
                        best_rej.append(float(info.split(":best_clean=")[1]))
                    continue
                for i, xy_i in enumerate(xys):
                    ob.set_obstacle_xy(env, xy_i, i)
                env.env.sim.forward()
                if ob.obstacle_contacts(env):
                    reasons["pillar_touches_scene"] += 1
                    continue
                n_ok += 1
                reasons["ok"] += 1
                fracs[round(info["frac"], 2)] += 1
                if "clean_min_clear" in info:
                    clears.append(info["clean_min_clear"])
                    held[info["held_obj"] is not None] += 1
            env.close()
        print(f"{suite_name}: {n_ok}/{suite.n_tasks * args.episodes} placed")
    print("reasons:", dict(reasons))
    print("gate frac of accepted:", dict(sorted(fracs.items())))
    if best_rej:
        b = np.array(best_rej)
        print(f"rejected by the clean check: best candidate clearance median {100 * np.median(b):.1f} cm; "
              f"would pass at 5 cm: {(b >= 0.05).sum()}, at 6 cm: {(b >= 0.06).sum()}, at 7.5 cm: {(b >= 0.075).sum()}")
    if clears:
        c = np.array(clears)
        print(f"baseline min clearance (accepted): median {100 * np.median(c):.1f} cm, min {100 * c.min():.1f}, "
              f"p10 {100 * np.percentile(c, 10):.1f}; grasp detected in {held[True]}/{held[True] + held[False]}")


if __name__ == "__main__":
    main()
