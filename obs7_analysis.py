"""obs7 analysis: pool runs obs7_<layout>_<arm>_s<seed> over seeds (and layouts), paired on identical placements.
Pre-registered primary tests: steerobj, steerexecobj, mppiobj vs cbfobj on safe_success, Holm over 3, plus a
task-clustered sign-flip permutation test. Secondary: cbfobj vs cbf (object-model ablation), replicate noise floor,
everything vs none. Metrics: safe (no contact at all), rsafe (no ROBOT contact), contact, robot contact,
object-only contact, stall.
Usage: python obs7_analysis.py --layouts gate17 gate20 --seeds 0 1 2
"""
import argparse
import glob
import itertools
import json
import math

import numpy as np

ARMS = ["none", "cbf", "cbfobj", "cbfobj_rep", "steerobj", "steerexecobj", "mppiobj", "carobj", "car2obj",
        "mppipers1", "mppipers5", "carvecobj", "cbfjac", "mppijac", "steerexecjac",
        "cbfjac_rel", "cbfjac_esc", "cbfjac_relesc", "steerexecjac_relesc", "mppijac_relesc", "carvecjac_relesc", "sota_cv", "sota_or", "sota_st", "sota_cv_t", "sota_or_t", "none_r2", "sota_cv_t_r2", "sota_cv_r2", "none_r8", "sota_r2", "sota_r8", "sota", "sota_s05", "sota_esc2", "sota_r4", "sota_gate2", "sota_rec", "sota_la", "sota_onm", "sota_onm_np"]
PRIMARY = [("steerobj", "cbfobj"), ("steerexecobj", "cbfobj"), ("mppiobj", "cbfobj"), ("carobj", "cbfobj"), ("car2obj", "cbfobj"),
           ("mppipers1", "cbfobj"), ("mppipers5", "cbfobj"), ("carvecobj", "cbfobj"),
           ("cbfjac", "cbfobj"), ("mppijac", "mppiobj"), ("steerexecjac", "steerexecobj"),
           ("cbfjac_rel", "cbfjac"), ("cbfjac_esc", "cbfjac"), ("cbfjac_relesc", "cbfjac"), ("steerexecjac_relesc", "steerexecjac"), ("mppijac_relesc", "mppijac"), ("cbfjac_relesc", "none"), ("mppijac_relesc", "cbfjac_relesc"), ("carvecjac_relesc", "cbfjac_relesc"), ("carvecjac_relesc", "carvecobj"),
           ("sota_cv", "none"), ("sota_or", "none"), ("sota_or", "sota_cv"), ("sota_st", "sota_cv"),
           ("sota_cv_t", "none"), ("sota_cv_t", "sota_cv"), ("sota_or_t", "sota_cv_t"),
           ("none_r2", "none"), ("sota_cv_t_r2", "sota_cv_t"), ("sota_cv_r2", "sota_cv"), ("sota_cv_t_r2", "none_r2"),
           ("sota_r2", "cbfjac_relesc"), ("sota_r8", "cbfjac_relesc"), ("sota_r2", "none_r2"), ("sota_r8", "none_r8"), ("none_r8", "none"), ("sota", "none"), ("sota_s05", "sota"), ("sota_esc2", "sota"), ("sota_r4", "sota"), ("sota_s05", "none"), ("sota_esc2", "none"), ("sota_r4", "none"), ("sota_gate2", "sota"), ("sota_gate2", "sota_esc2"), ("sota_gate2", "none"), ("sota_rec", "sota"), ("sota_rec", "sota_gate2"), ("sota_rec", "none"), ("sota_la", "sota"), ("sota_la", "sota_rec"), ("sota_la", "none"), ("sota_onm", "sota"), ("sota_onm_np", "sota"), ("sota_onm", "sota_onm_np"), ("sota_onm", "sota_s05"), ("sota_onm", "none"), ("sota_onm_np", "none")]
SECONDARY = [("cbfobj", "cbf"), ("cbfobj_rep", "cbfobj"), ("cbfobj", "none"), ("steerexecobj", "none"),
             ("mppiobj", "none"), ("steerexecobj", "steerobj"), ("mppiobj", "steerexecobj"), ("cbf", "none"),
             ("car2obj", "carobj"), ("carobj", "none"), ("car2obj", "mppiobj"),
             ("cbfjac", "none"), ("cbfjac_relesc", "none"), ("cbfjac_relesc", "cbfjac_rel"), ("cbfjac_relesc", "mppiobj"), ("mppijac", "cbfjac"), ("mppijac", "none"), ("steerexecjac", "cbfjac"),
             ("mppipers1", "mppiobj"), ("mppipers5", "mppiobj"), ("mppipers1", "mppipers5"), ("carvecobj", "car2obj"), ("carvecobj", "mppiobj")]


PREFIX = "obs7"  # run-dir prefix: obs7_<layout>_<arm>_s<seed> (pi0.5) or groot_... (GR00T N1.7), set by --prefix


def load(layout, arm, seed):
    rows = {}
    for f in glob.glob(f"runs/{PREFIX}_{layout}_{arm}_s{seed}/*/results_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if "skipped" not in r:
                rows[(layout, r["suite"], r["task_id"], r["episode"], seed)] = r
    return rows


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def clustered_p(diff_by_cluster, n_perm=20000, seed=0):
    """Sign-flip permutation on per-cluster mean paired differences (clusters = suite x task)."""
    d = np.array([np.mean(v) for v in diff_by_cluster.values()])
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return float(((np.abs((flips * d).mean(axis=1)) >= obs - 1e-12).sum() + 1) / (n_perm + 1))


def holm(ps):
    order = np.argsort(ps)
    out = np.zeros(len(ps))
    m = len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        out[i] = min(1.0, running)
    return out


METRICS = {
    "safe": lambda r: r["safe_success"],
    "rsafe": lambda r: r["success"] and not r["collided_robot"],
    "success": lambda r: r["success"],
    "contact": lambda r: r["collided_any"],
    "robot_contact": lambda r: r["collided_robot"],
    "obj_only": lambda r: r["collided_any"] and not r["collided_robot"],
    "stall": lambda r: (not r["success"]) and (not r["collided_any"]),
}


def report(data, keys, title):
    print(f"\n######## {title}: {len(keys)} paired episodes "
          f"({sum(k[1] == 'libero_spatial' for k in keys)} spatial / {sum(k[1] == 'libero_object' for k in keys)} object)")
    print(f"{'arm':14s}" + "".join(f"{m:>9s}" for m in METRICS) + f"{'ms':>7s}")
    for a in ARMS:
        if a not in data:
            continue
        rs = [data[a][k] for k in keys if k in data[a]]
        if not rs:
            continue
        print(f"{a:14s}" + "".join(f"{np.mean([f(r) for r in rs]):9.3f}" for f in METRICS.values())
              + f"{np.mean([r['infer_ms_mean'] for r in rs]):7.0f}")
    for metric in ("safe", "rsafe"):
        f = METRICS[metric]
        print(f"paired {metric}: A-only/B-only, McNemar p, task-clustered p")
        prim = []
        for a, b in PRIMARY + SECONDARY:
            if a not in data or b not in data:
                continue
            kk = [k for k in keys if k in data[a] and k in data[b]]
            if not kk:
                continue
            wa = sum(f(data[a][k]) and not f(data[b][k]) for k in kk)
            wb = sum(f(data[b][k]) and not f(data[a][k]) for k in kk)
            cl = {}
            for k in kk:
                cl.setdefault((k[1], k[2]), []).append(float(f(data[a][k])) - float(f(data[b][k])))
            p, pc = mcnemar(wa, wb), clustered_p(cl)
            tag = "PRIMARY" if (a, b) in PRIMARY else ""
            print(f"  {a:13s} vs {b:11s}: {wa:3d}/{wb:<3d} (n={len(kk)}) p={p:.3f}  clustered p={pc:.3f} {tag}")
            if tag:
                prim.append((a, b, p, pc))
        if prim:
            hp = holm([x[2] for x in prim]); hc = holm([x[3] for x in prim])
            for (a, b, p, pc), h1, h2 in zip(prim, hp, hc):
                print(f"    Holm(n): {a} vs {b}: McNemar {h1:.3f}, clustered {h2:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="+", default=["gate17", "gate20"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=None, help="restrict to these arms (default: all with data)")
    ap.add_argument("--prefix", default="obs7", help="run-dir prefix: obs7 (pi0.5 runs) or groot (GR00T N1.7 runs)")
    args = ap.parse_args()
    global PREFIX
    PREFIX = args.prefix
    data = {a: {} for a in ARMS}
    for L, s in itertools.product(args.layouts, args.seeds):
        for a in ARMS:
            data[a].update(load(L, a, s))
    data = {a: d for a, d in data.items() if d and (args.arms is None or a in args.arms)}
    core = [a for a in data if a != "cbfobj_rep"]  # every arm must cover all keys; rep is gate17 seed 0 only
    keys = sorted(set.intersection(*[set(data[a]) for a in core]))
    keys = [k for k in keys if all(data[a][k]["obstacle_xy"] == data[core[0]][k]["obstacle_xy"] for a in core)]
    report(data, keys, f"POOLED {args.layouts} seeds {args.seeds}")
    for L in args.layouts:
        report(data, [k for k in keys if k[0] == L], L)
    for suite in ("libero_spatial", "libero_object"):
        report(data, [k for k in keys if k[1] == suite], f"POOLED {suite}")
    if "cbfobj_rep" in data:
        rk = [k for k in keys if k in data["cbfobj_rep"]]
        d = sum(data["cbfobj_rep"][k]["safe_success"] != data["cbfobj"][k]["safe_success"] for k in rk)
        print(f"\nreplicate noise floor (cbfobj_rep vs cbfobj, same seed): {d}/{len(rk)} safe outcomes differ")


if __name__ == "__main__":
    main()
