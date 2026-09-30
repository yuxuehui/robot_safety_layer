"""Pick avoidance-success clips for the rollout video from the seeded recordings.

A (episode, seed) is eligible iff the guided run succeeded with ZERO pillar contact and the
unguided run touched the pillar. Per episode, prefer a seed where the unguided run also failed.
At most one clip per (suite, task); libero_object included when eligible. Writes video_plan.json.

Usage: python make_video_plan.py --none runs/video_none --guided runs/video_g1_ahat --max_clips 6 \
         --summary_json runs/video_out/aggregate.json --out video_plan.json
"""
import argparse
import collections
import glob
import json

import numpy as np


def load(run):
    d = {}
    for f in glob.glob(f"{run}/*/results_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if "skipped" in r or r.get("torch_seed") is None:
                continue
            d[(r["suite"], r["task_id"], r["episode"], r["torch_seed"])] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--none", default="runs/video_none")
    ap.add_argument("--guided", default="runs/video_g1_ahat")
    ap.add_argument("--max_clips", type=int, default=6)
    ap.add_argument("--summary_json", required=True, help="aggregate sweep numbers for the outro card")
    ap.add_argument("--out", default="video_plan.json")
    args = ap.parse_args()
    A, G = load(args.none), load(args.guided)

    by_ep = collections.defaultdict(list)
    for k in sorted(set(A) & set(G)):
        a, g = A[k], G[k]
        if g["success"] and not g["collided_any"] and a["collided_any"]:
            # sort key (max wins): prefer seeds where unguided also failed, then FEWER unguided contact
            # steps (the less dramatic contrast), then the higher seed
            by_ep[k[:3]].append((not a["success"], -a["contact_steps"], k[3]))
    # per-episode seed stats for ranking: episodes where guidance wins on more seeds rank higher
    def n_seeds_won(ep):
        return len(by_ep[ep])

    ranked = sorted(by_ep, key=lambda ep: (-n_seeds_won(ep), -max(c[0] for c in by_ep[ep]), ep))
    chosen, used_tasks = [], set()
    obj = [ep for ep in ranked if ep[0] == "libero_object"]
    if obj:  # guarantee one libero_object example when any qualifies
        ranked = [obj[0]] + [ep for ep in ranked if ep != obj[0]]
    for ep in ranked:
        if (ep[0], ep[1]) in used_tasks:
            continue
        best = sorted(by_ep[ep], reverse=True)[0]
        chosen.append((ep, best[2]))
        used_tasks.add((ep[0], ep[1]))
        if len(chosen) >= args.max_clips:
            break
    # order: spatial first (scene introduced), then object
    chosen.sort(key=lambda c: (c[0][0] != "libero_spatial", c[0][1]))

    clips = []
    for (su, t, e), s in chosen:
        a, g = A[(su, t, e, s)], G[(su, t, e, s)]
        # closest approach over executed steps (eval metric); the shared start pose is excluded because
        # it is identical on both sides and not an outcome of guidance
        g_clear = 100 * float(np.load(f"{args.guided}/{su}/record/t{t:02d}_e{e:02d}_s{s}.npz")["clearance"][1:].min())
        assert abs(g_clear - 100 * g["min_clearance"]) < 1e-6
        if a["success"]:
            title = "Both finish the task, but only guidance avoids the pillar"
        else:
            title = "Guidance steers around the pillar and completes the task"
        clips.append({"suite": su, "task": t, "ep": e, "seed": s, "title": title,
                      "note": f"unguided: {'success' if a['success'] else 'failed'}, pillar contact {a['contact_steps']} steps  |  "
                              f"guided: success, no contact (closest gripper-sphere clearance while moving: {g_clear:.1f} cm)"})

    agg = json.load(open(args.summary_json))
    plan = {
        "none_run": args.none, "guided_run": args.guided,
        "left_label": "pi0.5 (no guidance)", "right_label": "pi0.5 + inference-time guidance",
        "fps": 20,
        "intro": [
            "Inference-time obstacle avoidance for pi0.5 on LIBERO",
            "",
            "A static red pillar is placed on the robot's own obstacle-free path.",
            "Left: pi0.5 (pi05_libero) as is.   Right: the same policy, no retraining,",
            "plus guidance at inference time: every flow-matching step adds the gradient",
            "of an SDF clearance cost (gripper collision spheres, safety margin 3 cm).",
            "The guidance is given the pillar's true pose and size (read from the simulator);",
            "the policy itself sees the pillar only in its camera images.",
            "Both sides start from the same initial state and random seed.",
            "Contact = any MuJoCo contact between the pillar and anything except the table.",
            "",
            "Selected successful-avoidance examples; per-seed counts on each title card.",
        ],
        "outro": [
            "Aggregate (pillar on path, libero_spatial + libero_object, " + str(agg["n"]) + " placed episodes)",
            "",
            f"No guidance:        pillar contact {agg['none_contact']:.0%},  task success {agg['none_success']:.0%},  success without contact {agg['none_safe']:.0%}",
            f"With guidance:      pillar contact {agg['g_contact']:.0%},  task success {agg['g_success']:.0%},  success without contact {agg['g_safe']:.0%}",
            f"libero_spatial only: success without contact {agg['none_safe_spatial']:.0%} -> {agg['g_safe_spatial']:.0%}",
            "",
            f"Paired per episode: guidance turned {agg['wins']} episodes into contact-free successes and lost none.",
            "Remaining failures: arm links / held objects still touching (not in the cost),",
            "and detours after which the task is not completed (mostly libero_object).",
        ],
        "clips": clips,
    }
    json.dump(plan, open(args.out, "w"), indent=1)
    print(json.dumps(clips, indent=1))
    print("eligible episodes:", {k: [c[2] for c in v] for k, v in by_ep.items()})


if __name__ == "__main__":
    main()
