"""Merge the per-seed hand demo recordings into per-arm dirs and write compose_video plans (no task bar, hand texts).
Usage: python make_hand_plans.py sweep:2 reach:1   (motion:episode of libero_spatial task 0)
"""
import glob
import json
import os
import shutil
import sys

for spec in sys.argv[1:]:
    motion, ep = spec.split(":")
    ep = int(ep)
    for arm in ("none", "ours"):
        d = f"runs/video_ab/h{motion}_{arm}/libero_spatial"
        shutil.rmtree(f"runs/video_ab/h{motion}_{arm}", ignore_errors=True)
        os.makedirs(f"{d}/record", exist_ok=True)
        with open(f"{d}/results_all.jsonl", "w") as out:
            for s in (0, 1, 2):
                src = f"runs/video_hand/{motion}_{arm}_s{s}/libero_spatial"
                for f in glob.glob(f"{src}/record/t00_e{ep:02d}_s{s}.npz"):
                    shutil.copy(f, f"{d}/record/")
                for f in glob.glob(f"{src}/results_*.jsonl"):
                    for line in open(f):
                        r = json.loads(line)
                        if "skipped" not in r and r["episode"] == ep:
                            out.write(line)
    name = {"sweep": "hand sweeping across the workspace", "reach": "hand reaching for the same object"}[motion]
    plan = {"none_run": f"runs/video_ab/h{motion}_none", "guided_run": f"runs/video_ab/h{motion}_ours",
            "left_label": "pi0.5 (no guidance)", "right_label": "pi0.5 + inference-time guidance", "fps": 20,
            "d_safe_cm": 1.0, "card_sec": 1.5, "show_task": False, "obstacle": "hand",
            "clips": [{"suite": "libero_spatial", "task": 0, "ep": ep, "seed": s, "card": [f"Seed {s + 1} / 3"]} for s in (0, 1, 2)]}
    json.dump(plan, open(f"runs/video_ab/plan_h{motion}.json", "w"), indent=1)
    print("plan", f"runs/video_ab/plan_h{motion}.json", "->", name)
