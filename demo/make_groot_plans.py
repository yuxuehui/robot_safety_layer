"""Merge the per-seed GR00T demo recordings (runs/video_groot/<layout>_<none|ours>_s<seed>) into per-arm dirs and write
compose_video plans (no task bar, seed cards, GR00T labels). Usage: python make_groot_plans.py single:0:3 gate17:0:3 ...
(layout:task:episode of libero_spatial)."""
import glob
import json
import os
import shutil
import sys

ROOT = os.environ.get("VIDEO_ROOT", "runs/video_groot")  # per-seed recordings root (chosen set may be symlinked here)

SUBTITLE = ("20 Hz, real time. Clearance: gripper sphere model to {obs} (guidance cost also models arm links and the held "
            "object). Red border = physics contact.")
for spec in sys.argv[1:]:
    L, task, ep = spec.split(":")
    task, ep = int(task), int(ep)
    hand = L.startswith("h")
    for arm in ("none", "ours"):
        d = f"runs/video_ab/g{L}_{arm}/libero_spatial"
        shutil.rmtree(f"runs/video_ab/g{L}_{arm}", ignore_errors=True)
        os.makedirs(f"{d}/record", exist_ok=True)
        n = 0
        with open(f"{d}/results_all.jsonl", "w") as out:
            for s in (0, 1, 2):
                src = f"{ROOT}/{L}_{arm}_s{s}/libero_spatial"
                for f in glob.glob(f"{src}/record/t{task:02d}_e{ep:02d}_s{s}.npz"):
                    shutil.copy(f, f"{d}/record/"); n += 1
                for f in glob.glob(f"{src}/results_*.jsonl"):
                    for line in open(f):
                        r = json.loads(line)
                        if "skipped" not in r and r["task_id"] == task and r["episode"] == ep:
                            out.write(line)
        print(f"{L} {arm}: {n} recordings")
    plan = {"none_run": f"runs/video_ab/g{L}_none", "guided_run": f"runs/video_ab/g{L}_ours",
            "left_label": "GR00T N1.7 (no guidance)", "right_label": "GR00T N1.7 + inference-time guidance", "fps": 20,
            "d_safe_cm": 1.0, "card_sec": 1.5, "show_task": False, "obstacle": "hand" if hand else "pillar",
            "subtitle": SUBTITLE.format(obs="hand" if hand else "pillar"),
            "clips": [{"suite": "libero_spatial", "task": task, "ep": ep, "seed": s, "card": [f"Seed {s + 1} / 3"]} for s in (0, 1, 2)]}
    json.dump(plan, open(f"runs/video_ab/plan_g{L}.json", "w"), indent=1)
    print("plan", f"runs/video_ab/plan_g{L}.json")
