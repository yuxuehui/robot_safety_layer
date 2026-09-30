"""Derive 'clean' compose_video plans (no task bar, minimal seed cards) from runs/video_ab/plan_<layout>.json."""
import json
import sys

for L in sys.argv[1:]:
    plan = json.load(open(f"runs/video_ab/plan_{L}.json"))
    plan["show_task"] = False
    plan["card_sec"] = 1.5
    for c in plan["clips"]:
        c["card"] = [f"Seed {c['seed'] + 1} / 3"]
    json.dump(plan, open(f"runs/video_ab/plan_{L}_clean.json", "w"), indent=1)
    print("wrote", f"runs/video_ab/plan_{L}_clean.json")
