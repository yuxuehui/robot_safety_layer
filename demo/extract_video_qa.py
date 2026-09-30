"""Extract a QA bundle from runs/video_out: sampled frames, all-seed results, npz re-derived stats."""
import glob
import json
import os

import imageio
import numpy as np

out = "runs/video_out/qa"
os.makedirs(out, exist_ok=True)
r = imageio.get_reader("runs/video_out/rollouts_all.mp4")
frames = [f for f in r]
imageio.imwrite(f"{out}/full_intro.png", frames[60])
imageio.imwrite(f"{out}/full_outro.png", frames[-20])

man = json.load(open("runs/video_out/manifest.json"))
for c in man:
    clip_frames = [f for f in imageio.get_reader(f"runs/video_out/{c['file']}")]
    m = len(clip_frames)
    title_n = 100  # 5 s title card at 20 fps
    L = max(c["record_len"])
    contact_l = c["contact_frames_left"]
    idx = {50, title_n + L - 1, m - 1}
    idx |= {title_n + k for k in range(0, L, 25)}
    if contact_l:
        idx |= {title_n + k for k in contact_l[:: max(1, len(contact_l) // 4)][:5]}
    idx = sorted(i for i in idx if 0 <= i < m)
    for i in idx:
        imageio.imwrite(f"{out}/clip{c['clip']}_f{i:04d}.png", clip_frames[i])
    c["qa_frames"] = [f"clip{c['clip']}_f{i:04d}.png" for i in idx]
    c["video_frames_total"] = m
    c["title_card_frames"] = title_n
json.dump(man, open(f"{out}/manifest_qa.json", "w"), indent=1)

recs = []
for run in ["video_none", "video_g1_ahat"]:
    for f in glob.glob(f"runs/{run}/*/results_*.jsonl"):
        for line in open(f):
            x = json.loads(line)
            x["run"] = run
            recs.append(x)
json.dump(recs, open(f"{out}/video_results_all_seeds.json", "w"), indent=1)

chk = []
for c in man:
    for side, run in (("left", "video_none"), ("right", "video_g1_ahat")):
        z = np.load(f"runs/{run}/{c['suite']}/record/t{c['task']:02d}_e{c['ep']:02d}_s{c['seed']}.npz")
        hits = json.loads(str(z["hits"]))
        chk.append({
            "clip": c["clip"], "side": side, "n_frames": int(len(z["frames"])),
            "contact_frames": int(z["contact"].sum()),
            "min_clear_cm": float(100 * z["clearance"].min()),
            "first_contact_frame": int(np.argmax(z["contact"])) if z["contact"].any() else None,
            "geoms_touched": sorted({g for h in hits for g in h}),
        })
json.dump(chk, open(f"{out}/npz_rederived.json", "w"), indent=1)
print(len(os.listdir(out)), "qa files")
