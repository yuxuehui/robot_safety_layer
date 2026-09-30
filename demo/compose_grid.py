"""Compose one synchronized A/B grid video: rows = methods (no guidance / inference-time guidance),
columns = torch seeds, all on the SAME task + obstacle placement. Every frame's status text comes from the
recorded npz / results jsonl of that exact run (eval_obstacle.py --record).

Usage (pod3, openpi venv):
  python compose_grid.py --layout gate17 --suite libero_spatial --task 0 --ep 4 --seeds 0 1 2 \
      --runs runs/video_sep --out runs/video_sep/gate17_grid.mp4
"""
import argparse
import glob
import json
import pathlib

import imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf"
F = lambda size, bold=False: ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)

ROWS = [("none", "pi0.5  (no guidance)", (150, 150, 155)), ("ours", "pi0.5 + inference-time guidance", (80, 150, 240))]
PANEL = 384
GAP = 14
HEAD = 36
FOOT = 40
TOP = 58
BG, FG, DIM = (18, 18, 22), (235, 235, 235), (150, 150, 155)
RED, GREEN, ORANGE = (235, 64, 52), (60, 190, 110), (240, 160, 40)


def load(run, suite, task, ep, seed):
    z = np.load(f"{run}/{suite}/record/t{task:02d}_e{ep:02d}_s{seed}.npz")
    res = None
    for f in glob.glob(f"{run}/{suite}/results_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if "skipped" not in r and r["task_id"] == task and r["episode"] == ep:
                res = r
    assert res is not None, run
    return {"frames": z["frames"], "contact": z["contact"], "clearance": z["clearance"]}, res


def outcome(res):
    if res["safe_success"]:
        return "SUCCESS, no contact", GREEN
    if res["collided_any"]:
        return ("SUCCESS but CONTACT" if res["success"] else "CONTACT"), RED
    return "TIMEOUT (no contact)", ORANGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--runs", default="runs/video_sep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hold", type=float, default=2.5, help="seconds to hold the final frame")
    args = ap.parse_args()

    cells = {}
    for key, _, _ in ROWS:
        for s in args.seeds:
            cells[(key, s)] = load(f"{args.runs}/{args.layout}_{key}_s{s}", args.suite, args.task, args.ep, s)
    instruction = next(iter(cells.values()))[1]["task"]
    n_cols = len(args.seeds)
    W = n_cols * PANEL + (n_cols + 1) * GAP
    ROWT = 34  # full-width method title per row
    H = TOP + len(ROWS) * (ROWT + HEAD + PANEL + FOOT) + (len(ROWS) + 1) * GAP
    W += (-W) % 16; H += (-H) % 16
    T = max(len(c["frames"]) for c, _ in cells.values())
    n_frames = T + int(args.fps * args.hold)
    f_title, f_head, f_small, f_badge = F(22, True), F(17, True), F(15), F(16, True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8,
                           ffmpeg_params=["-pix_fmt", "yuv420p"])
    poster = None
    for k in range(n_frames):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((GAP, 14), f"Task: {instruction}", fill=FG, font=f_title)
        for ri, (key, label, col) in enumerate(ROWS):
            y0 = TOP + GAP + ri * (ROWT + HEAD + PANEL + FOOT + GAP)
            d.text((GAP, y0 + 4), label, fill=col, font=f_title)
            y0 += ROWT
            for ci, s in enumerate(args.seeds):
                x0 = GAP + ci * (PANEL + GAP)
                rec, res = cells[(key, s)]
                n = len(rec["frames"])
                j = min(k, n - 1)
                d.text((x0, y0 + 10), f"seed {s}", fill=DIM, font=f_head)
                fr = Image.fromarray(rec["frames"][j]).resize((PANEL, PANEL), Image.BILINEAR)
                img.paste(fr, (x0, y0 + HEAD))
                hit_now = bool(rec["contact"][j])
                hit_ever = bool(rec["contact"][: j + 1].any())
                if hit_ever:
                    d.rectangle([x0 - 3, y0 + HEAD - 3, x0 + PANEL + 2, y0 + HEAD + PANEL + 2], outline=RED, width=4 if hit_now else 2)
                yt = y0 + HEAD + PANEL + 8
                if k < n - 1:
                    clr = rec["clearance"][j]
                    txt = f"step {j:3d}/{res['steps']}    min clearance {100 * clr:5.1f} cm"
                    d.text((x0, yt), txt, fill=RED if hit_now else FG, font=f_small)
                    if hit_now:
                        d.text((x0 + PANEL - 92, yt), "CONTACT", fill=RED, font=f_badge)
                else:
                    txt, c = outcome(res)
                    d.text((x0, yt), f"{txt}   ({res['steps']} steps)", fill=c, font=f_badge)
        arr = np.asarray(img)
        if poster is None:
            poster = arr.copy()
        w.append_data(arr)
    w.close()
    Image.fromarray(poster).save(str(out.with_suffix(".png")))
    print(f"wrote {out} ({W}x{H}, {n_frames} frames @ {args.fps} fps) and poster {out.with_suffix('.png')}")
    for (key, s), (_, res) in sorted(cells.items()):
        print(f"  {key:5s} seed {s}: {outcome(res)[0]}, steps {res['steps']}, infer {res['infer_ms_mean']:.0f} ms")


if __name__ == "__main__":
    main()
