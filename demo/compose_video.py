"""Compose side-by-side rollout videos (no guidance | G1 guidance [| more arms]) from eval_obstacle.py --record output.

Usage (pod3, openpi venv):
  python demo/compose_video.py --plan demo/plans/<plan>.json --out runs/video_out   (from the project root)

plan JSON:
{
  "none_run": "runs/video_none", "guided_run": "runs/video_g1_ahat",
  "left_label": "pi0.5 (no guidance)", "right_label": "pi0.5 + G1 guidance",
  (or, for N panels: "runs": [...], "labels": [...])
  "fps": 20,
  "intro": ["line", ...], "outro": ["line", ...],
  "clips": [{"suite": "libero_spatial", "task": 0, "ep": 3, "seed": 1,
             "title": "...", "note": "..."}]
}
Every number printed on a frame comes from the recorded npz / results jsonl of that exact run.
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

PANEL = 512
GAP = 16
TOP = 60  # task instruction bar
HEAD = 40  # per-panel label
FOOT = 108  # per-panel status text + clearance timeline (H multiple of 8 for the encoder)
NPANEL = 2
W = NPANEL * PANEL + (NPANEL + 1) * GAP  # 1072 for two panels
H = TOP + HEAD + PANEL + FOOT + GAP  # 736 (TOP is set to 0 when the plan has "show_task": false)
BG = (18, 18, 22)
FG = (235, 235, 235)
DIM = (150, 150, 155)
RED = (235, 64, 52)
GREEN = (60, 190, 110)
ORANGE = (240, 160, 40)
BLUE = (80, 150, 240)
PURPLE = (170, 110, 230)
SHOW_TASK = True
D_SAFE_CM = 3.0  # dashed reference line in the clearance timeline; overridden by plan["d_safe_cm"]
SUBTITLE = ("20 Hz, real time. Clearance: gripper sphere model to pillar, approximate (as in the guidance cost; "
            "arm links and held objects not modelled). Red border = physics contact.")
OBSTACLE = "pillar"  # word used in the on-frame texts; plan["obstacle"] overrides (e.g. "hand")


def load_results(run, suite):
    out = {}
    for f in glob.glob(f"{run}/{suite}/results_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if "skipped" in r:
                continue
            out[(r["task_id"], r["episode"], r.get("torch_seed"))] = r
    return out


def load_record(run, suite, task, ep, seed):
    z = np.load(f"{run}/{suite}/record/t{task:02d}_e{ep:02d}_s{seed}.npz")
    return {
        "frames": z["frames"], "wrist": z["wrist"], "contact": z["contact"].astype(bool),
        "clear_cm": 100 * z["clearance"], "hits": json.loads(str(z["hits"])),
    }


def load_aux(run, suite, task, ep, seed):
    """Per-chunk sampler logs + top-view geometry of a recorded episode (None when the run has no traj logs)."""
    tag = f"t{task:02d}_e{ep:02d}_s{seed}"
    try:
        chunks = json.load(open(f"{run}/{suite}/traj/{tag}_chunks.json"))
        z = np.load(f"{run}/{suite}/traj/{tag}.npz")
    except FileNotFoundError:
        return None
    return {"chunks": chunks, "obstacle_xy": z["obstacle_xy"], "eef": z["eef_pos"]}


def chunk_of(aux, kk, exec_steps):
    """Chunk whose actions produced frame kk (frame k = state after k executed actions)."""
    if aux is None or not aux["chunks"]:
        return None
    return aux["chunks"][min(max(kk - 1, 0) // exec_steps, len(aux["chunks"]) - 1)]


def chunk_status(c):
    """(guidance push cm or None, CAR world vector (cm/step) or None, CAR gate fraction)."""
    if c is None:
        return None, None, 0.0
    push = c.get("push_cm") if (c.get("cost_first") or 0) > 0 else None
    gate = float(c.get("car_gate_frac") or 0.0)
    vec = c.get("car_vec_world_cm") if gate > 0 else None
    return push, vec, gate


def draw_topview(img, aux, kk, exec_steps, box=150, pillar_r_cm=2.5):
    """Top view (world x right, y up) of the pillars, the end-effector trail and the current sampler action."""
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0 = 10, PANEL - box - 10
    d.rectangle([x0, y0, x0 + box, y0 + box], fill=(0, 0, 0, 150), outline=(200, 200, 200, 200))
    pts = np.concatenate([aux["eef"][:, :2], aux["obstacle_xy"]], axis=0)
    lo, hi = pts.min(0) - 0.06, pts.max(0) + 0.06
    span = float(max(hi - lo)); ctr = (lo + hi) / 2
    sc = (box - 12) / span  # px per m
    P = lambda xy: (x0 + box / 2 + (xy[0] - ctr[0]) * sc, y0 + box / 2 - (xy[1] - ctr[1]) * sc)
    for o in aux["obstacle_xy"]:
        cx, cy = P(o); r = pillar_r_cm / 100 * sc
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 50, 40, 230))
    trail = [P(e) for e in aux["eef"][: kk + 1, :2]]
    if len(trail) > 1:
        d.line(trail, fill=(230, 230, 230, 200), width=2)
    cx, cy = trail[-1]
    push, vec, gate = chunk_status(chunk_of(aux, kk, exec_steps))
    if push is not None:
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=ORANGE + (255,), width=2)
    if vec is not None:
        ARROW_GAIN = 3.0  # the xy correction over the executed steps is only 1-3 cm: drawn x3 so it is readable
        dx, dy = ARROW_GAIN * vec[0] * exec_steps / 100 * sc, -ARROW_GAIN * vec[1] * exec_steps / 100 * sc
        ex, ey = cx + dx, cy + dy
        d.line([cx, cy, ex, ey], fill=PURPLE + (255,), width=3)
        ang = np.arctan2(dy, dx)
        for a in (ang + 2.6, ang - 2.6):
            d.line([ex, ey, ex + 7 * np.cos(a), ey + 7 * np.sin(a)], fill=PURPLE + (255,), width=3)
    d.ellipse([cx - 3.5, cy - 3.5, cx + 3.5, cy + 3.5], fill=(255, 255, 255, 255))
    d.text((x0 + 4, y0 + 2), "top view", font=F(12), fill=(255, 255, 255, 230))
    d.text((x0 + box - 4, y0 + box - 14), f"{span*100:.0f} cm", font=F(11), fill=(200, 200, 200, 230), anchor="ra")
    return push, vec, gate


def short_geom(name):
    return (name.replace("robot0_", "arm ").replace("gripper0_", "gripper ").replace("_collision", "")
            .replace("_", " "))


def clear_color(c):
    return GREEN if c >= D_SAFE_CM else (ORANGE if c >= 0 else RED)


def draw_panel(canvas, x0, rec, res, k, L, label, label_color, aux=None, exec_steps=5):
    d = ImageDraw.Draw(canvas)
    n = len(rec["frames"])
    kk = min(k, n - 1)
    ended = k >= n - 1 and k > 0
    # header
    d.rectangle([x0, TOP, x0 + PANEL, TOP + HEAD], fill=label_color)
    d.text((x0 + 12, TOP + 8), label, font=F(20, True), fill=(255, 255, 255))
    # frame + wrist inset
    img = Image.fromarray(rec["frames"][kk])
    # wrist inset bottom-right: top-right hid the pillar top in libero_object; bottom-right is floor/cabinet
    wrist = Image.fromarray(rec["wrist"][kk]).resize((136, 136))
    img.paste(Image.new("RGB", (140, 140), (255, 255, 255)), (PANEL - 150, PANEL - 150))
    img.paste(wrist, (PANEL - 148, PANEL - 148))
    di = ImageDraw.Draw(img)
    di.text((PANEL - 146, PANEL - 168), "wrist cam", font=F(13), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    if aux is not None and not (k >= n - 1 and k > 0):
        push, vec, gate = draw_topview(img, aux, kk, exec_steps)
        di = ImageDraw.Draw(img)
        yy = 8
        if push is not None:
            di.text((10, yy), f"guidance push {push:.1f} cm", font=F(17, True), fill=ORANGE, stroke_width=2, stroke_fill=(0, 0, 0)); yy += 24
        if vec is not None:
            di.text((10, yy), f"CAR correction {np.linalg.norm(vec):.1f} cm/step  (gate {100*gate:.0f} %)", font=F(17, True), fill=PURPLE,
                    stroke_width=2, stroke_fill=(0, 0, 0)); yy += 24
            di.text((10, yy), f"xy: purple arrow in top view (x3),  z: {vec[2]:+.1f} cm/step ({'up' if vec[2] > 0 else 'down'})", font=F(15), fill=PURPLE,
                    stroke_width=2, stroke_fill=(0, 0, 0))
    touching = bool(rec["contact"][kk])
    if touching:
        for w in range(7):
            di.rectangle([w, w, PANEL - 1 - w, PANEL - 1 - w], outline=RED)
    if ended:
        ov = Image.new("RGBA", img.size, (0, 0, 0, 110))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        di = ImageDraw.Draw(img)
        ok = res["success"]
        # two separate facts: task outcome, and pillar contact (not implying one caused the other)
        di.text((PANEL // 2, PANEL // 2 - 34), "TASK SUCCESS" if ok else "TASK NOT COMPLETED", font=F(36, True),
                fill=GREEN if ok else RED, anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
        if not ok:
            di.text((PANEL // 2, PANEL // 2 + 2), f"(time limit: {res['steps']} steps)", font=F(17), fill=FG,
                    anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
        hit = res["collided_any"]
        sub = (f"{OBSTACLE} contact during {res['contact_steps']} steps" if hit else f"no {OBSTACLE} contact")
        di.text((PANEL // 2, PANEL // 2 + 36), sub, font=F(22, True), fill=RED if hit else GREEN, anchor="mm",
                stroke_width=3, stroke_fill=(0, 0, 0))
    canvas.paste(img, (x0, TOP + HEAD))
    # footer text
    y = TOP + HEAD + PANEL + 6
    c = float(rec["clear_cm"][kk])
    d.text((x0 + 4, y), f"step {kk:3d}", font=F(17), fill=DIM)
    d.text((x0 + 100, y), f"gripper clearance to {OBSTACLE} {c:5.1f} cm", font=F(17, True), fill=clear_color(c))
    if touching:
        names = ", ".join(short_geom(h) for h in rec["hits"][kk][:2])
        d.text((x0 + 4, y + 24), f"CONTACT: {names}", font=F(17, True), fill=RED)
    elif any(rec["contact"][: kk + 1]):
        d.text((x0 + 4, y + 24), f"contact so far: {int(rec['contact'][: kk + 1].sum())} steps", font=F(16), fill=ORANGE)
    else:
        d.text((x0 + 4, y + 24), "no contact so far", font=F(16), fill=DIM)
    # clearance timeline; labels live in a right gutter so the trace never covers them
    GUT = 76
    ty0, ty1 = y + 50, y + 92
    px1 = x0 + PANEL - GUT
    d.rectangle([x0, ty0, px1, ty1], outline=(70, 70, 75))
    lo, hi = -5.0, 15.0
    ymap = lambda v: ty1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (ty1 - ty0)
    xmap = lambda i: x0 + i / max(L - 1, 1) * (px1 - x0)
    for ref, col in ((0.0, (120, 60, 60)), (D_SAFE_CM, (90, 90, 60))):
        yy = ymap(ref)
        for xs in range(x0, px1, 8):
            d.line([xs, yy, min(xs + 4, px1), yy], fill=col)
    if aux is not None:  # activity strips: orange = guidance pushed this chunk, purple = CAR correction gated on
        for ci, c in enumerate(aux["chunks"]):
            xa, xb = xmap(ci * exec_steps + 1), xmap(min((ci + 1) * exec_steps, L - 1))
            if (c.get("cost_first") or 0) > 0:
                d.rectangle([xa, ty0 + 1, xb, ty0 + 4], fill=ORANGE)
            if float(c.get("car_gate_frac") or 0) > 0:
                d.rectangle([xa, ty0 + 5, xb, ty0 + 8], fill=PURPLE)
    cc = rec["clear_cm"]
    for i in range(1, min(kk + 1, n)):
        d.line([xmap(i - 1), ymap(cc[i - 1]), xmap(i), ymap(cc[i])],
               fill=RED if rec["contact"][i] else (200, 200, 205), width=2)
    d.line([xmap(kk), ty0, xmap(kk), ty1], fill=(255, 255, 255))
    d.text((px1 + 4, ty0), "15 cm", font=F(11), fill=DIM, anchor="lt")
    d.text((px1 + 4, ymap(D_SAFE_CM) - 6), f"margin {D_SAFE_CM:g}", font=F(11), fill=(170, 170, 90), anchor="lm")
    d.text((px1 + 4, ymap(0) + 5), "0 cm", font=F(11), fill=(170, 90, 90), anchor="lm")


def clip_frames(recs, results, instruction, labels, colors, hold, auxs=None, exec_steps=5):
    L = max(len(r["frames"]) for r in recs)
    auxs = auxs or [None] * len(recs)
    for k in range(L + hold):
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)
        if SHOW_TASK:  # task line + subtitle (omitted when the slide already shows them)
            text, size = "Task: " + instruction, 22
            while size > 13 and d.textlength(text, font=F(size, True)) > W - 2 * GAP:
                size -= 1
            d.text((GAP, 10), text, font=F(size, True), fill=FG)
            sub, ssize = SUBTITLE, 13
            while ssize > 10 and d.textlength(sub, font=F(ssize)) > W - 2 * GAP:
                ssize -= 1
            d.text((GAP, 38), sub, font=F(ssize), fill=DIM)
        for j in range(len(recs)):
            # a panel whose episode ended earlier holds its last frame with the outcome stamp
            draw_panel(canvas, GAP + j * (PANEL + GAP), recs[j], results[j], k, L, labels[j], colors[j], auxs[j], exec_steps)
        yield np.asarray(canvas)


def card(lines, n_frames, big_first=True):
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    y = H // 2 - 24 * len(lines) + 40
    for i, line in enumerate(lines):
        if not line:
            y += 18
            continue
        bold = i == 0 and big_first
        size = 30 if bold else 20
        while size > 13 and d.textlength(line, font=F(size, bold)) > W - 2 * GAP:  # never clip at the edges
            size -= 1
        f = F(size, bold)
        col = FG if i == 0 else (200, 200, 205)
        d.text((W // 2, y), line, font=f, fill=col, anchor="mm")
        y += 48 if i == 0 else 32
    arr = np.asarray(canvas)
    for _ in range(n_frames):
        yield arr


def seed_summary(res_by_seed):
    n = len(res_by_seed)
    s = sum(r["success"] for r in res_by_seed)
    c = sum(r["collided_any"] for r in res_by_seed)
    return f"success {s}/{n}, touched pillar {c}/{n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fps = plan.get("fps", 20)
    global D_SAFE_CM, SUBTITLE, TOP, H, SHOW_TASK, OBSTACLE, NPANEL, W
    OBSTACLE = plan.get("obstacle", OBSTACLE)
    if not plan.get("show_task", True):
        SHOW_TASK = False
        TOP = GAP
        H = TOP + HEAD + PANEL + FOOT + GAP
        H += (-H) % 8
    D_SAFE_CM = float(plan.get("d_safe_cm", D_SAFE_CM))
    SUBTITLE = plan.get("subtitle", SUBTITLE)
    card_sec = float(plan.get("card_sec", 5))
    runs = plan.get("runs") or [plan["none_run"], plan["guided_run"]]
    labels = plan.get("labels") or [plan["left_label"], plan["right_label"]]
    NPANEL = len(runs)
    W = NPANEL * PANEL + (NPANEL + 1) * GAP
    colors = [(90, 90, 100), BLUE, (150, 90, 210), (40, 160, 150)][:NPANEL]
    overlay, exec_steps = bool(plan.get("overlay", False)), int(plan.get("exec_steps", 5))
    cache = {}

    writer_all = imageio.get_writer(out / "rollouts_all.mp4", fps=fps, codec="libx264", quality=8,
                                    macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for fr in card(plan["intro"], int(fps * 6)) if plan.get("intro") else []:
        writer_all.append_data(fr)
    manifest = []
    for ci, clip in enumerate(plan["clips"]):
        su, t, e, s = clip["suite"], clip["task"], clip["ep"], clip["seed"]
        for run in runs:
            if (run, su) not in cache:
                cache[(run, su)] = load_results(run, su)
        results = [cache[(run, su)][(t, e, s)] for run in runs]
        recs = [load_record(run, su, t, e, s) for run in runs]
        auxs = [load_aux(run, su, t, e, s) for run in runs] if overlay else None
        seeds = sorted({k[2] for k in cache[(runs[0], su)] if k[0] == t and k[1] == e})
        per_seed = [[cache[(run, su)][(t, e, x)] for x in seeds if (t, e, x) in cache[(run, su)]] for run in runs]
        title = clip.get("card") or ([f"Example {ci + 1}/{len(plan['clips'])}: {clip['title']}",
                 f"{su}, task {t}, episode {e}   (showing seed {s})",
                 clip.get("note", ""), "",
                 f"All {len(seeds)} seeds of this episode:"]
                 + [f"{labels[j]}: {seed_summary(per_seed[j])}" for j in range(NPANEL)])
        path = out / f"clip{ci + 1}_{su}_t{t}_e{e}_s{s}.mp4"
        w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8,
                               ffmpeg_params=["-pix_fmt", "yuv420p"])
        for fr in card(title, int(fps * card_sec)):
            w.append_data(fr); writer_all.append_data(fr)
        n_written = 0
        for fr in clip_frames(recs, results, results[0]["task"], labels, colors, hold=int(fps * 2.5), auxs=auxs, exec_steps=exec_steps):
            w.append_data(fr); writer_all.append_data(fr); n_written += 1
            if n_written in (1,) or n_written % 40 == 0:
                Image.fromarray(fr).save(out / f"qa_clip{ci + 1}_f{n_written:04d}.png")
        Image.fromarray(fr).save(out / f"qa_clip{ci + 1}_final.png")
        w.close()
        manifest.append({"clip": ci + 1, "file": path.name, **clip,
                         "panels": [{"label": labels[j], "run": runs[j],
                                     **{k: results[j][k] for k in ("success", "collided_any", "contact_steps", "steps", "min_clearance", "hit_geoms")},
                                     "per_seed": seed_summary(per_seed[j]),
                                     "contact_frames": np.flatnonzero(recs[j]["contact"]).tolist()} for j in range(NPANEL)],
                         "record_len": [len(r["frames"]) for r in recs]})
        print(f"clip {ci + 1}: {path.name}", flush=True)
    for fr in card(plan["outro"], int(fps * 8)) if plan.get("outro") else []:
        writer_all.append_data(fr)
    writer_all.close()
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print("wrote", out / "rollouts_all.mp4")


if __name__ == "__main__":
    main()
