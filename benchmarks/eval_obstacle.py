"""LIBERO + static pillar obstacle, pi05_libero with optional inference-time guidance.

Placement: for each (task, episode) the pillar is put on that episode's own
no-obstacle baseline EEF path (same init state + seed, from eval_libero.py), at a
fraction of the path's horizontal arc length, shifted sideways by --offset
(0 = directly on the path). Placements too close to task objects / the start pose,
or where the path already passes above the pillar top, are rejected.

Usage (from the repository root, `pip install -e .` done, openpi and LIBERO installed):
  python benchmarks/eval_obstacle.py --suite libero_spatial --tasks 0-4 --episodes 10 --guidance none
  python benchmarks/eval_obstacle.py ... --guidance g1 --scale 1.0 --calib runs/baseline/libero_spatial/action_model_calibration.npz
Obstacle-free baseline rollouts (eval_libero.py, --baseline_dir, default runs/baseline) must exist first.
"""
import argparse
import collections
import json
import os
import pathlib
import time

import imageio
import torch
import numpy as np
from libero.libero import benchmark

import guidance as gd
import hand as hd
import obstacle as ob
import safeguide as sg
from eval_libero import DUMMY_ACTION, MAX_STEPS, make_env, parse_range, policy_input
from openpi.policies import policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


def object_xy(env):
    e = env.env
    return {k: e.sim.data.body_xpos[e.obj_body_id[k]][:2].copy() for k in e.objects_dict}


def object_xyz(env):
    e = env.env
    return {k: e.sim.data.body_xpos[e.obj_body_id[k]].copy() for k in e.objects_dict}


def place_on_path(env, path, objs_xy, spec, frac, offset, min_obj_dist, min_start_dist):
    """Return (xy, info) or (None, reason)."""
    xy = path[:, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 0.10:
        return None, "path_too_short"
    for f in [frac] + [frac + d for d in (-0.05, 0.05, -0.1, 0.1, -0.15, 0.15, -0.2, 0.2)]:
        if not 0.1 <= f <= 0.9:
            continue
        i = int(np.searchsorted(s, f * s[-1]))
        i = int(np.clip(i, 3, len(path) - 4))
        d = xy[i + 3] - xy[i - 3]
        if np.linalg.norm(d) < 1e-4:
            continue
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        c = xy[i] + offset * n
        base_z = ob.support_z(env, c)  # table or floor height differs across LIBERO scenes
        if path[i, 2] > base_z + spec.height - 0.02:  # baseline already passes above the pillar here
            continue
        if min(np.linalg.norm(c - o) for o in objs_xy.values()) < min_obj_dist:
            continue
        if np.linalg.norm(c - xy[0]) < min_start_dist:
            continue
        return c, {"frac": f, "path_idx": i, "path_z": float(path[i, 2]), "base_z": base_z, "normal": n.tolist()}
    return None, "no_valid_candidate"


def baseline_clearance(env, path, actions, xys, s_gate, spec, base_z, objs_xy, args):
    """Clean-gate check: min over the obstacle-free baseline EEF path of (xy distance to a pillar surface minus
    the held object's footprint after the grasp), ignoring points within --clean_window arc length of the gate
    crossing and points that pass above the pillars. Returns (min_clear, grasp_step, held_name, r_obj)."""
    xy = path[:, :2]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    grasp = next((k for k in range(len(actions)) if actions[k, 6] > 0), None)
    held, r_obj, hh = None, 0.0, 0.0
    if grasp is not None and objs_xy:
        held = min(objs_xy, key=lambda k: np.linalg.norm(objs_xy[k] - xy[min(grasp, len(xy) - 1)]))
        r_obj, hh = ob.object_footprint(env, held)
    top = base_z + spec.height
    best = np.inf
    for i in range(len(xy)):
        if abs(s[i] - s_gate) < args.clean_window:
            continue
        carried = grasp is not None and i > grasp
        if path[i, 2] - (hh if carried else 0.0) > top + 0.02:
            continue
        d = min(np.linalg.norm(xy[i] - c) for c in xys) - spec.radius - (r_obj if carried else 0.0)
        best = min(best, d)
    return float(best), grasp, held, r_obj


def place_layout(env, path, objs_xy, spec, args, actions=None):
    """Pillar xy positions for --layout.  single: one pillar on the path (as before);
    gate: two pillars flanking the path point at +-gate_half_gap along the path normal (the
    gripper cannot pass between them, and their avoidance gradients point in opposite directions);
    seq: one pillar at --frac and a second one further along the path at --frac2."""
    if args.layout == "single":
        xy, info = place_on_path(env, path, objs_xy, spec, args.frac, args.offset, args.min_obj_dist, args.min_start_dist)
        return (None, info) if xy is None else ([xy], info)
    if args.layout == "gate":
        # try every candidate point along the path (nearest to --frac first) until both pillars fit
        import collections
        rej, best_clean = collections.Counter(), -np.inf
        for f in sorted(np.arange(0.15, 0.86, 0.05), key=lambda f: abs(f - args.frac)):
            c, info = place_on_path(env, path, objs_xy, spec, float(f), args.offset, 0.0, args.min_start_dist)
            if c is None or abs(info["frac"] - f) > 1e-6:
                rej["no_point"] += 1
                continue
            n = np.array(info["normal"])
            xys = [c + args.gate_half_gap * n, c - args.gate_half_gap * n]
            if any(min(np.linalg.norm(xy - o) for o in objs_xy.values()) < args.min_obj_dist for xy in xys):
                rej["near_object"] += 1
                continue
            if any(np.linalg.norm(xy - path[0, :2]) < args.min_start_dist for xy in xys):
                rej["near_start"] += 1
                continue
            if args.clean_gate > 0:
                s_arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))])
                mc, grasp, held, r_obj = baseline_clearance(env, path, actions, xys, float(s_arc[info["path_idx"]]),
                                                            spec, info["base_z"], objs_xy, args)
                info.update({"clean_min_clear": mc, "grasp_step": grasp, "held_obj": held, "held_r_xy": r_obj})
                best_clean = max(best_clean, mc)
                if mc < args.clean_gate:
                    rej["baseline_near_pillar"] += 1
                    continue
            info["gate_center"] = c.tolist()
            return xys, info
        # every candidate crossing failed: report the counts (and the best baseline clearance seen)
        return None, "gate_rejected:" + ",".join(f"{k}={v}" for k, v in sorted(rej.items())) + (
            f":best_clean={best_clean:.3f}" if np.isfinite(best_clean) else "")
    if args.layout == "seq":
        xy1, info1 = place_on_path(env, path, objs_xy, spec, args.frac, args.offset, args.min_obj_dist, args.min_start_dist)
        xy2, info2 = place_on_path(env, path, objs_xy, spec, args.frac2, args.offset, args.min_obj_dist, args.min_start_dist)
        if xy1 is None or xy2 is None:
            return None, "seq_no_valid_candidate"
        if np.linalg.norm(xy1 - xy2) < 2 * spec.radius + 0.10:
            return None, "seq_pillars_too_close"
        return [xy1, xy2], {"first": info1, "second": info2}
    raise ValueError(args.layout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replan_steps", type=int, default=5)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--baseline_dir", default="runs/baseline", help="obstacle-free rollouts from eval_libero.py")
    # obstacle
    ap.add_argument("--radius", type=float, default=0.025)
    ap.add_argument("--height", type=float, default=0.30)
    ap.add_argument("--invisible", action="store_true", help="collision-only pillar (not rendered)")
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--layout", default="single", choices=["single", "gate", "seq", "hand"])
    ap.add_argument("--hand_motion", default="sweep", choices=["sweep", "reach"], help="layout hand: scripted motion")
    ap.add_argument("--hand_visible", type=int, default=1, help="layout hand: 1 = skin-coloured (the policy sees it), 0 = collision-only")
    ap.add_argument("--hand_predict", default="cv", choices=["cv", "oracle", "static"], help="layout hand: pose prediction used by the guidance cost")
    ap.add_argument("--hand_speed", type=float, default=0.3, help="layout hand: m/s (x U(0.85,1.15) per episode)")
    ap.add_argument("--hand_trigger", type=float, default=0.18, help="layout hand: gripper distance (m) that triggers the motion")
    ap.add_argument("--hand_rest", type=float, default=0.45, help="layout hand: rest distance of the fingertip from the path (m)")
    ap.add_argument("--extra_steps", type=int, default=0, help="added to the suite's step limit (hand layout: the robot must wait for the hand)")
    ap.add_argument("--margin_hand", type=float, default=0.05, help="layout hand: extra radius added to the hand capsules in the cost (human safety margin)")
    ap.add_argument("--hand_margin_tau", type=float, default=0.3, help="layout hand: cost inflation += hand speed * tau (reaction time, s)")
    ap.add_argument("--gate_half_gap", type=float, default=0.08, help="gate: pillar centres at +-this along the path normal")
    ap.add_argument("--frac2", type=float, default=0.65, help="seq: arc fraction of the second pillar")
    ap.add_argument("--min_obj_dist", type=float, default=0.10)
    ap.add_argument("--min_start_dist", type=float, default=0.10)
    ap.add_argument("--clean_gate", type=float, default=0.0,
                    help="gate: require the obstacle-free baseline EEF path (dilated by the held object after the grasp) "
                         "to clear both pillar surfaces by this much everywhere except at the crossing (0 = off)")
    ap.add_argument("--clean_window", type=float, default=0.10, help="arc length (m) around the crossing exempt from --clean_gate")
    ap.add_argument("--model_object", type=int, default=0, help="1: add the held task object's collision spheres to the robot model")
    ap.add_argument("--arm_motion", default="alpha", choices=["alpha", "jac"],
                    help="arm-sphere motion model: heuristic alpha * dEEF, or Jacobian-based (dynamically consistent OSC inverse)")
    ap.add_argument("--log_arm_pred", type=int, default=0, help="1: log predicted vs actual arm-sphere motion per chunk (both models)")
    ap.add_argument("--arm_release", type=int, default=0, help="1: once the gripper has crossed the gate line, use --arm_margin_post for arm spheres")
    ap.add_argument("--arm_margin_post", type=float, default=0.0)
    ap.add_argument("--release_hyst", type=float, default=0.03, help="m past the gate line before the release latches")
    ap.add_argument("--escape_chunks", type=int, default=0, help="N>0: deadlock = guidance active on the last N chunks and EEF net displacement < --escape_min_disp -> lift")
    ap.add_argument("--escape_min_disp", type=float, default=0.02)
    ap.add_argument("--escape_lift", type=float, default=0.015, help="m per control step of upward offset during an escape")
    ap.add_argument("--escape_len", type=int, default=2, help="chunks per escape")
    ap.add_argument("--rewind", type=int, default=0, help="1: state-level recovery - rewind to the last in-distribution state when the policy dithers after a detour")
    ap.add_argument("--rewind_thr", type=float, default=0.06, help="m: nearest-neighbour distance to the baseline rollouts above which a state is OOD")
    ap.add_argument("--rewind_window", type=int, default=6)
    ap.add_argument("--rewind_net", type=float, default=0.08)
    ap.add_argument("--rewind_ratio", type=float, default=2.0)
    ap.add_argument("--rewind_quiet", type=int, default=2)
    ap.add_argument("--rewind_max", type=int, default=2)
    ap.add_argument("--rewind_step", type=float, default=0.012)
    ap.add_argument("--rewind_cooldown", type=int, default=6)
    ap.add_argument("--rewind_ood_window", type=int, default=3)
    ap.add_argument("--rewind_shortcut", type=int, default=1)
    ap.add_argument("--margin_obj", type=float, default=None, help="CBF/hinge margin for held-object spheres (default: margin_grip)")
    # guidance
    ap.add_argument("--guidance", default="none", choices=["none", "g1", "bestofn", "project", "oc", "car", "mppi"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--n_samples", type=int, default=8, help="bestofn")
    ap.add_argument("--project_steps", type=int, default=20, help="project")
    ap.add_argument("--project_lr", type=float, default=0.05, help="project")
    ap.add_argument("--oc_iters", type=int, default=10, help="oc: max iterations per chunk")
    ap.add_argument("--oc_lr", type=float, default=0.05, help="oc: step per unit-RMS gradient")
    ap.add_argument("--oc_reg", type=float, default=0.0, help="oc: lambda on sum ||u_k||^2")
    ap.add_argument("--oc_decay", type=float, default=1.0, help="oc: u <- decay*u - lr*g")
    ap.add_argument("--car_batch", type=int, default=64)
    ap.add_argument("--car_train_steps", type=int, default=1)
    ap.add_argument("--car_thr", type=float, default=0.15)
    ap.add_argument("--car_temp", type=float, default=0.1)
    ap.add_argument("--car_reward_temp", type=float, default=1.0)
    ap.add_argument("--car_obs_sigma", type=float, default=0.03)
    ap.add_argument("--grad_through_model", type=int, default=1)
    ap.add_argument("--schedule", default="const")
    ap.add_argument("--t_min", type=float, default=0.0)
    ap.add_argument("--d_safe", type=float, default=0.03)
    ap.add_argument("--cost_type", default="hinge", choices=["hinge", "energy", "cbf"])
    ap.add_argument("--margin_grip", type=float, default=None, help="per-sphere margin for gripper spheres (default d_safe)")
    ap.add_argument("--margin_arm", type=float, default=None, help="per-sphere margin for arm spheres (default d_safe)")
    ap.add_argument("--cbf_gamma", type=float, default=0.3)
    ap.add_argument("--cbf_vref", type=float, default=0.01)
    ap.add_argument("--steer", action="store_true", help="G1: steer, don't brake (tail braking -> latched sideways detour)")
    ap.add_argument("--steer_release", type=float, default=0.10)
    ap.add_argument("--steer_forget", type=float, default=0.35)
    ap.add_argument("--trigger_exec_only", action="store_true", help="ablation: guide only on executed-step violations")
    ap.add_argument("--car_zero_thr", type=float, default=0.0)
    ap.add_argument("--car_conflict", default="pillars", choices=["pillars", "progress", "both"])
    ap.add_argument("--car_prog_thr", type=float, default=0.75)
    ap.add_argument("--car_explore", type=float, default=0.0)
    ap.add_argument("--car_w_obs2", type=float, default=5.0)
    ap.add_argument("--car_w_prog", type=float, default=2.0)
    ap.add_argument("--car_w_lat", type=float, default=0.5)
    ap.add_argument("--mppi_samples", type=int, default=64)
    ap.add_argument("--mppi_iters", type=int, default=2)
    ap.add_argument("--mppi_sigma", type=float, default=0.01)
    ap.add_argument("--mppi_lambda", type=float, default=0.5)
    ap.add_argument("--mppi_w_prog", type=float, default=1.0)
    ap.add_argument("--mppi_w_ctrl", type=float, default=0.05)
    ap.add_argument("--mppi_w_obs", type=float, default=1.0)
    ap.add_argument("--car_vec_beta", type=float, default=0.5)
    ap.add_argument("--mppi_persist", type=float, default=0.0, help="MPPI per-episode correction vector EMA weight (0 = off)")
    ap.add_argument("--mppi_persist_decay", type=float, default=0.8)
    ap.add_argument("--mppi_w_dev", type=float, default=0.0)
    ap.add_argument("--mppi_ramp", type=int, default=1, help="1: constant + linear-ramp control (6-D), 0: constant (3-D)")
    ap.add_argument("--energy_sigma", type=float, default=0.05)
    ap.add_argument("--robot_model", default="gripper+arm", choices=["gripper", "gripper+arm"],
                    help="spheres in the guidance cost: gripper only (first sweeps) or gripper + forearm/wrist links")
    ap.add_argument("--policy", default="pi05", choices=["pi05", "groot"],
                    help="pi05: openpi pi0.5 in-process; groot: GR00T N1.7 through safeguide/server/groot_server.py (zmq)")
    ap.add_argument("--groot_host", default="127.0.0.1")
    ap.add_argument("--groot_port", type=int, default=5556)
    ap.add_argument("--engine", default="auto", choices=["auto", "legacy", "safeguide"],
                    help="sampler: the safeguide package (g1 / none) or the legacy research sampler (steer, oc, car, mppi)")
    ap.add_argument("--calib", default=None, help="action_model_calibration.npz (gain G); default 0.05*I")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--eps", default=None, help="explicit task:episode list, e.g. '0:3,8:3' (overrides --tasks/--episodes)")
    ap.add_argument("--torch_seed", type=int, default=None, help="seed torch/np per episode for reproducible sampling")
    ap.add_argument("--record", action="store_true",
                    help="save 512px agentview render + wrist frames + per-step contact/clearance to record/<tag>.npz")
    args = ap.parse_args()

    ckpt = None if args.policy == "groot" else (args.checkpoint or os.path.join(os.environ["OPENPI_DATA_HOME"], "openpi-assets/checkpoints/pi05_libero_pytorch"))
    out = pathlib.Path(args.out) / args.suite
    (out / "traj").mkdir(parents=True, exist_ok=True)
    if args.save_video:
        (out / "videos").mkdir(exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)

    np.random.seed(args.seed)
    if args.guidance != "none" and not args.calib:
        raise SystemExit("--calib is required for guided runs (nominal 0.05*I gain is ~4x too large)")
    G = np.load(args.calib)["G"] if args.calib else np.eye(3) * 0.012
    if args.policy == "groot":
        # GR00T N1.7 runs in its own venv behind safeguide/server/groot_server.py; the Supervisor stays here next to the sim
        # and ships the cost context with every observation. The server decides g1 / none; --guidance must agree.
        policy = None
        sampler = adapter = sg.RemoteGuidedPolicy(args.groot_host, args.groot_port, G)
        sampler.scale = args.scale  # the server applies this push size to this client's calls
        if sampler.info.get("guidance") != args.guidance:
            raise SystemExit(f"GR00T server runs guidance={sampler.info.get('guidance')!r} but --guidance {args.guidance}")
        engine = f"remote-safeguide ({sampler.info.get('model_path')}, guidance {args.guidance}, chunk {sampler.action_horizon}/{sampler.model_horizon})"
    else:
        policy = policy_config.create_trained_policy(_config.get_config("pi05_libero"), ckpt)
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path(ckpt) / "assets", "physical-intelligence/libero")
        # Always use our (uncompiled) sampler so guided / unguided runs share the exact same integrator.
        gcfg = gd.GuidanceConfig(mode=args.guidance, scale=args.scale,
                                 grad_through_model=bool(args.grad_through_model),
                                 schedule=args.schedule, t_min=args.t_min, n_samples=args.n_samples,
                                 project_steps=args.project_steps, project_lr=args.project_lr,
                                 oc_iters=args.oc_iters, oc_lr=args.oc_lr, oc_reg=args.oc_reg, oc_decay=args.oc_decay,
                                 car_batch=args.car_batch, car_train_steps=args.car_train_steps,
                                 car_thr=args.car_thr, car_temp=args.car_temp, car_reward_temp=args.car_reward_temp,
                                 car_obs_sigma=args.car_obs_sigma, car_zero_thr=args.car_zero_thr,
                                 steer=args.steer, steer_release=args.steer_release, steer_forget=args.steer_forget,
                                 trigger_exec_only=args.trigger_exec_only,
                                 car_conflict=args.car_conflict, car_prog_thr=args.car_prog_thr, car_explore=args.car_explore,
                                 car_w_obs2=args.car_w_obs2, car_w_prog=args.car_w_prog,
                                 car_w_lat=args.car_w_lat, mppi_samples=args.mppi_samples, mppi_iters=args.mppi_iters,
                                 mppi_sigma=args.mppi_sigma, mppi_lambda=args.mppi_lambda, mppi_w_prog=args.mppi_w_prog,
                                 mppi_w_ctrl=args.mppi_w_ctrl, mppi_w_dev=args.mppi_w_dev, mppi_ramp=bool(args.mppi_ramp), mppi_w_obs=args.mppi_w_obs,
                                 mppi_persist=args.mppi_persist, mppi_persist_decay=args.mppi_persist_decay,
                                 car_vec_beta=args.car_vec_beta)
        engine = args.engine
        if engine == "auto":
            engine = "safeguide" if args.guidance in ("none", "g1") and not args.steer else "legacy"
        adapter = sg.Pi05Adapter(policy, norm_stats, G)
        if engine == "safeguide":
            sampler = sg.Guide(adapter, sg.GuideConfig(mode=args.guidance, scale=args.scale,
                                                       grad_through_model=bool(args.grad_through_model),
                                                       t_min=args.t_min, schedule=args.schedule,
                                                       trigger_exec_only=args.trigger_exec_only))
            adapter.install(sampler)
        else:
            sampler = gd.install_guidance(policy, gcfg)
    print("guidance engine:", engine, flush=True)
    sup_cfg = sg.SupervisorConfig(
        exec_steps=args.replan_steps, d_safe=args.d_safe, cost_type=args.cost_type, energy_sigma=args.energy_sigma,
        margin_grip=args.margin_grip, margin_arm=args.margin_arm, margin_obj=args.margin_obj,
        cbf_gamma=args.cbf_gamma, cbf_vref=args.cbf_vref, include_arm=(args.robot_model == "gripper+arm"),
        arm_motion=args.arm_motion, model_object=bool(args.model_object), arm_release=bool(args.arm_release),
        arm_margin_post=args.arm_margin_post, release_hyst=args.release_hyst, escape_chunks=args.escape_chunks,
        escape_min_disp=args.escape_min_disp, escape_lift=args.escape_lift, escape_len=args.escape_len,
        rewind=(sg.core.recovery.RewindConfig(window=args.rewind_window, net_thr=args.rewind_net, ratio=args.rewind_ratio,
                                              quiet=args.rewind_quiet, ood_thr=args.rewind_thr, max_events=args.rewind_max,
                                              step=args.rewind_step, cooldown=args.rewind_cooldown, ood_window=args.rewind_ood_window,
                                              shortcut=bool(args.rewind_shortcut)) if args.rewind else None))
    json.dump({**vars(args), "G": np.asarray(G).tolist()}, open(out / "args.json", "w"), indent=1)
    print("action->EEF gain G:", np.asarray(G).round(4).tolist(), "| robot model:", args.robot_model, flush=True)

    spec = ob.ObstacleSpec(radius=args.radius, height=args.height, visible=not args.invisible)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = MAX_STEPS[args.suite] + args.extra_steps
    if args.eps:
        pairs = [tuple(int(x) for x in p.split(":")) for p in args.eps.split(",")]
        shard = "eps" + args.eps.replace(":", "-").replace(",", "_")
    else:
        pairs = [(t, e) for t in parse_range(args.tasks, suite.n_tasks) for e in range(args.episodes)]
        shard = "tasks" + args.tasks.replace(",", "_")
    if args.torch_seed is not None:
        shard += f"_seed{args.torch_seed}"
    results_f = open(out / f"results_{shard}.jsonl", "a")
    if args.record:
        (out / "record").mkdir(exist_ok=True)
    tasks_in_order = list(dict.fromkeys(t for t, _ in pairs))

    for task_id in tasks_in_order:
        task = suite.get_task(task_id)
        # the policy's own obstacle-free rollouts of this task define "in distribution" (OOD score + rewind targets)
        manifold = sg.core.recovery.ManifoldIndex.from_baseline(args.baseline_dir, args.suite, task_id)
        if manifold is None:
            print(f"task {task_id}: no baseline trajectories under {args.baseline_dir} - OOD score / rewind off", flush=True)
        init_states = suite.get_task_init_states(task_id)
        env = make_env(task, args.seed)
        hand_mode = args.layout == "hand"
        hand_spec = hd.HandSpec(visible=bool(args.hand_visible))
        if hand_mode:
            hd.install_hand(env, hand_spec)
        else:
            ob.install_obstacles(env, [spec] * (1 if args.layout == "single" else 2))
        for ep in [e for t, e in pairs if t == task_id]:
            tag = f"t{task_id:02d}_e{ep:02d}" + (f"_s{args.torch_seed}" if args.torch_seed is not None else "")
            rec = {"suite": args.suite, "task_id": task_id, "episode": ep, "task": task.language}
            if args.torch_seed is not None:
                s = args.torch_seed * 100003 + task_id * 1000 + ep
                torch.manual_seed(s)  # numpy RNG untouched: env.reset() randomness stays as in the baseline
                rec["torch_seed"] = args.torch_seed
            base = pathlib.Path(args.baseline_dir) / args.suite / "traj" / f"t{task_id:02d}_e{ep:02d}.npz"
            env.reset()
            obs = env.set_init_state(init_states[ep])
            if not base.exists():
                rec["skipped"] = "no_baseline_traj"
            else:
                bz = np.load(base)
                if hand_mode:
                    hm_rng = np.random.default_rng((args.torch_seed or 0) * 100003 + task_id * 1000 + ep)
                    try:
                        # two passes: crossing point first, then the support height under it
                        def obj_top(name):
                            pts_, r_ = ob.object_points(env, name)
                            return float((pts_[:, 2] + r_).max()) if len(pts_) else float(object_xyz(env)[name][2] + 0.04)
                        hm = hd.HandMotion(args.hand_motion, bz["eef_pos"], bz["actions"], object_xyz(env), ob.TABLE_TOP_Z,
                                           np.random.default_rng(0), speed=args.hand_speed, trigger_dist=args.hand_trigger, obj_top=obj_top, rest_offset=args.hand_rest)
                        sz_ = ob.support_z(env, hm.c[:2])
                        side0 = float(hm_rng.choice([-1.0, 1.0]))
                        cands = []
                        for side in (side0, -side0):  # prefer the seed's side; reject sides where the arm passes through scenery
                            hm_ = hd.HandMotion(args.hand_motion, bz["eef_pos"], bz["actions"], object_xyz(env), sz_,
                                                np.random.default_rng(hm_rng.integers(1 << 30)), speed=args.hand_speed,
                                                trigger_dist=args.hand_trigger, side=side, obj_top=obj_top, rest_offset=args.hand_rest)
                            n_scene, kt = 0, hm_.key_tips()
                            skip_pref = ("robot0", "gripper0") + tuple(env.env.objects_dict)
                            for a_, b_ in zip(kt[:-1], kt[1:]):  # sample the swept segments, not just their end points
                                for lam in np.linspace(0, 1, 6):
                                    hd.set_hand_pose(env, *hm_.pose_from_tip(a_ + lam * (b_ - a_)))
                                    n_scene += len([n for n in hd.hand_contacts(env) if not n.startswith(skip_pref)])
                            cands.append((n_scene, hm_))
                        cands.sort(key=lambda c: c[0])
                        hm = cands[0][1]
                        xys, info = [hm.c[:2]], hm.info() | {"scene_intersections": cands[0][0]}
                    except Exception as ex:  # e.g. path too short
                        xys, info = None, f"hand_{type(ex).__name__}"
                else:
                    xys, info = place_layout(env, bz["eef_pos"], object_xy(env), spec, args, actions=bz["actions"])
                if xys is None:
                    rec["skipped"] = info
            if "skipped" in rec:
                results_f.write(json.dumps(rec) + "\n"); results_f.flush(); print(json.dumps(rec), flush=True)
                continue
            if hand_mode:
                hd.set_hand_pose(env, *hm.pose_from_tip(hm.rest))
                hand_prefixes = hd.relevant_prefixes(env)
                pre = hd.hand_contacts(env, hand_prefixes)
            else:
                for i, xy_i in enumerate(xys):
                    ob.set_obstacle_xy(env, xy_i, idx=i)
                pre = ob.obstacle_contacts(env)  # pillars must not intersect fixtures/objects at placement
            if pre:
                rec["skipped"] = "pillar_touches_scene"
                rec["pillar_touches"] = sorted(set(pre))[:5]
                results_f.write(json.dumps(rec) + "\n"); results_f.flush(); print(json.dumps(rec), flush=True)
                continue
            xy = np.array(xys)
            rec["obstacle_xy"] = xy.round(4).tolist()
            rec["placement"] = info
            if hasattr(sampler, "car_reset"):
                sampler.car_reset(seed=(args.torch_seed or 0) * 100003 + task_id * 1000 + ep)  # g_psi trained per episode
            if hasattr(sampler, "reset_episode"):
                if args.torch_seed is not None:
                    sampler.reset_episode(seed=s)  # GR00T: the noise is drawn on the server
                else:
                    sampler.reset_episode()
            pillar_hits = [0] * len(xys)
            # contacts on EVERY physics substep (~25 per 20 Hz control step), not just the last one
            sim = env.env.sim
            sub = {"hits": [set() for _ in xys]}
            _orig_step = sim.step

            def _step_with_contacts(*a, _orig=_orig_step, **k):
                r = _orig(*a, **k)
                if hand_mode:
                    sub["hits"][0].update(hd.hand_contacts(env, hand_prefixes))
                else:
                    for i_p, h in enumerate(ob.obstacle_contacts_by_pillar(env)):
                        sub["hits"][i_p].update(h)
                return r

            # clearance metrics (measured, every control step): to the pillars, or to the hand's capsules
            if hand_mode:
                def grip_clr():
                    pts, radii = ob.gripper_points(env); return hd.points_clearance(pts, radii, *hm.hist[-1], hand_spec)
                def arm_clr():
                    pts, radii, _ = ob.arm_points(env); return hd.points_clearance(pts, radii, *hm.hist[-1], hand_spec)
                def obj_clr(name):
                    pts, radii = ob.object_points(env, name); return hd.points_clearance(pts, radii, *hm.hist[-1], hand_spec) if len(pts) else np.nan
                hm.hist.append(hm.pose_from_tip(hm.rest))
                hand_tips = []
            else:
                grip_clr, arm_clr, obj_clr = (lambda: ob.gripper_clearance(env)), (lambda: ob.arm_clearance(env)), (lambda name: ob.object_clearance(env, name))

            sim.step = _step_with_contacts

            plan = collections.deque()
            frames, eef, acts, clear, contact, arm_clear, obj_clear = [], [], [], [], [], [], []
            arm_pred, pend_ = [], None  # --log_arm_pred: [|dEEF|, err_static, err_alpha, err_jac (all arm), same for link5] in m
            gate_c, gate_d, gate_off = None, None, 0.0
            if args.layout == "gate" and "gate_center" in info:
                gate_c = np.array(info["gate_center"]); nn_ = np.array(info["normal"]); gate_d = np.array([nn_[1], -nn_[0]])
            elif args.layout == "single" and "normal" in info:
                # single pillar: the "gate line" is the cross-section through the pillar centre; release once the EEF is
                # past the pillar surface (+ hysteresis) along the path direction
                gate_c = np.array(xys[0]); nn_ = np.array(info["normal"]); gate_d = np.array([nn_[1], -nn_[0]]); gate_off = spec.radius
            robot = sg.MujocoPandaRobot(env)
            scene = (hd.HandCapsuleScene(hm, hand_spec, args.hand_predict, args.margin_hand, args.hand_margin_tau)
                     if hand_mode else ob.PillarScene(env))
            sup = sg.Supervisor(robot, scene, adapter.action_map(), sup_cfg, horizon=adapter.action_horizon, manifold=manifold)
            sup.reset(None if gate_c is None else (gate_c, gate_d, gate_off))

            chunk_logs, infer_ms = [], []
            hit_names = collections.Counter()
            rec_frames, rec_wrist, rec_hits = [], [], []

            def record_frame(obs, hits):
                # separate 512px render for the video only; the policy keeps its 256px input
                rec_frames.append(np.ascontiguousarray(
                    env.env.sim.render(camera_name="agentview", width=512, height=512)[::-1, ::-1]))
                rec_wrist.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
                rec_hits.append(sorted(set(hits)))

            done, t = False, 0
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(DUMMY_ACTION)
                    sub["hits"] = [set() for _ in xys]  # settling steps are not part of the rollout
                    t += 1
                    if args.record and t == args.num_steps_wait:
                        record_frame(obs, [])
                        clear_rec0 = grip_clr()
                    continue
                element, img = policy_input(obs, str(task.language), args.resize)
                if args.save_video:
                    frames.append(img)
                if not plan:
                    ctx = sup.before_chunk(bool(acts) and acts[-1][6] > 0)
                    ctx.obj_xy = np.array([o for o in object_xy(env).values()  # steer: skip the held object
                                           if np.linalg.norm(o - ob.eef_pos(env)[:2]) > 0.08])
                    sampler.ctx, sampler.escape_off = ctx, sup.escape_off
                    held = sup.held
                    if sup.rewind_actions is not None:  # state-level recovery: retrace the path, no policy call
                        racts = sup.rewind_actions
                        chunk_logs.append({"rewind": True, "rewind_steps": len(racts), "rewind_to_step": sup.rec.last_target_step,
                                           "ood_cm": None if sup.last_ood is None else 100 * sup.last_ood, "post_gate": sup.post_gate,
                                           "escape": False, "held_obj": held, "cost_first": None, "intent_cm": None})
                        plan.extend(racts)
                        sup.after_rewind()
                        print(f"  rewind #{sup.rec.events}: {len(racts)} steps back to step {sup.rec.last_target_step}", flush=True)
                    else:
                        if args.log_arm_pred:
                            ap_, _, aa_, _ = ob.arm_points(env, return_bodies=True)
                            pend_ = {"p0": ob.eef_pos(env), "arm0": ap_, "alpha": aa_, "M": ob.arm_motion_matrices(env), "t": t}
                        t0 = time.perf_counter()
                        if args.policy == "groot":
                            chunk = sg.remote.libero_action(sampler.infer(sg.remote.groot_obs(obs, str(task.language)))["actions"])
                            lg = sampler.last_log
                        else:
                            chunk = policy.infer(element)["actions"]
                            lg = sampler.last_log
                        infer_ms.append((time.perf_counter() - t0) * 1e3)
                        sup.after_chunk(lg)
                        chunk_logs.append({"post_gate": sup.post_gate, "escape": sampler.escape_off is not None}
                                          | {k: lg.get(k) for k in ("cost_final", "pred_min_clearance", "sample_ms",
                                                                  "oc_cost0", "oc_iters_used", "oc_best_cost", "oc_u_rms",
                                                                  "car_active_ratio", "car_loss", "car_gate_max", "car_gate_steps", "oc_best_it",
                                                                  "h_star", "viol_exec_cm", "viol_tail_cm", "bind_group", "bind_h",
                                                                  "c0_bind_cm", "push_cm", "intent_cm", "brake_frac",
                                                                  "steer_brake_steps", "latches", "latch_sides",
                                                                  "car_ess", "car_lat_spread_cm", "car_gate_frac", "car_resid_ratio",
                                                                  "guided_first")}
                                          | {k: v for k, v in lg.items() if k.startswith("mppi_") or k.startswith("car_vec")}
                                          | {"cost_first": (lg.get("cost_per_step") or [None])[0], "held_obj": held,
                                             "ood_cm": None if sup.last_ood is None else 100 * sup.last_ood,
                                             **(sup.rec.last_trigger or {}),
                                             "n_obj_spheres": int(sum(g == "obj" for g in (sampler.ctx.sphere_groups or [])))})
                        plan.extend(chunk[: args.replan_steps])
                action = np.asarray(plan.popleft())
                eef.append(ob.eef_pos(env)); acts.append(action.copy())
                sup.note_step(eef[-1], bool(action[6] > 0))
                if hand_mode:  # kinematic hand: pose for this control step, set before the physics substeps
                    h_el, h_tip = hm.update(ob.eef_pos(env), t - args.num_steps_wait, gripper_closed=bool(action[6] > 0))
                    hd.set_hand_pose(env, h_el, h_tip); hand_tips.append(h_tip)
                obs, _, done, _ = env.step(action.tolist())
                by_pillar = [sorted(h) for h in sub["hits"]]  # union over this control step's substeps
                sub["hits"] = [set() for _ in xys]
                hits = [n for h in by_pillar for n in h]
                for i, h in enumerate(by_pillar):
                    pillar_hits[i] += bool(h)
                hit_names.update(hits)
                contact.append(bool(hits)); clear.append(grip_clr()); arm_clear.append(arm_clr())
                if args.log_arm_pred and pend_ is not None and t + 1 - pend_["t"] == args.replan_steps:
                    d_ee = ob.eef_pos(env) - pend_["p0"]
                    act_ = ob.arm_points(env)[0] - pend_["arm0"]
                    pa = pend_["alpha"][:, None] * d_ee[None, :]
                    pj = np.einsum("pij,j->pi", pend_["M"], d_ee)
                    arm_pred.append([np.linalg.norm(d_ee), *[np.linalg.norm(act_ - x, axis=1).mean() for x in (0 * pa, pa, pj)],
                                     *[np.linalg.norm(act_[:5] - x[:5], axis=1).mean() for x in (0 * pa, pa, pj)]])
                    pend_ = None
                hnow = ob.held_object(env, action[6] > 0)
                obj_clear.append(obj_clr(hnow) if hnow is not None else np.nan)
                if args.record:
                    record_frame(obs, hits)
                t += 1
                if done:
                    break
            eef.append(ob.eef_pos(env))
            if args.record:
                # frame k shows the state after k executed actions (frame 0 = before the first action)
                np.savez_compressed(
                    out / "record" / f"{tag}.npz",
                    frames=np.array(rec_frames), wrist=np.array(rec_wrist),
                    contact=np.array([False] + contact), clearance=np.array([clear_rec0] + clear),
                    hits=np.array(json.dumps([[]] + rec_hits[1:])), eef_pos=np.array(eef),
                    chunk_pred_clearance=np.array([c["pred_min_clearance"] if c.get("pred_min_clearance") is not None else np.nan
                                                   for c in chunk_logs]),
                )

            robot_hit = any(n.startswith(("robot0", "gripper0")) for n in hit_names)
            rec.update({
                "success": bool(done), "steps": t - args.num_steps_wait,
                "collided_any": bool(hit_names), "collided_robot": robot_hit,
                "safe_success": bool(done) and not hit_names,
                "contact_steps": int(sum(contact)),
                "first_contact_step": int(np.argmax(contact)) if any(contact) else None,
                "hit_geoms": dict(hit_names.most_common(5)),
                "min_clearance": float(np.min(clear)) if clear else None,
                "min_arm_clearance": float(np.min(arm_clear)) if arm_clear else None,
                "min_obj_clearance": float(np.nanmin(obj_clear)) if obj_clear and np.isfinite(obj_clear).any() else None,
                "held_chunks": int(sum(h is not None for h in sup.held_log)), "held_objects": sorted({h for h in sup.held_log if h}),
                "collided_object_only": bool(hit_names) and not robot_hit,
                "post_gate_chunks": sup.post_gate_chunks, "escape_events": sup.escape_events,
                "rewind_events": sup.rec.events, "rewind_steps": sup.rec.steps_total,
                "ood_cm_max": max((c["ood_cm"] for c in chunk_logs if c.get("ood_cm") is not None), default=None),
                "ood_chunks": sum(1 for c in chunk_logs if (c.get("ood_cm") or 0) > 100 * args.rewind_thr),
                "hand": hm.info() if hand_mode else None,
                "arm_pred_err_cm": (100 * np.array(arm_pred).mean(axis=0)).round(3).tolist() if arm_pred else None,
                "contact_detection": "substep",
                "infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
                "pred_min_clearance_mean": float(np.mean([c["pred_min_clearance"] for c in chunk_logs if c.get("pred_min_clearance") is not None])) if any(c.get("pred_min_clearance") is not None for c in chunk_logs) else None,
                "contact_steps_per_pillar": pillar_hits,
                "car_trained_chunks": sum(1 for c in chunk_logs if c.get("car_loss") is not None),
                "car_gated_chunks": sum(1 for c in chunk_logs if (c.get("car_gate_steps") or 0) > 0),
                "oc_active_chunks": sum(1 for c in chunk_logs if (c.get("oc_iters_used") or 0) > 0),
                "oc_iters_total": int(sum((c.get("oc_iters_used") or 0) for c in chunk_logs)),
                "n_chunks": len(chunk_logs),
            })
            np.savez_compressed(out / "traj" / f"{tag}.npz", eef_pos=np.array(eef), actions=np.array(acts),
                                clearance=np.array(clear), contact=np.array(contact), obstacle_xy=xy,
                                hand_tip=np.array(hand_tips) if hand_mode else np.zeros((0, 3)))
            with open(out / "traj" / f"{tag}_chunks.json", "w") as f:
                json.dump(chunk_logs, f)
            if args.save_video:
                imageio.mimwrite(out / "videos" / f"{tag}_{'success' if done else 'failure'}{'_hit' if hit_names else ''}.mp4",
                                 frames, fps=10)
            results_f.write(json.dumps(rec) + "\n"); results_f.flush()
            print(json.dumps(rec), flush=True)
        env.close()


if __name__ == "__main__":
    main()
