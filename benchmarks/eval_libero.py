"""In-process LIBERO evaluation of pi05_libero (PyTorch), no websocket server.

Policy and simulator live in one process so that later the guidance cost can read
obstacle / end-effector state directly from the sim. Mirrors openpi's
examples/libero/main.py (same preprocessing, replan_steps, max_steps, wait steps).

Per episode it also stores (eef_pos, eef_quat, action) at every control step so the
action -> EEF displacement model used by the guidance cost can be calibrated.

Usage (from the repository root, in an environment with openpi and LIBERO installed):
  CUDA_VISIBLE_DEVICES=0 python benchmarks/eval_libero.py --suite libero_spatial --tasks 0-4 --episodes 10
The pi0.5 checkpoint defaults to $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero_pytorch (openpi's own
download location); pass --checkpoint to use another one.
"""
import argparse
import collections
import json
import math
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from openpi.policies import policy_config
from openpi.training import config as _config

DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RESOLUTION = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def quat2axisangle(quat):
    # Same as openpi examples/libero/main.py (robosuite convention, xyzw).
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def parse_range(s, n):
    if s == "all":
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_env(task, seed):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RESOLUTION, camera_widths=ENV_RESOLUTION)
    env.seed(seed)  # openpi: seed affects object positions even with fixed init state
    return env


def policy_input(obs, prompt, resize):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # rotate 180 to match training
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": prompt,
    }
    return element, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replan_steps", type=int, default=5)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--checkpoint", default=None, help="defaults to $OPENPI_DATA_HOME/.../pi05_libero_pytorch")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--policy", default="pi05", choices=["pi05", "groot"], help="groot: GR00T N1.7 via safeguide/server/groot_server.py")
    ap.add_argument("--groot_host", default="127.0.0.1")
    ap.add_argument("--groot_port", type=int, default=5556)
    ap.add_argument("--save_video", action="store_true")
    args = ap.parse_args()

    import os

    ckpt = None if args.policy == "groot" else (args.checkpoint or os.path.join(
        os.environ["OPENPI_DATA_HOME"], "openpi-assets/checkpoints/pi05_libero_pytorch"
    ))
    out = pathlib.Path(args.out) / args.suite
    (out / "traj").mkdir(parents=True, exist_ok=True)
    if args.save_video:
        (out / "videos").mkdir(exist_ok=True)

    np.random.seed(args.seed)
    if args.policy == "groot":
        from safeguide import remote as sgr

        policy = sgr.RemoteGuidedPolicy(args.groot_host, args.groot_port, G=np.eye(3) * 0.012)  # no cost context is sent: unguided
        print("GR00T server:", policy.info, flush=True)
    else:
        policy = policy_config.create_trained_policy(_config.get_config("pi05_libero"), ckpt)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = MAX_STEPS[args.suite]
    results_f = open(out / f"results_tasks{args.tasks.replace(',', '_')}.jsonl", "a")

    for task_id in parse_range(args.tasks, suite.n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env = make_env(task, args.seed)
        for ep in range(args.episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep])
            plan = collections.deque()
            frames, eef_pos, eef_quat, acts, infer_ms = [], [], [], [], []
            done, t = False, 0
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(DUMMY_ACTION)
                    t += 1
                    continue
                element, img = policy_input(obs, str(task.language), args.resize)
                if args.save_video:
                    frames.append(img)
                if not plan:
                    t0 = time.perf_counter()
                    if args.policy == "groot":
                        chunk = sgr.libero_action(policy.infer(sgr.groot_obs(obs, str(task.language)))["actions"])
                    else:
                        chunk = policy.infer(element)["actions"]
                    infer_ms.append((time.perf_counter() - t0) * 1e3)
                    plan.extend(chunk[: args.replan_steps])
                action = np.asarray(plan.popleft())
                eef_pos.append(obs["robot0_eef_pos"].copy())
                eef_quat.append(obs["robot0_eef_quat"].copy())
                acts.append(action.copy())
                obs, _, done, _ = env.step(action.tolist())
                t += 1
                if done:
                    break
            eef_pos.append(obs["robot0_eef_pos"].copy())
            eef_quat.append(obs["robot0_eef_quat"].copy())

            tag = f"t{task_id:02d}_e{ep:02d}"
            np.savez_compressed(
                out / "traj" / f"{tag}.npz",
                eef_pos=np.array(eef_pos), eef_quat=np.array(eef_quat), actions=np.array(acts),
            )
            if args.save_video:
                imageio.mimwrite(out / "videos" / f"{tag}_{'success' if done else 'failure'}.mp4", frames, fps=10)
            rec = {
                "suite": args.suite, "task_id": task_id, "episode": ep, "task": task.language,
                "success": bool(done), "steps": t - args.num_steps_wait,
                "infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
            }
            results_f.write(json.dumps(rec) + "\n")
            results_f.flush()
            print(json.dumps(rec), flush=True)
        env.close()


if __name__ == "__main__":
    main()
