"""Guided GR00T N1.7 policy server: the official gr00t PolicyServer plus the safeguide endpoints.

Runs inside the Isaac-GR00T uv venv (Python 3.12); the evaluator (any Python) talks to it through
safeguide.remote.RemoteGuidedPolicy. Per call the client sends the observation and the chunk cost context; the
server runs Gr00tPolicy.get_action with the action head's denoising loop replaced by the safeguide Guide.

  cd $WS/0Xuehui/Isaac-GR00T
  PYTHONPATH=$PROJ CUDA_VISIBLE_DEVICES=0 uv run python $PROJ/safeguide/server/groot_server.py \\
      --model-path checkpoints/GR00T-N1.7-LIBERO/libero_spatial --port 5556 --guidance g1
"""
import argparse
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    ap.add_argument("--host", default="*")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--guidance", default="g1", choices=["g1", "none"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--grad-through-model", type=int, default=0)
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--schedule", default="const")
    ap.add_argument("--num-steps", type=int, default=None, help="Euler steps (default: the checkpoint's num_inference_timesteps)")
    ap.add_argument("--strict", type=int, default=1)
    ap.add_argument("--rec_B", type=int, default=16)
    args = ap.parse_args()

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    from safeguide.adapters.groot_n1 import GrootN1Adapter
    from safeguide.core.guide import Guide, GuideConfig
    from safeguide.remote import ctx_from_wire, jsonable

    policy = Gr00tPolicy(embodiment_tag=args.embodiment_tag, model_path=args.model_path, device=args.device,
                         strict=bool(args.strict))
    tag = policy.embodiment_tag.value
    keys = list(policy.modality_configs["action"].modality_keys)
    valid_h = len(policy.modality_configs["action"].delta_indices)
    norm = policy.processor.state_action_processor.norm_params[tag]["action"]
    q01 = np.concatenate([np.atleast_1d(np.asarray(norm[k]["min"], float)) for k in keys])
    q99 = np.concatenate([np.atleast_1d(np.asarray(norm[k]["max"], float)) for k in keys])

    adapter = GrootN1Adapter(policy, action_map=None, num_steps=args.num_steps)
    guide = Guide(adapter, GuideConfig(mode=args.guidance, scale=args.scale, grad_through_model=bool(args.grad_through_model),
                                       t_min=args.t_min, schedule=args.schedule))
    adapter.install(guide)
    print(f"GR00T N1.7 {args.model_path}: embodiment {tag}, action keys {keys}, valid horizon {valid_h} of "
          f"{adapter.action_horizon}, {adapter.default_num_steps} Euler steps, guidance {args.guidance}", flush=True)
    print("q01", q01.round(4).tolist(), "\nq99", q99.round(4).tolist(), flush=True)

    def info():
        return {"action_horizon": valid_h, "model_horizon": adapter.action_horizon, "num_steps": adapter.default_num_steps,
                "q01": q01.tolist(), "q99": q99.tolist(), "action_keys": keys, "guidance": args.guidance,
                "model_path": args.model_path, "embodiment": tag}

    def get_action(obs=None, ctx=None, escape_off=None, scale=None):
        # gr00t's PolicyServer calls handlers as handler(**request["data"]), hence keyword parameters
        if scale is not None:  # per-client push size (requests are served one at a time, so this is race-free)
            guide.cfg.scale = float(scale)
        guide.ctx = ctx_from_wire(ctx) if ctx is not None else None
        guide.escape_off = None if escape_off is None else np.asarray(escape_off, dtype=float)
        action, _ = policy.get_action(obs)
        parts = []
        for k in keys:
            v = action.get(k, action.get(f"action.{k}"))
            v = np.asarray(v)
            parts.append(v[0] if v.ndim == 3 else v)  # (B=1, H, dim) -> (H, dim)
        arr = np.concatenate(parts, axis=-1).astype(np.float32)  # (H_valid, 7)
        return {"actions": arr, "log": jsonable(guide.last_log)}

    def reset(seed=None):
        if seed is not None:
            torch.manual_seed(int(seed))
        guide.reset_episode(seed)
        policy.reset()
        return {"status": "ok", "seed": seed}

    server = PolicyServer(policy, host=args.host, port=args.port)
    server.register_endpoint("safeguide_info", info, requires_input=False)
    server.register_endpoint("safeguide_get_action", get_action)
    server.register_endpoint("safeguide_reset", reset)
    with server:
        server.run()


if __name__ == "__main__":
    sys.exit(main())
