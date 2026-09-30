"""Bridge between an evaluator process and a guided policy that runs in another Python environment (GR00T N1.7 lives
in its own Python 3.12 uv venv, the LIBERO evaluator in the openpi Python 3.11 venv).

Wire format: msgpack + msgpack_numpy, the same as gr00t.policy.server_client.MsgSerializer, so the client can talk to a
gr00t PolicyServer that registered the `safeguide_*` endpoints (safeguide/server/groot_server.py). Per policy call the
client ships the observation plus the chunk cost context (robot spheres, obstacles, action map) built by the Supervisor
next to the simulator; the server runs the guided sampler and returns the decoded chunk and the guidance log.
"""
import dataclasses

import msgpack
import msgpack_numpy as mnp
import numpy as np

from .core.cost import ChunkCostContext, DeltaEEFMap

# ----------------------------------------------------------------------------- serialisation
def pack(obj):
    return msgpack.packb(obj, default=mnp.encode)


def unpack(buf):
    return msgpack.unpackb(buf, object_hook=mnp.decode, raw=False)


def jsonable(x):
    """numpy scalars / arrays -> Python types (for the log dict)."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def ctx_to_wire(ctx: ChunkCostContext) -> dict:
    d = {}
    for f in dataclasses.fields(ctx):
        v = getattr(ctx, f.name)
        if f.name == "action_map":
            if v is None:
                d[f.name] = None
            else:
                d[f.name] = {"kind": type(v).__name__} | {k: (np.asarray(x) if isinstance(x, np.ndarray) else x)
                                                          for k, x in dataclasses.asdict(v).items()}
        else:
            d[f.name] = v
    return d


def ctx_from_wire(d: dict) -> ChunkCostContext:
    d = dict(d)
    am = d.pop("action_map", None)
    if am is not None:
        am = dict(am)
        kind = am.pop("kind")
        if kind != "DeltaEEFMap":
            raise ValueError(f"unknown action map on the wire: {kind}")
        am = DeltaEEFMap(np.asarray(am["q01"]), np.asarray(am["q99"]), np.asarray(am["G"]), am.get("n_valid"))
    return ChunkCostContext(**d, action_map=am)


# ----------------------------------------------------------------------------- LIBERO <-> GR00T conventions
def groot_obs(obs, prompt):
    """robosuite/LIBERO observation -> the batched observation Gr00tPolicy (embodiment libero_sim) expects:
    video (B=1, T=1, 256, 256, 3) uint8 rotated 180 deg, state (1, 1, D) float32, language [[prompt]].
    Same conversion as gr00t/eval/sim/LIBERO/libero_env.py::_process_observation + the MultiStepWrapper batching."""
    from eval_libero import quat2axisangle

    xyz = np.asarray(obs["robot0_eef_pos"], np.float32)
    rpy = np.asarray(quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=float)), np.float32)
    grip = np.asarray(obs["robot0_gripper_qpos"], np.float32)
    st = lambda v: np.asarray(v, np.float32).reshape(1, 1, -1)
    return {
        "video": {"image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])[None, None],
                  "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])[None, None]},
        "state": {"x": st(xyz[0]), "y": st(xyz[1]), "z": st(xyz[2]),
                  "roll": st(rpy[0]), "pitch": st(rpy[1]), "yaw": st(rpy[2]), "gripper": st(grip)},
        "language": {"annotation.human.action.task_description": [[str(prompt)]]},
    }


def libero_action(chunk):
    """GR00T's decoded LIBERO actions (H, 7; gripper in the dataset's [0, 1], 1 = open) -> what OffScreenRenderEnv.step
    takes (gripper -1 = open, +1 = close), exactly as libero_env.py::step (normalize_gripper_action(binarize=True) then
    invert_gripper_action)."""
    a = np.array(chunk, dtype=np.float32, copy=True)
    a[..., -1] = 2.0 * (a[..., -1] - 0.0) / (1.0 - 0.0) - 1.0
    a[..., -1] = np.sign(a[..., -1])
    a[..., -1] = a[..., -1] * -1.0
    return a


# ----------------------------------------------------------------------------- client
class RemoteGuidedPolicy:
    """Client side of the bridge. Mirrors the attributes the evaluator uses on an in-process Guide / adapter:
    `.ctx`, `.escape_off`, `.last_log`, `.reset_episode()`, `.infer(obs)`, `.action_horizon`, `.action_map()`."""

    def __init__(self, host="127.0.0.1", port=5556, G=None, timeout_ms=180000):
        import zmq

        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._zmq = zmq
        self.context = zmq.Context()
        self._init_socket()
        self.ctx: ChunkCostContext | None = None
        self.escape_off = None
        self.scale = None  # guidance push size sent with every call (None = the server's default)
        self.last_log = {}
        self.info = self.call("safeguide_info", requires_input=False)
        self.action_horizon = int(self.info["action_horizon"])  # steps of the chunk that carry real actions
        self.model_horizon = int(self.info["model_horizon"])
        self.action_keys = list(self.info["action_keys"])
        if G is None:
            raise ValueError("RemoteGuidedPolicy needs the calibrated action -> EEF gain G")
        self._map = DeltaEEFMap(np.asarray(self.info["q01"], float), np.asarray(self.info["q99"], float),
                                np.asarray(G, float), n_valid=self.action_horizon)

    def _init_socket(self):
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            self.socket.send(pack(req))
            resp = unpack(self.socket.recv())
        except self._zmq.error.Again:
            self.socket.close()
            self._init_socket()
            raise TimeoutError(f"policy server {self.host}:{self.port} did not answer '{endpoint}' within {self.timeout_ms} ms")
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"policy server error on '{endpoint}': {resp['error']}")
        return resp

    def action_map(self):
        return self._map

    def reset_episode(self, seed=None):
        self.escape_off, self.last_log = None, {}
        self.call("safeguide_reset", {"seed": None if seed is None else int(seed)})

    def infer(self, gobs):
        """gobs: groot_obs(...). Returns {"actions": (H_valid, 7) float32} in GR00T's decoded convention."""
        data = {"obs": gobs,
                "ctx": None if self.ctx is None else ctx_to_wire(self.ctx),
                "escape_off": None if self.escape_off is None else np.asarray(self.escape_off, dtype=float),
                "scale": None if self.scale is None else float(self.scale)}
        resp = self.call("safeguide_get_action", data)
        self.last_log = resp.get("log") or {}
        return {"actions": np.asarray(resp["actions"], dtype=np.float32)}

    def close(self):
        self.socket.close()
        self.context.term()
