"""Template: plug a new flow-matching (or diffusion) action policy into the safety layer.

The layer only needs five model-specific things; everything else (constraint cost, guided integration, supervisor)
is model-agnostic. Reference implementations: safeguide/adapters/pi05_openpi.py (openpi pi0 / pi0.5) and
safeguide/adapters/groot_n1.py (Isaac-GR00T N1.7).
"""
import numpy as np
import torch

from safeguide.adapters.base import FlowPolicyAdapter
from safeguide.core.cost import DeltaEEFMap
from safeguide.core.flow import FlowSpec


class MyPolicyAdapter(FlowPolicyAdapter):
    # (1) Time convention of the model's flow. noise_at_one=True: x_{t=1} is noise and the sampler integrates t 1 -> 0
    #     (openpi: x_t = t eps + (1-t) a, v = eps - a). noise_at_one=False: x_{t=0} is noise, t 0 -> 1 (GR00T).
    #     Everything else about time (dt sign, clean-chunk estimate a_hat, push sign) follows from this flag.
    flow = FlowSpec("mypolicy", noise_at_one=True)

    def __init__(self, policy, action_horizon, action_dim, num_steps, action_lo, action_hi, G):
        self.policy = policy
        self.action_horizon = action_horizon  # H control steps per chunk (the padded length if the model pads)
        self.action_dim = action_dim  # D
        self.default_num_steps = num_steps  # Euler steps of the model's own sampler
        # (2) Action map: normalised action chunk -> end-effector path. For delta-EEF actions normalised per dimension to
        #     [-1, 1] by (lo, hi) (openpi quantiles, GR00T min/max) with a calibrated 3x3 gain G (metres of EEF motion per
        #     unit action per control step; fit with benchmarks/calibrate_action_model.py on unguided rollouts).
        #     Joint-space actions need a Jacobian-based ActionMap subclass instead.
        self._map = DeltaEEFMap(np.asarray(action_lo), np.asarray(action_hi), np.asarray(G))
        self._orig_sampler = None

    def batch_size(self, observation):
        return 1

    def sample_noise(self, batch_size, device):
        # (3) Draw the initial noise exactly like the model's own sampler (same generator, same shape) so that guided
        #     and unguided runs with the same seed stay paired.
        return torch.randn(batch_size, self.action_horizon, self.action_dim, device=device)

    def prepare(self, observation, device):
        # (4a) Everything the velocity field is conditioned on, computed ONCE per chunk: vision-language prefix / KV cache,
        #      state embedding, task tokens. Returned object is handed back to velocity() at every Euler step.
        return self.policy.encode(observation)

    def velocity(self, cond, x_t, t):
        # (4b) One velocity-field evaluation v_theta(x_t, t | cond): x_t (B, H, D), t (B,) -> (B, H, D).
        #      Called under torch.no_grad() unless GuideConfig.grad_through_model is set (then it must be differentiable).
        return self.policy.velocity(cond, x_t, t)

    def action_map(self):
        return self._map

    def install(self, guide):
        # (5) Route the policy's own inference path through guide.run(observation) -> (B, H, D) normalised chunk.
        #     Keep the policy's own post-processing (un-normalisation, clipping, gripper decoding) after it.
        self._orig_sampler = self.policy.sample_actions
        self.policy.sample_actions = lambda observation, **kw: guide.run(observation)
        return self

    def uninstall(self):
        if self._orig_sampler is not None:
            self.policy.sample_actions = self._orig_sampler
