"""openpi pi0 / pi0.5 (PyTorch) adapter.

openpi Policy.infer calls policy._sample_actions(device, observation, noise=None, num_steps=10); install() swaps
that callable for the Guide. Conditioning = the PaliGemma prefix KV cache (computed once per chunk, exactly as in
PI0Pytorch.sample_actions); velocity = PI0Pytorch.denoise_step.
"""
import numpy as np
import torch

from ..core.cost import DeltaEEFMap
from ..core.flow import OPENPI_FLOW
from .base import FlowPolicyAdapter


class Pi05Adapter(FlowPolicyAdapter):
    flow = OPENPI_FLOW

    def __init__(self, policy, norm_stats=None, G=None, default_num_steps=10):
        """policy: openpi Policy wrapping a PI0Pytorch model.  norm_stats: openpi norm stats of the checkpoint
        (norm_stats["actions"].q01/.q99).  G: (3, 3) calibrated action -> EEF gain (m per unit action per step)."""
        self.policy = policy
        self.model = policy._model
        self.model.eval()
        self.action_horizon = int(self.model.config.action_horizon)
        self.action_dim = int(self.model.config.action_dim)
        self.default_num_steps = default_num_steps
        self.norm_stats, self.G = norm_stats, G
        self._orig_sample_actions = None

    def batch_size(self, observation):
        return observation.state.shape[0]

    def sample_noise(self, batch_size, device):
        return self.model.sample_noise((batch_size, self.action_horizon, self.action_dim), device)

    def prepare(self, observation, device):
        m = self.model
        images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = m._prepare_attention_masks_4d(prefix_att_2d)
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = m.paligemma_with_expert.forward(
            attention_mask=att_4d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True,
        )
        return state, prefix_pad_masks, past_key_values

    def velocity(self, cond, x_t, t):
        state, prefix_pad_masks, pkv = cond
        return self.model.denoise_step(state, prefix_pad_masks, pkv, x_t, t)

    def expand_cond(self, cond, batch_size):
        import copy

        state, prefix_pad_masks, pkv = cond
        pkv_b = copy.deepcopy(pkv)
        pkv_b.batch_repeat_interleave(batch_size)
        return (state.expand(batch_size, *state.shape[1:]), prefix_pad_masks.expand(batch_size, *prefix_pad_masks.shape[1:]), pkv_b)

    def action_map(self):
        if self.norm_stats is None or self.G is None:
            raise ValueError("Pi05Adapter needs norm_stats and the calibrated gain G to build the action map")
        act = self.norm_stats["actions"]
        return DeltaEEFMap(np.asarray(act.q01)[:7], np.asarray(act.q99)[:7], np.asarray(self.G, dtype=float))

    def install(self, guide):
        if self._orig_sample_actions is None:
            self._orig_sample_actions = self.policy._sample_actions

        def _sample_actions(device, observation, noise=None, num_steps=10):
            return guide.run(observation, noise=noise, num_steps=num_steps, device=device)

        self.policy._sample_actions = _sample_actions
        return self

    def uninstall(self):
        if self._orig_sample_actions is not None:
            self.policy._sample_actions = self._orig_sample_actions
            self._orig_sample_actions = None
