"""Isaac-GR00T N1.7 adapter (Gr00tN1d7ActionHead, flow-matching DiT with a Cosmos-Reason2-2B / Qwen3-VL backbone).

Written against the official repo at github.com/NVIDIA/Isaac-GR00T, main branch, gr00t/model/gr00t_n1d7/gr00t_n1d7.py
(cloned 2026-09-30). velocity() reproduces one iteration of Gr00tN1d7ActionHead.get_action_with_features; prepare()
is Gr00tN1d7ActionHead._encode_features (vlln + VL self-attention on the backbone features, state tokens), computed once
per chunk exactly as the model does. Status: adapter code mirrors the source line by line but has NOT yet been run
against a checkpoint (environment install in progress on pod3).

N1.7 flow convention (forward()):  x_t = (1 - t) noise + t actions,  v = actions - noise,  t: 0 -> 1 in
num_inference_timesteps (4) Euler steps of dt = 1/N; the network receives the time as an integer bucket
int(t * num_timestep_buckets) (1000). No future tokens; positional embedding on the action tokens; the
AlternateVLDiT also takes image_mask / backbone_attention_mask from the backbone output.

LIBERO checkpoints (nvidia/GR00T-N1.7-LIBERO/<suite>): action_horizon 40 (8 executed by the reference eval), action
dim padded to max_action_dim 132 with the embodiment's 7 dims (x, y, z, roll, pitch, yaw, gripper) first, percentile
(q01/q99) normalisation -> DeltaEEFMap(q01, q99, G) with a calibrated gain G. RTC options (previous-chunk
inpainting) are not supported by the guided path: get_action falls back to the original when they are requested.
"""
import torch

from ..core.flow import GROOT_FLOW
from .base import FlowPolicyAdapter


class GrootN1Adapter(FlowPolicyAdapter):
    flow = GROOT_FLOW

    def __init__(self, policy_or_model, action_map, num_steps=None):
        """policy_or_model: gr00t.policy.Gr00tPolicy (has .model) or the Gr00tN1d7 model itself.
        action_map: ActionMap for this embodiment (normalisation statistics + action semantics)."""
        self.model = getattr(policy_or_model, "model", policy_or_model)
        self.head = self.model.action_head
        self.action_horizon = int(self.head.config.action_horizon)
        self.action_dim = int(self.head.action_dim)  # max_action_dim (padded); the embodiment's dims come first
        self.default_num_steps = int(num_steps or self.head.num_inference_timesteps)
        self._action_map = action_map
        self._orig_get_action = None
        self._last_cond = None

    # ------------------------------------------------------------------ FlowPolicyAdapter
    def batch_size(self, inputs):
        backbone_output, _ = inputs
        return backbone_output.backbone_features.shape[0]

    def sample_noise(self, batch_size, device):
        # get_action_with_features: torch.randn((B, action_horizon, action_dim), dtype=vl_embs.dtype)
        dtype = next(self.head.parameters()).dtype
        return torch.randn((batch_size, self.action_horizon, self.action_dim), dtype=dtype, device=device)

    def prepare(self, inputs, device):
        """inputs = (backbone_output, action_input) as passed to Gr00tN1d7ActionHead.get_action."""
        backbone_output, action_input = inputs
        feats = self.head._encode_features(backbone_output, action_input)  # processes backbone_output in place, as the model does
        cond = {"vl": feats.backbone_features, "state": feats.state_features, "emb": action_input.embodiment_id,
                "bo": backbone_output}
        self._last_cond = cond
        return cond

    def velocity(self, cond, x_t, t):
        head = self.head
        # continuous t in {0, 1/N, ...} -> the head's integer buckets (get_action: int(t_cont * num_timestep_buckets))
        t_disc = (t * head.num_timestep_buckets).long()
        af = head.action_encoder(x_t, t_disc, cond["emb"])
        if head.config.add_pos_embed:
            pos_ids = torch.arange(af.shape[1], dtype=torch.long, device=x_t.device)
            af = af + head.position_embedding(pos_ids).unsqueeze(0)
        sa = torch.cat((cond["state"], af), dim=1)
        if head.config.use_alternate_vl_dit:
            out = head.model(hidden_states=sa, encoder_hidden_states=cond["vl"], timestep=t_disc,
                             image_mask=cond["bo"].image_mask, backbone_attention_mask=cond["bo"].backbone_attention_mask)
        else:
            out = head.model(hidden_states=sa, encoder_hidden_states=cond["vl"], timestep=t_disc)
        pred = head.action_decoder(out, cond["emb"])
        return pred[:, -self.action_horizon:]

    def expand_cond(self, cond, batch_size):
        from transformers.feature_extraction_utils import BatchFeature

        ex = lambda v: v.expand(batch_size, *v.shape[1:]) if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 1 else v
        bo = BatchFeature(data={k: ex(v) for k, v in cond["bo"].items()})
        return {"vl": ex(cond["vl"]), "state": ex(cond["state"]), "emb": ex(cond["emb"]), "bo": bo}

    def action_map(self):
        return self._action_map

    # ------------------------------------------------------------------ hook
    def install(self, guide):
        """Replace the action head's denoising loop. Gr00tN1d7.get_action -> action_head.get_action(backbone_output,
        action_input, options); the policy's own pre/post-processing (Qwen3-VL processor, state normalisation,
        decode_action un-normalisation) is untouched, so the guided chunk flows through the reference pipeline."""
        if self._orig_get_action is None:
            self._orig_get_action = self.head.get_action
        adapter, orig = self, self._orig_get_action

        def _get_action(backbone_output, action_input, options=None):
            if "action" in action_input:  # RTC inpainting requested: not supported by the guided path
                return orig(backbone_output, action_input, options)
            from transformers.feature_extraction_utils import BatchFeature

            # Gr00tPolicy._get_action wraps the call in torch.inference_mode(), under which the cost gradient w.r.t.
            # a_hat cannot be taken; re-enter normal mode (no_grad for the network, enable_grad inside the Guide only
            # around the cost). Inference tensors from the backbone are valid inputs to normal-mode ops.
            with torch.inference_mode(False), torch.no_grad():
                x = guide.run((backbone_output, action_input), device=backbone_output.backbone_features.device)
            cond = adapter._last_cond
            return BatchFeature(data={"action_pred": x, "backbone_features": cond["vl"], "state_features": cond["state"]})

        self.head.get_action = _get_action
        return self

    def uninstall(self):
        if self._orig_get_action is not None:
            self.head.get_action = self._orig_get_action
            self._orig_get_action = None
