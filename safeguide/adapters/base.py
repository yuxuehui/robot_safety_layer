"""What the Guide needs from a flow-matching policy. Implement one subclass per model family."""
from ..core.cost import ActionMap
from ..core.flow import FlowSpec


class FlowPolicyAdapter:
    flow: FlowSpec  # time convention of the model's flow
    action_horizon: int  # H: control steps per chunk
    action_dim: int  # D
    default_num_steps: int  # Euler steps the model's own sampler uses

    def batch_size(self, observation) -> int:
        raise NotImplementedError

    def sample_noise(self, batch_size, device):
        """(B, H, D) initial noise, drawn exactly like the model's own sampler (keeps paired runs aligned)."""
        raise NotImplementedError

    def prepare(self, observation, device):
        """Everything the velocity field is conditioned on (vision-language prefix / KV cache, state embedding),
        computed once per chunk. Returned object is passed back to velocity()."""
        raise NotImplementedError

    def velocity(self, cond, x_t, t):
        """One velocity-field evaluation: (B, H, D) state x_t at flow time t (B,) -> (B, H, D). Must be
        differentiable w.r.t. x_t only when GuideConfig.grad_through_model is used."""
        raise NotImplementedError

    def expand_cond(self, cond, batch_size):
        """Repeat a batch-1 conditioning `batch_size` times (for batched candidate rollouts)."""
        raise NotImplementedError

    def action_map(self) -> ActionMap:
        """Normalised action chunk -> end-effector motion, for this checkpoint's normalisation and action semantics."""
        raise NotImplementedError

    def install(self, guide):
        """Route the policy's own inference path through guide.run(...). Returns self."""
        raise NotImplementedError

    def uninstall(self):
        raise NotImplementedError
